#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PCB Image Registration using Mutual Information (Grid + Coordinate Descent).

Registers a test image to a reference image by finding the optimal
(scaleX, scaleY, rotation, tx, ty) that maximizes mutual information,
using a coarse-to-fine image pyramid with grid search and coordinate
descent refinement.
"""

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

__all__ = [
    "HAS_CUPY",
    "register",
    "apply_registration",
    "align",
    "crop_to_valid_bbox",
    "save_params",
    "load_params",
    "to_gray",
]


def to_gray(img: NDArray) -> NDArray:
    """Convert image to uint8 grayscale."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def downsample_2x(gray: NDArray) -> NDArray:
    """2x downsample by 2x2 averaging."""
    h, w = gray.shape
    nh, nw = h // 2, w // 2
    return (
        gray[0 : nh * 2 : 2, 0 : nw * 2 : 2].astype(np.uint16)
        + gray[1 : nh * 2 : 2, 0 : nw * 2 : 2]
        + gray[0 : nh * 2 : 2, 1 : nw * 2 : 2]
        + gray[1 : nh * 2 : 2, 1 : nw * 2 : 2]
    ) // 4


def build_pyramid(gray: NDArray, levels: int) -> list[NDArray]:
    """Build image pyramid (returned coarse-to-fine)."""
    pyr = [gray]
    for _ in range(1, levels):
        prev = pyr[-1]
        if prev.shape[0] < 32 or prev.shape[1] < 32:
            break
        pyr.append(downsample_2x(prev).astype(np.uint8))
    pyr.reverse()
    return pyr


def transform_and_sample(
    test_gray: NDArray,
    ref_h: int,
    ref_w: int,
    scale_x: float,
    scale_y: float,
    rot_rad: float,
    tx: float,
    ty: float,
) -> tuple[NDArray, NDArray]:
    """Warp test image into reference frame with bilinear interpolation.

    Returns (warped, mask) both of shape (ref_h, ref_w).
    """
    th, tw = test_gray.shape
    out = np.zeros((ref_h, ref_w), dtype=np.uint8)
    mask = np.zeros((ref_h, ref_w), dtype=np.uint8)

    cx_ref, cy_ref = ref_w / 2.0, ref_h / 2.0
    cx_test, cy_test = tw / 2.0, th / 2.0
    cos_r, sin_r = math.cos(rot_rad), math.sin(rot_rad)

    # Build coordinate grids
    yy, xx = np.mgrid[0:ref_h, 0:ref_w]
    dx = xx - cx_ref
    dy = yy - cy_ref

    xt = scale_x * (cos_r * dx - sin_r * dy) + cx_test + tx
    yt = scale_y * (sin_r * dx + cos_r * dy) + cy_test + ty

    x0 = np.floor(xt).astype(np.int32)
    y0 = np.floor(yt).astype(np.int32)

    valid = (x0 >= 0) & (x0 < tw - 1) & (y0 >= 0) & (y0 < th - 1)

    fx = xt - x0
    fy = yt - y0

    # Bilinear interpolation (only at valid positions)
    x0v = x0[valid]
    y0v = y0[valid]
    fxv = fx[valid]
    fyv = fy[valid]

    idx = y0v * tw + x0v
    flat = test_gray.ravel()
    val = (
        (1 - fxv) * (1 - fyv) * flat[idx]
        + fxv * (1 - fyv) * flat[idx + 1]
        + (1 - fxv) * fyv * flat[idx + tw]
        + fxv * fyv * flat[idx + tw + 1]
    )
    out[valid] = np.round(val).astype(np.uint8)
    mask[valid] = 1

    return out, mask


def compute_mi(img_a: NDArray, img_b: NDArray, mask: NDArray, num_bins: int) -> float:
    """Compute mutual information between two images with coverage weighting."""
    h, w = img_a.shape
    n = h * w
    bin_size = 256.0 / num_bins

    a_bins = np.minimum((img_a[mask == 1].astype(np.float64) / bin_size).astype(int), num_bins - 1)
    b_bins = np.minimum((img_b[mask == 1].astype(np.float64) / bin_size).astype(int), num_bins - 1)
    count = len(a_bins)

    if count < n * 0.05:
        return -math.inf

    # Joint histogram
    joint = np.zeros((num_bins, num_bins), dtype=np.float64)
    np.add.at(joint, (a_bins, b_bins), 1)

    # Marginals
    hist_a = joint.sum(axis=1)
    hist_b = joint.sum(axis=0)

    # MI = sum p(a,b) * log(p(a,b) / (p(a) * p(b)))
    inv = 1.0 / count
    pab = joint * inv
    pa = hist_a * inv
    pb = hist_b * inv

    # Avoid log(0)
    outer = pa[:, None] * pb[None, :]
    valid = (pab > 0) & (outer > 0)
    mi = np.sum(pab[valid] * np.log(pab[valid] / outer[valid]))

    return float(mi * (count / n))


def eval_mi(
    test_gray: NDArray,
    ref_gray: NDArray,
    scale_x: float,
    scale_y: float,
    rot_rad: float,
    tx: float,
    ty: float,
    num_bins: int,
) -> float:
    """Evaluate MI for a given set of transform parameters."""
    ref_h, ref_w = ref_gray.shape
    warped, mask = transform_and_sample(test_gray, ref_h, ref_w, scale_x, scale_y, rot_rad, tx, ty)
    return compute_mi(ref_gray, warped, mask, num_bins)


def _gpu_precompute(ref_gpu, test_gpu, num_bins: int) -> dict:
    """Precompute level-invariant GPU arrays shared across all evaluations."""
    rh, rw = ref_gpu.shape
    th, tw = test_gpu.shape
    bin_size = 256.0 / num_bins

    test_f = test_gpu.astype(cp.float32).ravel()
    ref_bins = cp.minimum(
        (ref_gpu.astype(cp.float32) / bin_size).astype(cp.int32),
        num_bins - 1,
    ).ravel()

    yy, xx = cp.mgrid[0:rh, 0:rw].astype(cp.float32)
    dx = (xx - rw / 2.0).ravel()
    dy = (yy - rh / 2.0).ravel()

    return {
        "rh": rh,
        "rw": rw,
        "th": th,
        "tw": tw,
        "n_pixels": int(rh * rw),
        "bin_size": bin_size,
        "num_bins": num_bins,
        "test_f": test_f,
        "ref_bins": ref_bins,
        "dx": dx,
        "dy": dy,
        "cx_test": tw / 2.0,
        "cy_test": th / 2.0,
    }


def gpu_batch_eval_mi(gpu_ctx: dict, combos: NDArray, chunk_size: int | None = None) -> NDArray:
    """Evaluate MI for N parameter combinations on GPU (cupy).

    combos: shape (N, 5), columns = [sx, sy, rot_rad, tx, ty].
    Returns: (N,) numpy float32 MI values.
    """
    th, tw = gpu_ctx["th"], gpu_ctx["tw"]
    n_pixels = gpu_ctx["n_pixels"]
    num_bins = gpu_ctx["num_bins"]
    bin_size = gpu_ctx["bin_size"]
    test_f = gpu_ctx["test_f"]
    ref_bins = gpu_ctx["ref_bins"]
    dx, dy = gpu_ctx["dx"], gpu_ctx["dy"]
    cx_test, cy_test = gpu_ctx["cx_test"], gpu_ctx["cy_test"]

    combos_gpu = cp.asarray(combos, dtype=cp.float32)
    n_combos = int(combos_gpu.shape[0])

    if chunk_size is None:
        free, _ = cp.cuda.Device().mem_info
        per_combo = n_pixels * 4 * 12 + num_bins * num_bins * 4
        chunk_size = max(1, int(free * 0.25 // per_combo))
        chunk_size = min(chunk_size, n_combos)

    results = cp.empty(n_combos, dtype=cp.float32)
    min_count = n_pixels * 0.05

    for start in range(0, n_combos, chunk_size):
        end = min(start + chunk_size, n_combos)
        b = end - start
        batch = combos_gpu[start:end]
        sx = batch[:, 0:1]
        sy = batch[:, 1:2]
        rot = batch[:, 2:3]
        tx = batch[:, 3:4]
        ty = batch[:, 4:5]
        cos_r = cp.cos(rot)
        sin_r = cp.sin(rot)

        xt = sx * (cos_r * dx - sin_r * dy) + cx_test + tx
        yt = sy * (sin_r * dx + cos_r * dy) + cy_test + ty

        x0 = cp.floor(xt).astype(cp.int32)
        y0 = cp.floor(yt).astype(cp.int32)
        valid = (x0 >= 0) & (x0 < tw - 1) & (y0 >= 0) & (y0 < th - 1)
        fx = xt - x0
        fy = yt - y0

        x0c = cp.clip(x0, 0, tw - 2)
        y0c = cp.clip(y0, 0, th - 2)
        idx00 = y0c * tw + x0c
        val = (
            (1 - fx) * (1 - fy) * test_f[idx00]
            + fx * (1 - fy) * test_f[idx00 + 1]
            + (1 - fx) * fy * test_f[idx00 + tw]
            + fx * fy * test_f[idx00 + tw + 1]
        )

        warped_bins = cp.minimum((val / bin_size).astype(cp.int32), num_bins - 1)
        del xt, yt, x0, y0, fx, fy, x0c, y0c, idx00, val

        global_idx = (
            cp.arange(b, dtype=cp.int32).reshape(b, 1) * (num_bins * num_bins)
            + ref_bins * num_bins
            + warped_bins
        )
        weights = valid.astype(cp.float32).ravel()
        hist = cp.bincount(
            global_idx.ravel(), weights=weights, minlength=b * num_bins * num_bins
        ).reshape(b, num_bins, num_bins)
        del global_idx, warped_bins, weights, valid

        count = hist.sum(axis=(1, 2))
        h_a = hist.sum(axis=2)
        h_b = hist.sum(axis=1)
        count_safe = cp.maximum(count, 1.0)
        pab = hist / count_safe.reshape(b, 1, 1)
        pa = h_a / count_safe.reshape(b, 1)
        pb = h_b / count_safe.reshape(b, 1)
        outer = pa.reshape(b, num_bins, 1) * pb.reshape(b, 1, num_bins)

        safe = (pab > 0) & (outer > 0)
        safe_pab = cp.where(safe, pab, 1.0)
        safe_outer = cp.where(safe, outer, 1.0)
        log_term = cp.where(safe, cp.log(safe_pab / safe_outer), 0.0)
        mi = (pab * log_term).sum(axis=(1, 2))
        mi = mi * (count / n_pixels)
        mi = cp.where(count >= min_count, mi, -cp.inf)

        results[start:end] = mi

    return cp.asnumpy(results)


def grid_search_gpu(
    gpu_ctx: dict,
    sx_vals: NDArray,
    sy_vals: NDArray,
    rot_vals: NDArray,
    tx_vals: NDArray,
    ty_vals: NDArray,
) -> tuple[float, float, float, float, float, float]:
    """GPU grid search: full cartesian product of all 5 parameter vectors."""
    mesh = np.meshgrid(sx_vals, sy_vals, rot_vals, tx_vals, ty_vals, indexing="ij")
    combos = np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)
    n = combos.shape[0]
    print(
        f"  GPU full grid: {len(sx_vals)}sX × {len(sy_vals)}sY × {len(rot_vals)}rot "
        f"× {len(tx_vals)}tx × {len(ty_vals)}ty = {n} evals"
    )

    t0 = time.time()
    mi_vals = gpu_batch_eval_mi(gpu_ctx, combos)
    elapsed = time.time() - t0

    best_idx = int(np.argmax(mi_vals))
    best_mi = float(mi_vals[best_idx])
    bp = combos[best_idx]
    print(
        f"  GPU done in {elapsed:.2f}s  best MI={best_mi:.6f}  "
        f"sX={bp[0]:.4f} sY={bp[1]:.4f} rot={math.degrees(bp[2]):.2f}° "
        f"tx={bp[3]:.1f} ty={bp[4]:.1f}"
    )
    return best_mi, float(bp[0]), float(bp[1]), float(bp[2]), float(bp[3]), float(bp[4])


def grid_search(
    test_gray: NDArray,
    ref_gray: NDArray,
    sx_vals: NDArray,
    sy_vals: NDArray,
    rot_vals: NDArray,
    tx_vals: NDArray,
    ty_vals: NDArray,
    num_bins: int,
) -> tuple[float, float, float, float, float, float]:
    """Two-phase CPU grid search.

    Phase A: search (scaleX, scaleY, rot) with coarse (tx, ty).
    Phase B: fix best scale/rot, search fine (tx, ty).
    """
    # Phase A: coarse shift sampling
    tx_coarse_step = max(1, len(tx_vals) // 11)
    ty_coarse_step = max(1, len(ty_vals) // 11)
    tx_coarse = tx_vals[::tx_coarse_step]
    ty_coarse = ty_vals[::ty_coarse_step]

    total_a = len(sx_vals) * len(sy_vals) * len(rot_vals) * len(tx_coarse) * len(ty_coarse)
    print(
        f"  Phase A: {len(sx_vals)}sX × {len(sy_vals)}sY × {len(rot_vals)}rot "
        f"× {len(tx_coarse)}tx × {len(ty_coarse)}ty = {total_a} evals"
    )

    best_mi = -math.inf
    best_sx = best_sy = 1.0
    best_rot = best_tx = best_ty = 0.0
    count = 0

    for sx in sx_vals:
        for sy in sy_vals:
            for rot in rot_vals:
                for txv in tx_coarse:
                    for tyv in ty_coarse:
                        mi = eval_mi(test_gray, ref_gray, sx, sy, rot, txv, tyv, num_bins)
                        if mi > best_mi:
                            best_mi = mi
                            best_sx, best_sy, best_rot = sx, sy, rot
                            best_tx, best_ty = txv, tyv
                        count += 1
        if count % max(1, total_a // 5) < len(sy_vals) * len(rot_vals) * len(tx_coarse) * len(
            ty_coarse
        ):
            print(f"    A: {count}/{total_a}  MI={best_mi:.6f}")

    print(
        f"  Phase A best: MI={best_mi:.6f}  sX={best_sx:.4f} sY={best_sy:.4f} "
        f"rot={math.degrees(best_rot):.2f}° tx={best_tx:.1f} ty={best_ty:.1f}"
    )

    # Phase B: fine shift search with best scale/rot
    total_b = len(tx_vals) * len(ty_vals)
    print(f"  Phase B: {len(tx_vals)}tx × {len(ty_vals)}ty = {total_b} evals")

    for txv in tx_vals:
        for tyv in ty_vals:
            mi = eval_mi(test_gray, ref_gray, best_sx, best_sy, best_rot, txv, tyv, num_bins)
            if mi > best_mi:
                best_mi = mi
                best_tx, best_ty = txv, tyv

    print(f"  Phase B best: MI={best_mi:.6f}  tx={best_tx:.1f} ty={best_ty:.1f}")

    return best_mi, best_sx, best_sy, best_rot, best_tx, best_ty


def coordinate_descent(
    test_gray: NDArray,
    ref_gray: NDArray,
    sx: float,
    sy: float,
    rot_rad: float,
    tx: float,
    ty: float,
    num_bins: int,
    step_sx: float,
    step_sy: float,
    step_r: float,
    step_t: float,
    max_iter: int = 200,
    verbose: bool = True,
    gpu_ctx: dict | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Coordinate descent refinement of registration parameters."""

    def _eval(sx_, sy_, rot_, tx_, ty_):
        if gpu_ctx is not None:
            combo = np.array([[sx_, sy_, rot_, tx_, ty_]], dtype=np.float32)
            return float(gpu_batch_eval_mi(gpu_ctx, combo, chunk_size=1)[0])
        return eval_mi(test_gray, ref_gray, sx_, sy_, rot_, tx_, ty_, num_bins)

    best_mi = _eval(sx, sy, rot_rad, tx, ty)

    for iteration in range(max_iter):
        improved = False
        params = [
            ("sX", lambda: sx, None, step_sx),
            ("sY", lambda: sy, None, step_sy),
            ("rot", lambda: rot_rad, None, step_r),
            ("tx", lambda: tx, None, step_t),
            ("ty", lambda: ty, None, step_t),
        ]

        for name, getter, _, step in params:
            orig = getter()
            for direction in [1, -1]:
                # Set the parameter
                new_val = orig + direction * step
                if name == "sX":
                    sx = new_val
                elif name == "sY":
                    sy = new_val
                elif name == "rot":
                    rot_rad = new_val
                elif name == "tx":
                    tx = new_val
                elif name == "ty":
                    ty = new_val

                mi = _eval(sx, sy, rot_rad, tx, ty)
                if mi > best_mi + 1e-8:
                    best_mi = mi
                    improved = True
                    break
                else:
                    # Restore
                    if name == "sX":
                        sx = orig
                    elif name == "sY":
                        sy = orig
                    elif name == "rot":
                        rot_rad = orig
                    elif name == "tx":
                        tx = orig
                    elif name == "ty":
                        ty = orig

        if not improved:
            step_sx *= 0.5
            step_sy *= 0.5
            step_r *= 0.5
            step_t *= 0.5
            if step_sx < 0.001 and step_sy < 0.001 and step_r < 0.0002 and step_t < 0.25:
                if verbose:
                    print(f"    Converged at iteration {iteration}")
                break

        if verbose and iteration % 10 == 0:
            print(
                f"    refine #{iteration}  MI={best_mi:.6f}  "
                f"sX={sx:.4f} sY={sy:.4f} rot={math.degrees(rot_rad):.2f}° "
                f"tx={tx:.1f} ty={ty:.1f}"
            )

    return best_mi, sx, sy, rot_rad, tx, ty


def register(
    ref_gray: NDArray,
    test_gray: NDArray,
    sx_min: float = 0.9,
    sx_max: float = 1.1,
    sx_step: float = 0.02,
    sy_min: float = 0.9,
    sy_max: float = 1.1,
    sy_step: float = 0.02,
    rot_min_deg: float = -1.0,
    rot_max_deg: float = 1.0,
    rot_step_deg: float = 0.05,
    shift_range: int = 200,
    shift_step: int = 4,
    num_bins: int = 64,
    pyr_levels: int = 3,
    use_gpu: bool = False,
) -> dict:
    """Run full registration pipeline: pyramid + grid search + coordinate descent.

    Returns dict with keys: scaleX, scaleY, rotation_deg, tx, ty, mi_before, mi_after.
    """
    DEG2RAD = math.pi / 180.0

    use_gpu = use_gpu and HAS_CUPY
    if use_gpu:
        dev = cp.cuda.Device()
        attrs = cp.cuda.runtime.getDeviceProperties(dev.id)
        dev_name = attrs["name"].decode() if isinstance(attrs["name"], bytes) else attrs["name"]
        print(f"GPU acceleration enabled (cupy on {dev_name})")

    print("Building image pyramids...")
    ref_pyr = build_pyramid(ref_gray, pyr_levels)
    test_pyr = build_pyramid(test_gray, pyr_levels)
    total_levels = len(ref_pyr)
    print(
        f"Pyramid: {total_levels} levels — "
        + ", ".join(f"{p.shape[1]}×{p.shape[0]}" for p in ref_pyr)
    )

    cur_sx, cur_sy = 1.0, 1.0
    cur_rot = 0.0
    cur_tx, cur_ty = 0.0, 0.0
    best_mi = -math.inf

    for lvl in range(total_levels):
        ref_lvl = ref_pyr[lvl]
        test_lvl = test_pyr[lvl]
        rh, rw = ref_lvl.shape
        level_factor = 2 ** (total_levels - 1 - lvl)

        print(f"\n--- Level {lvl + 1}/{total_levels} ({rw}×{rh}) ---")

        gpu_ctx = None
        if use_gpu:
            gpu_ctx = _gpu_precompute(cp.asarray(ref_lvl), cp.asarray(test_lvl), num_bins)

        if lvl == 0:
            # Build parameter grids
            sr = int(math.ceil(shift_range / level_factor))
            ss = max(1, int(math.ceil(shift_step / level_factor * 2)))

            sx_vals = np.arange(sx_min, sx_max + 0.001, sx_step * 2)
            sy_vals = np.arange(sy_min, sy_max + 0.001, sy_step * 2)
            rot_vals = np.arange(rot_min_deg, rot_max_deg + 0.001, rot_step_deg) * DEG2RAD
            tx_vals = np.arange(-sr, sr + 1, ss, dtype=np.float64)
            ty_vals = np.arange(-sr, sr + 1, ss, dtype=np.float64)

            if use_gpu:
                best_mi, cur_sx, cur_sy, cur_rot, cur_tx, cur_ty = grid_search_gpu(
                    gpu_ctx,
                    sx_vals,
                    sy_vals,
                    rot_vals,
                    tx_vals,
                    ty_vals,
                )
            else:
                best_mi, cur_sx, cur_sy, cur_rot, cur_tx, cur_ty = grid_search(
                    test_lvl,
                    ref_lvl,
                    sx_vals,
                    sy_vals,
                    rot_vals,
                    tx_vals,
                    ty_vals,
                    num_bins,
                )
        else:
            # Scale up shifts from previous level
            cur_tx *= 2
            cur_ty *= 2
            print(f"  Scale-up from previous level: tx={cur_tx:.1f} ty={cur_ty:.1f}")

        # Coordinate descent refinement
        step_sx = sx_step if lvl == 0 else sx_step / 2
        step_sy = sy_step if lvl == 0 else sy_step / 2
        step_r = (rot_step_deg * DEG2RAD) if lvl == 0 else (rot_step_deg * DEG2RAD / 2)
        step_t = max(1, int(math.ceil(shift_step / level_factor))) if lvl == 0 else 2.0

        best_mi, cur_sx, cur_sy, cur_rot, cur_tx, cur_ty = coordinate_descent(
            test_lvl,
            ref_lvl,
            cur_sx,
            cur_sy,
            cur_rot,
            cur_tx,
            cur_ty,
            num_bins,
            step_sx,
            step_sy,
            step_r,
            step_t,
            gpu_ctx=gpu_ctx,
        )

        print(
            f"  Level {lvl + 1} done — MI={best_mi:.6f}  "
            f"sX={cur_sx:.4f} sY={cur_sy:.4f} rot={math.degrees(cur_rot):.2f}° "
            f"tx={cur_tx:.1f} ty={cur_ty:.1f}"
        )

    # Compute initial MI for comparison
    mi_before = eval_mi(test_gray, ref_gray, 1.0, 1.0, 0, 0, 0, num_bins)
    rot_deg = math.degrees(cur_rot)

    result = {
        "scaleX": cur_sx,
        "scaleY": cur_sy,
        "rotation_deg": rot_deg,
        "tx": cur_tx,
        "ty": cur_ty,
        "mi_before": mi_before,
        "mi_after": best_mi,
    }

    improvement = (best_mi - mi_before) / abs(mi_before or 1) * 100
    print(f"\n{'=' * 40}")
    print(f"Scale X:  {cur_sx:.4f}")
    print(f"Scale Y:  {cur_sy:.4f}")
    print(f"Rotation: {rot_deg:.2f}°")
    print(f"Shift X:  {cur_tx:.2f} px")
    print(f"Shift Y:  {cur_ty:.2f} px")
    print(f"MI before: {mi_before:.6f}")
    print(f"MI after:  {best_mi:.6f}")
    print(f"Improvement: {improvement:.1f}%")

    return result


def apply_registration(
    test_img: NDArray, ref_shape: tuple[int, int], params: dict
) -> tuple[NDArray, NDArray]:
    """Apply registration transform to warp the test image into the ref frame.

    Preserves color if the input has 3 channels (warps each channel). Returns
    ``(warped, mask)`` where ``mask`` marks the geometrically valid region
    (1 = inside test image bounds, 0 = out of bounds).
    """
    ref_h, ref_w = ref_shape
    sx = params["scaleX"]
    sy = params["scaleY"]
    rot = math.radians(params["rotation_deg"])
    tx = params["tx"]
    ty = params["ty"]

    if test_img.ndim == 2:
        warped, mask = transform_and_sample(test_img, ref_h, ref_w, sx, sy, rot, tx, ty)
        return warped, mask

    channels = []
    mask = None
    for c in range(test_img.shape[2]):
        w, m = transform_and_sample(test_img[..., c], ref_h, ref_w, sx, sy, rot, tx, ty)
        channels.append(w)
        if mask is None:
            mask = m
    warped = np.stack(channels, axis=-1)
    return warped, mask


def crop_to_valid_bbox(
    ref_img: NDArray,
    warped: NDArray,
    mask: NDArray,
    margin: int = 0,
) -> tuple[NDArray, NDArray, tuple[int, int, int, int]]:
    """Crop ``ref_img`` and ``warped`` to the bbox where ``mask`` is valid.

    Both outputs share the same H×W; ``ref_img`` is only cropped (no warp,
    no resize) so component positions still match ``warped`` 1:1. Returns
    ``(ref_cropped, warped_cropped, bbox)`` with ``bbox = (x0, y0, x1, y1)``.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Registration mask has no valid pixels; cannot crop.")
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    return ref_img[y0:y1, x0:x1], warped[y0:y1, x0:x1], (x0, y0, x1, y1)


def align(
    ref: NDArray | str | Path,
    test: NDArray | str | Path,
    *,
    no_resize: bool = False,
    use_gpu: bool | None = None,
    sx_range: tuple[float, float, float] = (0.9, 1.1, 0.02),
    sy_range: tuple[float, float, float] = (0.9, 1.1, 0.02),
    rot_range_deg: tuple[float, float, float] = (-1.0, 1.0, 0.05),
    shift_range: int = 200,
    shift_step: int = 4,
    num_bins: int = 64,
    pyr_levels: int = 3,
) -> tuple[NDArray, dict]:
    """High-level entry point for a workflow.

    Accepts either numpy arrays or file paths for ``ref`` and ``test``.
    Returns ``(warped, mask, params)`` where ``warped`` is the test image
    warped into the reference frame, ``mask`` marks the geometrically valid
    overlap region, and ``params`` holds the 5-DOF affine + MI before/after.

    ``use_gpu=None`` auto-detects cupy; pass True/False to force.
    """
    ref_img = _load_image(ref)
    test_img = _load_image(test)

    ref_gray = to_gray(ref_img)
    test_gray = to_gray(test_img)

    rh, rw = ref_gray.shape
    th, tw = test_gray.shape
    if not no_resize and (rw, rh) != (tw, th):
        test_gray = cv2.resize(test_gray, (rw, rh), interpolation=cv2.INTER_LINEAR)
        test_for_apply = cv2.resize(test_img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    else:
        test_for_apply = test_img

    resolved_gpu = HAS_CUPY if use_gpu is None else (use_gpu and HAS_CUPY)

    params = register(
        ref_gray,
        test_gray,
        sx_min=sx_range[0],
        sx_max=sx_range[1],
        sx_step=sx_range[2],
        sy_min=sy_range[0],
        sy_max=sy_range[1],
        sy_step=sy_range[2],
        rot_min_deg=rot_range_deg[0],
        rot_max_deg=rot_range_deg[1],
        rot_step_deg=rot_range_deg[2],
        shift_range=shift_range,
        shift_step=shift_step,
        num_bins=num_bins,
        pyr_levels=pyr_levels,
        use_gpu=resolved_gpu,
    )
    warped, mask = apply_registration(test_for_apply, ref_gray.shape, params)
    return warped, mask, params


def _load_image(src: NDArray | str | Path) -> NDArray:
    if isinstance(src, (str, Path)):
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {src}")
        return img
    return src


def save_params(params: dict, path: str | Path) -> None:
    """Persist registration parameters to a JSON file."""
    serializable = {
        k: (float(v) if isinstance(v, (np.floating, float, int)) else v) for k, v in params.items()
    }
    Path(path).write_text(json.dumps(serializable, indent=2))


def load_params(path: str | Path) -> dict:
    """Load registration parameters previously saved with :func:`save_params`."""
    return json.loads(Path(path).read_text())


def save_blink_gif(
    ref: NDArray, aligned: NDArray, out_path: str | Path, duration_ms: int = 500
) -> None:
    """Write a 2-frame blink GIF alternating ``ref`` and ``aligned``.

    Both inputs must be the same H×W; accepts gray or color (BGR/BGRA).
    """
    from PIL import Image

    def _label(img: NDArray, text: str) -> NDArray:
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            bgr = img.copy()
        cv2.rectangle(bgr, (0, 0), (180, 24), (0, 0, 0), -1)
        cv2.putText(
            bgr, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA
        )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    frames = [
        Image.fromarray(_label(ref, "REFERENCE")),
        Image.fromarray(_label(aligned, "ALIGNED")),
    ]
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
