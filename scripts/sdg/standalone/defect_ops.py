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

"""
Defect operations: shift, tombstone, sideflip via pose_ops.

Tombstone defect uses manual pivot rotation around the component's bottom edge,
computing bounding box to find the long axis and rotating around the fixed end.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import omni.usd
from omni.replicator.core.scripts.functional.modify import TagCache, TransformToken, pose_ops
from pxr import UsdGeom

try:
    from omni.replicator.core.functional import modify as rep_modify
except ImportError:  # Kit --exec init-timing: lazy sys.modules alias not yet set
    from omni.replicator.core.scripts.functional import modify as rep_modify

logger = logging.getLogger(__name__)

_pose_ops_context_warmed = False


def _warm_pose_ops_context() -> None:
    """Force warp to bind to the active CUDA context before the first pose_ops.

    Replicator's pose_ops native binding launches a CUDA kernel for the transform
    apply. On multi-GPU hosts (observed on 4xH100 NVL with --gpus all) the first
    launch can race with warp's lazy CUDA module load and print
    `[CUDA LAUNCH ERROR] Transform ops kernel launch failed: invalid resource handle`
    to stderr. The kernel succeeds on Replicator's internal retry, so output
    counts still match, but the log noise misleads SQA into flagging a failure.

    wp.init() + wp.synchronize_device() forces warp to materialize its CUDA
    stream + module bindings on the active device first, so pose_ops sees a
    valid resource handle on its first launch. Idempotent + cached via the
    module-level flag.
    """
    global _pose_ops_context_warmed
    if _pose_ops_context_warmed:
        return
    try:
        import warp as wp
        wp.init()
        wp.synchronize_device()
        _pose_ops_context_warmed = True
        logger.debug("pose_ops CUDA context warmed via warp.synchronize_device()")
    except Exception as e:
        logger.debug("pose_ops CUDA context warm-up skipped: %s", e)


def _clear_tag_cache() -> None:
    """Clear TagCache private caches for Fabric/USD sync."""
    try:
        TagCache._cache.clear()
        TagCache._path_order_cache.clear()
    except AttributeError:
        logger.warning("TagCache private API changed — cache not cleared")


def _get_prim_and_transform(prim_path: str) -> tuple:
    """Get prim and its original transform as row-major 4x4 numpy array and flat list."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        logger.error("Prim not found: %s", prim_path)
        return None, None, None
    matrix = prim.GetAttribute("xformOp:transform").Get()
    M = np.array([[float(v) for v in row] for row in matrix])
    flat = M.flatten().tolist()
    return prim, flat, M


def _compute_tombstone_transform(
    prim: object,
    M_orig: np.ndarray,
    angle_min: float,
    angle_max: float,
) -> tuple[list[float], dict[str, Any]]:
    """Compute tombstone pivot rotation matrix.

    Finds the long axis via bounding box, picks a random end to lift,
    and rotates around the opposite (fixed) end as pivot.

    Returns:
        final_flat: flat 16-element list of the final 4x4 transform
        params: dict with angle_deg, sign, long_axis for record storage
    """
    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    local_bbox = bbox_cache.ComputeLocalBound(prim)
    bbox_range = local_bbox.GetRange()
    vmin = np.array(bbox_range.GetMin())
    vmax = np.array(bbox_range.GetMax())
    center = (vmin + vmax) / 2.0
    half = (vmax - vmin) / 2.0

    long_axis = int(np.argmax(half))
    axis_name = ["X", "Y", "Z"][long_axis]

    sign = int(np.random.choice([-1, 1]))
    angle_deg = float(np.random.uniform(angle_min, angle_max)) * sign
    angle_rad = np.radians(angle_deg)

    # Pivot point: the end that stays on the board (opposite to lifted end)
    pivot_vec = np.zeros(3)
    pivot_vec[long_axis] = -sign
    pivot_point = center + half * pivot_vec

    # Rotation axis: perpendicular to long axis and board normal (Z)
    # long_axis=X -> rotate around Y; long_axis=Y -> rotate around X; long_axis=Z -> rotate around X
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.eye(4)
    if long_axis == 0:  # X -> rotate around Y
        R[0, 0] = c
        R[0, 2] = -s
        R[2, 0] = s
        R[2, 2] = c
    elif long_axis == 1:  # Y -> rotate around X
        R[1, 1] = c
        R[1, 2] = s
        R[2, 1] = -s
        R[2, 2] = c
    else:  # Z -> rotate around X
        R[1, 1] = c
        R[1, 2] = s
        R[2, 1] = -s
        R[2, 2] = c

    # Pivot rotation: T(-P) @ R @ T(+P) @ M_orig
    T_neg = np.eye(4)
    T_neg[3, 0:3] = -pivot_point
    T_pos = np.eye(4)
    T_pos[3, 0:3] = pivot_point

    M_final = T_neg @ R @ T_pos @ M_orig
    final_flat = M_final.flatten().tolist()

    lifted_end = f"{axis_name}+" if sign > 0 else f"{axis_name}-"
    logger.info("Tombstone: axis=%s, angle=%.1f deg, lifted=%s", axis_name, angle_deg, lifted_end)

    params = {"angle_deg": angle_deg, "sign": sign, "long_axis": long_axis}
    return final_flat, params


def _apply_semantic(prim: object, defect_type: str) -> None:
    """Apply semantic label + clear TagCache for Fabric/USD sync."""
    if not prim.HasAttribute("semantics:labels:defect"):
        rep_modify.semantics(prim, value={"defect": defect_type})
    _clear_tag_cache()


def prepare_defects(
    component_pool: list[str],
    defects_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select components, generate random params, apply defects, and store params for re-apply.

    Returns:
        defect_records: list of dicts with path, defect type, and stored transform params
    """
    _warm_pose_ops_context()
    defect_records: list[dict[str, Any]] = []
    used_paths: set[str] = set()

    for defect_type, cfg in defects_cfg.items():
        if not cfg.get("enabled", False):
            continue

        available = [p for p in component_pool if p not in used_paths]
        # Optional per-defect filter: only consider prim paths containing one of
        # the listed component_types tokens. Used by polarity-sensitive defects
        # that must target a specific subset of the global pool.
        type_filter = cfg.get("component_types")
        if type_filter:
            available = [p for p in available if any(t in p for t in type_filter)]
        if not available:
            logger.info("%s: no eligible components after filter; skipping", defect_type)
            continue
        n = max(1, int(len(component_pool) * cfg["ratio"]))
        n = min(n, len(available))
        selected = np.random.choice(available, size=n, replace=False).tolist()
        used_paths.update(selected)

        for prim_path in selected:
            prim, original_flat, M_orig = _get_prim_and_transform(prim_path)
            if prim is None:
                continue

            _apply_semantic(prim, defect_type)

            record: dict[str, Any] = {
                "path": prim_path,
                "defect": defect_type,
                "original_flat": original_flat,
            }

            if defect_type == "shift":
                t = cfg["translate_range"]
                rz = cfg["rotate_z_range"]
                translate = np.random.uniform([-t, -t, 0], [t, t, 0], size=(1, 3))
                rotate_z = np.random.uniform(-rz, rz, size=1)
                record["translate"] = translate
                record["rotate_z"] = rotate_z
                transform_list = [
                    (TransformToken.TRANSFORM, original_flat),
                    (TransformToken.TRANSLATE, translate),
                    (TransformToken.ROTATE_Z, rotate_z),
                    (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
                ]
                pose_ops([prim], transform_list)

            elif defect_type == "tombstone":
                final_flat, ts_params = _compute_tombstone_transform(
                    prim, M_orig, cfg["angle_min"], cfg["angle_max"]
                )
                record["tombstone_final_flat"] = final_flat
                record["tombstone_params"] = ts_params
                transform_list = [
                    (TransformToken.TRANSFORM, final_flat),
                ]
                pose_ops([prim], transform_list)

            elif defect_type == "sideflip":
                sign = np.random.choice([-1, 1])
                angle = np.random.uniform(cfg["angle_min"], cfg["angle_max"], size=1) * sign
                record["rotate_x"] = angle
                transform_list = [
                    (TransformToken.TRANSFORM, original_flat),
                    (TransformToken.ROTATE_X, angle),
                    (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
                ]
                pose_ops([prim], transform_list)

            elif defect_type == "reverse_polarity":
                # Fixed 180-degree Z rotation around the component centroid.
                # Targets only polarity-sensitive component types via the
                # per-defect component_types filter above.
                angle = np.array([180.0])
                record["rotate_z"] = angle
                transform_list = [
                    (TransformToken.TRANSFORM, original_flat),
                    (TransformToken.ROTATE_Z, angle),
                    (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
                ]
                pose_ops([prim], transform_list)

            defect_records.append(record)

        logger.info("%s: %d components (%.1f%%)", defect_type, n, cfg["ratio"] * 100)

    logger.info("Total defects: %d / %d components", len(defect_records), len(component_pool))
    for d in defect_records[:10]:
        logger.info("  %12s -> %s", d["defect"], d["path"].split("/")[-1])
    if len(defect_records) > 10:
        logger.info("  ... and %d more", len(defect_records) - 10)

    return defect_records


def reapply_defects(defect_records: list[dict[str, Any]]) -> None:
    """Re-apply all defect transforms using stored params (called before each step_async)."""
    _warm_pose_ops_context()
    stage = omni.usd.get_context().get_stage()
    for record in defect_records:
        prim = stage.GetPrimAtPath(record["path"])
        if not prim.IsValid():
            continue

        original_flat = record["original_flat"]
        defect_type = record["defect"]

        if defect_type == "shift":
            transform_list = [
                (TransformToken.TRANSFORM, original_flat),
                (TransformToken.TRANSLATE, record["translate"]),
                (TransformToken.ROTATE_Z, record["rotate_z"]),
                (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
            ]
        elif defect_type == "tombstone":
            transform_list = [
                (TransformToken.TRANSFORM, record["tombstone_final_flat"]),
            ]
        elif defect_type == "sideflip":
            transform_list = [
                (TransformToken.TRANSFORM, original_flat),
                (TransformToken.ROTATE_X, record["rotate_x"]),
                (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
            ]
        elif defect_type == "reverse_polarity":
            transform_list = [
                (TransformToken.TRANSFORM, original_flat),
                (TransformToken.ROTATE_Z, record["rotate_z"]),
                (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
            ]
        else:
            continue

        pose_ops([prim], transform_list)


def restore_defects(defect_records: list[dict[str, Any]]) -> None:
    """Reset each previously-flipped prim back to its original xform.

    Used by the per-trigger re-randomization path in sdg_pipeline.py so
    that triggers don't accumulate flips from prior triggers' selections.
    Walks the previous trigger's records and writes the cached
    ``original_flat`` back as a single TRANSFORM op (mirrors the shape
    reapply_defects writes).
    """
    stage = omni.usd.get_context().get_stage()
    for record in defect_records:
        prim = stage.GetPrimAtPath(record["path"])
        if not prim.IsValid():
            continue
        transform_list = [(TransformToken.TRANSFORM, record["original_flat"])]
        pose_ops([prim], transform_list)


def pretag_defect_eligible(
    component_pool: list[str],
    defects_cfg: dict[str, Any],
) -> int:
    """Pre-tag every defect-eligible prim with its defect semantic at init.

    Used by the ``randomize_defects_per_trigger: true`` code path. Without
    this, ``rep_modify.semantics`` calls made inside the trigger loop
    don't propagate to the colorize semseg annotator (which caches its
    prim -> class map when the render product is set up). Tagging all
    candidate prims BEFORE the first render bakes them into that cache,
    so each trigger's colorize PNG shows every eligible prim coloured as
    its defect class. The actual rotation still varies per trigger via
    pose_ops; the bbox annotator (which doesn't cache the map) remains
    the authoritative per-frame "which prims actually got rotated"
    ground truth — read ``bounding_box_2d_tight_prim_paths_*.json``.

    Returns the number of prims tagged.
    """
    stage = omni.usd.get_context().get_stage()
    tagged = 0
    for defect_type, cfg in defects_cfg.items():
        if not cfg.get("enabled", False):
            continue
        type_filter = cfg.get("component_types")
        eligible = component_pool
        if type_filter:
            eligible = [p for p in eligible if any(t in p for t in type_filter)]
        for prim_path in eligible:
            prim = stage.GetPrimAtPath(prim_path)
            if prim is None or not prim.IsValid():
                continue
            _apply_semantic(prim, defect_type)
            tagged += 1
    logger.info(
        "Pre-tagged %d defect-eligible component(s) "
        "(randomize_defects_per_trigger mode)",
        tagged,
    )
    return tagged
