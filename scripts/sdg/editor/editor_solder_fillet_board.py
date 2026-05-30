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

"""
Solder Fillet Board Placer — Script Editor script.

Finds all _0603_H100 capacitor instances in camera view and places two
solder fillet meshes (one per pad end) for each.

UI controls: shape, noise, material, camera path.
Buttons: "Generate Fillets" (places/regenerates) | "Clear" (removes all).

Paste into Omniverse Composer Script Editor with temp_scene.usd open.
"""

import os

import numpy as np
import omni.ui as ui
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

# ── Fractal noise ─────────────────────────────────────────────────────────────


def _fractal_noise_2d(x, y, scale, octaves, seed=42):
    noise = np.zeros_like(x, dtype=float)
    amplitude = 1.0
    frequency = float(scale)
    np.random.seed(seed)
    for _ in range(max(1, int(octaves))):
        px = np.random.random() * 2 * np.pi
        py = np.random.random() * 2 * np.pi
        ang = np.random.random() * 2 * np.pi
        xr = x * np.cos(ang) - y * np.sin(ang)
        yr = x * np.sin(ang) + y * np.cos(ang)
        noise += amplitude * (
            np.sin(2 * np.pi * frequency * xr + px) * np.cos(2 * np.pi * frequency * yr + py)
        )
        amplitude *= 0.5
        frequency *= 2.0
    return noise


# ── Warp GPU acceleration ────────────────────────────────────────────────────
# The fillet's per-grid-point profile (sigmoid plateau × exp decay × fractal
# noise) is embarrassingly parallel — each (i, j) is independent. Move the
# whole grid kernel onto cuda:0 via warp; falls back to NumPy when warp/cuda
# isn't available, so the function is safe to call on CPU-only test rigs.
#
# Set FILLET_BACKEND=cpu in the environment to force the NumPy fallback path
# (useful for debugging or perf comparison).

try:  # warp is bundled with Kit; outside Kit it may be missing.
    import warp as _wp

    _WP_OK = True
except Exception:  # noqa: BLE001
    _wp = None
    _WP_OK = False

_WP_MAX_OCTAVES = 16  # bumped from PARAMS.noise_octaves default of 4; covers any
# reasonable user setting without dynamic shape compilation
_WP_INIT_DONE = False


def _wp_init() -> bool:
    """Initialize warp once. Returns True if a CUDA device is available; False
    otherwise (caller should fall back to NumPy)."""
    global _WP_INIT_DONE
    if not _WP_OK:
        return False
    if _WP_INIT_DONE:
        return True
    try:
        _wp.init()
        # If there's no CUDA device, get_preferred_device returns "cpu" — in
        # that case there's no point routing through warp (its overhead beats
        # NumPy on CPU-bound problems this small).
        dev = str(_wp.get_preferred_device())
        if not dev.startswith("cuda"):
            return False
        _WP_INIT_DONE = True
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[Fillet/Warp] init failed, falling back to CPU: {exc!r}")
        return False


if _WP_OK:

    @_wp.kernel
    def _wp_fillet_z_kernel(
        z_out: _wp.array2d(dtype=_wp.float32),  # noqa: F821 — annotated by warp
        long_e: _wp.float32,
        short: _wp.float32,
        x_len: _wp.float32,
        decay: _wp.float32,
        smooth_y: _wp.float32,
        plat_start: _wp.float32,
        plat_end: _wp.float32,
        bump: _wp.float32,
        noise_amp: _wp.float32,
        n_octaves: _wp.int32,
        noise_freq0: _wp.float32,
        rng_px: _wp.array(dtype=_wp.float32),  # noqa: F821 — len = _WP_MAX_OCTAVES
        rng_py: _wp.array(dtype=_wp.float32),
        rng_ang: _wp.array(dtype=_wp.float32),
        r: _wp.int32,
    ):
        """One thread per grid cell. Writes z (height) into z_out[i, j].

        Mirrors the NumPy expression in :func:`_compute_geometry`:
          xn = x / long_e
          yn = y / short
          dx = exp(-decay * xn)
          lr = sigmoid(smooth_y * (yn - plat_start))
          rf = sigmoid(smooth_y * (yn - plat_end))
          delta = long_e * bump * dx * lr * (1 - rf)
          noise = sum_octaves( amp_k · sin(2πf_k·xr+px) · cos(2πf_k·yr+py) )
          z = delta + noise_amp * noise
        """
        i, j = _wp.tid()
        if i >= r or j >= r:
            return

        # Linear x/y across the grid: x in [0, x_len], y in [0, short].
        # The Python code uses ``np.linspace(0, x_len, r)`` — last sample is
        # exactly x_len, so spacing is x_len / (r - 1) when r >= 2.
        denom = _wp.float32(_wp.max(r - 1, 1))
        x = _wp.float32(j) * (x_len / denom)
        y = _wp.float32(i) * (short / denom)

        xn = x / long_e
        yn = y / short

        dx = _wp.exp(-decay * xn)
        lr = _wp.float32(1.0) / (_wp.float32(1.0) + _wp.exp(-smooth_y * (yn - plat_start)))
        rf = _wp.float32(1.0) / (_wp.float32(1.0) + _wp.exp(-smooth_y * (yn - plat_end)))
        delta = long_e * bump * dx * lr * (_wp.float32(1.0) - rf)

        # Fractal noise (mirror of _fractal_noise_2d): per-octave rotation +
        # phase offsets are sampled CPU-side and passed in; here we only
        # consume them deterministically.
        noise = _wp.float32(0.0)
        amp = _wp.float32(1.0)
        freq = noise_freq0
        TWO_PI = _wp.float32(6.2831853)
        for k in range(n_octaves):
            ang = rng_ang[k]
            px = rng_px[k]
            py = rng_py[k]
            xr = xn * _wp.cos(ang) - yn * _wp.sin(ang)
            yr = xn * _wp.sin(ang) + yn * _wp.cos(ang)
            noise += amp * (_wp.sin(TWO_PI * freq * xr + px) * _wp.cos(TWO_PI * freq * yr + py))
            amp = amp * _wp.float32(0.5)
            freq = freq * _wp.float32(2.0)

        z_out[i, j] = delta + noise_amp * noise


def _compute_z_warp(p: "FilletParams") -> "np.ndarray | None":
    """GPU path. Returns the (r, r) z-grid as a NumPy array, or None if warp
    isn't usable (caller should fall back to NumPy)."""
    if not _wp_init():
        return None

    r = max(4, int(p.resolution))
    long_e = float(p.long_edge)
    short = long_e * 0.6
    x_len = long_e * float(p.x_scale)
    n_oct = max(1, min(_WP_MAX_OCTAVES, int(p.noise_octaves)))

    # Reproduce CPU side's fractal-noise RNG draw order so warp output bit
    # matches the NumPy reference (same seed → same shape).
    np.random.seed(42)
    rng_px = np.zeros(_WP_MAX_OCTAVES, dtype=np.float32)
    rng_py = np.zeros(_WP_MAX_OCTAVES, dtype=np.float32)
    rng_ang = np.zeros(_WP_MAX_OCTAVES, dtype=np.float32)
    for k in range(n_oct):
        rng_px[k] = float(np.random.random() * 2 * np.pi)
        rng_py[k] = float(np.random.random() * 2 * np.pi)
        rng_ang[k] = float(np.random.random() * 2 * np.pi)

    try:
        z_dev = _wp.zeros((r, r), dtype=_wp.float32, device="cuda:0")
        rng_px_d = _wp.array(rng_px, dtype=_wp.float32, device="cuda:0")
        rng_py_d = _wp.array(rng_py, dtype=_wp.float32, device="cuda:0")
        rng_ang_d = _wp.array(rng_ang, dtype=_wp.float32, device="cuda:0")
        _wp.launch(
            kernel=_wp_fillet_z_kernel,
            dim=(r, r),
            inputs=[
                z_dev,
                np.float32(long_e),
                np.float32(short),
                np.float32(x_len),
                np.float32(p.decay),
                np.float32(p.smooth_y),
                np.float32(p.plat_start),
                np.float32(p.plat_end),
                np.float32(p.bump),
                np.float32(p.noise_amp),
                np.int32(n_oct),
                np.float32(p.noise_scale),
                rng_px_d,
                rng_py_d,
                rng_ang_d,
                np.int32(r),
            ],
            device="cuda:0",
        )
        z_host = z_dev.numpy().astype(np.float64, copy=False)
        return z_host
    except Exception as exc:  # noqa: BLE001
        print(f"[Fillet/Warp] launch failed, falling back to CPU: {exc!r}")
        return None


# ── Parameters ────────────────────────────────────────────────────────────────


class FilletParams:
    # Shape (map to editor_solder_fillet.py fields)
    long_edge = 9.0  # long_edge_length
    x_scale = 0.2  # x_length_scale  → x_length = long_edge * x_scale
    plat_start = 0.2  # platform_start
    plat_end = 0.8  # platform_end
    # Defaults match the production reference run's tuned single-shot
    # values (board_0505 fixed-camera config). YAML pipeline pulls
    # ±10 % randomization ranges from ``randomize_fillet:`` — the
    # editor stays at the midpoint scalars below.
    smooth_y = 6.44  # smoothness_y
    bump = 0.38  # delta_max_scale
    decay = 10.8  # delta_x_decay
    # Noise
    noise_scale = 4.0
    noise_octaves = 4.0
    noise_amp = 0.01  # absolute amplitude (NOT scaled by long_edge)
    resolution = 20.0
    # USD scale applied to every fillet prim (xformOp:transform bakes this in)
    fillet_sx = 0.14
    fillet_sy = 0.14
    fillet_sz = 0.24
    # Material (UsdPreviewSurface). mat_color is the baseColor shown on
    # every fillet prim — defaults to mid-grey silver (0.85) to mimic
    # real solder. Override via yaml ``solder_fillet.mat_color`` /
    # editor "Fillet color" slider for vantablack-style black fillets
    # (e.g. when using a fully black PCBA reference).
    mat_color = (0.85, 0.85, 0.85)
    mat_metallic = 1.0
    mat_roughness = 0.2
    # Position fine-tune
    inward_offset_mm = 2.6  # absolute inward offset (mm); used when adaptive_scale=False
    inward_frac = 0.15  # fraction of each component's half-length used as inward offset
    # when adaptive_scale=True (replaces inward_offset_mm per-component)
    z_offset_mm = 10.0  # move both fillets upward (along Z, mm)
    # Adaptive sizing: scale fillet sx/sy to match each component's world-space bbox width.
    # Also applies inward_frac instead of inward_offset_mm.
    # When True, fillet_sx/sy from PARAMS are ignored; sz and z_offset_mm are still used.
    # Default is False so the editor's "Scale X/Y" and "Inward (mm)" sliders take
    # effect directly. The pipeline overrides this to True via yaml (see
    # sdg_pipeline.py: `_sf_cfg.get("adaptive_scale", True)`), so production
    # behaviour is unchanged.
    adaptive_scale = False
    # Scene — must match the PCBA assembly being scanned (top side)
    camera_path = "/World/camera_light/Camera"  # capital C matches config
    pcba_root = "/World/pcba_main_s_detail/PCBA/tn__60014242BASEA04_fM9E"
    comp_types = ["_0603_H100"]  # component type substrings to match; overridden by pipeline


PARAMS = FilletParams()

FILLET_ROOT = "/World/SolderFillets"
MAT_PATH = "/World/SolderFilletMat"
COMP_TYPE = "_0603_H100"


# ── Mesh geometry ─────────────────────────────────────────────────────────────

# Cache for the per-call invariants: when ``FilletParams`` doesn't change
# between successive calls (the common case — every fillet placed in one
# ``generate_fillets()`` pass uses the same global PARAMS), reuse the
# previous geometry instead of recomputing the grid + Vt arrays.
_GEOM_CACHE: "dict[tuple, tuple]" = {}
_GEOM_CACHE_MAX = 4


def _params_cache_key(p: "FilletParams") -> tuple:
    return (
        float(p.long_edge),
        float(p.x_scale),
        float(p.plat_start),
        float(p.plat_end),
        float(p.smooth_y),
        float(p.bump),
        float(p.decay),
        float(p.noise_scale),
        int(p.noise_octaves),
        float(p.noise_amp),
        int(p.resolution),
    )


def _compute_geometry(p: FilletParams, backend: "str | None" = None, use_cache: bool = True):
    """Compute the per-grid-point fillet profile and triangulation.

    ``backend`` selects the math kernel:
      * ``None`` (default) — pick warp+CUDA if available, else NumPy.
        Override globally via env var ``FILLET_BACKEND=cpu|warp``.
      * ``"cpu"`` — force the NumPy fallback path.
      * ``"warp"`` — force the warp path; if warp/CUDA isn't usable,
        falls back to NumPy and prints a one-line warning.

    ``use_cache`` (default True): when set, identical successive calls
    skip the recompute entirely. The cache is keyed on every
    geometry-affecting param (long_edge / x_scale / plat_start / plat_end
    / smooth_y / bump / decay / noise_* / resolution), so as soon as any
    of them change the cache invalidates correctly.
    """
    if backend is None:
        backend = os.environ.get("FILLET_BACKEND", "auto").lower()

    key = _params_cache_key(p) if use_cache else None
    if key is not None and key in _GEOM_CACHE:
        return _GEOM_CACHE[key]

    long_e = float(p.long_edge)
    short = long_e * 0.6
    x_len = long_e * float(p.x_scale)
    r = max(4, int(p.resolution))

    # Build the (r, r) X / Y grids once on CPU — used both for the
    # ``pts`` array (regardless of compute backend) and for the NumPy
    # fallback. Indexing convention matches np.meshgrid(..., indexing='xy'),
    # i.e. X[i, j] varies with j (column), Y[i, j] varies with i (row),
    # to keep the warp kernel addressing scheme consistent.
    x_lin = np.linspace(0.0, x_len, r, dtype=np.float64)
    y_lin = np.linspace(0.0, short, r, dtype=np.float64)
    X, Y = np.meshgrid(x_lin, y_lin)

    z = None
    if backend in ("auto", "warp"):
        z = _compute_z_warp(p)
        if z is None and backend == "warp":
            print("[Fillet/Warp] warp unavailable — using NumPy")

    if z is None:
        # CPU fallback (the original NumPy implementation).
        xn = X / long_e
        yn = Y / short
        dx = np.exp(-float(p.decay) * xn)
        lr = 1.0 / (1.0 + np.exp(-float(p.smooth_y) * (yn - float(p.plat_start))))
        rf = 1.0 / (1.0 + np.exp(-float(p.smooth_y) * (yn - float(p.plat_end))))
        dy = lr * (1.0 - rf)
        delta = long_e * float(p.bump) * dx * dy
        noise = _fractal_noise_2d(xn, yn, p.noise_scale, p.noise_octaves)
        z = delta + float(p.noise_amp) * noise

    # Bulk-build a Vt.Vec3fArray straight from the (r, r) X/Y/z grids —
    # an order-of-magnitude faster than a Python list comp constructing
    # r*r Gf.Vec3f objects one at a time.
    flat = np.empty((r * r, 3), dtype=np.float32)
    flat[:, 0] = X.reshape(-1).astype(np.float32, copy=False)
    flat[:, 1] = Y.reshape(-1).astype(np.float32, copy=False)
    flat[:, 2] = z.reshape(-1).astype(np.float32, copy=False)
    try:
        from pxr import Vt

        pts = Vt.Vec3fArray.FromNumpy(flat)
    except Exception:  # noqa: BLE001
        # Older USD without FromNumpy — fall back to list comp.
        pts = [
            Gf.Vec3f(float(flat[k, 0]), float(flat[k, 1]), float(flat[k, 2]))
            for k in range(flat.shape[0])
        ]

    # Triangulation indices: for each (r-1) × (r-1) quad, two triangles.
    # Build with vectorised arange + reshape in O(quads), no Python loop.
    n_quads = (r - 1) * (r - 1)
    qi, qj = np.meshgrid(np.arange(r - 1), np.arange(r - 1), indexing="ij")
    a = (qi * r + qj).reshape(-1)
    b = (qi * r + qj + 1).reshape(-1)
    c = ((qi + 1) * r + qj).reshape(-1)
    d = ((qi + 1) * r + qj + 1).reshape(-1)
    idx = np.empty(n_quads * 6, dtype=np.int32)
    idx[0::6], idx[1::6], idx[2::6] = a, b, c
    idx[3::6], idx[4::6], idx[5::6] = b, d, c
    idx = idx.tolist()
    counts = [3] * (n_quads * 2)

    out = (pts, counts, idx)
    if key is not None:
        if len(_GEOM_CACHE) >= _GEOM_CACHE_MAX:
            _GEOM_CACHE.pop(next(iter(_GEOM_CACHE)))
        _GEOM_CACHE[key] = out
    return out


# ── Material ──────────────────────────────────────────────────────────────────

_shader_prim = [None]  # mutable ref so callbacks can update it


def _ensure_material(stage, p: FilletParams):
    base_rgb = (
        Gf.Vec3f(*[float(x) for x in p.mat_color])
        if not isinstance(p.mat_color, Gf.Vec3f)
        else p.mat_color
    )
    existing = stage.GetPrimAtPath(MAT_PATH)
    if existing.IsValid() and _shader_prim[0] is not None:
        s = _shader_prim[0]
        s.GetInput("metallic").Set(float(p.mat_metallic))
        s.GetInput("roughness").Set(float(p.mat_roughness))
        # baseColor input may not exist if first call set it; guard.
        bc = s.GetInput("baseColor")
        if bc:
            bc.Set(base_rgb)
        else:
            s.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set(base_rgb)
        return UsdShade.Material(existing)

    mat = UsdShade.Material.Define(stage, MAT_PATH)
    shader = UsdShade.Shader.Define(stage, f"{MAT_PATH}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(p.mat_metallic))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(p.mat_roughness))
    shader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set(base_rgb)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _shader_prim[0] = shader
    return mat


# ── Single fillet mesh ────────────────────────────────────────────────────────


def _write_mesh(stage, path, p: FilletParams, mat):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        mesh = UsdGeom.Mesh.Define(stage, path)
        UsdShade.MaterialBindingAPI(mesh).Bind(mat)
    else:
        mesh = UsdGeom.Mesh(prim)
    pts, counts, idx = _compute_geometry(p)
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(idx)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return mesh


def _set_transform(stage, path, mat4: Gf.Matrix4d):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(mat4)


def _fillet_world_matrix(
    origin: Gf.Vec3d,
    x_dir: Gf.Vec3d,  # fillet local X in world space (unit vector)
    y_dir: Gf.Vec3d,  # fillet local Y in world space (unit vector)
    z_dir: Gf.Vec3d,  # fillet local Z in world space (unit vector)
    sx: float,
    sy: float,
    sz: float,
) -> Gf.Matrix4d:
    """Row-vector convention: p_world = p_local * M."""
    return Gf.Matrix4d(
        x_dir[0] * sx,
        x_dir[1] * sx,
        x_dir[2] * sx,
        0.0,
        y_dir[0] * sy,
        y_dir[1] * sy,
        y_dir[2] * sy,
        0.0,
        z_dir[0] * sz,
        z_dir[1] * sz,
        z_dir[2] * sz,
        0.0,
        origin[0],
        origin[1],
        origin[2],
        1.0,
    )


# ── Camera frustum ────────────────────────────────────────────────────────────


def _mm_to_scene(stage, mm: float) -> float:
    """Convert physical mm to scene units using stage metersPerUnit."""
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    return mm * 0.001 / mpu


def _get_frustum(stage, camera_path):
    """
    Return GfFrustum in world space (scene units), or None.
    Compatible with pxr v0.25 / USD 21.x where GfCamera.GetFrustum() may not exist.
    """
    import math as _math

    cam_prim = stage.GetPrimAtPath(camera_path)
    if not cam_prim.IsValid():
        print(f"[Fillet] Camera not found: {camera_path} — including all instances.")
        return None

    cam_schema = UsdGeom.Camera(cam_prim)
    xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_mat = xfc.GetLocalToWorldTransform(cam_prim)

    # Try GetCamera().GetFrustum() first (USD ≥ 22.03)
    try:
        gf_cam = cam_schema.GetCamera(Usd.TimeCode.Default())
        frustum = gf_cam.GetFrustum()
        frustum.Transform(world_mat)
        return frustum
    except AttributeError:
        pass

    # Fallback: build GfFrustum from USD camera attributes.
    # USD stores hap/vap/fl in *tenths of scene units* (film-format compat).
    # Divide by 10 to convert to scene units before building the frustum.
    fl = cam_schema.GetFocalLengthAttr().Get()
    hap = cam_schema.GetHorizontalApertureAttr().Get()
    vap = cam_schema.GetVerticalApertureAttr().Get()
    crng = cam_schema.GetClippingRangeAttr().Get()
    proj = cam_schema.GetProjectionAttr().Get()

    fl = float(fl) / 10.0 if fl is not None else 5.0
    hap = float(hap) / 10.0 if hap is not None else 3.6
    vap = float(vap) / 10.0 if vap is not None else 2.4
    near = float(crng[0]) if crng is not None else 1.0
    far = float(crng[1]) if crng is not None else 2_000_000.0

    try:
        frustum = Gf.Frustum()
        if proj == "orthographic":
            frustum.SetOrthographic(-hap / 2, hap / 2, -vap / 2, vap / 2, near, far)
        else:
            fov_v = _math.degrees(2.0 * _math.atan(vap / (2.0 * fl)))
            aspect = hap / vap
            try:
                frustum.SetPerspective(fov_v, True, aspect, near, far)
            except TypeError:
                frustum.SetPerspective(fov_v, aspect, near, far)
        frustum.Transform(world_mat)
        return frustum
    except Exception as e:
        print(f"[Fillet] Frustum build failed: {e} — including all instances.")
        return None


def _in_frustum(frustum, world_pos: Gf.Vec3d, margin_scene: float = 0.0) -> bool:
    if frustum is None:
        return True
    rng = Gf.Range3d(
        world_pos - Gf.Vec3d(margin_scene, margin_scene, margin_scene),
        world_pos + Gf.Vec3d(margin_scene, margin_scene, margin_scene),
    )
    return frustum.Intersects(Gf.BBox3d(rng))


# ── Instance discovery ────────────────────────────────────────────────────────


def _find_instances(stage, camera_path, pcba_root, margin_mm: float = 5.0, comp_types=None):
    """
    Return list of (prim, world_mat) for all instance prims whose USD path
    contains any string in comp_types, are within the camera frustum, and
    belong to pcba_root.

    comp_types: list of substrings to match (e.g. ["_0603_H100", "_0402_H060"]).
                Defaults to [COMP_TYPE] for backward compatibility.
    margin_mm:  physical bbox expansion in mm around each component centre.
    """
    if comp_types is None:
        comp_types = [COMP_TYPE]
    frustum = _get_frustum(stage, camera_path)
    margin_scene = _mm_to_scene(stage, margin_mm)
    xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
    result = []
    for prim in stage.TraverseAll():
        path_str = str(prim.GetPath())
        if not any(ct in path_str for ct in comp_types):
            continue
        if pcba_root and not path_str.startswith(pcba_root):
            continue
        # Instance check: USD native instances (IsInstance()) OR
        # `tn__`-prefixed Xformable instance prims (newer assets that
        # use plain references instead of `instanceable=true`). Without
        # the fallback, assets_final/spark_lighting.usd reports 0
        # in-view instances even though component Xforms exist.
        name = prim.GetName()
        if not (prim.IsInstance() or (name.startswith("tn__") and UsdGeom.Xformable(prim))):
            continue
        world_mat = xfc.GetLocalToWorldTransform(prim)
        pos = world_mat.ExtractTranslation()
        if _in_frustum(frustum, pos, margin_scene=margin_scene):
            result.append((prim, world_mat))
    return result


# ── Component axis extraction ─────────────────────────────────────────────────


def _component_axes(world_mat: Gf.Matrix4d, bbox_range: Gf.Range3d):
    """
    Return (long_dir, width_dir, up_dir, half_len, half_width) in world space.
    Identifies long axis from world-space bbox (works for 0°/90° rotations).
    """
    # Local axes from world matrix (row-vector convention → rows = local axes)
    lx = Gf.Vec3d(world_mat[0][0], world_mat[0][1], world_mat[0][2])
    ly = Gf.Vec3d(world_mat[1][0], world_mat[1][1], world_mat[1][2])
    lz = Gf.Vec3d(world_mat[2][0], world_mat[2][1], world_mat[2][2])

    # Determine which world dimension is the long one from bbox extents
    size = bbox_range.GetSize()
    if size[0] >= size[1]:
        long_world = Gf.Vec3d(1, 0, 0)
    else:
        long_world = Gf.Vec3d(0, 1, 0)

    lxn = lx.GetNormalized()
    lyn = ly.GetNormalized()
    lzn = lz.GetNormalized()

    # Pick the local axis that best aligns with the world long direction
    if abs(Gf.Dot(lxn, long_world)) >= abs(Gf.Dot(lyn, long_world)):
        long_dir = lxn
        width_dir = lyn
    else:
        long_dir = lyn
        width_dir = lxn

    up_dir = lzn
    half_len = max(size[0], size[1]) / 2.0
    half_width = min(size[0], size[1]) / 2.0

    return long_dir, width_dir, up_dir, half_len, half_width


# ── Bbox (native-instance safe) ───────────────────────────────────────────────


def _compute_world_bbox(inst_prim, world_mat):
    """
    World-space GfRange3d for a native instance prim.

    BBoxCache.ComputeWorldBound on a native instance (IsInstance()==True) in
    USD 21.x returns the prototype's LOCAL bbox — coordinates near the prototype
    origin, not the instance's world position.

    Fix: compute the prototype's local bbox with ComputeLocalBound, then
    transform its 8 corners through the instance's world_mat.
    """
    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=True,
    )

    # For native instances use the prototype prim; otherwise use prim itself
    compute_target = inst_prim
    if inst_prim.IsInstance():
        proto = inst_prim.GetPrototype()
        if proto and proto.IsValid():
            compute_target = proto

    try:
        local_bbox = bc.ComputeLocalBound(compute_target)
        local_rng = local_bbox.GetRange()
    except Exception:
        local_rng = Gf.Range3d()

    if local_rng.IsEmpty():
        return Gf.Range3d()

    # Transform 8 corners of local bbox to world space via instance world_mat
    lmin, lmax = local_rng.GetMin(), local_rng.GetMax()
    world_pts = [
        world_mat.Transform(Gf.Vec3d(x, y, z))
        for x in (lmin[0], lmax[0])
        for y in (lmin[1], lmax[1])
        for z in (lmin[2], lmax[2])
    ]
    xs = [pt[0] for pt in world_pts]
    ys = [pt[1] for pt in world_pts]
    zs = [pt[2] for pt in world_pts]
    return Gf.Range3d(
        Gf.Vec3d(min(xs), min(ys), min(zs)),
        Gf.Vec3d(max(xs), max(ys), max(zs)),
    )


# ── Per-instance fillet placement ─────────────────────────────────────────────


def _place_fillets(stage, inst_prim, world_mat, idx, p: FilletParams, mat):
    rng = _compute_world_bbox(inst_prim, world_mat)
    if rng.IsEmpty():
        print(f"  [Fillet] empty bbox for {inst_prim.GetPath()} — skipping")
        return

    min_pt = Gf.Vec3d(rng.GetMin())
    max_pt = Gf.Vec3d(rng.GetMax())
    center = (min_pt + max_pt) * 0.5

    long_dir, width_dir, up_dir, half_len, half_width = _component_axes(world_mat, rng)

    # Fillet scale: adaptive (fit to component bbox) or fixed from PARAMS
    if getattr(p, "adaptive_scale", False):
        mesh_width = float(p.long_edge) * 0.6
        sy = (half_width * 2.0) / mesh_width if mesh_width > 0 else p.fillet_sy
        sx = sy  # keep X/Y scale equal (square pad footprint)
        sz = p.fillet_sz
    else:
        sx, sy, sz = p.fillet_sx, p.fillet_sy, p.fillet_sz
    y_world = float(p.long_edge) * 0.6 * sy  # scaled width of fillet

    # Center fillet across component width
    width_offset = -y_world / 2.0

    # Base Z: bottom of bbox (PCB surface level)
    base_z = rng.GetMin()[2]

    for side, sign in (("L", -1.0), ("R", +1.0)):
        path = f"{FILLET_ROOT}/fillet_{idx:04d}_{side}"
        _write_mesh(stage, path, p, mat)

        if getattr(p, "adaptive_scale", False):
            inward = half_len * float(getattr(p, "inward_frac", 0.15))
        else:
            inward = _mm_to_scene(stage, p.inward_offset_mm)
        inner_edge = center + long_dir * (sign * (half_len - inward))

        origin_xy = inner_edge + width_dir * width_offset
        z_off = _mm_to_scene(stage, p.z_offset_mm)
        origin = Gf.Vec3d(origin_xy[0], origin_xy[1], base_z + z_off)

        # For right (+1): fillet x axis points rightward (outward from body)
        # For left  (-1): fillet x axis points leftward (outward from body)
        fx = long_dir * sign  # outward direction
        fy = width_dir
        fz = up_dir

        mat4 = _fillet_world_matrix(origin, fx, fy, fz, sx, sy, sz)
        _set_transform(stage, path, mat4)

    print(f"  [Fillet] {inst_prim.GetPath().name}  → L+R placed at idx {idx}")


# ── Top-level actions ─────────────────────────────────────────────────────────


def generate_fillets():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[Fillet] No stage.")
        return

    if not stage.GetPrimAtPath(FILLET_ROOT).IsValid():
        UsdGeom.Xform.Define(stage, FILLET_ROOT)

    mat = _ensure_material(stage, PARAMS)
    comp_types = list(getattr(PARAMS, "comp_types", [COMP_TYPE]))
    instances = _find_instances(stage, PARAMS.camera_path, PARAMS.pcba_root, comp_types=comp_types)
    type_label = f"{len(comp_types)} types" if len(comp_types) > 1 else comp_types[0]
    print(
        f"[Fillet] {len(instances)} {type_label} instances in view → generating {len(instances) * 2} fillets…"
    )

    for idx, (prim, wmat) in enumerate(instances):
        _place_fillets(stage, prim, wmat, idx, PARAMS, mat)

    print("[Fillet] Done.")


def clear_fillets():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    root = stage.GetPrimAtPath(FILLET_ROOT)
    if root.IsValid():
        stage.RemovePrim(FILLET_ROOT)
        print(f"[Fillet] Cleared {FILLET_ROOT}")
    mat = stage.GetPrimAtPath(MAT_PATH)
    if mat.IsValid():
        stage.RemovePrim(MAT_PATH)
        _shader_prim[0] = None


def _get_viewport_camera_path():
    """Return the USD prim path of the active viewport camera, or None."""
    try:
        import omni.kit.viewport.utility as vp_util

        vp = vp_util.get_active_viewport()
        if vp is not None:
            return str(vp.camera_path)
    except Exception:
        pass
    return None


def list_inview_components():
    """Print _0603_H100 instances visible in the current viewport camera."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[Fillet] No stage.")
        return

    cam_path = _get_viewport_camera_path()
    if cam_path:
        print(f"[Fillet] Viewport camera: {cam_path}")
        margin_mm = 1.0
    else:
        cam_path = PARAMS.camera_path
        print(f"[Fillet] (viewport cam unavailable, falling back to: {cam_path})")
        margin_mm = 5.0

    instances = _find_instances(stage, cam_path, PARAMS.pcba_root, margin_mm=margin_mm)
    print(f"[Fillet] {len(instances)} {COMP_TYPE} instances in view:")
    for prim, _ in instances:
        print(f"  {prim.GetPath()}")


# ── UI ────────────────────────────────────────────────────────────────────────

_auto_regen = [False]  # mutable so closures can read it


def _auto_update_shape():
    """Compute geometry once, push to every existing fillet mesh — O(N) writes, 0 stage traversals."""
    stage = omni.usd.get_context().get_stage()
    if not stage:
        return
    root = stage.GetPrimAtPath(FILLET_ROOT)
    if not root.IsValid():
        return
    pts, counts, idx = _compute_geometry(PARAMS)
    for child in root.GetChildren():
        if child.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(child)
            m.GetPointsAttr().Set(pts)
            m.GetFaceVertexCountsAttr().Set(counts)
            m.GetFaceVertexIndicesAttr().Set(idx)


def _auto_update_material():
    stage = omni.usd.get_context().get_stage()
    if stage and stage.GetPrimAtPath(MAT_PATH).IsValid():
        _ensure_material(stage, PARAMS)


def _auto_update_full():
    """Position/scale changed — need to recompute transforms, so full regeneration."""
    stage = omni.usd.get_context().get_stage()
    if stage and stage.GetPrimAtPath(FILLET_ROOT).IsValid():
        generate_fillets()


def _slider(label, key, lo, hi, default, fmt="{:.3f}", regen="shape"):
    """regen: 'shape' | 'material' | 'full'"""
    lbl_ref = [None]
    with ui.HStack(height=20, spacing=4):
        ui.Label(label, width=120)
        s = ui.FloatSlider(min=lo, max=hi)
        s.model.set_value(float(default))
        lbl_ref[0] = ui.Label(fmt.format(default), width=48)

    def _on(_=None):
        v = s.model.get_value_as_float()
        setattr(PARAMS, key, v)
        lbl_ref[0].text = fmt.format(v)
        if _auto_regen[0]:
            if regen == "shape":
                _auto_update_shape()
            elif regen == "material":
                _auto_update_material()
            else:
                _auto_update_full()

    s.model.add_value_changed_fn(_on)
    return s


def _section(title, collapsed=False):
    return ui.CollapsableFrame(title, collapsed=collapsed, style={"font_size": 13, "margin": 0})


def build_ui():
    win = ui.Window("Solder Fillet Placer", width=440, height=520)
    win.visible = True

    with win.frame:
        with ui.ScrollingFrame(
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
        ):
            with ui.VStack(spacing=2, style={"margin": 4}):
                with _section("Camera / PCBA", collapsed=True):
                    with ui.VStack(spacing=2, style={"margin": 4}):
                        with ui.HStack(height=20, spacing=4):
                            ui.Label("Camera", width=70)
                            cam_f = ui.StringField()
                            cam_f.model.set_value(PARAMS.camera_path)
                            cam_f.model.add_value_changed_fn(
                                lambda m: setattr(PARAMS, "camera_path", m.get_value_as_string())
                            )
                        with ui.HStack(height=20, spacing=4):
                            ui.Label("PCBA root", width=70)
                            root_f = ui.StringField()
                            root_f.model.set_value(PARAMS.pcba_root)
                            root_f.model.add_value_changed_fn(
                                lambda m: setattr(PARAMS, "pcba_root", m.get_value_as_string())
                            )

                with _section("Shape"):
                    with ui.VStack(spacing=2, style={"margin": 4}):
                        _slider("Long edge", "long_edge", 2, 20, PARAMS.long_edge, "{:.1f}")
                        _slider("X scale", "x_scale", 0.05, 0.8, PARAMS.x_scale)
                        _slider("Plat start", "plat_start", 0.0, 1.0, PARAMS.plat_start)
                        _slider("Plat end", "plat_end", 0.0, 1.0, PARAMS.plat_end)
                        _slider("Smooth Y", "smooth_y", 1.0, 30.0, PARAMS.smooth_y, "{:.1f}")
                        _slider("Bump", "bump", 0.0, 1.0, PARAMS.bump)
                        _slider("Decay", "decay", 1.0, 40.0, PARAMS.decay, "{:.1f}")

                with _section("Noise"):
                    with ui.VStack(spacing=2, style={"margin": 4}):
                        _slider("Scale", "noise_scale", 0.0, 1.5, PARAMS.noise_scale)
                        _slider("Octaves", "noise_octaves", 1, 8, PARAMS.noise_octaves, "{:.0f}")
                        _slider("Amplitude", "noise_amp", 0.0, 0.05, PARAMS.noise_amp, "{:.4f}")
                        _slider("Resolution", "resolution", 10, 80, PARAMS.resolution, "{:.0f}")

                with _section("Scale & Position"):
                    with ui.VStack(spacing=2, style={"margin": 4}):
                        _slider(
                            "Scale X/Y",
                            "fillet_sx",
                            0.01,
                            0.5,
                            PARAMS.fillet_sx,
                            "{:.5f}",
                            regen="full",
                        )
                        _slider(
                            "Scale Z",
                            "fillet_sz",
                            0.01,
                            1.0,
                            PARAMS.fillet_sz,
                            "{:.4f}",
                            regen="full",
                        )
                        _slider(
                            "Inward (mm)",
                            "inward_offset_mm",
                            -3.0,
                            6.0,
                            PARAMS.inward_offset_mm,
                            "{:.3f}",
                            regen="full",
                        )
                        _slider(
                            "Z offset (mm)",
                            "z_offset_mm",
                            -2.0,
                            15.0,
                            PARAMS.z_offset_mm,
                            "{:.3f}",
                            regen="full",
                        )

                with _section("Material", collapsed=True):
                    with ui.VStack(spacing=2, style={"margin": 4}):
                        _slider(
                            "Metallic",
                            "mat_metallic",
                            0.0,
                            1.0,
                            PARAMS.mat_metallic,
                            regen="material",
                        )
                        _slider(
                            "Roughness",
                            "mat_roughness",
                            0.0,
                            1.0,
                            PARAMS.mat_roughness,
                            regen="material",
                        )

                ui.Spacer(height=4)
                with ui.HStack(height=28, spacing=4):
                    ui.Button("Generate", clicked_fn=generate_fillets, width=ui.Fraction(3))
                    ui.Button("Clear", clicked_fn=clear_fillets, width=ui.Fraction(1))
                with ui.HStack(height=28, spacing=4):
                    ui.Button(
                        "List In-View", clicked_fn=list_inview_components, width=ui.Fraction(1)
                    )
                ui.Spacer(height=2)
                with ui.HStack(height=22, spacing=6):
                    cb = ui.CheckBox(width=18)
                    cb.model.set_value(False)
                    ui.Label("Auto-update on slider change")

                    def _on_auto(m):
                        _auto_regen[0] = m.get_value_as_bool()

                    cb.model.add_value_changed_fn(_on_auto)

    return win


# Keep Scale Y in sync with Scale X (fillet_sx slider sets both)
def _patched_setattr(self, key, val):
    object.__setattr__(self, key, val)
    if key == "fillet_sx":
        object.__setattr__(self, "fillet_sy", val)


FilletParams.__setattr__ = _patched_setattr


# ── Execute ───────────────────────────────────────────────────────────────────

if not os.getenv("FILLET_PIPELINE_MODE"):
    try:
        _fillet_win = build_ui()
        globals()["_fillet_win"] = _fillet_win
        print(
            "[Fillet] UI ready. Press 'Generate Fillets' to place fillets for in-view _0603_H100 components."
        )
    except Exception as _e:
        import traceback

        traceback.print_exc()
