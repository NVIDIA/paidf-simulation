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
tin_noise_patch.py — perlin-noise bumps on capacitor / resistor tin
terminals.

Logic mirrors `editor_solder_fillet_board.py` but the per-grid-point
profile has **no half-moon slope** — it's a flat baseline (z = 0)
plus the same fractal-noise displacement, so the result is a thin
"normal-map-like" geometric patch sitting on top of each tin face.

Pipeline integration: call ``generate_tin_patches()`` once before each
scan-grid capture (paired with ``clear_tin_patches()``). Both share
the in-view component discovery from the fillet module so we only do
the camera-frustum walk once.

YAML block (top-level config key ``tin_noise``):

    tin_noise:
      enabled: true
      noise_scale: 0.7
      noise_octaves: 4
      noise_amp: 0.05         # mm-ish — height of the bump (scene units)
      resolution: 24
      patch_x_frac: 0.30      # patch length along component long axis
                              # (fraction of half-length per side; the
                              # patch sits right at the component end)
      patch_y_frac: 1.0       # patch width as a fraction of component width
      z_offset_mm: 0.001      # epsilon above tin top to avoid z-fighting
      side_z_scale: 0.7       # how much taller (in z) the bumps are along
                              # the side-faces relative to the top

Per-trigger randomization (optional):

    randomize_tin_noise:
      noise_scale: [0.4, 1.0]
      noise_octaves: [3, 6]
      noise_amp:    [0.03, 0.08]
      patch_x_frac: [0.25, 0.40]

Backends honor ``FILLET_BACKEND`` env var (auto / cpu / warp). Warp
kernel + per-call cache live in ``editor_solder_fillet_board`` and are
reused here.
"""

from __future__ import annotations

import os

import numpy as np
import omni.usd

# Reuse all the noise / warp / cache machinery from the fillet module.
from editor_solder_fillet_board import (
    _component_axes,
    _compute_world_bbox,
    _compute_z_warp,
    _find_instances,
    _fractal_noise_2d,
    _mm_to_scene,
    _set_transform,
    _wp_init,
)
from pxr import Gf, Sdf, UsdGeom, UsdShade

TIN_ROOT = "/World/TinNoisePatches"
MAT_PATH = "/World/TinNoisePatchMat"


class TinNoiseParams:
    """Mirrors the FilletParams shape but the geometry kernel uses a
    flat baseline (no plateau / no exp decay). Defaults tuned to be
    visible at typical macro-camera distance without changing the
    component silhouette."""

    enabled = True
    noise_scale = 4.27
    noise_octaves = 5.14
    noise_amp = 0.18  # scene units (mm) — bump height
    resolution = 75.76
    # How big the patch is at each component end (sits flush against
    # the end face). x is along the component's long axis; y across.
    patch_x_frac = 0.30
    patch_y_frac = 1.0
    z_offset_mm = 0.001  # epsilon above tin top to dodge z-fighting
    # Material — silvery metal mirroring the original tin.
    mat_metallic = 1.0
    mat_roughness = 0.25
    # Scene context — overridden by pipeline from CFG (camera_path,
    # pcba_root, comp_types).
    camera_path = "/World/camera_light/Camera"
    pcba_root = "/World/pcba_main_s_detail/PCBA/tn__60014242BASEA04_fM9E"
    comp_types = ["_0603_H100"]


PARAMS = TinNoiseParams()


# Single source of truth for "yaml/UI dict → TinNoiseParams". Both the
# pipeline (CFG['tin_noise']) and the editor UI sliders write through
# this helper so they cannot drift. Keys are the canonical yaml names
# (``noise_amp``, ``noise_scale``, …); unknown / missing keys are
# left at the param's previous value, so partial updates from a UI
# slider only touch what changed.
_PARAM_KEYS: tuple[str, ...] = (
    "noise_scale",
    "noise_octaves",
    "noise_amp",
    "resolution",
    "patch_x_frac",
    "patch_y_frac",
    "z_offset_mm",
    "metallic",
    "roughness",
    "camera_path",
    "pcba_root",
    "comp_types",
)


def apply_params_from_dict(p: "TinNoiseParams", d: dict) -> None:
    """Mutate ``p`` in place from a dict of canonical keys.

    * Float-valued keys are coerced to ``float``.
    * ``noise_octaves`` and ``resolution`` are coerced to ``int``-castable
      floats (the kernel handles the int conversion internally).
    * Slider-friendly aliases ``mat_metallic``/``mat_roughness`` from
      the editor UI map to canonical ``metallic``/``roughness``.
    * Unknown keys are silently ignored so the same helper can absorb
      either a yaml block (with ``enabled``, ``noise_*``, …) or a UI
      slider dict (without ``enabled``).
    """
    if "metallic" in d:
        p.mat_metallic = float(d["metallic"])
    if "roughness" in d:
        p.mat_roughness = float(d["roughness"])
    if "mat_metallic" in d:
        p.mat_metallic = float(d["mat_metallic"])
    if "mat_roughness" in d:
        p.mat_roughness = float(d["mat_roughness"])

    for k in ("noise_scale", "noise_amp", "z_offset_mm", "patch_x_frac", "patch_y_frac"):
        if k in d:
            setattr(p, k, float(d[k]))
    if "noise_octaves" in d:
        p.noise_octaves = float(d["noise_octaves"])
    if "resolution" in d:
        p.resolution = float(d["resolution"])

    if "camera_path" in d:
        p.camera_path = str(d["camera_path"])
    if "pcba_root" in d:
        p.pcba_root = str(d["pcba_root"])
    if "comp_types" in d and d["comp_types"]:
        p.comp_types = list(d["comp_types"])


def params_to_dict(p: "TinNoiseParams") -> dict:
    """Inverse of :func:`apply_params_from_dict` — used by the
    equivalence test to compare UI vs pipeline state without touching
    Kit-only attributes."""
    return {
        "noise_scale": float(p.noise_scale),
        "noise_octaves": float(p.noise_octaves),
        "noise_amp": float(p.noise_amp),
        "resolution": float(p.resolution),
        "patch_x_frac": float(p.patch_x_frac),
        "patch_y_frac": float(p.patch_y_frac),
        "z_offset_mm": float(p.z_offset_mm),
        "metallic": float(p.mat_metallic),
        "roughness": float(p.mat_roughness),
        "camera_path": str(p.camera_path),
        "pcba_root": str(p.pcba_root),
        "comp_types": list(p.comp_types),
    }


# ── Cache (mirrors the fillet module's cache) ────────────────────────────────
_GEOM_CACHE: "dict[tuple, tuple]" = {}
_GEOM_CACHE_MAX = 4


def _params_cache_key(p: "TinNoiseParams", w: float, h: float) -> tuple:
    return (
        float(p.noise_scale),
        int(p.noise_octaves),
        float(p.noise_amp),
        int(p.resolution),
        float(w),
        float(h),
    )


# ── Flat noise patch geometry ────────────────────────────────────────────────


def _compute_patch_geometry(
    p: TinNoiseParams, w: float, h: float, backend: "str | None" = None, use_cache: bool = True
):
    """Build a flat (w × h) patch in local space, displaced in z by
    fractal noise only. Returns (pts, counts, idx).

    ``backend``: ``"cpu"``, ``"warp"``, or ``None``/``"auto"`` (env
    var ``FILLET_BACKEND`` honored).
    """
    if backend is None:
        backend = os.environ.get("FILLET_BACKEND", "auto").lower()

    key = _params_cache_key(p, w, h) if use_cache else None
    if key is not None and key in _GEOM_CACHE:
        return _GEOM_CACHE[key]

    r = max(4, int(p.resolution))
    x_lin = np.linspace(-w * 0.5, w * 0.5, r, dtype=np.float64)
    y_lin = np.linspace(-h * 0.5, h * 0.5, r, dtype=np.float64)
    X, Y = np.meshgrid(x_lin, y_lin)

    # Normalised grid coords for the noise function — fractal_noise_2d
    # was authored expecting the grid in [0, 1] roughly; keep it
    # consistent so the same FILLET_BACKEND= warp path works.
    long_e = max(w, 1e-6)
    short = max(h, 1e-6)
    xn = (X + w * 0.5) / long_e
    yn = (Y + h * 0.5) / short

    z = None
    if backend in ("auto", "warp"):
        # Build a transient FilletParams-shaped stand-in so we can
        # call the existing GPU kernel. The fillet kernel's ``bump``
        # multiplies the *plateau × decay* term, which we want to be
        # zero — so set bump=0 to skip it (kernel still computes the
        # noise term, which is what we want).
        class _Stub:
            pass

        s = _Stub()
        s.long_edge = long_e
        s.x_scale = w / long_e if long_e > 0 else 1.0
        s.plat_start = 0.5
        s.plat_end = 0.5
        s.smooth_y = 1.0
        s.bump = 0.0  # ← kills the half-moon slope
        s.decay = 0.0
        s.noise_scale = p.noise_scale
        s.noise_octaves = p.noise_octaves
        s.noise_amp = p.noise_amp
        s.resolution = r
        z = _compute_z_warp(s)
        if z is None and backend == "warp":
            print("[TinNoise/Warp] warp unavailable — using NumPy")

    if z is None:
        noise = _fractal_noise_2d(xn, yn, p.noise_scale, p.noise_octaves)
        z = float(p.noise_amp) * noise

    flat = np.empty((r * r, 3), dtype=np.float32)
    flat[:, 0] = X.reshape(-1).astype(np.float32, copy=False)
    flat[:, 1] = Y.reshape(-1).astype(np.float32, copy=False)
    flat[:, 2] = z.reshape(-1).astype(np.float32, copy=False)
    try:
        from pxr import Vt

        pts = Vt.Vec3fArray.FromNumpy(flat)
    except Exception:  # noqa: BLE001
        pts = [
            Gf.Vec3f(float(flat[k, 0]), float(flat[k, 1]), float(flat[k, 2]))
            for k in range(flat.shape[0])
        ]

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


# ── Material ─────────────────────────────────────────────────────────────────


def _ensure_material(stage, p: TinNoiseParams) -> UsdShade.Material:
    existing = stage.GetPrimAtPath(MAT_PATH)
    if existing.IsValid():
        return UsdShade.Material(existing)
    mat = UsdShade.Material.Define(stage, MAT_PATH)
    sh = UsdShade.Shader.Define(stage, f"{MAT_PATH}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(p.mat_metallic))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(p.mat_roughness))
    sh.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.78, 0.78, 0.80))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _write_mesh(stage, path, pts, counts, idx, mat):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        mesh = UsdGeom.Mesh.Define(stage, path)
        UsdShade.MaterialBindingAPI(mesh).Bind(mat)
    else:
        mesh = UsdGeom.Mesh(prim)
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(idx)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return mesh


def _patch_world_matrix(
    origin: Gf.Vec3d,
    x_dir: Gf.Vec3d,
    y_dir: Gf.Vec3d,
    z_dir: Gf.Vec3d,
    sx: float = 1.0,
    sy: float = 1.0,
    sz: float = 1.0,
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


# ── Place patches at each in-view component ──────────────────────────────────


def _place_patches(stage, inst_prim, world_mat, idx, p: TinNoiseParams, mat):
    rng = _compute_world_bbox(inst_prim, world_mat)
    if rng.IsEmpty():
        return

    long_dir, width_dir, up_dir, half_len, half_width = _component_axes(world_mat, rng)
    if half_len <= 0 or half_width <= 0:
        return

    # Patch dimensions in scene units.
    patch_w = float(p.patch_x_frac) * (2.0 * half_len) * 0.5  # half-len × frac
    patch_h = float(p.patch_y_frac) * (2.0 * half_width) * 1.0
    if patch_w < 1e-4 or patch_h < 1e-4:
        return

    pts, counts, indices = _compute_patch_geometry(p, patch_w, patch_h)

    center = Gf.Vec3d(rng.GetMin()) * 0.5 + Gf.Vec3d(rng.GetMax()) * 0.5
    top_z = float(rng.GetMax()[2])
    z_off = _mm_to_scene(stage, p.z_offset_mm)

    for side, sign in (("L", -1.0), ("R", +1.0)):
        # Patch centred at the end of the component, offset inward by half
        # the patch width so its outer edge aligns with the component end.
        end_x_world = sign * (half_len - patch_w * 0.5)
        origin_xy = center + long_dir * end_x_world
        origin = Gf.Vec3d(origin_xy[0], origin_xy[1], top_z + z_off)

        path = f"{TIN_ROOT}/patch_{idx:04d}_{side}"
        _write_mesh(stage, path, pts, counts, indices, mat)
        mat4 = _patch_world_matrix(origin, long_dir, width_dir, up_dir)
        _set_transform(stage, path, mat4)


# ── Top-level actions ────────────────────────────────────────────────────────


def generate_tin_patches():
    if not getattr(PARAMS, "enabled", False):
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[TinNoise] No stage.")
        return

    if not stage.GetPrimAtPath(TIN_ROOT).IsValid():
        UsdGeom.Xform.Define(stage, TIN_ROOT)

    mat = _ensure_material(stage, PARAMS)
    comp_types = list(getattr(PARAMS, "comp_types", ["_0603_H100"]))
    instances = _find_instances(stage, PARAMS.camera_path, PARAMS.pcba_root, comp_types=comp_types)
    print(
        f"[TinNoise] {len(instances)} components in view → generating {len(instances) * 2} patches…"
    )
    for idx, (prim, wmat) in enumerate(instances):
        _place_patches(stage, prim, wmat, idx, PARAMS, mat)
    print("[TinNoise] Done.")


def clear_tin_patches():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    root = stage.GetPrimAtPath(TIN_ROOT)
    if not root.IsValid():
        return
    stage.RemovePrim(root.GetPath())


# ── Normal-map texture generation ────────────────────────────────────────────
#
# Surface-mesh patches above were the original path. The user prefers to
# treat tin-bump as a per-fragment normal map applied directly to every
# tin shader (no new geometry, no in-frame walks), so the editor and
# pipeline now use ``make_tin_normalmap`` to bake a tileable normal map
# from the same fractal noise and stamp it as ``inputs:normalmap_texture``
# on every tin shader via ``component_material_override._State``.

import hashlib  # noqa: E402
import tempfile  # noqa: E402

_NORMALMAP_DIR = os.path.join(tempfile.gettempdir(), "pcb_aoi_tin_normals")
os.makedirs(_NORMALMAP_DIR, exist_ok=True)


def _normalmap_cache_key(
    p: "TinNoiseParams", size: int, seed: int = 42, smooth_sigma: float = 0.0
) -> str:
    keys = (
        float(p.noise_scale),
        int(p.noise_octaves),
        float(p.noise_amp),
        size,
        int(seed),
        float(smooth_sigma),
    )
    return hashlib.md5(repr(keys).encode()).hexdigest()[:12]


def _smooth_z_periodic(z: np.ndarray, sigma: float) -> np.ndarray:
    """3-tap separable [1, 2, 1] / 4 lowpass on z, applied N times to
    approximate a gaussian of std-dev ``sigma`` (in pixels). Wraparound
    via ``np.roll`` so the smoothing preserves tileability.

    Used to kill single-pixel spikes in the noise heightmap before
    computing the normal-map gradient — high-frequency spikes were
    showing up as bright sparkles in the rendered tin (very-glancing
    surface normals → metallic Fresnel hot pixel).
    """
    if sigma <= 0.0:
        return z
    # gaussian std for one [1,2,1]/4 pass ≈ 1/sqrt(2). N passes ≈ sigma^2*2.
    n_passes = max(1, int(round(float(sigma) * float(sigma) * 2.0)))
    for _ in range(n_passes):
        z = 0.25 * np.roll(z, 1, axis=0) + 0.5 * z + 0.25 * np.roll(z, -1, axis=0)
        z = 0.25 * np.roll(z, 1, axis=1) + 0.5 * z + 0.25 * np.roll(z, -1, axis=1)
    return z


def _value_noise_tileable_layer(size: int, n: int, rng: "np.random.Generator") -> np.ndarray:
    """One octave of tileable value noise.

    Place i.i.d. uniform random values on an ``n × n`` integer grid that
    wraps (toroidal), then smoothstep-bilinear-interpolate to
    ``size × size``. Wrap is implicit in the modulo index lookup, so
    the output is seamless on the unit square — no sinusoid grid
    artifacts because the spatial structure is driven by random grid
    values, not by ``sin·cos`` standing waves.
    """
    g = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float64)
    # Sample positions in grid-cell coordinates [0, n).
    t = np.linspace(0.0, float(n), size, endpoint=False, dtype=np.float64)
    i0 = (np.floor(t).astype(np.int64)) % n
    i1 = (i0 + 1) % n
    f = t - np.floor(t)
    # Smoothstep s(f) = 3f² − 2f³ → C¹-continuous bilinear interp,
    # the canonical "value noise" smoothing (Perlin-style).
    s = f * f * (3.0 - 2.0 * f)

    # 2D sampling via outer-index broadcasting (no Python loops).
    sx = s[None, :]
    sy = s[:, None]
    G00 = g[np.ix_(i0, i0)]
    G01 = g[np.ix_(i0, i1)]
    G10 = g[np.ix_(i1, i0)]
    G11 = g[np.ix_(i1, i1)]
    return (1.0 - sy) * ((1.0 - sx) * G00 + sx * G01) + sy * ((1.0 - sx) * G10 + sx * G11)


def _tileable_fractal_noise_2d(size: int, scale: float, octaves: int, seed: int = 42) -> np.ndarray:
    """Seamless fractal value noise on the unit square.

    fBm-style sum of value-noise octaves: each octave doubles the
    grid resolution and halves the amplitude, giving the rough/smooth
    multi-scale character of natural terrain. Tileability is provided
    by the toroidal index wrap inside :func:`_value_noise_tileable_layer`,
    so the resulting heightmap can be safely used as a tiling normal
    map without seams.

    ``scale`` sets the grid size of the BASE octave (in cells per unit
    period). ``scale=2`` means 2 random control points per side at the
    coarsest octave — so one big "valley/peak" pair across the texture.
    """
    rng = np.random.default_rng(int(seed))
    base = max(2, int(round(float(scale))))
    z = np.zeros((size, size), dtype=np.float64)
    amp = 1.0
    for k in range(max(1, int(octaves))):
        n = base * (1 << k)  # 2^k doubling per octave
        if n >= size:
            break  # finer than 1 grid-cell per pixel — stop
        z += amp * _value_noise_tileable_layer(size, n, rng)
        amp *= 0.5
    return z


def make_tin_normalmap(
    p: "TinNoiseParams",
    size: int = 512,
    force_rebuild: bool = False,
    seed: "int | None" = None,
    smooth_sigma: float = 1.0,
) -> str:
    """Bake a SEAMLESS RGB normal map from ``p``'s fractal noise + amp.

    Returns the absolute path to the PNG. Path is cache-keyed on the
    shape-affecting params **plus seed and smooth_sigma** so repeat calls
    with identical params return the existing file (sub-ms). When
    ``force_rebuild=True``, writes a new file even if the cached path
    exists — used by the editor when OmniPBR's texture cache might still
    hold the previous bytes.

    ``seed``: explicit RNG seed for the underlying value-noise grid. Pass
    a per-sample integer to vary the noise pattern across frames (the
    cache key includes it). When ``None``, the historical fixed seed
    (42) is used so output is bit-stable for back-compat.

    ``smooth_sigma``: Gaussian-equivalent std-dev (in pixels) applied to
    the heightmap *before* the normal-map gradient. Kills single-pixel
    spikes that appeared as bright sparkles on metallic tin (steep
    gradient → near-tangent normal → Fresnel hot pixel). Default 1.0 px
    smoothes one pixel without visibly softening the texture.

    The math is intentionally pure-numpy on a 512×512 grid (≈ 80 ms on
    CPU); this runs once per slider drag (or per scan-grid sample when
    randomized), not per ray, so a warp port wouldn't pay back.
    """
    seed_used = 42 if seed is None else int(seed)
    key = _normalmap_cache_key(p, size, seed=seed_used, smooth_sigma=smooth_sigma)
    out_path = os.path.join(_NORMALMAP_DIR, f"tin_normal_{key}.png")
    if os.path.exists(out_path) and not force_rebuild:
        return out_path

    # Tileable heightmap on a unit grid.
    z = _tileable_fractal_noise_2d(
        size, p.noise_scale, int(p.noise_octaves), seed=seed_used
    ).astype(np.float32)
    z *= float(p.noise_amp)

    # Smooth out single-pixel spikes — sharp z transitions blow up the
    # gradient and produce bright sparkles on metallic tin under PT.
    z = _smooth_z_periodic(z, float(smooth_sigma)).astype(np.float32)

    # Periodic (wraparound) central differences — np.gradient at the
    # edges falls back to one-sided diff which would re-introduce the
    # seam we just got rid of in z.
    dx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) * 0.5 * (size * 0.5)
    dy = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) * 0.5 * (size * 0.5)
    # Clamp gradient magnitude — caps the worst-case slope so even a
    # rare residual jump can't drive nz below ~0.3 (≈ 70° tilt). Keeps
    # the firefly filter from being our only line of defense.
    grad_clamp = 3.0
    dx = np.clip(dx, -grad_clamp, grad_clamp)
    dy = np.clip(dy, -grad_clamp, grad_clamp)
    nz = np.ones_like(z, dtype=np.float32)
    inv_len = 1.0 / np.sqrt(dx * dx + dy * dy + nz * nz)
    nx = -dx * inv_len
    ny = -dy * inv_len
    nz = nz * inv_len

    rgb = np.empty((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    try:
        from PIL import Image

        Image.fromarray(rgb, mode="RGB").save(out_path)
    except Exception as exc:  # pragma: no cover — PIL absence is rare in Kit
        raise RuntimeError(f"tin_noise_patch: PIL unavailable: {exc!r}")
    return out_path
