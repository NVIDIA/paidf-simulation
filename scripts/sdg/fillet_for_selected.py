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

"""Generate solder fillets for the SELECTED capacitor instance,
plus disable existing scene lights and spawn the RGB ring rig.

Paste into Omniverse Script Editor (Window > Script Editor) with one
capacitor instance (or a child mesh under one) selected, then run.

What it does:
  1) Walks the selection up to the nearest `tn__*` instance Xform.
  2) Reuses pipeline code paths:
       - editor_solder_fillet_board._place_fillets   (fillets)
       - common.disable_all_lights                    (kill scene lights)
       - common.build_ring_lights                     (RGB ring rig)
  3) Prints a diagnostic so you can see exactly where the fillets land
     vs. where the cap's two ends are — useful to verify `inward` is
     placing fillets inside the cap.

Tunable defaults below; edit and re-run.
"""
import sys
import importlib

import omni.usd
from pxr import Usd, UsdGeom, Gf

# ─── User-tunable defaults ────────────────────────────────────────────────
# Set USE_CURRENT_PARAMS=True (default) and the script uses whatever the
# in-Kit UI sliders are currently set to — drag the `Inward frac` slider
# FIRST in the UI, then run this script. The override dict below is
# IGNORED in that mode (the active PARAMS are printed in the diagnostic).
#
# Flip USE_CURRENT_PARAMS=False to forcibly stamp the FILLET_PARAMS_OVERRIDES
# values into PARAMS before placing fillets (handy if you want a known-good
# starting point regardless of current slider state).
USE_CURRENT_PARAMS = True

FILLET_PARAMS_OVERRIDES = {
    "adaptive_scale": True,    # use inward_frac (bbox-proportional)
    "inward_frac":    0.15,    # 15% of half-length inward from each end
    "z_offset_mm":    0.5,     # just above cap top (was 10 mm = clipped on near plane)
    "fillet_sz":      0.24,
}

CLEAR_FILLETS_FIRST = False   # True → nuke /World/SolderFillets first
DISABLE_SCENE_LIGHTS = True   # True → disable existing lights in scene
BUILD_RING_LIGHT = True       # True → build new RGB ring rig

# ─── Tin perlin normal-map (mirrors pipeline tin_noise + set_tin_normalmap) ─
# Pipeline bakes ONE perlin PNG and stamps it as `inputs:normalmap_texture`
# on every tin shader. Setup also runs vantablack body override (currently
# False here — keep authored body colors; we only want the tin path
# discovery to enable normal-map stamping). Flip APPLY_TIN_PERLIN=False to
# skip.
APPLY_TIN_PERLIN        = True
USE_CURRENT_TIN_PARAMS  = True   # True → use live tin_noise_patch.PARAMS
                                 # (whatever the UI sliders set); False → stamp
                                 # the TIN_NOISE_OVERRIDES dict below.
TIN_NOISE_OVERRIDES = {
    "noise_amp":     0.18,
    "noise_scale":   4.27,
    "noise_octaves": 5.14,
    "resolution":    75.76,
}
TIN_BUMP_FACTOR    = 0.24        # OmniPBR normal-map intensity (independent of amp)
TIN_TEXTURE_SCALE  = 17.62       # world-space UV tile (triplanar projection)
TIN_VANTABLACK_BODY = False      # True → also stamp body→(0,0,0)+metallic=0.89

RING_LIGHT_ROOT = "/World/camera_light/aoi_ring_light"
RIG_DOME_RADIUS = 184.0
RIG_LIGHT_RADIUS = 18.0
RIG_LIGHT_INTENSITY = 6204.0
RIG_CONE_ANGLE = 120.0
RIG_CONE_SOFTNESS = 1.0
# ──────────────────────────────────────────────────────────────────────────


def _find_instance_ancestor(prim):
    """Walk up until a `tn__`-prefixed Xformable is found."""
    cur = prim
    while cur.IsValid():
        if cur.GetName().startswith("tn__") and UsdGeom.Xformable(cur):
            return cur
        cur = cur.GetParent()
    return None


def main():
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("No USD stage open.")

    sel = ctx.get_selection().get_selected_prim_paths()
    if not sel:
        raise RuntimeError("Select a capacitor instance (or a child mesh under one) first.")

    sel_prim = stage.GetPrimAtPath(sel[0])
    if not sel_prim.IsValid():
        raise RuntimeError(f"Selected path is not a valid prim: {sel[0]}")

    inst_prim = _find_instance_ancestor(sel_prim)
    if inst_prim is None:
        raise RuntimeError(
            f"Couldn't find a `tn__`-prefixed instance ancestor of {sel[0]}."
        )

    # ── Load pipeline modules from the repo ──────────────────────────────
    import os as _os
    _root = _os.environ.get("PAIDF_SIM_ROOT")
    if not _root:
        raise RuntimeError("PAIDF_SIM_ROOT not set — point it at your paidf-simulation repo root.")
    for repo_path in (
        f"{_root}/scripts/sdg/editor",
        f"{_root}/scripts/sdg/standalone",
    ):
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    try:
        import editor_solder_fillet_board as fb
        importlib.reload(fb)
        import common as sdg_common
        importlib.reload(sdg_common)
        import tin_noise_patch as tnp
        importlib.reload(tnp)
        import component_material_override as cmo
        importlib.reload(cmo)
    except ImportError as e:
        raise RuntimeError(f"Couldn't import pipeline modules: {e}")

    # ── 1) Disable existing scene lights ─────────────────────────────────
    if DISABLE_SCENE_LIGHTS:
        n = sdg_common.disable_all_lights(stage)
        print(f"[script] disabled {n} existing scene light(s)")

    # ── 2) Build RGB ring light rig ──────────────────────────────────────
    if BUILD_RING_LIGHT:
        # Ensure parent xform exists
        parent_path = "/".join(RING_LIGHT_ROOT.split("/")[:-1])
        if not stage.GetPrimAtPath(parent_path).IsValid():
            UsdGeom.Xform.Define(stage, parent_path)
        sdg_common.build_ring_lights(
            stage,
            RING_LIGHT_ROOT,
            dome_radius=RIG_DOME_RADIUS,
            light_radius=RIG_LIGHT_RADIUS,
            light_intensity=RIG_LIGHT_INTENSITY,
            cone_angle=RIG_CONE_ANGLE,
            cone_softness=RIG_CONE_SOFTNESS,
        )
        print(f"[script] built RGB ring rig under {RING_LIGHT_ROOT}")

    # ── 3a) Tin perlin normal map (mirror pipeline tin_noise stamping) ───
    if APPLY_TIN_PERLIN:
        try:
            mat_state = cmo._State()
            ok, info = mat_state.setup(stage, vantablack_body=TIN_VANTABLACK_BODY)
            if not ok:
                print(f"[script] tin normalmap skipped — setup failed: {info}")
            else:
                if not USE_CURRENT_TIN_PARAMS:
                    tnp.apply_params_from_dict(tnp.PARAMS, TIN_NOISE_OVERRIDES)
                png = tnp.make_tin_normalmap(tnp.PARAMS, force_rebuild=True)
                mat_state.set_tin_normalmap(
                    png,
                    bump_factor=TIN_BUMP_FACTOR,
                    texture_scale=TIN_TEXTURE_SCALE,
                    project_uvw=True,
                )
                print(
                    f"[script] tin normalmap → {png}\n"
                    f"[script]   amp={tnp.PARAMS.noise_amp:.3f} scale={tnp.PARAMS.noise_scale:.2f} "
                    f"oct={tnp.PARAMS.noise_octaves:.2f} res={tnp.PARAMS.resolution:.0f}\n"
                    f"[script]   bump={TIN_BUMP_FACTOR}  uv_scale={TIN_TEXTURE_SCALE}  "
                    f"applied to {len(mat_state.tin_paths)} tin shaders"
                )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[script] tin normalmap failed: {exc!r}")

    # ── 3b) Apply FILLET_PARAMS overrides (or keep current slider values) ─
    if USE_CURRENT_PARAMS:
        print(
            "[script] USE_CURRENT_PARAMS=True — using live UI slider values; "
            "FILLET_PARAMS_OVERRIDES IGNORED"
        )
    else:
        for k, v in FILLET_PARAMS_OVERRIDES.items():
            if hasattr(fb.PARAMS, k):
                setattr(fb.PARAMS, k, v)
            else:
                print(f"[script] WARN unknown FILLET_PARAMS key: {k}")

    if CLEAR_FILLETS_FIRST:
        root_prim = stage.GetPrimAtPath(fb.FILLET_ROOT)
        if root_prim.IsValid():
            stage.RemovePrim(fb.FILLET_ROOT)
            print(f"[script] cleared {fb.FILLET_ROOT}")

    if not stage.GetPrimAtPath(fb.FILLET_ROOT).IsValid():
        UsdGeom.Xform.Define(stage, fb.FILLET_ROOT)
    mat = fb._ensure_material(stage, fb.PARAMS)

    # ── 4) Diagnostic + place fillets ────────────────────────────────────
    xform_cache = UsdGeom.XformCache()
    world_mat = xform_cache.GetLocalToWorldTransform(inst_prim)
    rng = fb._compute_world_bbox(inst_prim, world_mat)
    if rng.IsEmpty():
        raise RuntimeError(f"World bbox of {inst_prim.GetPath()} is empty.")

    min_pt = Gf.Vec3d(rng.GetMin())
    max_pt = Gf.Vec3d(rng.GetMax())
    center = (min_pt + max_pt) * 0.5
    size = max_pt - min_pt
    long_dir, width_dir, up_dir, half_len, half_width = fb._component_axes(world_mat, rng)

    # Compute where the cap's two ends and the fillet origins WILL land,
    # so you can verify by-hand that `inward` is doing what's intended.
    if getattr(fb.PARAMS, "adaptive_scale", False):
        inward = half_len * float(getattr(fb.PARAMS, "inward_frac", 0.15))
    else:
        inward = fb._mm_to_scene(stage, fb.PARAMS.inward_offset_mm)

    end_L = center + long_dir * (-half_len)
    end_R = center + long_dir * (+half_len)
    fillet_L = center + long_dir * (-(half_len - inward))
    fillet_R = center + long_dir * (+(half_len - inward))

    def _v(p):
        return f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})"

    print(
        f"\n[script] instance: {inst_prim.GetPath()}\n"
        f"[script]   world bbox  center={_v(center)} size={_v(size)}\n"
        f"[script]   long_dir    {_v(long_dir)}    half_len={half_len:.4f}\n"
        f"[script]   width_dir   {_v(width_dir)}   half_width={half_width:.4f}\n"
        f"[script]   inward      {inward:.4f}  (adaptive_scale={fb.PARAMS.adaptive_scale}, "
        f"inward_frac={fb.PARAMS.inward_frac}, inward_offset_mm={fb.PARAMS.inward_offset_mm})\n"
        f"[script]   cap_end_L   {_v(end_L)}\n"
        f"[script]   cap_end_R   {_v(end_R)}\n"
        f"[script]   fillet_L    {_v(fillet_L)}    (distance from end_L: {inward:.4f})\n"
        f"[script]   fillet_R    {_v(fillet_R)}    (distance from end_R: {inward:.4f})\n"
        f"[script]   z_offset_mm {fb.PARAMS.z_offset_mm}  base_z={min_pt[2]:.4f}  "
        f"=> fillet_z_origin={min_pt[2] + fb._mm_to_scene(stage, fb.PARAMS.z_offset_mm):.4f}\n"
    )

    fb._place_fillets(stage, inst_prim, world_mat, 0, fb.PARAMS, mat)
    print(f"[script] placed two fillets under {fb.FILLET_ROOT}/fillet_0000_{{L,R}}\n")


if __name__ == "__main__":
    main()
