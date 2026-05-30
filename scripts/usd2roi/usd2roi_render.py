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

"""cad2roi (usd2roi) — Stage 1: USD render only.

One of three entry points (render / register / crop). Reads the same YAML
config used by the others; reads only the ``scene``, ``camera``,
``resolution``, ``semantics``, ``writer``, and ``output`` sections.

Outputs:
    <output.dir>/sdg/
        rgb_0000.png
        semantic_segmentation_0000.png + _labels.json
        bounding_box_2d_tight_0000.{npy,json,_prim_paths.json}  (if writer.bbox=true)
        metadata.txt
        semantic_stats.json   (this script — quick summary of rules applied)

Run via Isaac-Sim (or its bundled Kit App)::

    /isaac-sim/kit/kit /isaac-sim/apps/isaacsim.exp.base.kit --no-window --exec \\
        "scripts/usd2roi/usd2roi_render.py --config <usd2roi_target.yaml>"

After this completes, run :mod:`usd2roi_register` (host python) to do MI
registration against ``yaml.real_image``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

import carb.settings
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
import yaml
from pxr import Gf, UsdGeom, UsdLux

# Local imports (this file lives in scripts/usd2roi/)
try:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
except ValueError:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_rules import _uninstance_ancestor, find_matching_prims

logger = logging.getLogger("usd2roi_render")


# === Generated camera constants ===
CAMERA_PRIM_PATH = "/World/cad2roi_camera"
CAMERA_Z = 5000.0
CAMERA_FOCAL_LENGTH = 50.0
CAMERA_ROTATION_XYZ_DEFAULT = (0.0, 0.0, 90.0)


# === CLI ===
parser = argparse.ArgumentParser(description="cad2roi USD render (Kit-only)")
parser.add_argument("--config", type=str, required=True, help="YAML config path")
_cli_args, _ = parser.parse_known_args()

with open(_cli_args.config) as f:
    CFG = yaml.safe_load(f)

print(f"[usd2roi_render] Loaded config: {_cli_args.config}", flush=True)

_app = omni.kit.app.get_app()


# === Helpers ===


def _abs_or_passthrough(p: str) -> str:
    if p.startswith("omniverse://"):
        return p
    return os.path.abspath(os.path.expanduser(p))


def _create_orthographic_camera(
    stage: Any,
    translate_xy: tuple[float, float],
    horizontal_aperture: float,
    resolution: tuple[int, int],
    rotation_xyz: tuple[float, float, float] = CAMERA_ROTATION_XYZ_DEFAULT,
) -> str:
    """Author an orthographic camera at CAMERA_PRIM_PATH using pxr.

    ``rep.create.camera`` doesn't support orthographic projection, so we
    author the Camera prim directly and pass its path as a string to the
    render product.
    """
    cam = UsdGeom.Camera.Define(stage, CAMERA_PRIM_PATH)
    cam.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
    cam.GetFocalLengthAttr().Set(float(CAMERA_FOCAL_LENGTH))
    cam.GetHorizontalApertureAttr().Set(float(horizontal_aperture))
    res_w, res_h = int(resolution[0]), int(resolution[1])
    v_aperture = float(horizontal_aperture) * (res_h / res_w)
    cam.GetVerticalApertureAttr().Set(v_aperture)

    xformable = UsdGeom.Xformable(cam.GetPrim())
    xformable.SetXformOpOrder([])
    for attr_name in (
        "xformOp:translate",
        "xformOp:rotateXYZ",
        "xformOp:rotateZYX",
        "xformOp:rotateX",
        "xformOp:rotateY",
        "xformOp:rotateZ",
        "xformOp:scale",
    ):
        if cam.GetPrim().HasAttribute(attr_name):
            cam.GetPrim().RemoveProperty(attr_name)
    t_op = xformable.AddTranslateOp()
    t_op.Set(Gf.Vec3d(float(translate_xy[0]), float(translate_xy[1]), CAMERA_Z))
    r_op = xformable.AddRotateXYZOp()
    r_op.Set(Gf.Vec3f(*rotation_xyz))

    print(
        f"[usd2roi_render] Camera {CAMERA_PRIM_PATH}: ortho, "
        f"focal={CAMERA_FOCAL_LENGTH}, hAp={horizontal_aperture:.2f}, "
        f"vAp={v_aperture:.2f}, t=({translate_xy[0]:.2f}, {translate_xy[1]:.2f}, {CAMERA_Z:.0f}), "
        f"rot={tuple(rotation_xyz)}",
        flush=True,
    )
    return CAMERA_PRIM_PATH


async def _open_usd_stage(usd_path: str) -> Any:
    ctx = omni.usd.get_context()
    ctx.disable_save_to_recent_files()
    ok, err = await ctx.open_stage_async(usd_path)
    ctx.enable_save_to_recent_files()
    if not ok:
        raise RuntimeError(f"Cannot open USD: {usd_path} ({err})")
    while not _app.is_app_ready() or not ctx.get_stage():
        await _app.next_update_async()
    while ctx.get_stage_loading_status()[2] > 0:
        await _app.next_update_async()
    print(f"[usd2roi_render] Opened stage: {usd_path}", flush=True)
    return ctx.get_stage()


# === Render ===


async def run_render() -> None:
    output_root = _abs_or_passthrough(CFG["output"]["dir"])
    sdg_dir = os.path.join(output_root, "sdg")
    os.makedirs(sdg_dir, exist_ok=True)

    # --- Step 1: open USD ---
    carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
    scene_path = _abs_or_passthrough(CFG["scene"])
    stage = await _open_usd_stage(scene_path)

    carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
    # --- Step 1.5: defensive 0-light fallback ---
    # Unlike sdg_pipeline.py (which builds a 900-DiskLight rig from the yaml
    # `rig:` + `lighting:` blocks via common.build_ring_lights), this script
    # renders against whatever lights the scene file authored. A bare PCB CAD
    # without authored lights renders near-black RGB — drop in a single
    # DomeLight so the user gets something usable instead of debugging an
    # empty frame. Author real lights in the scene to override.
    n_lights = sum(1 for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI))
    if n_lights == 0:
        fallback = UsdLux.DomeLight.Define(stage, "/World/_usd2roi_fallback_light")
        fallback.CreateIntensityAttr(2500.0)
        print(
            "[usd2roi_render] WARNING: scene has 0 authored lights; "
            "added fallback DomeLight at /World/_usd2roi_fallback_light "
            "(intensity=2500). Author proper lighting in the scene for "
            "production runs.",
            flush=True,
        )
    else:
        print(f"[usd2roi_render] Scene has {n_lights} authored light(s).", flush=True)

    # --- Step 2: ortho camera (pxr; rep.create.camera has no ortho) ---
    cam_cfg = CFG["camera"]
    resolution = tuple(CFG["resolution"])
    translate_xy = tuple(cam_cfg["translate"])
    if len(translate_xy) != 2:
        raise ValueError(
            f"camera.translate must be [x, y] (z is fixed at {CAMERA_Z}); got {translate_xy}"
        )
    rotation_xyz = tuple(cam_cfg.get("rotation_xyz", CAMERA_ROTATION_XYZ_DEFAULT))
    if len(rotation_xyz) != 3:
        raise ValueError(
            f"camera.rotation_xyz must be [rx, ry, rz] in degrees; got {rotation_xyz}"
        )
    cam_path = _create_orthographic_camera(
        stage,
        translate_xy=translate_xy,
        horizontal_aperture=float(cam_cfg["horizontal_aperture"]),
        resolution=resolution,
        rotation_xyz=rotation_xyz,
    )

    # --- Step 3: collect rules → (label_tuples, prim_paths) groups (with uninstance) ---
    sem_rules = CFG.get("semantics", []) or []
    rule_groups: list[tuple[list[tuple[str, str]], list[str]]] = []
    n_uninstanced = 0
    for rule in sem_rules:
        prims = find_matching_prims(stage, rule["match"])
        if not prims:
            continue
        for prim in prims:
            anc = _uninstance_ancestor(prim)
            if anc is not None:
                n_uninstanced += 1
        paths = [str(p.GetPath()) for p in prims]
        labels = [(str(k), str(v)) for k, v in (rule.get("labels") or {}).items()]
        if labels:
            rule_groups.append((labels, paths))
    print(
        f"[usd2roi_render] Semantics: {len(sem_rules)} rule(s) -> {len(rule_groups)} group(s), "
        f"{n_uninstanced} uninstance(s)",
        flush=True,
    )

    # Settle stage mutations from the uninstance loop. Each uninstance copies a
    # LibRef prototype into a unique prim, which triggers a fresh material
    # binding + MDL JIT compile for that prim's body shader. Spark scenes
    # produce ~5800 uninstances, i.e. ~5800 new MDL compile units pending.
    # Without explicit drain, the next step_async (warmup or capture) races
    # against MDL JIT and component bodies render empty on cold GPUs while
    # already-compiled materials (PASTEMASK pad, IC's direct-mesh shader)
    # still render correctly — the failure signature is class-asymmetric:
    # `[ic, pad]` instead of `[capacitor, ic, pad, solder]`.
    #
    # Drain by polling Kit's stage-loading status + advancing the app event
    # loop until the post-uninstance settle quiesces. This is the same
    # idiom used after USD payload load (lines 127-129).
    _app = omni.kit.app.get_app()
    _ctx = omni.usd.get_context()
    for _ in range(60):
        await _app.next_update_async()
    while _ctx.get_stage_loading_status()[2] > 0:
        await _app.next_update_async()
    print(
        f"[usd2roi_render] Post-uninstance settle done (pending stage ops drained)",
        flush=True,
    )

    # --- Step 4: build Replicator graph + render one frame ---
    writer_cfg = dict(CFG["writer"])
    writer_cfg.pop("semantic_filter_predicate", None)

    render_product = rep.create.render_product(cam_path, resolution)

    for labels, paths in rule_groups:
        with rep.get.prim_at_path(paths):
            rep.modify.semantics(labels)

    # Warmup so all annotators have buffers ready before writer captures.
    # Required for RGB at higher resolutions — fewer steps risk RGB silently
    # being skipped while seg still writes.
    #
    # WORKAROUND for OMPE-90559 / NVBug-5986841 (Replicator >= 1.13.16):
    # step_async short-circuits to a no-render fast path when
    # AnnotatorRegistry has no attached annotators, and _initialize_async
    # skips the renderer-prep sequence under the same condition. Without
    # an annotator attached during warmup the orchestrator ends up partially
    # initialized — when the writer later attaches, the renderer never
    # produces LdrColorSD / SemanticSegmentation render vars and the next
    # step_async hangs forever waiting for outputs.
    #
    # Attaching a throwaway rgb annotator across warmup forces
    # has_attached_annotators() == True so the full init/step path runs.
    # Detach + delete before writer.attach() so writer outputs are
    # unaffected. Pass render_product.path (str) to .detach(); a Replicator
    # 1.13.16 bug in annotators.py:1457 only unwraps HydraTexture for
    # single-object inputs.
    # Pass rt_subframes per step to give MDL programs enough render-cycle
    # budget to JIT-compile component-body shaders before the capture step.
    # Default rt_subframes=1 (preview path) silently misses the compile
    # window on cold-boot containers — Hydra captures the next frame with
    # those shaders still compiling, and component bodies render as empty
    # while semantic-seg writes correctly. sdg_pipeline.py's scan loop
    # absorbs this race over 100 frames; the single-shot Day-1 render
    # doesn't, so we explicitly mirror sdg_pipeline.py's warmup shape here.
    # See troubleshooting.md `[day1]` "component bodies are missing".
    _warmup_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    _rp_path = render_product.path
    _warmup_annot.attach(_rp_path)
    try:
        for _ in range(10):
            await rep.orchestrator.step_async(rt_subframes=4, delta_time=0.0)
    finally:
        _warmup_annot.detach(_rp_path)
        del _warmup_annot

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=sdg_dir, **writer_cfg)
    writer.attach([render_product])

    rep.set_global_seed(0)
    await rep.orchestrator.step_async()

    rgb_check = os.path.join(sdg_dir, "rgb_0000.png")
    if not os.path.exists(rgb_check):
        print("[usd2roi_render] RGB missing after first capture; retrying once.", flush=True)
        await rep.orchestrator.step_async()

    writer.detach()
    render_product.destroy()
    print(f"[usd2roi_render] Render done -> {sdg_dir}", flush=True)

    # Persist a quick summary of what was labelled (helps debug crop later).
    with open(os.path.join(sdg_dir, "semantic_stats.json"), "w") as f:
        json.dump(
            {
                "n_rules": len(sem_rules),
                "n_groups": len(rule_groups),
                "n_uninstanced_ancestors": n_uninstanced,
                "groups": [
                    {"labels": dict(labels), "n_prims": len(paths), "sample_paths": paths[:5]}
                    for labels, paths in rule_groups
                ],
            },
            f,
            indent=2,
        )

    _app.shutdown()


# Module-level scheduling (Kit App --exec convention)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
asyncio.ensure_future(run_render())
