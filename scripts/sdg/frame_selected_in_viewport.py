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

"""Frame the active viewport's camera on the currently-selected prim(s).

Paste into the Omniverse Script Editor (Window > Script Editor) and run with
the prim(s) you want to focus on selected in the viewport / stage tree.

Mirrors the pipeline's `auto_locate_component` rule:
  - camera XY  = world-aligned bbox center of the selection
  - camera Z   = bbox z_max + max(5 mm, 1.5 × bbox z-extent)  (clearance over the top face)
  - projection = orthographic
  - horizontal_aperture = (longer XY dim) × 4/3   (1/3 total padding, ~17% each side)
  - vertical_aperture   = horizontal_aperture × (viewport_height / viewport_width)
  - if bbox Y > bbox X, the aperture pair is swapped so the bbox long-axis
    lines up with the viewport long-axis.
"""
import omni.usd
from pxr import Gf, Usd, UsdGeom

try:
    from omni.kit.viewport.utility import get_active_viewport
except ImportError:
    raise RuntimeError(
        "Couldn't import omni.kit.viewport.utility — make sure you're running this "
        "inside an Omniverse app with a viewport (e.g., Isaac Sim, USD Composer)."
    )


def frame_active_viewport_on_selection() -> None:
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("No USD stage open.")

    sel = ctx.get_selection().get_selected_prim_paths()
    if not sel:
        raise RuntimeError("Select one or more prims in the viewport / stage tree first.")

    prims = [stage.GetPrimAtPath(p) for p in sel if stage.GetPrimAtPath(p).IsValid()]
    if not prims:
        raise RuntimeError(f"None of the selected paths resolve to valid prims: {sel}")

    # World-aligned union bbox over the selection
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True
    )
    total = Gf.BBox3d()
    first = True
    for prim in prims:
        b = bbox_cache.ComputeWorldBound(prim)
        total = b if first else Gf.BBox3d.Combine(total, b)
        first = False
    rng = total.ComputeAlignedRange()
    if rng.IsEmpty():
        raise RuntimeError(
            "Selection's world-aligned bbox is empty — selected prims may have no "
            "geometry or no resolved extents."
        )

    center = rng.GetMidpoint()
    size = rng.GetSize()
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])

    # Padding rule: at least 1/2 padding (= bbox/2 per dim) on BOTH dims.
    # Locked frame aspect = viewport (vap/hap = vh/vw). Solve:
    #   hap >= sx * 3/2
    #   vap = hap * (vh/vw) >= sy * 3/2
    # => hap = max(sx * 3/2, sy * 3/2 * (vw/vh)); vap follows.
    # The dim whose bbox/viewport ratio is dominant gets EXACTLY 1/2 padding;
    # the other gets MORE padding. No clipping regardless of bbox aspect.
    vp = get_active_viewport()
    res = getattr(vp, "resolution", (1920, 1080))
    vw, vh = float(res[0]), float(res[1])
    pad = 3.0 / 2.0
    hap = max(sx * pad, sy * pad * (vw / vh))
    vap = hap * (vh / vw)

    # Camera Z = above the bbox top with min 5 mm clearance, or 1.5 × bbox z-extent.
    z_top = cz + (sz * 0.5)
    z_above = z_top + max(5.0, sz * 1.5)

    cam_path = vp.camera_path
    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim.IsValid():
        raise RuntimeError(f"Active viewport camera prim not found: {cam_path}")

    cam = UsdGeom.Camera(cam_prim)
    cam.GetProjectionAttr().Set("orthographic")
    # USD camera aperture is stored in "tenths of scene units" (older
    # film-format convention). Multiply scene-unit values by 10.
    cam.GetHorizontalApertureAttr().Set(float(hap * 10.0))
    cam.GetVerticalApertureAttr().Set(float(vap * 10.0))

    # Ensure a translate op and set it. If the camera has a parent xform, we
    # set translate on the camera itself; if the camera is its own xform, same path.
    xform = UsdGeom.Xformable(cam_prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(cx, cy, z_above))

    print(
        f"[frame] selection bbox: center=({cx:.3f}, {cy:.3f}, {cz:.3f}) "
        f"size=({sx:.3f} x {sy:.3f} x {sz:.3f})"
    )
    print(
        f"[frame] camera moved: pos=({cx:.3f}, {cy:.3f}, {z_above:.3f}) "
        f"ortho aperture=({hap:.3f} x {vap:.3f}) "
        f"viewport={int(vw)}x{int(vh)} aspect={vh/vw:.4f}"
    )
    print(f"[frame] applied to: {cam_path}")


if __name__ == "__main__":
    frame_active_viewport_on_selection()
