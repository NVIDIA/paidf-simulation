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

"""cad2roi (usd2roi) — Stage 2: registration + sdg_crop.

One of three entry points (render / register / crop). Reads the same YAML
config used by the others; reads only the ``real_image`` and
``registration`` sections.

Inputs (must exist on disk before running):
    <output.dir>/sdg/rgb_0000.png        (from usd2roi_render.py)
    <output.dir>/sdg/semantic_segmentation_0000.png
    <output.dir>/sdg/bounding_box_2d_tight_0000.npy   (optional; pass-through)
    <real_image>                                      (the real photo)

Outputs:
    <output.dir>/aligned/
        ref_crop.png         synth, cropped to MI valid-overlap bbox (not warped)
        aligned_crop.png     real warped into ref frame, same crop
        params.json          5-DOF affine + MI before/after
        blink.gif            ref ↔ aligned alternating (for visual QA)
        semantic_segmentation_0000.png  (sdg_crop'd, crop-local coords)
        bounding_box_2d_tight_0000.*    (sdg_crop'd, if bbox npy was present)
        sdg_crop_stats.json

Run on host python (uses cupy if available; no Kit needed)::

    python3 scripts/usd2roi/usd2roi_register.py --config <usd2roi_target.yaml>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import yaml

# Local sibling imports (scripts/usd2roi/)
try:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
except ValueError:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from registration import HAS_CUPY, align, crop_to_valid_bbox, save_blink_gif, save_params
from sdg_crop import crop_sdg


def _abs(p: str) -> str:
    if p.startswith("omniverse://"):
        return p
    return os.path.abspath(os.path.expanduser(p))


def main() -> int:
    ap = argparse.ArgumentParser(description="cad2roi MI registration + sdg_crop (host python)")
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_root = _abs(cfg["output"]["dir"])
    sdg_dir = os.path.join(output_root, "sdg")
    aligned_dir = os.path.join(output_root, "aligned")

    sdg_rgb = os.path.join(sdg_dir, "rgb_0000.png")
    real_path = _abs(cfg["real_image"])
    if not os.path.exists(sdg_rgb):
        print(f"ERROR: SDG render not found at {sdg_rgb}; run usd2roi_render.py first.")
        return 2
    if not os.path.exists(real_path):
        print(f"ERROR: real_image not found: {real_path}")
        return 2

    os.makedirs(aligned_dir, exist_ok=True)

    reg = cfg.get("registration", {}) or {}
    gpu_pref = reg.get("gpu", "auto")
    use_gpu = HAS_CUPY if gpu_pref == "auto" else bool(gpu_pref)
    print(f"[usd2roi_register] gpu={use_gpu} (cupy available: {HAS_CUPY})", flush=True)
    print(f"[usd2roi_register] ref={sdg_rgb}\n                   test={real_path}")

    # === Step 1: MI registration ===
    t0 = time.perf_counter()
    warped, mask, params = align(
        sdg_rgb,
        real_path,
        no_resize=reg.get("no_resize", True),
        sx_range=tuple(reg.get("sx_range", [0.9, 1.1, 0.02])),
        sy_range=tuple(reg.get("sy_range", [0.9, 1.1, 0.02])),
        rot_range_deg=tuple(reg.get("rot_range_deg", [-1.0, 1.0, 0.05])),
        shift_range=int(reg.get("shift_range", 200)),
        shift_step=int(reg.get("shift_step", 4)),
        pyr_levels=int(reg.get("pyr_levels", 3)),
        num_bins=int(reg.get("bins", 64)),
        use_gpu=use_gpu,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[usd2roi_register] MI {params.get('mi_before', float('nan')):.4f} -> "
        f"{params.get('mi_after', float('nan')):.4f}  "
        f"sx={params['scaleX']:.3f} sy={params['scaleY']:.3f} "
        f"rot={params['rotation_deg']:.2f} t=({params['tx']:.1f}, {params['ty']:.1f})  "
        f"({elapsed:.1f}s)"
    )

    min_mi = reg.get("min_mi")
    if min_mi is not None and float(params.get("mi_after", -1.0)) < float(min_mi):
        print(f"ERROR: MI after ({params['mi_after']:.4f}) below min_mi={min_mi}; aborting.")
        return 2

    # === Step 2: crop ref + warped to valid-overlap bbox ===
    ref_full = cv2.imread(sdg_rgb, cv2.IMREAD_UNCHANGED)
    ref_crop, aligned_crop, bbox = crop_to_valid_bbox(ref_full, warped, mask)
    ref_crop_path = os.path.join(aligned_dir, "ref_crop.png")
    aligned_crop_path = os.path.join(aligned_dir, "aligned_crop.png")
    cv2.imwrite(ref_crop_path, ref_crop)
    cv2.imwrite(aligned_crop_path, aligned_crop)
    save_params(params, os.path.join(aligned_dir, "params.json"))
    save_blink_gif(ref_crop, aligned_crop, os.path.join(aligned_dir, "blink.gif"))
    print(f"[usd2roi_register] aligned/ written; bbox={bbox}")

    # === Step 3: crop SDG annotations (semseg + bbox) to align bbox ===
    crop_cfg = cfg.get("crop", {}) or {}
    xform_depth = int(crop_cfg.get("xform_depth", 7))
    sdg_stats = crop_sdg(
        sdg_dir,
        aligned_dir,
        bbox,
        frame_idx="0000",
        xform_depth=xform_depth,
        outputs=("semseg", "bbox"),
    )
    with open(os.path.join(aligned_dir, "sdg_crop_stats.json"), "w") as f:
        json.dump(sdg_stats, f, indent=2)
    print(
        f"[usd2roi_register] sdg_crop: {sdg_stats['totals']['total']} total, "
        f"{sdg_stats['totals']['kept']} kept, {sdg_stats['totals']['dropped']} dropped"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
