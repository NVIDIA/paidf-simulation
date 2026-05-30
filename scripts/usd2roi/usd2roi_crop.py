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

"""cad2roi (usd2roi) — Stage 3: per-ROI label-based crop.

One of three entry points (render / register / crop). Reads the same YAML
config used by the others; reads only the ``crop`` and ``output`` sections.

Two input modes (selected by ``crop.source``):

* ``crop.source: aligned`` (default) — Day-1 flow: read register-stage
  outputs from ``<output.dir>/aligned/``. Single frame
  (``semantic_segmentation_<pattern>.png``, default pattern ``0000``)
  paired with ``ref_crop.png`` / ``aligned_crop.png``.
* ``crop.source: sdg`` (or any other subdir) — Day-0 flow: read render-stage
  outputs from ``<output.dir>/sdg/``. ``crop.pattern`` typically a glob
  like ``x*_y*`` to walk every scan-grid cell. Each cell's ``rgb_<idx>.png``
  is used as both ref and real (no real photo in Day 0).

Outputs:

  Single-frame (default)::
      <output.dir>/crop/component/normal_img/NNNN.png
      <output.dir>/crop/component/cad_mask/NNNN_cad_mask.png
      <output.dir>/crop/component/semantic_segmentation_labels.json

  Single-frame with ``crop.class_dirs`` set, one subdir per class group;
  NNNN counter restarts per subdir. Single shared labels.json at crop root::
      <output.dir>/crop/<class_dir>/normal_img/NNNN.png
      <output.dir>/crop/<class_dir>/cad_mask/NNNN_cad_mask.png
      <output.dir>/crop/semantic_segmentation_labels.json

  Multi-frame (multiple seg files matched), one subdir per cell::
      <output.dir>/crop/component/<frame_idx>/normal_img/NNNN.png
      <output.dir>/crop/component/<frame_idx>/cad_mask/NNNN_cad_mask.png
      <output.dir>/crop/component/<frame_idx>/semantic_segmentation_labels.json

  Multi-frame with ``crop.class_dirs`` (Day-0 per-cell layout): each scan-grid
  cell gets its own ``<frame_idx>`` subdir under the class bucket so the
  ``x<i>_y<j>`` cell encoding is preserved. NNNN resets per ``(class, cell)``.
  A single unioned labels.json sits at crop root::
      <output.dir>/crop/<class_dir>/<frame_idx>/normal_img/NNNN.png
      <output.dir>/crop/<class_dir>/<frame_idx>/cad_mask/NNNN_cad_mask.png
      <output.dir>/crop/semantic_segmentation_labels.json

  Bridge dirs follow the same flat / per-cell layout under
  ``<output.dir>/crop/bridge/`` when ``crop.bridge: true``.

YAML schema for this stage::

    crop:
      source: aligned                       # aligned (Day 1) | sdg (Day 0); default aligned
      pattern: "0000"                       # filename glob suffix for seg files; default "0000"
      classes: [capacitor, ic]              # anchor classes; pad/solder etc. join via labelled CC
      morph_kernel: 2                       # close radius (px); merges adjacent labels
      min_area: 50                          # px²; reject smaller connected components
      max_area: null                        # px²; null = no cap
      offset: 10                            # px padding around each ROI bbox
      edge_skip: true                       # drop ROIs whose tight bbox touches an image edge
      min_coverage: 0.2                     # min labelled-pixel fraction inside crop box (0 = off)
      max_emit: null                        # int = global cap on emitted ROIs across all cells; null = no cap
      # Optional per-class output routing (single-frame only). Multiple classes
      # can share one subdir; classes not listed fall back to component/.
      # NNNN counter restarts per subdir.
      class_dirs:
        capacitor: passive_component
        ic: ic
      # Optional pairwise bridge crops:
      bridge: false
      bridge_dis: 20                        # px edge-to-edge distance threshold
      bridge_classes: []                    # classes participating in bridge pairing

Run::

    python3 scripts/usd2roi/usd2roi_crop.py --config <usd2roi_target.yaml>
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import os
import re
import shutil
import sys
from collections import Counter

import yaml

try:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
except ValueError:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from component_crop import crop_bridge_pairs, crop_rois_by_label


def _abs(p: str) -> str:
    if p.startswith("omniverse://"):
        return p
    return os.path.abspath(os.path.expanduser(p))


def main() -> int:
    ap = argparse.ArgumentParser(description="cad2roi label-based per-ROI crop (host python)")
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_root = _abs(cfg["output"]["dir"])
    crop_cfg = cfg.get("crop", {}) or {}

    # Source / pattern selectors. Defaults preserve the Day-1 register-then-crop
    # behaviour (read aligned/, single frame, suffix _0000) so existing yamls
    # don't need to add new fields.
    crop_source = crop_cfg.get("source", "aligned")
    crop_pattern = str(crop_cfg.get("pattern", "0000"))

    input_dir = os.path.join(output_root, crop_source)
    crop_root = os.path.join(output_root, "crop")
    component_dir = os.path.join(crop_root, "component")
    bridge_dir = os.path.join(crop_root, "bridge")

    # Find all seg files matching the pattern.
    seg_glob = os.path.join(input_dir, f"semantic_segmentation_{crop_pattern}.png")
    seg_files = sorted(glob_mod.glob(seg_glob))
    if not seg_files:
        print(f"ERROR: no seg files match {seg_glob}")
        if crop_source == "aligned":
            print("       Did you run usd2roi_register.py first?")
        else:
            print(f"       Did you run usd2roi_render.py first to populate {input_dir}/?")
        return 2

    target_classes = list(crop_cfg.get("classes", []) or [])
    if not target_classes:
        print("ERROR: crop.classes is empty in yaml; specify which class labels to extract.")
        print("       Example: crop:\\n  classes: [capacitor, solder, pad, ic]")
        return 2

    multi_frame = len(seg_files) > 1
    bridge_enabled = bool(crop_cfg.get("bridge", False))
    bridge_dis = float(crop_cfg.get("bridge_dis", 20))
    bridge_classes = list(crop_cfg.get("bridge_classes", []) or [])

    edge_skip = bool(crop_cfg.get("edge_skip", True))
    min_coverage = float(crop_cfg.get("min_coverage", 0.2))
    max_emit_total = crop_cfg.get("max_emit")  # int or None — global cap across cells
    class_dirs = crop_cfg.get("class_dirs") or None
    if class_dirs and not isinstance(class_dirs, dict):
        print("ERROR: crop.class_dirs must be a mapping of {class: subdir_name}.")
        return 2

    print(
        f"[usd2roi_crop] source={crop_source} pattern='{crop_pattern}' "
        f"({len(seg_files)} seg file{'s' if multi_frame else ''}) "
        f"classes={target_classes} "
        f"morph_kernel={crop_cfg.get('morph_kernel', 2)} "
        f"min_area={crop_cfg.get('min_area', 50)} "
        f"max_area={crop_cfg.get('max_area')} "
        f"offset={crop_cfg.get('offset', 10)} "
        f"edge_skip={edge_skip} "
        f"min_coverage={min_coverage} "
        f"max_emit={max_emit_total}"
    )
    if class_dirs:
        print(f"[usd2roi_crop] class_dirs: {class_dirs}")
    if bridge_enabled:
        print(f"[usd2roi_crop] bridge: dis<={bridge_dis} classes={bridge_classes or '(any)'}")

    # When class_dirs is set, NNNN runs per subdir across all cells so the
    # Day-0 scan_grid output is a flat per-class dataset. Counter + labels
    # union are caller-owned and shared across the cell loop.
    class_emit_counter: Counter | None = Counter() if class_dirs else None
    labels_union: dict | None = {} if class_dirs else None

    # Aggregate stats across frames (single-frame mode = one iteration).
    total = {
        "emitted": 0,
        "skipped_min": 0,
        "skipped_max": 0,
        "skipped_edge": 0,
        "skipped_low_coverage": 0,
        "n_components": 0,
        "bridges": 0,
        "bridge_candidates": 0,
        "bridge_in_range": 0,
    }

    seg_idx_re = re.compile(r"semantic_segmentation_(.+)\.png$")
    for seg_file in seg_files:
        if max_emit_total is not None and total["emitted"] >= max_emit_total:
            break
        m = seg_idx_re.search(os.path.basename(seg_file))
        if not m:
            print(f"  skipping (cannot extract idx): {seg_file}")
            continue
        frame_idx = m.group(1)
        labels_file = os.path.join(input_dir, f"semantic_segmentation_labels_{frame_idx}.json")
        if not os.path.exists(labels_file):
            print(f"  skipping {frame_idx}: missing labels file {labels_file}")
            continue

        # Pick ref/real per source mode.
        if crop_source == "aligned":
            # Day 1: register stage produced ref_crop.png + aligned_crop.png.
            ref_path = os.path.join(input_dir, "ref_crop.png")
            real_path = os.path.join(input_dir, "aligned_crop.png")
            for f in (ref_path, real_path):
                if not os.path.exists(f):
                    print(f"ERROR: required file missing: {f}")
                    print("       Did you run usd2roi_register.py first?")
                    return 2
        else:
            # Day 0: no real photo — synth render is both ref and real.
            ref_path = os.path.join(input_dir, f"rgb_{frame_idx}.png")
            if not os.path.exists(ref_path):
                print(f"  skipping {frame_idx}: missing rgb {ref_path}")
                continue
            real_path = ref_path

        # Per-cell subdir when multi-frame; flat layout for single-frame.
        frame_component = os.path.join(component_dir, frame_idx) if multi_frame else component_dir
        frame_bridge = os.path.join(bridge_dir, frame_idx) if multi_frame else bridge_dir

        if multi_frame:
            print(f"[usd2roi_crop] cell {frame_idx}:")

        remaining_emit = max_emit_total - total["emitted"] if max_emit_total is not None else None
        # When class_dirs is set, the worker treats output_dir as the parent
        # (crop_root) and routes each ROI under it. Otherwise pass the
        # frame-local component dir as today.
        worker_output_dir = crop_root if class_dirs else frame_component
        stats = crop_rois_by_label(
            aligned_dir=input_dir,
            ref_crop_path=ref_path,
            real_crop_path=real_path,
            output_dir=worker_output_dir,
            target_classes=target_classes,
            morph_kernel=int(crop_cfg.get("morph_kernel", 2)),
            min_area=int(crop_cfg.get("min_area", 50)),
            max_area=crop_cfg.get("max_area"),
            offset=int(crop_cfg.get("offset", 10)),
            edge_skip=edge_skip,
            min_coverage=min_coverage,
            max_emit=remaining_emit,
            frame_idx=frame_idx,
            class_dirs=class_dirs,
            class_emit_counter=class_emit_counter,
            multi_frame=multi_frame,
        )
        total["emitted"] += stats["n_emitted"]
        total["skipped_min"] += stats["skipped_min_area"]
        total["skipped_max"] += stats["skipped_max_area"]
        total["skipped_edge"] += stats.get("skipped_edge", 0)
        total["skipped_low_coverage"] += stats.get("skipped_low_coverage", 0)
        total["n_components"] += stats["n_components_total"]
        if class_dirs:
            # Union of color->class entries across all cells; same RGBA -> same
            # class within a Replicator run, so dict.update() is conflict-free.
            with open(labels_file) as f:
                labels_union.update(json.load(f))
        else:
            shutil.copyfile(
                labels_file,
                os.path.join(frame_component, "semantic_segmentation_labels.json"),
            )

        if bridge_enabled:
            bridge_stats = crop_bridge_pairs(
                rois=stats["rois"],
                ref_crop_path=ref_path,
                real_crop_path=real_path,
                seg_crop_path=seg_file,
                output_dir=frame_bridge,
                bridge_dis=bridge_dis,
                bridge_classes=bridge_classes or None,
                offset=int(crop_cfg.get("offset", 10)),
            )
            total["bridges"] += bridge_stats["n_emitted"]
            total["bridge_candidates"] += bridge_stats["n_candidates"]
            total["bridge_in_range"] += bridge_stats["n_pairs_in_range"]
            # crop_bridge_pairs only creates frame_bridge when it emits a pair;
            # ensure it exists so the labels copy is unconditional.
            os.makedirs(frame_bridge, exist_ok=True)
            shutil.copyfile(
                labels_file,
                os.path.join(frame_bridge, "semantic_segmentation_labels.json"),
            )

    if class_dirs:
        # Write merged labels.json once at crop root after all cells processed.
        with open(os.path.join(crop_root, "semantic_segmentation_labels.json"), "w") as f:
            json.dump(labels_union, f, indent=2)

    if bridge_enabled:
        print(
            f"[usd2roi_crop] bridges: {total['bridges']} emitted "
            f"(candidates={total['bridge_candidates']}, "
            f"in_range={total['bridge_in_range']})"
        )

    print(
        f"[usd2roi_crop] emitted={total['emitted']}, "
        f"skipped_min={total['skipped_min']}, "
        f"skipped_max={total['skipped_max']}, "
        f"skipped_edge={total['skipped_edge']}, "
        f"skipped_low_coverage={total['skipped_low_coverage']}, "
        f"n_components_total={total['n_components']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
