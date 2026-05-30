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
Build a ChangeNet-style pair dataset from good + defect pipeline outputs.

Two output granularities:
  --mode frame      (default) full-frame golden/defect/mask triples
  --mode component  per-component crops; bbox = union of good+defect bboxes
                    for components carrying a non-empty `defect` label

Pair sources supported in both modes:
  - Pose defects (shift/tombstone/sideflip): mask is in the defect dir
  - Missing components: mask is in the good (reference) dir
Mask location is auto-detected by checking both directories.

Usage (frame mode):
    python build_pair_dataset.py \
        --good  sdg_test_output/pair_test/good/trigger_0000 \
        --defect sdg_test_output/pair_test/defect/trigger_0000 \
        --output Pair-dataset

Usage (component mode):
    python build_pair_dataset.py \
        --good  sdg_test_output/pair_test/good/trigger_0000 \
        --defect sdg_test_output/pair_test/defect/trigger_0000 \
        --output Pair-dataset-component \
        --mode component --offset 10

Output structure (both modes share golden/defect/mask layout):
    <output>/
        golden/   {idx}[_{comp}]_{LightMode}.png
        defect/   {idx}[_{comp}]_{LightMode}.png
        mask/     {idx}[_{comp}]_{LightMode}.png + .json
    Component mode adds {comp} (component xform name) to filenames.

Light mode suffix is determined from metadata.json (lighting.ring_light)
or can be overridden with --light-mode.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image


def find_frames(trigger_dir: str, prefix: str, ext: str) -> dict[str, str]:
    """Find files matching {prefix}_{frame}.{ext} where frame is either
    NNNN (older writer naming) or x{a}_y{b} (auto scan_grid renaming).
    Returns {frame_idx_str: path}."""
    pattern = os.path.join(trigger_dir, f"{prefix}_*.{ext}")
    files: dict[str, str] = {}
    for path in glob.glob(pattern):
        match = re.search(rf"{prefix}_(\d+|x\d+_y\d+)\.{ext}$", path)
        if match:
            files[match.group(1)] = path
    return files


def _classify_lighting(lighting: dict) -> str:
    """SolderLight when any layer is colored (R/G/B-saturated); WhiteLight when all 3
    layers are near-neutral (R≈G≈B). Falls back to SolderLight when the metadata
    schema is partial / unknown."""
    layers = lighting.get("layers") or {}
    saturated = 0
    for spec in layers.values():
        colors = [
            float(spec.get("color_r", [0, 0])[0]),
            float(spec.get("color_g", [0, 0])[0]),
            float(spec.get("color_b", [0, 0])[0]),
        ]
        # White-balanced setups use all-similar low-floor colors (e.g. 0.8–1.0
        # across R/G/B); RGB ring uses a one-channel-dominant per layer.
        if max(colors) - min(colors) > 0.5:
            saturated += 1
    return "SolderLight" if saturated >= 2 else "WhiteLight"


def detect_light_mode(trigger_dir: str) -> str:
    """Detect light mode from metadata.json. Prefer the explicit
    `config.lighting.ring_light` toggle when present (it's authoritative);
    fall back to inferring from yaml color ranges when the toggle is absent."""
    for meta_path in (
        os.path.join(trigger_dir, "metadata.json"),
        os.path.join(os.path.dirname(trigger_dir), "metadata.json"),
    ):
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            cfg_light = meta.get("config", {}).get("lighting", {})
            if "ring_light" in cfg_light:
                return "SolderLight" if cfg_light["ring_light"] else "WhiteLight"
            return _classify_lighting(cfg_light)
    return "SolderLight"


def build_pair_dataset(
    good_dir: str, defect_dir: str, output_dir: str, light_mode: str | None = None
) -> None:
    golden_dir = os.path.join(output_dir, "golden")
    defect_out = os.path.join(output_dir, "defect")
    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(golden_dir, exist_ok=True)
    os.makedirs(defect_out, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Determine light mode suffix
    if light_mode is None:
        light_mode = detect_light_mode(defect_dir)
    print(f"Light mode: {light_mode}")

    good_rgb = find_frames(good_dir, "rgb", "png")
    defect_rgb = find_frames(defect_dir, "rgb", "png")

    # Auto-detect mask source: check defect dir first, fall back to good dir
    defect_seg = find_frames(defect_dir, "semantic_segmentation", "png")
    defect_labels = find_frames(defect_dir, "semantic_segmentation_labels", "json")
    if not defect_seg:
        defect_seg = find_frames(good_dir, "semantic_segmentation", "png")
        defect_labels = find_frames(good_dir, "semantic_segmentation_labels", "json")
        if defect_seg:
            print("Mask source: good dir (missing mode)")
        else:
            print("Warning: no semantic_segmentation found in either directory")
    else:
        print("Mask source: defect dir (pose defect mode)")

    common_frames = sorted(set(good_rgb) & set(defect_rgb))
    if not common_frames:
        print(f"No matching frames found between {good_dir} and {defect_dir}")
        return

    print(f"Found {len(common_frames)} paired frames")

    copied = 0
    skipped = 0
    for idx in common_frames:
        # idx is the captured suffix string: "0000" (older) or "x0_y0" (auto-grid).
        frame_id = idx

        # Check if mask has any defect pixels
        if idx not in defect_seg or idx not in defect_labels:
            skipped += 1
            print(f"  Skipping frame {frame_id}: missing segmentation or labels")
            continue

        with open(defect_labels[idx]) as f:
            labels = json.load(f)

        # Find defect colors (entries with "defect" key, not "class")
        defect_colors = set()
        for color_str, info in labels.items():
            if "defect" in info:
                rgba = tuple(int(x) for x in color_str.strip("()").split(", "))
                defect_colors.add(rgba)

        if not defect_colors:
            skipped += 1
            print(f"  Skipping frame {frame_id}: no defect labels")
            continue

        # Check if any defect-colored pixel exists in the mask
        mask_img = np.array(Image.open(defect_seg[idx]))
        has_defect = False
        for rgba in defect_colors:
            match = np.all(mask_img == np.array(rgba, dtype=np.uint8), axis=2)
            if match.any():
                has_defect = True
                break

        if not has_defect:
            skipped += 1
            print(f"  Skipping frame {frame_id}: no defect pixels visible")
            continue

        suffix = f"{frame_id}_{light_mode}"

        # golden
        shutil.copy2(good_rgb[idx], os.path.join(golden_dir, f"{suffix}.png"))

        # defect
        shutil.copy2(defect_rgb[idx], os.path.join(defect_out, f"{suffix}.png"))

        # mask
        shutil.copy2(defect_seg[idx], os.path.join(mask_dir, f"{suffix}.png"))
        if idx in defect_labels:
            shutil.copy2(defect_labels[idx], os.path.join(mask_dir, f"{suffix}.json"))

        copied += 1

    print(f"Done -> {output_dir}")
    print(f"  {copied} pairs copied, {skipped} skipped (blank mask)")
    print(f"  golden: {copied} images")
    print(f"  defect: {copied} images")
    print(f"  mask:   {copied} masks + labels")


def extract_component_key(prim_path: str, xform_depth: int) -> str:
    """Extract component Xform path from a full prim path."""
    parts = prim_path.split("/")
    if len(parts) >= xform_depth:
        return "/".join(parts[:xform_depth])
    return prim_path


def _build_components_index(
    anno_dir: str,
    frame_idx: str,
    xform_depth: int,
) -> dict[str, dict[str, Any]] | None:
    """Load bbox+prim_paths and merge sub-mesh bboxes by component xform.

    Returns {comp_key: {x_min, y_min, x_max, y_max, sid}}, or None if files missing.
    """
    bbox_path = os.path.join(anno_dir, f"bounding_box_2d_tight_{frame_idx}.npy")
    prim_paths_path = os.path.join(anno_dir, f"bounding_box_2d_tight_prim_paths_{frame_idx}.json")
    if not os.path.exists(bbox_path) or not os.path.exists(prim_paths_path):
        return None
    bboxes = np.load(bbox_path)
    with open(prim_paths_path) as f:
        prim_paths = json.load(f)

    components: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "x_min": 99999,
            "y_min": 99999,
            "x_max": 0,
            "y_max": 0,
            "sid": None,
        }
    )
    for entry, ppath in zip(bboxes, prim_paths):
        comp_key = extract_component_key(ppath, xform_depth)
        c = components[comp_key]
        c["x_min"] = min(c["x_min"], int(entry["x_min"]))
        c["y_min"] = min(c["y_min"], int(entry["y_min"]))
        c["x_max"] = max(c["x_max"], int(entry["x_max"]))
        c["y_max"] = max(c["y_max"], int(entry["y_max"]))
        c["sid"] = str(int(entry["semanticId"]))
    return components


def build_pair_dataset_component(
    good_dir: str,
    defect_dir: str,
    output_dir: str,
    offset: int = 10,
    class_filter: list[str] | None = None,
    light_mode: str | None = None,
    xform_depth: int = 7,
) -> None:
    """Per-component golden/defect/mask triples cropped at union(good, defect) bbox.

    Only components with a non-empty `defect` label in defect_dir's labels are
    emitted. Mask comes from defect_dir's semantic_segmentation. If a component
    has no matching prim_path in good_dir, falls back to defect-only bbox.
    """
    if light_mode is None:
        light_mode = detect_light_mode(defect_dir)
    print(f"Light mode: {light_mode}")

    golden_out = os.path.join(output_dir, "golden")
    defect_out = os.path.join(output_dir, "defect")
    mask_out = os.path.join(output_dir, "mask")
    os.makedirs(golden_out, exist_ok=True)
    os.makedirs(defect_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    bbox_files = sorted(
        [
            f
            for f in os.listdir(defect_dir)
            if f.startswith("bounding_box_2d_tight_") and f.endswith(".npy")
        ]
    )
    if not bbox_files:
        print(f"No bounding_box_2d_tight_*.npy files found in {defect_dir}")
        return

    total = 0
    skipped_edge = 0
    skipped_coverage = 0
    skipped_no_good_bbox = 0
    for bbox_file in bbox_files:
        m = re.match(r"bounding_box_2d_tight_(\d+|x\d+_y\d+)\.npy$", bbox_file)
        if not m:
            continue
        frame_idx = m.group(1)

        defect_comps = _build_components_index(defect_dir, frame_idx, xform_depth)
        if defect_comps is None:
            continue
        good_comps = _build_components_index(good_dir, frame_idx, xform_depth)

        label_file = os.path.join(defect_dir, f"bounding_box_2d_tight_labels_{frame_idx}.json")
        if not os.path.exists(label_file):
            continue
        with open(label_file) as f:
            labels = json.load(f)

        defect_rgb_path = os.path.join(defect_dir, f"rgb_{frame_idx}.png")
        good_rgb_path = os.path.join(good_dir, f"rgb_{frame_idx}.png")
        sem_path = os.path.join(defect_dir, f"semantic_segmentation_{frame_idx}.png")
        sem_labels_path = os.path.join(defect_dir, f"semantic_segmentation_labels_{frame_idx}.json")
        if not (
            os.path.exists(defect_rgb_path)
            and os.path.exists(good_rgb_path)
            and os.path.exists(sem_path)
        ):
            continue

        defect_rgb_img = Image.open(defect_rgb_path)
        good_rgb_img = Image.open(good_rgb_path)
        sem_img = Image.open(sem_path)
        img_w, img_h = defect_rgb_img.size

        sem_labels_blob: str | None = None
        if os.path.exists(sem_labels_path):
            with open(sem_labels_path) as f:
                sem_labels_blob = f.read()

        frame_crops = 0
        for comp_key, dc in defect_comps.items():
            label_info = labels.get(dc["sid"], {})
            label_class = label_info.get("class", "")
            label_defect = label_info.get("defect", "")

            if not label_defect:
                continue

            if class_filter:
                match = any(f in label_class or f in label_defect for f in class_filter)
                if not match:
                    continue

            # Union bbox (good ∪ defect); fallback to defect-only when good lacks the prim_path
            gc = good_comps.get(comp_key) if good_comps else None
            if gc is None:
                skipped_no_good_bbox += 1
                u_x1, u_y1 = dc["x_min"], dc["y_min"]
                u_x2, u_y2 = dc["x_max"], dc["y_max"]
            else:
                u_x1 = min(dc["x_min"], gc["x_min"])
                u_y1 = min(dc["y_min"], gc["y_min"])
                u_x2 = max(dc["x_max"], gc["x_max"])
                u_y2 = max(dc["y_max"], gc["y_max"])

            if u_x1 <= 0 or u_y1 <= 0 or u_x2 >= img_w - 1 or u_y2 >= img_h - 1:
                skipped_edge += 1
                continue

            x1 = max(0, u_x1 - offset)
            y1 = max(0, u_y1 - offset)
            x2 = min(img_w, u_x2 + offset)
            y2 = min(img_h, u_y2 + offset)
            crop_box = (x1, y1, x2, y2)

            sem_crop = sem_img.crop(crop_box)
            sem_arr = np.array(sem_crop)
            if sem_arr.shape[-1] == 4:
                nonzero = np.count_nonzero(sem_arr[:, :, 3])
            else:
                nonzero = np.count_nonzero(sem_arr.max(axis=2))
            coverage = nonzero / (sem_arr.shape[0] * sem_arr.shape[1])
            if coverage < 0.2:
                skipped_coverage += 1
                continue

            comp_name = comp_key.split("/")[-1]
            suffix = f"{frame_idx}_{comp_name}_{light_mode}"

            good_rgb_img.crop(crop_box).save(os.path.join(golden_out, f"{suffix}.png"))
            defect_rgb_img.crop(crop_box).save(os.path.join(defect_out, f"{suffix}.png"))
            sem_crop.save(os.path.join(mask_out, f"{suffix}.png"))
            if sem_labels_blob is not None:
                with open(os.path.join(mask_out, f"{suffix}.json"), "w") as f:
                    f.write(sem_labels_blob)

            frame_crops += 1

        total += frame_crops
        if frame_crops > 0:
            print(f"  Frame {frame_idx}: {frame_crops} component pairs cropped")

    print(f"Done -> {output_dir}")
    print(
        f"  {total} component pairs (skipped: {skipped_edge} edge, "
        f"{skipped_coverage} low-coverage, {skipped_no_good_bbox} no-good-bbox)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ChangeNet pair dataset from good + defect outputs"
    )
    parser.add_argument("--good", required=True, help="Good pipeline trigger directory")
    parser.add_argument("--defect", required=True, help="Defect pipeline trigger directory")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument(
        "--light-mode",
        default=None,
        choices=["SolderLight", "WhiteLight"],
        help="Light mode suffix (auto-detected from metadata.json if omitted)",
    )
    parser.add_argument(
        "--mode",
        default="frame",
        choices=["frame", "component"],
        help="frame: full-frame triples (default). "
        "component: per-component crops with union(good, defect) bbox.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=10,
        help="(component mode) extra pixels around the union bbox (default 10)",
    )
    parser.add_argument(
        "--class-filter",
        nargs="+",
        default=None,
        help="(component mode) only emit components whose class/defect "
        "contains any of these substrings",
    )
    parser.add_argument(
        "--xform-depth",
        type=int,
        default=7,
        help="(component mode) USD hierarchy depth for component xform extraction",
    )
    args = parser.parse_args()

    if args.mode == "component":
        build_pair_dataset_component(
            good_dir=args.good,
            defect_dir=args.defect,
            output_dir=args.output,
            offset=args.offset,
            class_filter=args.class_filter,
            light_mode=args.light_mode,
            xform_depth=args.xform_depth,
        )
    else:
        build_pair_dataset(args.good, args.defect, args.output, args.light_mode)


if __name__ == "__main__":
    main()
