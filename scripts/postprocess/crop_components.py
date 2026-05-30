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
Crop individual components from SDG output based on bounding_box_2d_tight.

Groups bbox entries by component Xform (using prim_paths), merges sub-mesh
bboxes into a single per-component bbox, then crops specified image types.

Usage (normal defect mode):
    python crop_components.py \
        --input /path/to/trigger_0000 \
        --output /path/to/cropped \
        --crops rgb semantic_segmentation component_instance \
        --offset 10

Usage (missing mode -- bbox/semantic from reference, RGB from defective):
    python crop_components.py \
        --input /path/to/defective \
        --reference /path/to/reference \
        --output /path/to/cropped \
        --crops rgb semantic_segmentation component_instance \
        --offset 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CROPS: tuple[str, ...] = ("rgb", "semantic_segmentation", "instance_id_segmentation")


def extract_component_key(prim_path: str, xform_depth: int) -> str:
    """Extract component Xform path from a full prim path."""
    parts = prim_path.split("/")
    if len(parts) >= xform_depth:
        return "/".join(parts[:xform_depth])
    return prim_path


def crop_components(
    input_dir: str,
    output_dir: str,
    offset: int = 10,
    class_filter: list[str] | None = None,
    reference_dir: str | None = None,
    crop_types: tuple[str, ...] | list[str] | None = None,
    component_xform_depth: int = 7,
) -> None:
    """Crop components from SDG output.

    Args:
        input_dir: trigger folder containing rgb (and bbox/seg if no reference_dir)
        output_dir: output folder for cropped images
        offset: extra pixels around bbox on each side (default 10)
        class_filter: if set, only crop components whose label contains this string
        reference_dir: if set, read bbox/labels/prim_paths/semantic from here,
                       rgb from input_dir (missing mode). Also crops reference rgb as "ok".
        crop_types: list of image type prefixes to crop (e.g. ["rgb", "semantic_segmentation"])
    """
    if crop_types is None:
        crop_types = DEFAULT_CROPS

    # bbox/labels/semantic come from reference_dir if provided, otherwise input_dir
    anno_dir = reference_dir if reference_dir else input_dir

    bbox_files = sorted(
        [
            f
            for f in os.listdir(anno_dir)
            if f.startswith("bounding_box_2d_tight_") and f.endswith(".npy")
        ]
    )

    if not bbox_files:
        logger.error("No bounding_box_2d_tight_*.npy files found.")
        return

    # Determine which crop type to use for semantic coverage filtering
    sem_type: str | None = None
    for ct in crop_types:
        if "semantic" in ct:
            sem_type = ct
            break

    total_crops = 0
    created_dirs: set[str] = set()

    for bbox_file in bbox_files:
        frame_idx = bbox_file.replace("bounding_box_2d_tight_", "").replace(".npy", "")

        # Load bbox + prim_paths + labels (from anno_dir)
        bboxes = np.load(os.path.join(anno_dir, bbox_file))

        prim_paths_file = os.path.join(
            anno_dir, f"bounding_box_2d_tight_prim_paths_{frame_idx}.json"
        )
        if not os.path.exists(prim_paths_file):
            logger.warning("prim_paths not found for frame %s, skipping", frame_idx)
            continue
        with open(prim_paths_file) as f:
            prim_paths = json.load(f)

        label_file = os.path.join(anno_dir, f"bounding_box_2d_tight_labels_{frame_idx}.json")
        if not os.path.exists(label_file):
            continue
        with open(label_file) as f:
            labels = json.load(f)

        # Load images for each crop type
        # rgb always from input_dir, others from anno_dir
        images: dict[str, Image.Image | None] = {}
        for ct in crop_types:
            if ct == "rgb":
                path = os.path.join(input_dir, f"{ct}_{frame_idx}.png")
            else:
                path = os.path.join(anno_dir, f"{ct}_{frame_idx}.png")
            images[ct] = Image.open(path) if os.path.exists(path) else None

        # In missing mode, also load reference rgb for "ok" crops
        ref_rgb_img: Image.Image | None = None
        if reference_dir:
            ref_rgb_path = os.path.join(reference_dir, f"rgb_{frame_idx}.png")
            ref_rgb_img = Image.open(ref_rgb_path) if os.path.exists(ref_rgb_path) else None

        if images.get("rgb") is None:
            continue

        img_w, img_h = images["rgb"].size

        # Load semantic image for coverage filtering
        sem_img = images.get(sem_type) if sem_type else None

        # Group bboxes by component Xform
        components: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"x_min": 99999, "y_min": 99999, "x_max": 0, "y_max": 0, "sid": None}
        )

        for entry, ppath in zip(bboxes, prim_paths):
            comp_key = extract_component_key(ppath, component_xform_depth)
            c = components[comp_key]
            c["x_min"] = min(c["x_min"], int(entry["x_min"]))
            c["y_min"] = min(c["y_min"], int(entry["y_min"]))
            c["x_max"] = max(c["x_max"], int(entry["x_max"]))
            c["y_max"] = max(c["y_max"], int(entry["y_max"]))
            c["sid"] = str(int(entry["semanticId"]))

        # Crop each component
        frame_crops = 0
        for comp_key, c in components.items():
            label_info = labels.get(c["sid"], {})
            label_class = label_info.get("class", "")
            label_defect = label_info.get("defect", "")

            if class_filter:
                match = any(f in label_class or f in label_defect for f in class_filter)
                if not match:
                    continue

            # Skip components whose bbox touches the image edge (incomplete/cut-off)
            if (
                c["x_min"] <= 0
                or c["y_min"] <= 0
                or c["x_max"] >= img_w - 1
                or c["y_max"] >= img_h - 1
            ):
                continue

            # Apply offset with boundary clamping
            x1 = max(0, c["x_min"] - offset)
            y1 = max(0, c["y_min"] - offset)
            x2 = min(img_w, c["x_max"] + offset)
            y2 = min(img_h, c["y_max"] + offset)
            crop_box = (x1, y1, x2, y2)

            # Build filename from component name
            comp_name = comp_key.split("/")[-1]  # e.g., tn__0402_H060_339_
            prefix = f"frame{frame_idx}_{comp_name}"

            # Skip component if semantic coverage < 20%
            if sem_img is not None:
                sem_crop = sem_img.crop(crop_box)
                sem_arr = np.array(sem_crop)
                if sem_arr.shape[-1] == 4:
                    nonzero = np.count_nonzero(sem_arr[:, :, 3])
                else:
                    nonzero = np.count_nonzero(sem_arr.max(axis=2))
                coverage = nonzero / (sem_arr.shape[0] * sem_arr.shape[1])
                if coverage < 0.2:
                    continue
            else:
                continue

            if reference_dir:
                # Missing mode: save defective rgb as "missing", reference rgb as "ok"
                for folder, rgb_src in [("missing", images.get("rgb")), ("ok", ref_rgb_img)]:
                    if rgb_src is None:
                        continue
                    if folder not in created_dirs:
                        for ct in crop_types:
                            os.makedirs(os.path.join(output_dir, folder, ct), exist_ok=True)
                        created_dirs.add(folder)
                    rgb_src.crop(crop_box).save(
                        os.path.join(output_dir, folder, "rgb", f"{prefix}.png")
                    )
                    for ct in crop_types:
                        if ct == "rgb":
                            continue
                        img = images.get(ct)
                        if img is not None:
                            img.crop(crop_box).save(
                                os.path.join(output_dir, folder, ct, f"{prefix}.png")
                            )
            else:
                # Normal mode: use label as folder name
                safe_class = label_class.replace(",", "_").replace(" ", "_")
                if label_defect:
                    label_folder = label_defect
                else:
                    label_folder = safe_class

                if label_folder not in created_dirs:
                    for ct in crop_types:
                        os.makedirs(os.path.join(output_dir, label_folder, ct), exist_ok=True)
                    created_dirs.add(label_folder)

                for ct in crop_types:
                    img = images.get(ct)
                    if img is not None:
                        img.crop(crop_box).save(
                            os.path.join(output_dir, label_folder, ct, f"{prefix}.png")
                        )

            frame_crops += 1

        total_crops += frame_crops
        if frame_crops > 0:
            logger.info("  Frame %s: %d components cropped", frame_idx, frame_crops)

    logger.info("Total: %d crops -> %s", total_crops, output_dir)

    # Post-filter: remove crops whose saved semantic image has <20% coverage
    if sem_type is None:
        return

    removed = 0
    for root, dirs, files in os.walk(output_dir):
        if os.path.basename(root) != sem_type:
            continue
        for fname in files:
            sem_file = os.path.join(root, fname)
            sem_arr = np.array(Image.open(sem_file))
            if sem_arr.shape[-1] == 4:
                nonzero = np.count_nonzero(sem_arr[:, :, 3])
            else:
                nonzero = np.count_nonzero(sem_arr.max(axis=2))
            coverage = nonzero / (sem_arr.shape[0] * sem_arr.shape[1])
            if coverage < 0.2:
                label_dir = os.path.dirname(root)
                for ct in crop_types:
                    p = os.path.join(label_dir, ct, fname)
                    if os.path.exists(p):
                        os.remove(p)
                removed += 1
    if removed > 0:
        logger.info("Post-filter: Removed %d crops with empty semantic masks", removed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Crop components from SDG output")
    parser.add_argument("--input", type=str, required=True, help="Input trigger folder")
    parser.add_argument("--output", type=str, required=True, help="Output folder for crops")
    parser.add_argument(
        "--offset", type=int, default=10, help="Extra pixels around bbox (default 10)"
    )
    parser.add_argument(
        "--crops",
        type=str,
        nargs="+",
        default=list(DEFAULT_CROPS),
        help=f"Image type prefixes to crop (default: {DEFAULT_CROPS})",
    )
    parser.add_argument(
        "--class-filter",
        type=str,
        nargs="+",
        default=None,
        help="Only crop components whose class/defect contains any of these strings",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference dir (missing mode): bbox/semantic from here, RGB from --input",
    )
    parser.add_argument(
        "--xform-depth",
        type=int,
        default=7,
        help="USD hierarchy depth for component Xform extraction (default 7)",
    )
    args = parser.parse_args()

    crop_components(
        args.input,
        args.output,
        args.offset,
        args.class_filter,
        args.reference,
        args.crops,
        args.xform_depth,
    )
