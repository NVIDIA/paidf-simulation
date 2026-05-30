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

"""Crop an SDG frame's RGB + annotations to a bbox in reference coordinates.

Designed to pair with :func:`cad2roi.registration.crop_to_valid_bbox`: once an
alignment produces a valid-overlap bbox in the reference (SDG) frame, this
module crops every annotation so component positions stay consistent with the
cropped image.

Bbox handling:
  * Per-mesh bboxes are merged into per-component bboxes by grouping on
    ``prim_paths[:xform_depth]`` joined with ``/`` (same policy as
    ``scripts/postprocess/crop_components.py``).
  * Only components whose merged bbox lies **fully inside** the crop region
    are kept; partially clipped ones are dropped.
  * Kept bboxes are translated into crop-local coordinates.
  * ``occlusionRatio`` is reset to ``-1.0`` after merging (no longer meaningful).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_OUTPUTS: tuple[str, ...] = ("rgb", "semseg", "bbox")
ALL_OUTPUTS: tuple[str, ...] = ("rgb", "semseg", "bbox", "instance_seg")

# Per-category file rules: (image_stems_to_crop, passthrough_jsons_to_copy)
_OUTPUT_SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "rgb": (("rgb",), ()),
    "semseg": (("semantic_segmentation",), ("semantic_segmentation_labels",)),
    "instance_seg": (("instance_id_segmentation",), ("instance_id_segmentation_mapping",)),
    "bbox": ((), ()),  # handled separately
}


def _component_key(prim_path: str, xform_depth: int) -> str:
    parts = prim_path.split("/")
    if len(parts) >= xform_depth:
        return "/".join(parts[:xform_depth])
    return prim_path


def _merge_to_components(
    bboxes: np.ndarray, prim_paths: list[str], xform_depth: int
) -> dict[str, dict[str, Any]]:
    components: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "x_min": 10**9,
            "y_min": 10**9,
            "x_max": -1,
            "y_max": -1,
            "sid": None,
        }
    )
    for entry, ppath in zip(bboxes, prim_paths):
        key = _component_key(ppath, xform_depth)
        c = components[key]
        c["x_min"] = min(c["x_min"], int(entry["x_min"]))
        c["y_min"] = min(c["y_min"], int(entry["y_min"]))
        c["x_max"] = max(c["x_max"], int(entry["x_max"]))
        c["y_max"] = max(c["y_max"], int(entry["y_max"]))
        c["sid"] = int(entry["semanticId"])
    return components


def crop_sdg(
    src_dir: str | Path,
    dst_dir: str | Path,
    bbox: tuple[int, int, int, int],
    frame_idx: str = "0000",
    xform_depth: int = 7,
    include_loose: bool = True,
    outputs: tuple[str, ...] | list[str] = DEFAULT_OUTPUTS,
) -> dict:
    """Crop selected SDG annotations in a frame to ``bbox`` (x0, y0, x1, y1).

    ``outputs`` selects which categories to process. Choices:
    ``"rgb"``, ``"semseg"``, ``"bbox"``, ``"instance_seg"``. Default is
    ``("rgb", "semseg", "bbox")``.

    Returns a stats dict with per-class kept/dropped counts and the crop size.
    """
    unknown = [o for o in outputs if o not in ALL_OUTPUTS]
    if unknown:
        raise ValueError(f"Unknown outputs: {unknown}. Choose from {ALL_OUTPUTS}.")
    outputs = tuple(outputs)

    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = (int(v) for v in bbox)
    crop_w, crop_h = x1 - x0, y1 - y0

    # --- images + passthrough JSONs (gated by outputs) ---
    for category in outputs:
        image_stems, passthrough_jsons = _OUTPUT_SPECS[category]
        for stem in image_stems:
            p = src / f"{stem}_{frame_idx}.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(
                    str(dst / f"{stem}_{frame_idx}.png"), img[y0:y1, x0:x1]
                )
        for stem in passthrough_jsons:
            p = src / f"{stem}_{frame_idx}.json"
            if p.exists():
                (dst / f"{stem}_{frame_idx}.json").write_text(p.read_text())

    # --- metadata (always copied) ---
    meta = src / "metadata.txt"
    if meta.exists():
        (dst / "metadata.txt").write_text(meta.read_text())

    # --- bbox kinds ---
    if "bbox" not in outputs:
        return {
            "bbox": [x0, y0, x1, y1],
            "crop_size": [crop_w, crop_h],
            "frame_idx": frame_idx,
            "xform_depth": xform_depth,
            "outputs": list(outputs),
            "per_class": {},
            "totals": {"total": 0, "kept": 0, "dropped": 0},
        }

    kinds = ["tight"] + (["loose"] if include_loose else [])
    stats_total: dict[str, int] = defaultdict(int)
    stats_kept: dict[str, int] = defaultdict(int)
    stats_dropped: dict[str, int] = defaultdict(int)

    for kind in kinds:
        bb_path = src / f"bounding_box_2d_{kind}_{frame_idx}.npy"
        pp_path = src / f"bounding_box_2d_{kind}_prim_paths_{frame_idx}.json"
        lb_path = src / f"bounding_box_2d_{kind}_labels_{frame_idx}.json"
        if not (bb_path.exists() and pp_path.exists()):
            continue

        bboxes = np.load(bb_path)
        prim_paths = json.loads(pp_path.read_text())
        labels = json.loads(lb_path.read_text()) if lb_path.exists() else {}
        components = _merge_to_components(bboxes, prim_paths, xform_depth)

        kept_entries = []
        kept_prim_paths = []
        for comp_key, c in components.items():
            cls = labels.get(str(c["sid"]), {}).get("class", "?")
            stats_total[cls] += 1
            inside = c["x_min"] >= x0 and c["y_min"] >= y0 and c["x_max"] <= x1 and c["y_max"] <= y1
            if inside:
                kept_entries.append(
                    (
                        c["sid"],
                        c["x_min"] - x0,
                        c["y_min"] - y0,
                        c["x_max"] - x0,
                        c["y_max"] - y0,
                        -1.0,
                    )
                )
                kept_prim_paths.append(comp_key)
                stats_kept[cls] += 1
            else:
                stats_dropped[cls] += 1

        new_bboxes = np.array(kept_entries, dtype=bboxes.dtype)
        np.save(dst / f"bounding_box_2d_{kind}_{frame_idx}.npy", new_bboxes)
        (dst / f"bounding_box_2d_{kind}_prim_paths_{frame_idx}.json").write_text(
            json.dumps(kept_prim_paths, indent=2)
        )
        if lb_path.exists():
            (dst / f"bounding_box_2d_{kind}_labels_{frame_idx}.json").write_text(
                lb_path.read_text()
            )

    return {
        "bbox": [x0, y0, x1, y1],
        "crop_size": [crop_w, crop_h],
        "frame_idx": frame_idx,
        "xform_depth": xform_depth,
        "outputs": list(outputs),
        "per_class": {
            cls: {
                "total": stats_total[cls],
                "kept": stats_kept.get(cls, 0),
                "dropped": stats_dropped.get(cls, 0),
            }
            for cls in sorted(stats_total)
        },
        "totals": {
            "total": sum(stats_total.values()),
            "kept": sum(stats_kept.values()),
            "dropped": sum(stats_dropped.values()),
        },
    }
