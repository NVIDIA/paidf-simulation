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

"""Per-ROI cropping by semantic-mask connected components.

Used by :mod:`usd2roi_crop`. Emits flat (real_crop, seg_crop) pairs for
anomaly-detection datasets: each ROI becomes one ``NNNN.png`` (real photo
crop) under ``normal_img/`` plus one ``NNNN_cad_mask.png`` (semantic seg
crop) under ``cad_mask/``.

Algorithm — anchor-on-target, expand-to-labelled-neighbourhood:

  1. From ``semantic_segmentation_labels_*.json``:
     * **Target colors** — colors whose ``class`` field includes any class in
       ``target_classes`` (anchors; e.g. ``capacitor`` / ``ic``).
     * **Labelled colors** — every color with a non-empty class that isn't
       ``BACKGROUND`` / ``UNLABELLED`` (anchors plus their adjacent
       support meshes such as ``pad`` / ``solder``).
  2. Build two pixel masks: ``target_mask`` (anchor pixels) and
     ``labelled_mask`` (everything labelled, including non-anchor support).
  3. Morphological close on ``labelled_mask`` so meshes within ``morph_kernel``
     px get merged into one blob.
  4. ``cv2.connectedComponents`` on the closed *labelled* mask. Each
     connected blob is a candidate ROI covering an anchor and any labelled
     pixels adjacent / connected to it.
  5. Keep only blobs that contain at least one anchor pixel — pure-pad /
     pure-silkscreen / etc. regions with no anchor are dropped (no noise
     ROIs from blank board areas).
  6. Filter remaining blobs by min/max area, drop blobs whose tight bbox
     touches an image edge (``edge_skip``; partial / cut-off components),
     compute bbox + offset padding, drop crops with labelled-pixel
     coverage below ``min_coverage`` (mostly-blank crops), and save the
     real / seg images at that bbox.

The bbox naturally grows to enclose ``pad`` + ``solder`` belonging to the
anchor component (because they share a labelled CC). Listing only anchor
classes in ``target_classes`` (e.g. ``[capacitor, ic]``) is enough — the
adjacent pad / solder pixels are pulled in by the labelled-CC step.

Output structure (default, ``class_dirs=None``)::

    <output_dir>/
      normal_img/0001.png             real photo cropped to ROI
      cad_mask/0001_cad_mask.png      semantic seg cropped (RGBA preserved)

When ``class_dirs`` is supplied, ``output_dir`` is treated as the parent
(``crop/``) and each ROI is routed to ``<output_dir>/<class_dirs[dominant_class]>/``.
Classes not in ``class_dirs`` fall back to ``component/``.

Two layout sub-modes under ``class_dirs``:

* ``multi_frame=False`` (default, single-frame Day-1) — flat:

    <output_dir>/<sub>/normal_img/NNNN.png
    <output_dir>/<sub>/cad_mask/NNNN_cad_mask.png

  NNNN counter restarts per ``<sub>``; with a shared ``class_emit_counter``
  the caller can keep NNNN running per ``<sub>`` across multiple frames.

* ``multi_frame=True`` (Day-0 scan grid) — per-cell:

    <output_dir>/<sub>/<frame_idx>/normal_img/NNNN.png
    <output_dir>/<sub>/<frame_idx>/cad_mask/NNNN_cad_mask.png

  NNNN counter resets per ``(<sub>, <frame_idx>)`` — every cell starts fresh
  so each scan-grid cell keeps its own ROI numbering. ``rois[*]['name']``
  becomes ``"<sub>/<frame_idx>/NNNN"``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _parse_rgba(color_str: str) -> tuple[int, int, int, int]:
    """Parse '(r, g, b, a)' string -> tuple of ints."""
    parts = color_str.strip("() ").split(",")
    return tuple(int(x.strip()) for x in parts[:4])


def _color_classes(class_str: str) -> set[str]:
    """Comma-separated 'a,b,c' -> {'a','b','c'} (Replicator multi-label seg)."""
    return {c.strip() for c in (class_str or "").split(",") if c.strip()}


def crop_rois_by_label(
    aligned_dir: str,
    ref_crop_path: str,
    real_crop_path: str,
    output_dir: str,
    target_classes: list[str],
    morph_kernel: int = 2,
    min_area: int = 50,
    max_area: int | None = None,
    offset: int = 10,
    edge_skip: bool = True,
    min_coverage: float = 0.2,
    max_emit: int | None = None,
    frame_idx: str = "0000",
    class_dirs: dict[str, str] | None = None,
    fallback_dir: str = "component",
    class_emit_counter: Counter | None = None,
    multi_frame: bool = False,
) -> dict[str, Any]:
    """Emit per-ROI (real, seg) pairs by anchor-rooted labelled connected components.

    Each ROI is a labelled CC that contains at least one ``target_classes``
    pixel ("anchor"). The bbox grows naturally to enclose any non-anchor
    labelled pixels (e.g. ``pad`` / ``solder``) in the same CC, so listing
    only the anchor component types (e.g. ``[capacitor, ic]``) is enough —
    surrounding support meshes are pulled in automatically. Pure-pad or
    pure-silkscreen regions with no anchor are dropped (no noise ROIs from
    blank board areas).

    Args:
        aligned_dir: contains ``semantic_segmentation_<idx>.png`` and labels JSON.
        ref_crop_path: synth crop. Currently only used for size validation;
            not written to disk.
        real_crop_path: real crop (Day 1) or synth render (Day 0).
        output_dir: ``normal_img/NNNN.png`` and ``cad_mask/NNNN_cad_mask.png``
            are written here.
        target_classes: anchor classes — list only the *primary* component
            types (e.g. ``[capacitor, ic]``); pads / solder around them are
            included automatically via the labelled CC.
        morph_kernel: structuring element radius for ``MORPH_CLOSE`` on the
            labelled mask; merges blobs that sit ``≤ kernel`` px apart.
            ``0`` disables the close.
        min_area / max_area: pixel-area filters on the labelled CC
            (``None`` for max_area = no upper bound).
        offset: padding (px) added around each ROI's bbox before crop.
        edge_skip: when True, drop ROIs whose tight (pre-offset) bbox
            touches any image edge (likely partial / cut-off components
            spanning into the next scan-grid cell).
        min_coverage: minimum fraction of *labelled* pixels inside the
            offset-padded crop box (0..1). Drop crops that are mostly
            blank board. ``0`` disables the filter.
        max_emit: cap the number of ROIs emitted from this frame. ``None``
            = no cap. Iteration stops as soon as the cap is reached, so
            cells are processed in connected-component order and trailing
            CCs in the same frame are dropped silently.
        frame_idx: matches the seg / labels filename suffix.
        class_dirs: optional ``{class_name: subdir_name}`` mapping. When set,
            ``output_dir`` is treated as the *parent* (typically ``crop/``)
            and each ROI is saved under ``output_dir/<subdir>/normal_img/``.
            NNNN counter restarts per subdir. Classes not in the mapping
            fall back to ``fallback_dir``. When ``None`` (default), all ROIs
            go directly under ``output_dir/normal_img/`` with one global
            counter (backward-compatible behaviour).
        fallback_dir: subdir name for classes not listed in ``class_dirs``.
            Only used when ``class_dirs`` is set.
        class_emit_counter: optional ``Counter`` shared across multiple calls
            so NNNN continues running per subdir across cells (Day-0 multi-cell
            use case). When ``None``, a fresh counter is used per call.
            Only relevant when ``class_dirs`` is set, and only meaningful when
            ``multi_frame=False`` — in per-cell layout (``multi_frame=True``)
            the counter key is ``(subdir, frame_idx)`` so each cell starts
            fresh regardless of the shared counter.
        multi_frame: when ``True`` together with ``class_dirs``, insert a
            ``frame_idx`` subdir between the class subdir and the
            ``normal_img/`` / ``cad_mask/`` leaves, so each scan-grid cell
            keeps a distinct bucket per class — ``crop/<sub>/<frame_idx>/
            normal_img/NNNN.png``. NNNN resets per ``(class, cell)``.
            When ``False`` (default), the flat layout is used:
            ``crop/<sub>/normal_img/NNNN.png``.

    Returns:
        stats dict with per-ROI metadata + counts of skipped components.
        ``stats['rois']`` is kept (used by :func:`crop_bridge_pairs` for
        pairing) but not written to disk by callers.
    """
    target_set = set(target_classes)
    BG_LABELS = {"BACKGROUND", "UNLABELLED"}

    seg_path = os.path.join(aligned_dir, f"semantic_segmentation_{frame_idx}.png")
    labels_path = os.path.join(aligned_dir, f"semantic_segmentation_labels_{frame_idx}.json")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"missing seg: {seg_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"missing labels: {labels_path}")

    # Always create output dirs up-front so callers can copy adjacent files
    # (labels JSON, etc.) into them even when this cell yields zero ROIs —
    # off-board scan cells with no anchor colors are a normal case.
    if class_dirs is None:
        normal_dir = os.path.join(output_dir, "normal_img")
        mask_dir = os.path.join(output_dir, "cad_mask")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
    elif not multi_frame:
        # Single-frame class_dirs: pre-create every listed destination so an
        # empty-ROI cell still leaves the directory layout intact (the
        # single-frame Day-1 caller copies adjacent files into it). In the
        # multi-frame per-cell layout the labels.json sits at crop root, so
        # empty cells don't need their subdirs pre-created — the emit-loop's
        # lazy mkdir keeps the tree to only cells that actually produced ROIs.
        for sub in set(class_dirs.values()):
            os.makedirs(os.path.join(output_dir, sub, "normal_img"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, sub, "cad_mask"), exist_ok=True)

    with open(labels_path) as f:
        labels = json.load(f)

    seg_img = Image.open(seg_path).convert("RGBA")
    seg_arr = np.array(seg_img)  # (H, W, 4)

    # --- Step 1: identify target (anchor) colors and the broader labelled-color set ---
    target_color_to_classes: dict[tuple[int, int, int, int], set[str]] = {}
    labelled_colors: list[tuple[int, int, int, int]] = []
    for color_str, info in labels.items():
        # Union of `class:` and `defect:` semantics so anchors can match either
        # field. defect-mode renders only author the `defect:` semantic on
        # mutated components (no class), so target_classes can be e.g.
        # [shift, tombstone, sideflip] to anchor on defect pose types directly.
        cls_set = _color_classes(info.get("class", "")) | _color_classes(info.get("defect", ""))
        if not cls_set or cls_set & BG_LABELS:
            continue
        rgba = _parse_rgba(color_str)
        labelled_colors.append(rgba)
        if cls_set & target_set:
            target_color_to_classes[rgba] = cls_set

    if not target_color_to_classes:
        logger.info(
            "No labels JSON colors match target classes %s in frame %s",
            list(target_classes),
            frame_idx,
        )
        return {
            "n_emitted": 0,
            "n_components_total": 0,
            "skipped_min_area": 0,
            "skipped_max_area": 0,
            "skipped_no_target": 0,
            "skipped_edge": 0,
            "skipped_low_coverage": 0,
            "target_classes": list(target_classes),
            "target_colors": 0,
            "rois": [],
        }

    # --- Step 2: build target_mask (anchor pixels) and labelled_mask (any labelled pixel)
    # plus a per-pixel color id over the target colors only (used by dominant_class voting).
    H, W = seg_arr.shape[:2]
    target_mask = np.zeros((H, W), dtype=np.uint8)
    labelled_mask = np.zeros((H, W), dtype=np.uint8)
    color_idx = np.zeros((H, W), dtype=np.uint16)  # 0 = non-target; 1..N = target_color_list[i-1]
    target_color_list = list(target_color_to_classes.keys())
    for i, rgba in enumerate(target_color_list, start=1):
        match = np.all(seg_arr == np.array(rgba, dtype=np.uint8), axis=-1)
        target_mask[match] = 255
        color_idx[match] = i
    for rgba in labelled_colors:
        match = np.all(seg_arr == np.array(rgba, dtype=np.uint8), axis=-1)
        labelled_mask[match] = 255

    # --- Step 3: morphological close on the LABELLED mask so adjacent
    # support meshes (pad / solder right next to the anchor) merge into one
    # CC together with the anchor.
    if morph_kernel and morph_kernel > 0:
        k = int(morph_kernel)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)
        )
        labelled_closed = cv2.morphologyEx(
            labelled_mask, cv2.MORPH_CLOSE, kernel
        )
    else:
        labelled_closed = labelled_mask

    # --- Step 4: connected components on the labelled mask. Each CC may
    # contain anchor + adjacent support; CCs with no anchor are filtered
    # out in Step 5.
    n_labels, labels_img = cv2.connectedComponents(labelled_closed)

    # Load images (must be the same H×W as seg)
    ref_img = Image.open(ref_crop_path).convert("RGBA")
    real_img = Image.open(real_crop_path).convert("RGBA")
    if ref_img.size != (W, H):
        raise ValueError(f"ref_crop size {ref_img.size} doesn't match seg {(W, H)}")
    if real_img.size != (W, H):
        raise ValueError(f"real_crop size {real_img.size} doesn't match seg {(W, H)}")

    # --- Step 5: emit ROIs ---
    emitted = 0  # global tally (drives max_emit, return stats)
    # per-subdir NNNN counter when class_dirs set; can be caller-owned for
    # cross-cell continuity in Day-0 multi-frame mode.
    emitted_per_dir: Counter = class_emit_counter if class_emit_counter is not None else Counter()
    skipped_min = 0
    skipped_max = 0
    skipped_no_target = 0
    skipped_edge = 0
    skipped_low_coverage = 0
    rois: list[dict[str, Any]] = []

    for comp_id in range(1, n_labels):
        comp_mask = labels_img == comp_id

        # Anchor filter: drop labelled CCs that contain no target-class pixel
        # (pure pad / pure silkscreen / etc. — no anchor = no ROI).
        comp_color_ids = color_idx[comp_mask]
        comp_color_ids = comp_color_ids[comp_color_ids > 0]
        if comp_color_ids.size == 0:
            skipped_no_target += 1
            continue

        area = int(comp_mask.sum())
        if area < min_area:
            skipped_min += 1
            continue
        if max_area is not None and area > max_area:
            skipped_max += 1
            continue

        # Inner bbox = tight bounding box of the labelled CC (anchor + adjacent
        # labelled support meshes are all included because they share this CC).
        rows, cols = np.where(comp_mask)
        y_min, y_max = int(rows.min()), int(rows.max())
        x_min, x_max = int(cols.min()), int(cols.max())

        # Edge filter: tight bbox touches any image edge -> likely a partial
        # component cut by the scan-grid cell boundary; skip rather than emit
        # an incomplete ROI.
        if edge_skip and (x_min <= 0 or y_min <= 0 or x_max >= W - 1 or y_max >= H - 1):
            skipped_edge += 1
            continue

        # Outer bbox = inner + offset, clamped to image
        x1 = max(0, x_min - offset)
        y1 = max(0, y_min - offset)
        x2 = min(W, x_max + 1 + offset)
        y2 = min(H, y_max + 1 + offset)
        crop_box = (x1, y1, x2, y2)

        # Coverage filter: drop crops where the offset-padded bbox is mostly
        # blank board (labelled fraction below threshold).
        if min_coverage > 0:
            crop_labelled = labelled_closed[y1:y2, x1:x2]
            box_area = crop_labelled.size
            if box_area > 0:
                coverage = float((crop_labelled > 0).sum()) / box_area
                if coverage < min_coverage:
                    skipped_low_coverage += 1
                    continue

        # Dominant class: vote by pixel count, anchor-classes only.
        per_color_counts = Counter(comp_color_ids.tolist())
        class_pixel_counts: Counter = Counter()
        for cid, cnt in per_color_counts.items():
            for cls in target_color_to_classes[target_color_list[cid - 1]]:
                if cls in target_set:
                    class_pixel_counts[cls] += cnt
        if not class_pixel_counts:
            skipped_no_target += 1
            continue
        dominant_class = class_pixel_counts.most_common(1)[0][0]

        if class_dirs is None:
            local_normal, local_mask = normal_dir, mask_dir
            idx_num = emitted + 1
            roi_name = f"{idx_num:04d}"
        else:
            target_sub = class_dirs.get(dominant_class, fallback_dir)
            if multi_frame:
                target_root = os.path.join(output_dir, target_sub, frame_idx)
                counter_key = f"{target_sub}/{frame_idx}"
                name_prefix = f"{target_sub}/{frame_idx}"
            else:
                target_root = os.path.join(output_dir, target_sub)
                counter_key = target_sub
                name_prefix = target_sub
            local_normal = os.path.join(target_root, "normal_img")
            local_mask = os.path.join(target_root, "cad_mask")
            os.makedirs(local_normal, exist_ok=True)
            os.makedirs(local_mask, exist_ok=True)
            emitted_per_dir[counter_key] += 1
            idx_num = emitted_per_dir[counter_key]
            roi_name = f"{name_prefix}/{idx_num:04d}"

        idx = f"{idx_num:04d}"
        real_img.crop(crop_box).save(os.path.join(local_normal, f"{idx}.png"))
        seg_img.crop(crop_box).save(os.path.join(local_mask, f"{idx}_cad_mask.png"))

        rois.append(
            {
                "name": roi_name,
                "bbox_local": [x1, y1, x2, y2],
                "bbox_inner": [x_min, y_min, x_max, y_max],
                "area_pixels": area,
                "dominant_class": dominant_class,
                "pixel_count_per_class": dict(class_pixel_counts.most_common()),
            }
        )
        emitted += 1

        if max_emit is not None and emitted >= max_emit:
            break

    stats = {
        "target_classes": list(target_classes),
        "target_colors": len(target_color_to_classes),
        "n_components_total": n_labels - 1,
        "n_emitted": emitted,
        "skipped_min_area": skipped_min,
        "skipped_max_area": skipped_max,
        "skipped_no_target": skipped_no_target,
        "skipped_edge": skipped_edge,
        "skipped_low_coverage": skipped_low_coverage,
        "morph_kernel": morph_kernel,
        "min_area": min_area,
        "max_area": max_area,
        "offset": offset,
        "edge_skip": edge_skip,
        "min_coverage": min_coverage,
        "rois": rois,
    }
    logger.info(
        "[crop_rois] emitted=%d, skipped_min=%d, skipped_max=%d, "
        "skipped_edge=%d, skipped_low_coverage=%d, n_components_total=%d",
        emitted,
        skipped_min,
        skipped_max,
        skipped_edge,
        skipped_low_coverage,
        n_labels - 1,
    )
    return stats


def _bbox_edge_distance(a: list[int], b: list[int]) -> float:
    """Shortest Euclidean distance between two axis-aligned bboxes [x1,y1,x2,y2].

    0 if the rectangles overlap or touch.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
    dy = max(0.0, max(by1 - ay2, ay1 - by2))
    return float((dx * dx + dy * dy) ** 0.5)


def crop_bridge_pairs(
    rois: list[dict[str, Any]],
    ref_crop_path: str,
    real_crop_path: str,
    seg_crop_path: str,
    output_dir: str,
    bridge_dis: float,
    bridge_classes: list[str] | None = None,
    offset: int = 10,
) -> dict[str, Any]:
    """Emit bridge crops: union-bbox of every pair of ROIs within ``bridge_dis``.

    Operates on the ROI list returned by :func:`crop_rois_by_label`. Distance
    is bbox edge-to-edge (touching = 0). When ``bridge_classes`` is set, both
    ROIs of a pair must have ``dominant_class`` in that list to qualify.

    Output: ``output_dir/normal_img/NNNN.png`` and
    ``output_dir/cad_mask/NNNN_cad_mask.png`` — same flat layout as
    :func:`crop_rois_by_label`.
    """
    candidates = list(rois)
    if bridge_classes:
        target = set(bridge_classes)
        candidates = [r for r in candidates if r.get("dominant_class") in target]

    if len(candidates) < 2:
        logger.info("[crop_bridge] only %d candidate(s); no pairs to emit", len(candidates))
        return {
            "n_candidates": len(candidates),
            "n_pairs_in_range": 0,
            "n_emitted": 0,
            "bridge_dis": bridge_dis,
            "bridge_classes": list(bridge_classes or []),
            "bridges": [],
        }

    ref_img = Image.open(ref_crop_path).convert("RGBA")
    real_img = Image.open(real_crop_path).convert("RGBA")
    seg_img = Image.open(seg_crop_path).convert("RGBA")
    W, H = ref_img.size
    if real_img.size != (W, H) or seg_img.size != (W, H):
        raise ValueError(
            f"size mismatch: ref={ref_img.size} real={real_img.size} seg={seg_img.size}"
        )

    normal_dir = os.path.join(output_dir, "normal_img")
    mask_dir = os.path.join(output_dir, "cad_mask")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    emitted = 0
    n_pairs_in_range = 0
    bridges: list[dict[str, Any]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            dist = _bbox_edge_distance(a["bbox_inner"], b["bbox_inner"])
            if dist > bridge_dis:
                continue
            n_pairs_in_range += 1

            ax1, ay1, ax2, ay2 = a["bbox_inner"]
            bx1, by1, bx2, by2 = b["bbox_inner"]
            ix_min, iy_min = min(ax1, bx1), min(ay1, by1)
            ix_max, iy_max = max(ax2, bx2), max(ay2, by2)

            x1 = max(0, ix_min - offset)
            y1 = max(0, iy_min - offset)
            x2 = min(W, ix_max + 1 + offset)
            y2 = min(H, iy_max + 1 + offset)
            crop_box = (x1, y1, x2, y2)

            idx = f"{emitted + 1:04d}"
            real_img.crop(crop_box).save(os.path.join(normal_dir, f"{idx}.png"))
            seg_img.crop(crop_box).save(os.path.join(mask_dir, f"{idx}_cad_mask.png"))

            cls_pair = sorted([a["dominant_class"], b["dominant_class"]])
            bridges.append(
                {
                    "name": idx,
                    "bbox_local": [x1, y1, x2, y2],
                    "bbox_inner": [ix_min, iy_min, ix_max, iy_max],
                    "distance_px": dist,
                    "pair": [a["name"], b["name"]],
                    "pair_classes": cls_pair,
                }
            )
            emitted += 1

    stats = {
        "n_candidates": len(candidates),
        "n_pairs_in_range": n_pairs_in_range,
        "n_emitted": emitted,
        "bridge_dis": bridge_dis,
        "bridge_classes": list(bridge_classes or []),
        "offset": offset,
        "bridges": bridges,
    }
    logger.info(
        "[crop_bridge] emitted=%d, candidates=%d, dis<=%g",
        emitted,
        len(candidates),
        bridge_dis,
    )
    return stats
