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
Missing component operations: select, hide, restore visibility.
"""

from __future__ import annotations

import logging

import numpy as np
import omni.usd
from pxr import UsdGeom

try:
    from omni.replicator.core.functional import modify as rep_modify
except ImportError:  # Kit --exec init-timing: lazy sys.modules alias not yet set
    from omni.replicator.core.scripts.functional import modify as rep_modify

logger = logging.getLogger(__name__)


def select_missing_components(
    component_pool: list[str],
    missing_cfg: dict[str, float],
) -> list[str]:
    """Randomly select components to hide. Returns list of prim paths."""
    ratio = missing_cfg["ratio"]
    n = max(1, int(len(component_pool) * ratio))
    selected = np.random.choice(component_pool, size=n, replace=False).tolist()
    logger.info("Missing: %d components selected (%.1f%%)", n, ratio * 100)
    for p in selected[:10]:
        logger.info("  -> %s", p.split("/")[-1])
    if len(selected) > 10:
        logger.info("  ... and %d more", len(selected) - 10)
    return selected


def apply_missing_semantics(stage: object, prim_paths: list[str]) -> None:
    """Apply semantic label defect=missing to selected components."""
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and not prim.HasAttribute("semantics:labels:defect"):
            rep_modify.semantics(prim, value={"defect": "missing"})
    logger.info("Applied defect=missing semantic to %d components", len(prim_paths))


def hide_components(stage: object, prim_paths: list[str]) -> None:
    """Make components invisible."""
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()


def restore_components(stage: object, prim_paths: list[str]) -> None:
    """Restore component visibility."""
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeVisible()


def build_writer_kwargs(writer_cfg: dict, output_dir: str) -> dict:
    """Build BasicWriter.initialize() kwargs from config dict."""
    kwargs: dict = {"output_dir": output_dir}
    for key in [
        "rgb",
        "image_output_format",
        "bounding_box_2d_tight",
        "bounding_box_2d_loose",
        "bounding_box_3d",
        "semantic_segmentation",
        "colorize_semantic_segmentation",
        "instance_id_segmentation",
        "colorize_instance_id_segmentation",
        "distance_to_camera",
        "distance_to_image_plane",
        "colorize_depth",
        "semantic_types",
        "frame_padding",
    ]:
        if key in writer_cfg:
            kwargs[key] = writer_cfg[key]
    return kwargs
