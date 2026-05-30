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

"""cad2roi: USD → registration → per-component crop pipeline."""

from .component_crop import crop_rois_by_label
from .registration import (
    HAS_CUPY,
    align,
    apply_registration,
    crop_to_valid_bbox,
    load_params,
    register,
    save_blink_gif,
    save_params,
    to_gray,
)
from .sdg_crop import crop_sdg
from .semantic_rules import (
    apply_semantic_rules,
    find_matching_prims,
    glob_to_regex,
)

__all__ = [
    "HAS_CUPY",
    "align",
    "apply_registration",
    "apply_semantic_rules",
    "crop_rois_by_label",
    "crop_sdg",
    "crop_to_valid_bbox",
    "find_matching_prims",
    "glob_to_regex",
    "load_params",
    "register",
    "save_blink_gif",
    "save_params",
    "to_gray",
]
