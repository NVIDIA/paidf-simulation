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
"""Write a prompt_metadata.json sidecar into an SDG output dir.

Invoked by the /simulation skill at Stage 5 (after the derived YAML is
written but before / alongside the docker / isaac-sim.sh run).
The sidecar captures everything needed to reproduce the run from English:
the natural prompt, the parsed intent, the chosen configs, the overrides,
and the execution environment.

Usage (from the skill or a backfill):
    python3 write_prompt_metadata.py \\
        --output-dir sdg_test_output/<slug> \\
        --prompt "generate 3 good PCB images" \\
        --flow good \\
        --lighting scene \\
        --base configs/examples/flow1_good_image/good_image.yaml \\
        --derived configs/runs/<slug>.yaml \\
        --pcba-target configs/pcba_target.yaml \\
        --pcb-usd-path <path/to/your/board.usd> \\
        --mode local \\
        --overrides 'max_image_count: 3
output: ${PAIDF_SIM_ROOT}/sdg_test_output/<slug>'
"""

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None  # overrides become an opaque string

SKILL_NAME = "generate"
SKILL_VERSION = "1.0.0"
METADATA_VERSION = "0.2.0"


def _count_frames(out_dir: Path) -> Dict[str, int]:
    if not out_dir.exists():
        return {"rgb": 0, "semantic_segmentation": 0, "bounding_box_2d_tight": 0}
    return {
        "rgb": len(list(out_dir.rglob("rgb_*.png"))),
        "semantic_segmentation": len(list(out_dir.rglob("semantic_segmentation_*.png"))),
        "bounding_box_2d_tight": len(list(out_dir.rglob("bounding_box_2d_tight_*.npy"))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True, help="Run output dir (will be created if absent).")
    ap.add_argument("--prompt", required=True, help="The user's natural-language prompt.")
    ap.add_argument("--flow", required=True, choices=["good", "good_fixed", "defect", "missing", "lighting"])
    ap.add_argument("--lighting", default="scene", choices=["scene", "ring", "dome", "scene_whitened"])
    ap.add_argument("--count", type=int, default=None, help="Requested image count (max_image_count).")
    ap.add_argument("--board", default="Spark", help="Board name parsed from intent.")
    ap.add_argument("--base", required=True, help="Path to the canonical config used as the base.")
    ap.add_argument("--derived", required=True, help="Path to the derived YAML (configs/runs/...).")
    ap.add_argument("--pcba-target", required=True, help="Path to pcba_target YAML (--pcba-config).")
    ap.add_argument("--pcb-usd-path", required=True, help="Resolved PCB_USD_PATH used at runtime.")
    ap.add_argument("--mode", choices=["docker", "local"], default="local")
    ap.add_argument("--launcher", default=None, help="Docker image tag or isaac-sim.sh path.")
    ap.add_argument(
        "--overrides",
        default="",
        help="YAML / JSON string of the override dict the skill applied (e.g. max_image_count, defects.*.enabled).",
    )

    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides: Any = args.overrides
    if overrides and yaml is not None:
        try:
            overrides = yaml.safe_load(overrides) or {}
        except yaml.YAMLError:
            pass  # keep as raw string

    # Derive replication helpers.
    # Convention: replication_prompt ALWAYS shows the default (container) form
    # with an OUTPUT_PATH placeholder, so the doc reader is guided to the
    # production-style invocation regardless of how this particular run was
    # executed. The `replication_note` below describes both modes; `execution`
    # records what actually ran.
    replication_prompt = f"/simulation {args.prompt}, to OUTPUT_PATH"

    launcher_str = args.launcher or (
        "paidf-simulation:sqa"
        if args.mode == "docker"
        else os.environ.get("ISAAC_SIM_PATH", "<set $ISAAC_SIM_PATH>")
    )

    note_lines = [
        "Replace OUTPUT_PATH with an absolute host path where the renders should land",
        f"  (default if you omit `to OUTPUT_PATH`: ${{PAIDF_SIM_ROOT}}/sdg_test_output/<auto-slug>).",
        "",
        "Default execution mode is Docker (container). Prerequisites:",
        f"  - docker image: paidf-simulation:sqa (or paidf-simulation:local-sqa-test as fallback)",
        f"  - PCB_USD_PATH (-e flag at docker run): {args.pcb_usd_path}",
        "  - PAIDF_SIM_ROOT bind-mounted as /workspace/paidf-simulation inside the container",
        "  - assets dir bind-mounted read-only (the USD has peer references).",
        "",
        "To run on the host instead (no container), append `, local` to the prompt.",
        "Local prerequisites:",
        f"  - launcher: $ISAAC_SIM_PATH (your Isaac-Sim install's isaac-sim.sh)",
        f"  - PCB_USD_PATH set in shell to: {args.pcb_usd_path}",
        "  - PAIDF_SIM_ROOT set in shell to your paidf-simulation repo root.",
        "",
        f"This particular run actually executed via: {args.mode} ({launcher_str})",
    ]
    replication_note = chr(10).join(note_lines)

    metadata: Dict[str, Any] = {
        "metadata_version": METADATA_VERSION,
        "timestamp": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "skill": {"name": SKILL_NAME, "version": SKILL_VERSION},
        "natural_prompt": args.prompt,
        "replication_prompt": replication_prompt,
        "replication_note": replication_note,
        "normalized_intent": {
            "flow": args.flow,
            "lighting": args.lighting,
            "count": args.count,
            "board": args.board,
        },
        "config": {
            "base": args.base,
            "derived": args.derived,
            "pcba_target": args.pcba_target,
            "pcb_usd_path": args.pcb_usd_path,
        },
        "overrides": overrides,
        "execution": {
            "mode": args.mode,
            "launcher": launcher_str,
        },
        "outputs": {
            "dir": str(out_dir),
            "frames": _count_frames(out_dir),
        },
    }

    dst = out_dir / "prompt_metadata.json"
    dst.write_text(json.dumps(metadata, indent=2, sort_keys=False))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
