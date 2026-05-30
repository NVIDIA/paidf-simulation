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

"""Isaac Sim standalone shim — pre-warp + SimulationApp + runpy(target_script).

Usage:
    python3 run_sdg_standalone.py <target_script.py> [target's argv...]

The first positional arg is the script to runpy under SimulationApp; the rest
are forwarded as the target's own sys.argv. Examples:

    # Drive sdg_pipeline.py (Path A SDG flows 1/2/3/5):
    python3 run_sdg_standalone.py scripts/sdg/standalone/sdg_pipeline.py         --config configs/flow1_good_image/good_image.yaml         --pcba-config configs/pcba_target.yaml

    # Drive usd2roi_render.py (Path A flow 4 stage 1):
    python3 run_sdg_standalone.py scripts/usd2roi/usd2roi_render.py         --config configs/cad2roi/day1/replicator/usd2roi_target.yaml

Why this shim exists:
- Kit's ext loader concurrently imports omni.warp.core -> warp/__init__.py,
  which has an internal circular import (warp/__init__.py imports from
  warp._src.types, which imports warp). Under concurrent ext load, the partial
  warp gets cached missing int32 / context / kernel, which then cascades:
  omni.warp.core ext fails -> omni.replicator.core ext fails -> SDG dead.
- This shim pre-imports warp into sys.modules *before* SimulationApp boots
  Kit, so Kit's ext loader finds it already complete and the race is bypassed.
- The shim runs the target script via runpy(run_name="__main__"), so any
  if __name__ == "__main__": block in the target fires normally and
  argparse sees argv exactly as if the target were invoked directly.

Use this for any Kit-side script (anything that does import omni.replicator
or import omni.kit.* at module load). Host-python stages (register / crop /
postprocess) should be run with --entrypoint python3 instead — they don't
need Kit and would waste ~90s of SimulationApp boot each call.
"""

import glob
import importlib.util
import os
import runpy
import sys

if len(sys.argv) < 2:
    print("[Shim] usage: run_sdg_standalone.py <target_script.py> [target argv...]", flush=True)
    sys.exit(2)

print("[Shim] start", flush=True)

# --- Pre-warp ---
warp_candidates = glob.glob("/isaac-sim/extscache/omni.warp.core-*")
if warp_candidates:
    warp_parent = warp_candidates[0]
    warp_init = os.path.join(warp_parent, "warp", "__init__.py")
    warp_pkg_dir = os.path.join(warp_parent, "warp")
    if os.path.exists(warp_init):
        if warp_parent not in sys.path:
            sys.path.insert(0, warp_parent)
        spec = importlib.util.spec_from_file_location(
            "warp",
            warp_init,
            submodule_search_locations=[warp_pkg_dir],
        )
        warp_mod = importlib.util.module_from_spec(spec)
        sys.modules["warp"] = warp_mod
        spec.loader.exec_module(warp_mod)
        print(
            "[Shim] pre-warp loaded; int32=",
            hasattr(warp_mod, "int32"),
            "context=",
            hasattr(warp_mod, "context"),
            "kernel=",
            hasattr(warp_mod, "kernel"),
            flush=True,
        )

# --- SimulationApp boots Kit ---
from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})
print(
    "[Shim] sim ready; rep has create=",
    hasattr(sys.modules.get("omni.replicator.core"), "create"),
    flush=True,
)

# --- Resolve target + runpy ---
target = sys.argv[1]
# Resolve relative paths against CWD (which docker run sets to /workspace/paidf-simulation
# via the Dockerfile WORKDIR), so callers can pass either an absolute path or a
# repo-relative path like 'scripts/sdg/standalone/sdg_pipeline.py'.
if not os.path.isabs(target):
    target = os.path.abspath(target)
if not os.path.exists(target):
    print(f"[Shim] target not found: {target}", flush=True)
    sim_app.close()
    sys.exit(2)

# Forward argv to the target's __main__: sys.argv[0] = target, rest = its flags.
sys.argv = [target] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(target))

try:
    print(f"[Shim] runpy {target}", flush=True)
    runpy.run_path(target, run_name="__main__")
    print("[Shim] runpy returned", flush=True)
    while sim_app.is_running():
        sim_app.update()
finally:
    print("[Shim] close", flush=True)
    sim_app.close()
