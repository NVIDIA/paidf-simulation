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
Unified SDG Pipeline — supports good / defect / missing modes via config.
Run via Isaac Sim Kit App.

The pipeline type is determined by the `pipeline_type` field in the YAML config:
  - good:    single-pass scan, no defects
  - defect:  single-pass scan with pose_ops defects (shift/tombstone/sideflip)
  - missing: two-pass scan (reference + defective with hidden components)

Usage:
    /isaac-sim/kit/kit /isaac-sim/apps/isaacsim.exp.base.kit --no-window --exec \
        "path/sdg_pipeline.py --config path/config.yaml"

Output frame indices: optional YAML ``seed`` (default 0) skips the first ``seed`` scan-grid cells on
trigger 0 only, and is the first rgb/bbox index; each written frame increments (aligned with scan order).
Optional ``max_image_count`` (-1 = no limit) caps total writer frames; ``missing`` counts reference +
defective as two frames per cell. Optional ``random_seed`` (default 0) controls lighting / camera /
defect-prep / augmentation NumPy streams so changing only ``seed`` does not change scene randomization.

Benchmark (init + per output frame):
    /isaac-sim/kit/kit /isaac-sim/apps/isaacsim.exp.base.kit --no-window --exec \
        "path/sdg_pipeline.py --config path/config.yaml --benchmark"
    Optional: --benchmark-json /tmp/sdg_bench.json
    With --benchmark: phase_once_s (one-time setup) and phase_per_trigger_avg_s (mean per trigger).
    Capture steps use per-frame averages within each trigger (not full loop totals).
    Default “to first frame” time: kit_to_first_output_s (IApp.get_time_since_start_s; Kit start
    → first writer step). Optional OS wall-clock metrics: add --benchmark-wall-clock; for
    host-to-first-frame also pass --benchmark-wall-start-file /path/to/timestamp.txt (e.g.
    date +%s.%N > file before launching Kit).
"""

import argparse
from pathlib import Path
import asyncio
import json
import os
import statistics
import sys
import time
from typing import Any

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
import yaml
from pxr import Gf, Usd, UsdGeom

# Ensure local imports work when executed via --exec
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Also expose paidf-simulation/scripts/ for top-level helpers
_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
# Editor-side modules (fillet, tin perlin, material overrides) live in
# scripts/sdg/editor/ — same dir the Script Editor pastes from.
_EDITOR_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../editor"))
if _EDITOR_DIR not in sys.path:
    sys.path.insert(0, _EDITOR_DIR)

from common import (
    apply_semantics,
    build_component_pool,
    build_ring_lights,
    build_scan_positions,
    cap_positions_for_max_outputs,
    configure_pathtracing,
    disable_all_lights,
    find_translate_op,
    grid_index_for_position,
    open_usd_stage,
    randomize_lighting,
    rename_outputs_to_grid_index,
    resolve_scan_grid_from_stage,
    save_metadata,
    scan_positions_for_trigger,
    seed_trigger_numpy,
    set_replicator_seed_for_output_frame,
    set_writer_start_frame_index,
    setup_augmentation,
)
from randomize import (
    randomize_fillet_shape as _randomize_fillet_shape,
)
from randomize import (
    randomize_material_params as _randomize_material_params,
)
from randomize import (
    randomize_rig as _randomize_rig,
)
from randomize import (
    randomize_tin_noise_shape as _randomize_tin_noise_shape,
)
from randomize import (
    sample as _sample,
)

# === Parse CLI ===
parser = argparse.ArgumentParser(description="Unified SDG Pipeline")
parser.add_argument("--config", type=str, required=True, help="YAML config file path")
parser.add_argument(
    "--pcba-config",
    type=str,
    default=None,
    help=(
        "Optional YAML with USD/scene-bound fields "
        "(scene, pcba_root, component_types, camera_path, camera_xform_path, "
        "ring_light_root, horizontal_aperture). Each key must appear in "
        "exactly one of --config / --pcba-config; pipeline raises on overlap."
    ),
)
parser.add_argument(
    "--benchmark",
    action="store_true",
    help="Print timing: pipeline init→first output frame, and per-frame step_async stats",
)
parser.add_argument(
    "--benchmark-json",
    type=str,
    default=None,
    help="Write benchmark results to this JSON path",
)
parser.add_argument(
    "--benchmark-wall-clock",
    action="store_true",
    help="With --benchmark: also record OS wall-clock to first frame (script parse anchor; optional file)",
)
parser.add_argument(
    "--benchmark-wall-start-file",
    type=str,
    default=None,
    help="With --benchmark-wall-clock: file with one float Unix epoch (e.g. date +%%s.%%N > file) for host→first frame",
)
_cli_args, _ = parser.parse_known_args()


def _expand_env_in_cfg(obj: Any) -> Any:
    """Apply os.path.expandvars recursively (yaml.safe_load does not expand ``${VAR}``)."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_in_cfg(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_in_cfg(x) for x in obj]
    return obj


_WALL_T0 = (
    time.time() if (_cli_args.benchmark and _cli_args.benchmark_wall_clock) else None
)

with open(_cli_args.config) as f:
    CFG = yaml.safe_load(f)

if _cli_args.pcba_config:
    with open(_cli_args.pcba_config) as f:
        _pcba_cfg = yaml.safe_load(f) or {}
    # Enforce strict boundary: PCBA_CONFIG holds USD/scene-bound fields
    # (scene path, prim paths, component_types, camera intrinsics);
    # SDG_CONFIG holds pipeline/render settings (resolution, scan_grid,
    # writer, lighting, defects). Each key belongs to exactly one file —
    # if you find yourself wanting to override on the other side, the key
    # is on the wrong side.
    _conflicts = sorted(set(_pcba_cfg) & set(CFG))
    if _conflicts:
        raise RuntimeError(
            f"Config boundary violation: keys {_conflicts} appear in both "
            f"--config ({_cli_args.config}) and --pcba-config "
            f"({_cli_args.pcba_config}). PCBA_CONFIG holds USD/scene-bound "
            "fields; SDG_CONFIG holds pipeline/render settings. "
            "Move each key to exactly one file."
        )
    CFG.update(_pcba_cfg)
    print(
        f"[Pipeline] Loaded pcba target: {_cli_args.pcba_config} "
        f"(merged keys: {sorted(_pcba_cfg.keys())})"
    )

# --- component_types keyword resolution -----------------------------------
# pcba_target.yaml may write `component_types: ALL` or `0` instead of a list.
# Resolve against the master registry at configs/components.yaml.
_ct = CFG.get("component_types")
if isinstance(_ct, str):
    _registry_path = (
        Path(__file__).resolve().parent.parent.parent.parent / "configs" / "components.yaml"
    )
    with open(_registry_path) as _rf:
        _registry = yaml.safe_load(_rf) or {}
    _subsets = _registry.get("subsets") or {}
    if _ct == "ALL":
        CFG["component_types"] = list(_registry.get("all") or [])
    elif _ct == "0":
        CFG["component_types"] = []
    elif _ct in _subsets:
        CFG["component_types"] = list(_subsets[_ct])
    else:
        raise ValueError(
            f"Unknown component_types keyword: {_ct!r}. "
            f"Allowed: ALL, 0, or a key in {_registry_path}'s `subsets`."
        )
    print(
        f"[Pipeline] Resolved component_types keyword {_ct!r} -> "
        f"{len(CFG['component_types'])} types"
    )

CFG = _expand_env_in_cfg(CFG)

_WALL_FILE_EPOCH = None
if _cli_args.benchmark and _cli_args.benchmark_wall_clock and _cli_args.benchmark_wall_start_file:
    try:
        with open(_cli_args.benchmark_wall_start_file) as _wf:
            _WALL_FILE_EPOCH = float(_wf.read().strip().split()[0])
    except (OSError, ValueError) as _e:
        print(f"[Benchmark] Ignoring --benchmark-wall-start-file: {_e}")
elif _cli_args.benchmark and _cli_args.benchmark_wall_start_file:
    print("[Benchmark] Ignoring --benchmark-wall-start-file (requires --benchmark-wall-clock)")

PIPELINE_TYPE = CFG.get("pipeline_type", "good")
_seed_raw = int(CFG.get("seed", 0))
if _seed_raw < 0:
    print("[Pipeline] Warning: seed < 0 is treated as 0")
IMAGE_SEED = max(0, _seed_raw)
# Lighting / defects / missing selection use this base so changing `seed` alone does not change scene randomization.
RANDOM_SEED = int(CFG.get("random_seed", 0))
MAX_IMAGE_COUNT = int(CFG.get("max_image_count", -1))
print(
    f"[Pipeline] Loaded config: {_cli_args.config} (type: {PIPELINE_TYPE}); "
    f"image_seed={IMAGE_SEED}, random_seed={RANDOM_SEED}, max_image_count={MAX_IMAGE_COUNT}"
)

if PIPELINE_TYPE not in ("good", "defect", "missing"):
    raise ValueError(f"Unknown pipeline_type: {PIPELINE_TYPE}. Must be good, defect, or missing.")

# Conditional imports
if PIPELINE_TYPE == "defect":
    from defect_ops import (
        prepare_defects,
        reapply_defects,
        restore_defects,
        pretag_defect_eligible,
    )
elif PIPELINE_TYPE == "missing":
    from missing_ops import (
        apply_missing_semantics,
        hide_components,
        restore_components,
        select_missing_components,
    )

# Suppress UI startup in imported scripts when running as pipeline
os.environ["FILLET_PIPELINE_MODE"] = "1"

# When Kit caches a negative ``ModuleNotFoundError`` for a script file
# that landed after Kit started, ``import`` keeps failing even after
# the file appears. Invalidate the path-importer cache so freshly-added
# modules are picked up on first run.
import importlib  # noqa: E402

importlib.invalidate_caches()

try:
    from editor_solder_fillet_board import (
        PARAMS as FILLET_PARAMS,
    )
    from editor_solder_fillet_board import (
        clear_fillets as _do_clear_fillets,
    )
    from editor_solder_fillet_board import (
        generate_fillets as _do_generate_fillets,
    )

    _FILLET_AVAILABLE = True
except Exception as _fe:
    print(f"[Pipeline] Solder fillet import failed (disabled): {_fe}")
    _FILLET_AVAILABLE = False
    FILLET_PARAMS = None

try:
    from component_material_override import _State as _MatOverrideState

    _MAT_AVAILABLE = True
except Exception as _me:
    print(f"[Pipeline] Material override import failed (disabled): {_me}")
    _MAT_AVAILABLE = False

try:
    from board_material_override import (
        _State as _BoardMatOverrideState,
    )
    from board_material_override import (
        apply_params_from_dict as _apply_board_params,
    )

    _BOARD_MAT_AVAILABLE = True
except Exception as _bme:
    print(f"[Pipeline] Board material override import failed (disabled): {_bme}")
    _BOARD_MAT_AVAILABLE = False
    _apply_board_params = None

try:
    from tin_noise_patch import (
        PARAMS as TIN_PARAMS,
    )
    from tin_noise_patch import (
        apply_params_from_dict as _apply_tin_params,
    )
    from tin_noise_patch import (
        make_tin_normalmap as _do_make_tin_normalmap,
    )

    _TIN_AVAILABLE = True
except Exception as _te:
    print(f"[Pipeline] Tin noise import failed (disabled): {_te}")
    _TIN_AVAILABLE = False
    TIN_PARAMS = None
    _apply_tin_params = None

# Configure solder fillet PARAMS from YAML (stage not needed)
_sf_cfg = CFG.get("solder_fillet", {})
_FILLET_ENABLED = _FILLET_AVAILABLE and bool(_sf_cfg.get("enabled", False))
if _FILLET_ENABLED and FILLET_PARAMS is not None:
    FILLET_PARAMS.plat_start = float(_sf_cfg.get("platform_start", FILLET_PARAMS.plat_start))
    FILLET_PARAMS.plat_end = float(_sf_cfg.get("platform_end", FILLET_PARAMS.plat_end))
    FILLET_PARAMS.smooth_y = float(_sf_cfg.get("smoothness_y", FILLET_PARAMS.smooth_y))
    FILLET_PARAMS.bump = float(_sf_cfg.get("bump", FILLET_PARAMS.bump))
    FILLET_PARAMS.decay = float(_sf_cfg.get("delta_x_decay", FILLET_PARAMS.decay))
    FILLET_PARAMS.noise_scale = float(_sf_cfg.get("noise_scale", FILLET_PARAMS.noise_scale))
    FILLET_PARAMS.noise_octaves = float(_sf_cfg.get("noise_octaves", FILLET_PARAMS.noise_octaves))
    FILLET_PARAMS.noise_amp = float(_sf_cfg.get("noise_amp", FILLET_PARAMS.noise_amp))
    FILLET_PARAMS.resolution = float(_sf_cfg.get("resolution", FILLET_PARAMS.resolution))
    FILLET_PARAMS.inward_offset_mm = float(
        _sf_cfg.get("inward_offset_mm", FILLET_PARAMS.inward_offset_mm)
    )
    FILLET_PARAMS.inward_frac = float(_sf_cfg.get("inward_frac", FILLET_PARAMS.inward_frac))
    FILLET_PARAMS.z_offset_mm = float(_sf_cfg.get("z_offset_mm", FILLET_PARAMS.z_offset_mm))
    FILLET_PARAMS.long_edge = float(_sf_cfg.get("long_edge", FILLET_PARAMS.long_edge))
    FILLET_PARAMS.x_scale = float(_sf_cfg.get("x_scale", FILLET_PARAMS.x_scale))
    FILLET_PARAMS.fillet_sx = float(_sf_cfg.get("fillet_sx", FILLET_PARAMS.fillet_sx))
    FILLET_PARAMS.fillet_sz = float(_sf_cfg.get("fillet_sz", FILLET_PARAMS.fillet_sz))
    if "mat_color" in _sf_cfg:
        FILLET_PARAMS.mat_color = tuple(_sf_cfg["mat_color"])
    if "mat_metallic" in _sf_cfg:
        FILLET_PARAMS.mat_metallic = float(_sf_cfg["mat_metallic"])
    if "mat_roughness" in _sf_cfg:
        FILLET_PARAMS.mat_roughness = float(_sf_cfg["mat_roughness"])
    FILLET_PARAMS.camera_path = CFG.get("camera_path", FILLET_PARAMS.camera_path)
    FILLET_PARAMS.pcba_root = CFG.get("pcba_root", FILLET_PARAMS.pcba_root)
    if CFG.get("component_types"):
        FILLET_PARAMS.comp_types = list(CFG["component_types"])
    FILLET_PARAMS.adaptive_scale = bool(_sf_cfg.get("adaptive_scale", True))
    print(
        f"[Pipeline] Solder fillets enabled (inward_frac={FILLET_PARAMS.inward_frac}, "
        f"z_off={FILLET_PARAMS.z_offset_mm}mm, {len(FILLET_PARAMS.comp_types)} comp types, "
        f"adaptive_scale={FILLET_PARAMS.adaptive_scale})"
    )

# Tin perlin normal map — bake ONE tileable normal-map PNG from the
# fractal noise and stamp it as ``inputs:normalmap_texture`` on every
# tin shader via component_material_override._State.set_tin_normalmap.
# All tin meshes across the whole board share that single texture
# binding (instance-style sharing at the material level — 1 texture
# covers thousands of tin pad faces).
_tn_cfg = CFG.get("tin_noise", {})
_TIN_ENABLED = _TIN_AVAILABLE and bool(_tn_cfg.get("enabled", False))
if _TIN_ENABLED and TIN_PARAMS is not None:
    _tn_input = dict(_tn_cfg)
    _tn_input.setdefault("camera_path", CFG.get("camera_path", TIN_PARAMS.camera_path))
    _tn_input.setdefault("pcba_root", CFG.get("pcba_root", TIN_PARAMS.pcba_root))
    if CFG.get("component_types"):
        _tn_input.setdefault("comp_types", list(CFG["component_types"]))
    _apply_tin_params(TIN_PARAMS, _tn_input)
    print(
        f"[Pipeline] Tin normal-map enabled (amp={TIN_PARAMS.noise_amp}, "
        f"scale={TIN_PARAMS.noise_scale}, octaves={int(TIN_PARAMS.noise_octaves)}, "
        f"r={int(TIN_PARAMS.resolution)})"
    )

_app = omni.kit.app.get_app()


def _make_bench_state():
    return {
        "t_pipeline_start": None,
        "output_frame_times": [],
        "pipeline_to_first_output_s": None,
        "kit_to_first_output_s": None,
        "wall_script_to_first_output_s": None,
        "wall_file_to_first_output_s": None,
        "phase_once_s": {},
        "phase_per_trigger_s": {},
    }


def _bench_pt_append(bench, key, dt_s):
    if bench is None:
        return
    bench["phase_per_trigger_s"].setdefault(key, []).append(float(dt_s))


def _bench_phase_avg(phase_per_trigger):
    out = {}
    for key, vals in phase_per_trigger.items():
        if vals:
            out[key] = statistics.mean(vals)
    return out


async def _orchestrator_step(rt_subframes, delta_time, bench=None, record_output_frame=False):
    """One Replicator step; optionally record timings for benchmark output frames only."""
    if bench is None or not record_output_frame:
        await rep.orchestrator.step_async(rt_subframes=rt_subframes, delta_time=delta_time)
        return
    t0 = time.perf_counter()
    await rep.orchestrator.step_async(rt_subframes=rt_subframes, delta_time=delta_time)
    dt = time.perf_counter() - t0
    if bench["pipeline_to_first_output_s"] is None:
        bench["pipeline_to_first_output_s"] = time.perf_counter() - bench["t_pipeline_start"]
        bench["kit_to_first_output_s"] = float(_app.get_time_since_start_s())
        if _cli_args.benchmark_wall_clock:
            if _WALL_T0 is not None:
                bench["wall_script_to_first_output_s"] = time.time() - _WALL_T0
            if _WALL_FILE_EPOCH is not None:
                bench["wall_file_to_first_output_s"] = time.time() - _WALL_FILE_EPOCH
    bench["output_frame_times"].append(dt)


def _print_and_save_benchmark(bench):
    fts = bench["output_frame_times"]
    n = len(fts)
    lines = ["[Benchmark] ---", f"  output_frames_recorded: {n}"]
    pfirst = bench["pipeline_to_first_output_s"]
    if pfirst is not None:
        lines.append(
            f"  pipeline_to_first_output_s (run_pipeline → first writer step done): {pfirst:.4f}"
        )
    else:
        lines.append("  pipeline_to_first_output_s: (no output step completed)")
    kfirst = bench["kit_to_first_output_s"]
    if kfirst is not None:
        lines.append(
            "  kit_to_first_output_s (IApp.get_time_since_start_s at first writer step done): "
            f"{kfirst:.4f}"
        )
    ws = bench["wall_script_to_first_output_s"]
    if ws is not None:
        lines.append(
            "  wall_script_to_first_output_s (after CLI parse, before YAML → first writer step done): "
            f"{ws:.4f}"
        )
    wf = bench["wall_file_to_first_output_s"]
    if wf is not None:
        lines.append(
            f"  wall_file_to_first_output_s (timestamp file → first writer step done): {wf:.4f}"
        )
    if n == 0:
        lines.append("  per_frame_s: (no output steps — check pipeline / errors)")
    elif n == 1:
        lines.append(f"  per_frame_s: mean/min/max = {fts[0]:.4f} / {fts[0]:.4f} / {fts[0]:.4f}")
    else:
        lines.append(
            f"  per_frame_s: mean={statistics.mean(fts):.4f}  stdev={statistics.stdev(fts):.4f}  "
            f"min={min(fts):.4f}  max={max(fts):.4f}"
        )
    po = bench.get("phase_once_s") or {}
    pt = bench.get("phase_per_trigger_s") or {}
    pt_avg = _bench_phase_avg(pt)
    if po:
        lines.append("  phase_once_s:")
        for k in sorted(po.keys()):
            lines.append(f"    {k}: {po[k]:.4f}")
    if pt_avg:
        lines.append("  phase_per_trigger_avg_s:")
        for k in sorted(pt_avg.keys()):
            lines.append(f"    {k}: {pt_avg[k]:.4f}")

    report = "\n".join(lines)
    print(report)

    payload = {
        "output_frames_recorded": n,
        "pipeline_to_first_output_s": pfirst,
        "kit_to_first_output_s": bench["kit_to_first_output_s"],
        "per_frame_seconds": fts,
        "phase_once_s": po,
        "phase_per_trigger_avg_s": pt_avg,
    }
    if _cli_args.benchmark_wall_clock:
        payload["wall_script_to_first_output_s"] = bench["wall_script_to_first_output_s"]
        payload["wall_file_to_first_output_s"] = bench["wall_file_to_first_output_s"]
    if n >= 2:
        payload["per_frame_stats"] = {
            "mean": statistics.mean(fts),
            "stdev": statistics.stdev(fts),
            "min": min(fts),
            "max": max(fts),
        }
    elif n == 1:
        payload["per_frame_stats"] = {"mean": fts[0], "min": fts[0], "max": fts[0]}

    if _cli_args.benchmark_json:
        out_path = os.path.abspath(_cli_args.benchmark_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Benchmark] Wrote {_cli_args.benchmark_json}")


# === Writer helpers ===


def _get_writers(writer_cfg: dict[str, Any], output_dir: str) -> list:
    """Initialize the BasicWriter for the standard annotators."""
    cfg = dict(writer_cfg)
    cfg.pop("semantic_filter_predicate", None)
    basic_writer = rep.WriterRegistry.get("BasicWriter")
    basic_writer.initialize(output_dir=output_dir, **cfg)
    return [basic_writer]


# === Per-trigger randomization helpers — extracted to randomize.py ===
# (sdg_pipeline only needs to bind the editor-loaded FILLET_PARAMS / TIN_PARAMS /
#  _do_make_tin_normalmap into the call sites below.)


async def _scan_good(
    trigger_dir: str,
    render_product: object,
    translate_op: object,
    positions: list[tuple[float, float, float]],
    pt_total_spp: int,
    trigger_idx: int,
    num_triggers: int,
    light_meta: dict[str, Any],
    aug_meta: dict[str, Any],
    image_seed: int,
    output_cursor: dict[str, int],
    meta_filename: str = "metadata.json",
    bench: dict[str, Any] | None = None,
    samples_per_position: int = 1,
    on_per_sample_randomize=None,  # callable() -> dict (new light_meta)
) -> None:
    t_ws = time.perf_counter()
    writers = _get_writers(CFG["writer"], trigger_dir)
    for w in writers:
        set_writer_start_frame_index(w, image_seed + output_cursor["next"])
        w.attach([render_product])
    _bench_pt_append(bench, "scan_writer_setup_s", time.perf_counter() - t_ws)

    rec = bench is not None
    _cap_dts: list[float] = []
    _frame_to_grid: list[tuple[int, int, int]] = []
    sample_meta: list[dict[str, Any]] = []
    for i, (x, y, z) in enumerate(positions):
        translate_op.Set(Gf.Vec3d(x, y, z))
        for s in range(max(1, int(samples_per_position))):
            # Re-seed numpy explicitly per sample so back-to-back
            # samples can't share a stale RNG state. ``rep.set_global_seed``
            # alone doesn't always force numpy.random to advance between
            # samples — that's why earlier sample 2 / sample 3 sometimes
            # produced bit-identical fillet + tin params (mean abs diff
            # 0.37 between the two PNGs). Seed = image_seed*1000 +
            # i*samples_per_position + s gives every (position, sample)
            # pair a unique deterministic stream.
            np.random.seed(int(image_seed) * 1000 + i * int(samples_per_position) + s)
            # When samples_per_position > 1 we re-randomize lighting +
            # material + rig + fillet for every sample (the callback
            # provided by run_pipeline does the work). The first sample
            # of the first position keeps the trigger-level randomization
            # already in effect.
            if (
                samples_per_position > 1
                and (i > 0 or s > 0)
                and on_per_sample_randomize is not None
            ):
                light_meta = on_per_sample_randomize() or light_meta
            if _FILLET_ENABLED:
                _do_clear_fillets()
                _do_generate_fillets()
            _frame_num = image_seed + output_cursor["next"]
            set_replicator_seed_for_output_frame(_frame_num)
            _t_iter = time.perf_counter()
            await _orchestrator_step(pt_total_spp, 0.0, bench=bench, record_output_frame=rec)
            _idx = grid_index_for_position(x, y, CFG["scan_grid"])
            if _idx is not None:
                _frame_to_grid.append((_frame_num, _idx[0], _idx[1]))
            output_cursor["next"] += 1
            _cap_dts.append(time.perf_counter() - _t_iter)
            sample_meta.append({"frame": _frame_num, "lighting": light_meta})
        if (i + 1) % 20 == 0:
            print(f"  [{trigger_idx + 1}/{num_triggers}] scan {i + 1}/{len(positions)}")
    if _cap_dts:
        _bench_pt_append(bench, "scan_capture_per_frame_avg_s", statistics.mean(_cap_dts))

    t_td = time.perf_counter()
    # Drain BackendDispatch so async PNG encodes for the last frame land
    # before we detach writers. Without this the last RGB is dropped while
    # bbox/segmentation (smaller, synchronous) still land.
    await rep.orchestrator.wait_until_complete_async()
    for w in writers:
        w.detach()
    _bench_pt_append(bench, "scan_teardown_s", time.perf_counter() - t_td)

    if _frame_to_grid and CFG.get("rename_to_grid_index", False):
        _renamed = rename_outputs_to_grid_index(
            trigger_dir,
            _frame_to_grid,
            frame_padding=int(CFG["writer"].get("frame_padding", 4)),
        )
        print(f"[Pipeline] Renamed {_renamed} output file(s) to x_idx/y_idx naming")

    metadata = {
        "trigger_idx": trigger_idx,
        "num_positions": len(positions),
        "samples_per_position": int(samples_per_position),
        "config": CFG,
        "lighting": light_meta,
        "samples": sample_meta,  # one entry per rendered frame
        "augmentation": aug_meta,
        "frame_to_grid": [
            {"frame": fn, "x_idx": xi, "y_idx": yi} for (fn, xi, yi) in _frame_to_grid
        ],
    }
    save_metadata(trigger_dir, metadata, meta_filename)


async def _scan_defect(
    trigger_dir: str,
    render_product: object,
    translate_op: object,
    positions: list[tuple[float, float, float]],
    pt_total_spp: int,
    trigger_idx: int,
    num_triggers: int,
    light_meta: dict[str, Any],
    aug_meta: dict[str, Any],
    defect_records: list[dict[str, Any]],
    image_seed: int,
    output_cursor: dict[str, int],
    meta_filename: str = "metadata.json",
    bench: dict[str, Any] | None = None,
) -> None:
    t_ws = time.perf_counter()
    writers = _get_writers(CFG["writer"], trigger_dir)
    for w in writers:
        set_writer_start_frame_index(w, image_seed + output_cursor["next"])
        w.attach([render_product])
    _bench_pt_append(bench, "scan_writer_setup_s", time.perf_counter() - t_ws)

    rec = bench is not None
    _cap_dts: list[float] = []
    _frame_to_grid: list[tuple[int, int, int]] = []
    for i, (x, y, z) in enumerate(positions):
        translate_op.Set(Gf.Vec3d(x, y, z))
        if _FILLET_ENABLED:
            _do_clear_fillets()
            _do_generate_fillets()
        reapply_defects(defect_records)
        _frame_num = image_seed + output_cursor["next"]
        set_replicator_seed_for_output_frame(_frame_num)
        _t_iter = time.perf_counter()
        await _orchestrator_step(pt_total_spp, 0.0, bench=bench, record_output_frame=rec)
        _idx = grid_index_for_position(x, y, CFG["scan_grid"])
        if _idx is not None:
            _frame_to_grid.append((_frame_num, _idx[0], _idx[1]))
        output_cursor["next"] += 1
        _cap_dts.append(time.perf_counter() - _t_iter)
        if (i + 1) % 20 == 0:
            print(f"  [{trigger_idx + 1}/{num_triggers}] scan {i + 1}/{len(positions)}")
    if _cap_dts:
        _bench_pt_append(bench, "scan_capture_per_frame_avg_s", statistics.mean(_cap_dts))

    t_td = time.perf_counter()
    # Drain BackendDispatch so async PNG encodes for the last frame land
    # before we detach writers.
    await rep.orchestrator.wait_until_complete_async()
    for w in writers:
        w.detach()
    _bench_pt_append(bench, "scan_teardown_s", time.perf_counter() - t_td)

    if _frame_to_grid and CFG.get("rename_to_grid_index", False):
        _renamed = rename_outputs_to_grid_index(
            trigger_dir,
            _frame_to_grid,
            frame_padding=int(CFG["writer"].get("frame_padding", 4)),
        )
        print(f"[Pipeline] Renamed {_renamed} output file(s) to x_idx/y_idx naming")

    defect_meta = [{"path": r["path"], "defect": r["defect"]} for r in defect_records]
    metadata = {
        "trigger_idx": trigger_idx,
        "num_positions": len(positions),
        "config": CFG,
        "lighting": light_meta,
        "augmentation": aug_meta,
        "defects": defect_meta,
        "frame_to_grid": [
            {"frame": fn, "x_idx": xi, "y_idx": yi} for (fn, xi, yi) in _frame_to_grid
        ],
    }
    save_metadata(trigger_dir, metadata, meta_filename)


async def _scan_missing(
    trigger_dir: str,
    render_product: object,
    translate_op: object,
    positions: list[tuple[float, float, float]],
    pt_total_spp: int,
    trigger_idx: int,
    num_triggers: int,
    light_meta: dict[str, Any],
    aug_meta: dict[str, Any],
    stage: object,
    component_pool: list[str],
    image_seed: int,
    output_cursor: dict[str, int],
    meta_filename: str = "metadata.json",
    bench: dict[str, Any] | None = None,
) -> None:
    t0 = time.perf_counter()
    ref_dir = os.path.join(trigger_dir, "reference")
    def_dir = os.path.join(trigger_dir, "defective")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(def_dir, exist_ok=True)

    missing_paths = select_missing_components(component_pool, CFG["missing"])
    apply_missing_semantics(stage, missing_paths)

    # --- Pass 1: Reference (all visible) ---
    pass_start = output_cursor["next"]
    print(f"[Pipeline] Pass 1: reference (all visible, {len(positions)} positions)")
    ref_writers = _get_writers(CFG["writer"]["reference"], ref_dir)
    for w in ref_writers:
        set_writer_start_frame_index(w, image_seed + output_cursor["next"])
        w.attach([render_product])
    _bench_pt_append(bench, "missing_prepare_s", time.perf_counter() - t0)

    rec = bench is not None
    _ref_dts: list[float] = []
    _frame_to_grid: list[tuple[int, int, int]] = []
    for i, (x, y, z) in enumerate(positions):
        translate_op.Set(Gf.Vec3d(x, y, z))
        if _FILLET_ENABLED:
            _do_clear_fillets()
            _do_generate_fillets()
        _frame_num = image_seed + output_cursor["next"]
        set_replicator_seed_for_output_frame(_frame_num)
        _t_iter = time.perf_counter()
        await _orchestrator_step(pt_total_spp, 0.0, bench=bench, record_output_frame=rec)
        _idx = grid_index_for_position(x, y, CFG["scan_grid"])
        if _idx is not None:
            _frame_to_grid.append((_frame_num, _idx[0], _idx[1]))
        output_cursor["next"] += 1
        _ref_dts.append(time.perf_counter() - _t_iter)
        if (i + 1) % 20 == 0:
            print(f"  [{trigger_idx + 1}/{num_triggers}] reference {i + 1}/{len(positions)}")
    if _ref_dts:
        _bench_pt_append(bench, "missing_reference_per_frame_avg_s", statistics.mean(_ref_dts))

    t_gap = time.perf_counter()
    # Drain BackendDispatch so the reference pass's last-frame writes
    # (rgb / seg / labels) land before we detach.
    await rep.orchestrator.wait_until_complete_async()
    for w in ref_writers:
        w.detach()
    if _frame_to_grid and CFG.get("rename_to_grid_index", False):
        rename_outputs_to_grid_index(
            ref_dir,
            _frame_to_grid,
            frame_padding=int(CFG["writer"]["reference"].get("frame_padding", 4)),
        )

    # --- Hide missing components ---
    print(f"[Pipeline] Hiding {len(missing_paths)} components")
    hide_components(stage, missing_paths)

    # --- Pass 2: Defective (components hidden) ---
    # Reset frame index so defective pass matches reference pass numbering
    output_cursor["next"] = pass_start
    print(f"[Pipeline] Pass 2: defective ({len(positions)} positions)")
    def_writers = _get_writers(CFG["writer"]["defective"], def_dir)
    for w in def_writers:
        set_writer_start_frame_index(w, image_seed + output_cursor["next"])
        w.attach([render_product])
    _bench_pt_append(bench, "missing_pass_gap_s", time.perf_counter() - t_gap)

    _def_dts: list[float] = []
    _frame_to_grid_def: list[tuple[int, int, int]] = []
    for i, (x, y, z) in enumerate(positions):
        translate_op.Set(Gf.Vec3d(x, y, z))
        if _FILLET_ENABLED:
            _do_clear_fillets()
            _do_generate_fillets()
        _frame_num = image_seed + output_cursor["next"]
        set_replicator_seed_for_output_frame(_frame_num)
        _t_iter = time.perf_counter()
        await _orchestrator_step(pt_total_spp, 0.0, bench=bench, record_output_frame=rec)
        _idx = grid_index_for_position(x, y, CFG["scan_grid"])
        if _idx is not None:
            _frame_to_grid_def.append((_frame_num, _idx[0], _idx[1]))
        output_cursor["next"] += 1
        _def_dts.append(time.perf_counter() - _t_iter)
        if (i + 1) % 20 == 0:
            print(f"  [{trigger_idx + 1}/{num_triggers}] defective {i + 1}/{len(positions)}")
    if _def_dts:
        _bench_pt_append(bench, "missing_defective_per_frame_avg_s", statistics.mean(_def_dts))

    t_fin = time.perf_counter()

    # --- Restore visibility ---
    restore_components(stage, missing_paths)
    print("[Pipeline] Components restored")

    # Drain BackendDispatch so the defective pass's last-frame rgb encode
    # lands before we detach. (Previously this was masked by piggy-backing
    # on the USD mutations from restore_components, which gave the writer
    # queue time to drain.)
    await rep.orchestrator.wait_until_complete_async()
    for w in def_writers:
        w.detach()
    if _frame_to_grid_def and CFG.get("rename_to_grid_index", False):
        rename_outputs_to_grid_index(
            def_dir,
            _frame_to_grid_def,
            frame_padding=int(CFG["writer"]["defective"].get("frame_padding", 4)),
        )
    _bench_pt_append(bench, "missing_finalize_s", time.perf_counter() - t_fin)

    metadata = {
        "trigger_idx": trigger_idx,
        "defect_type": "missing",
        "num_positions": len(positions),
        "config": CFG,
        "lighting": light_meta,
        "augmentation": aug_meta,
        "missing_components": missing_paths,
        "frame_to_grid": [
            {"frame": fn, "x_idx": xi, "y_idx": yi} for (fn, xi, yi) in _frame_to_grid
        ],
    }
    save_metadata(trigger_dir, metadata, meta_filename)


# === Main pipeline ===
# === Main pipeline ===
async def run_pipeline():
    bench = _make_bench_state() if _cli_args.benchmark else None
    if bench is not None:
        bench["t_pipeline_start"] = time.perf_counter()
        _bc = "[Benchmark] Enabled (output-frame timings; warmup steps excluded)"
        if _cli_args.benchmark_wall_clock:
            _bc += "; wall-clock metrics on"
        print(_bc)

    usd_path = CFG["scene"]
    if not usd_path.startswith("omniverse://"):
        usd_path = os.path.abspath(usd_path)
    output_root = CFG["output"]
    num_triggers = CFG["num_triggers"]
    camera_path = CFG["camera_path"]
    camera_xform_path = CFG["camera_xform_path"]
    resolution = tuple(CFG["resolution"])

    # Yield 200 main-loop ticks before opening the stage to allow Kit to settle
    for _ in range(200):
        await _app.next_update_async()

    _tp = time.perf_counter()
    await open_usd_stage(_app, usd_path)
    if bench is not None:
        bench["phase_once_s"]["open_usd_stage_s"] = time.perf_counter() - _tp

    _tp = time.perf_counter()
    # configure_pathtracing returns the rt_subframes value for step_async:
    # PathTracing → total_spp; RealTime → realtime.subframes (default 1).
    pt_total_spp = configure_pathtracing(CFG)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available.")

    cam = UsdGeom.Camera.Get(stage, camera_path)
    if not cam.GetPrim().IsValid():
        print(f"[Pipeline] Camera not found at '{camera_path}'; creating camera on session layer.")
        xform = UsdGeom.Xform.Define(stage, camera_xform_path)
        if not any(op.GetOpName() == "xformOp:translate" for op in xform.GetOrderedXformOps()):
            xform.AddTranslateOp()
        cam = UsdGeom.Camera.Define(stage, camera_path)
    # Camera intrinsics are all optional. If omitted, the camera's USD-authored
    # value is used as-is. For orthographic + auto scan_grid (x_num / y_num),
    # resolve_scan_grid_from_stage will grow horizontal_aperture to bbox/num so
    # a missing or tiny value still ends up tiling the board cleanly. Specify
    # here only if you want to override the USD value (e.g. footprint floor
    # for cell overlap, or perspective focal length / vap for FOV control).
    if "horizontal_aperture" in CFG:
        cam.GetHorizontalApertureAttr().Set(float(CFG["horizontal_aperture"]))
    if "vertical_aperture" in CFG:
        cam.GetVerticalApertureAttr().Set(float(CFG["vertical_aperture"]))
    if "focal_length" in CFG:
        cam.GetFocalLengthAttr().Set(float(CFG["focal_length"]))
    # Projection: "perspective" (default) or "orthographic". Ortho mode
    # matches the editor's "Zoom to capacitor" button so editor viewport
    # ↔ pipeline output are at exactly the same camera. Set both
    # h_aperture == v_aperture to get a square ortho region.
    # Pipeline runs strictly in orthographic mode (per project decision
    # . focal_length still settable for documentation but has
    # no rendering effect under ortho projection.
    cam.GetProjectionAttr().Set("orthographic")

    # --- auto_locate_component: random-instance fixed-camera framing -----
    # When CFG has `auto_locate_component: <substring>`, the pipeline:
    #   1. Finds prim instances under pcba_root whose path contains the substring.
    #   2. Picks one at random (seeded by `random_seed` for reproducibility).
    #   3. Computes its world-aligned bounding box.
    #   4. Sets horizontal_aperture so the LONGER dim of the bbox + 1/3 padding
    #      (≈17% per side, i.e. aperture = longer * 4/3) fills the frame.
    #   5. Sets vertical_aperture to maintain 16:9 (or matches if `resolution`
    #      is square).
    #   6. Rewrites `scan_grid` to a single cell centered on the component XY.
    #
    # Closes the long-standing "component-name -> camera-position lookup" gap.
    # Skip this block if scan_grid is already explicitly multi-cell (user
    # didn't ask for auto-locate) OR if auto_locate_component is absent.
    if (
        "auto_locate_prim_path" in CFG
        or "auto_locate_component" in CFG
        or "component_list_override" in CFG
    ):
        import random as _random
        # Precedence (first present wins):
        #   1. auto_locate_prim_path  -- exact USD path; pinpoint single prim
        #   2. component_list_override -- list of substrings, random instance
        #   3. auto_locate_component   -- single substring, random instance
        if "auto_locate_prim_path" in CFG:
            exact_path = str(CFG["auto_locate_prim_path"])
            chosen = stage.GetPrimAtPath(exact_path)
            if not chosen.IsValid():
                raise RuntimeError(
                    f"auto_locate_prim_path: {exact_path!r} not found in stage"
                )
            if not UsdGeom.Xformable(chosen):
                raise RuntimeError(
                    f"auto_locate_prim_path: {exact_path!r} is not Xformable"
                )
            print(
                f"[Pipeline] auto_locate_prim_path: pinned to exact path "
                f"{chosen.GetPath()}"
            )
        else:
            if "component_list_override" in CFG:
                _target_substrs = list(CFG["component_list_override"]) or []
                print(f"[Pipeline] component_list_override: {_target_substrs}")
            else:
                _target_substrs = [str(CFG["auto_locate_component"])]
            pcba_root_path = CFG.get("pcba_root")
            if not pcba_root_path:
                raise RuntimeError(
                    "auto_locate_component / component_list_override requires "
                    "pcba_root in CFG"
                )
            root_prim = stage.GetPrimAtPath(pcba_root_path)
            if not root_prim.IsValid():
                raise RuntimeError(
                    f"auto_locate_component: pcba_root {pcba_root_path!r} not in stage"
                )
            # Tight candidate selection: only INSTANCE prims (tn__ prefix).
            candidates = []
            for prim in Usd.PrimRange(root_prim):
                name = prim.GetName()
                if not name.startswith("tn__"):
                    continue
                if any(ts in name for ts in _target_substrs) and UsdGeom.Xformable(prim):
                    candidates.append(prim)
            if not candidates:
                # Loose fallback if no tn__-prefixed instance matches.
                for prim in Usd.PrimRange(root_prim):
                    name = prim.GetName()
                    if any(ts in name for ts in _target_substrs) and UsdGeom.Xformable(prim):
                        candidates.append(prim)
            if not candidates:
                raise RuntimeError(
                    f"auto_locate_component: no Xformable prim matching any of "
                    f"{_target_substrs!r} under {pcba_root_path!r}"
                )
            rng = _random.Random(RANDOM_SEED)
            chosen = rng.choice(candidates)
            target_substr = "|".join(_target_substrs)
            print(
                f"[Pipeline] auto_locate_component: picked {chosen.GetPath()} "
                f"from {len(candidates)} candidate(s) matching {target_substr!r}"
            )
        # World-space axis-aligned bounding box.
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True
        )
        bbox = bbox_cache.ComputeWorldBound(chosen)
        bbox_range = bbox.ComputeAlignedRange()
        if bbox_range.IsEmpty():
            raise RuntimeError(
                f"auto_locate_component: bbox of {chosen.GetPath()} is empty "
                "(prim has no geometry or hasn't been instanced yet)"
            )
        center = bbox_range.GetMidpoint()
        size = bbox_range.GetSize()
        cx, cy = float(center[0]), float(center[1])
        sx, sy = float(size[0]), float(size[1])
        # Aperture = longer dim * 4/3 (≈33% total padding, ≈17% per side).
        # Padding rule: at least 1/2 padding (= bbox/2 per dim) on BOTH dims.
        # Locked frame aspect = viewport (vap/hap = vh/vw). Solve:
        #   hap >= sx * 3/2            (X needs >=1/3 padding)
        #   vap = hap * (vh/vw) >= sy * 3/2   (Y needs >=1/3 padding too)
        # => hap = max(sx * 3/2, sy * 3/2 * (vw/vh)); vap follows.
        # The dim whose bbox/viewport ratio is dominant gets EXACTLY 1/2 padding;
        # the other gets MORE padding. No clipping on either axis regardless of
        # bbox aspect vs viewport aspect.
        res = CFG.get("resolution", [1920, 1080])
        try:
            vw, vh = float(res[0]), float(res[1])
        except (IndexError, TypeError):
            vw, vh = 1920.0, 1080.0
        pad = 3.0 / 2.0  # 1/2 padding: bbox is 2/3 of frame (padding = bbox/2)
        hap = max(sx * pad, sy * pad * (vw / vh))
        vap = hap * (vh / vw)
        aspect = vh / vw
        # Override the scan_grid to a single cell at (cx, cy).
        sg = CFG.get("scan_grid", {}) or {}
        z_val = float(sg.get("z", 10.0))
        CFG["scan_grid"] = {
            "x_start": cx, "x_end": cx,
            "y_start": cy, "y_end": cy,
            "step": 1.0, "z": z_val,
        }
        # USD camera aperture is stored in "tenths of scene units" (older
        # film-format convention). Convert from scene units (mm) to USD value
        # by multiplying by 10 -- e.g. a 2.0 mm wide ortho frame needs
        # cam.GetHorizontalApertureAttr().Set(20.0).
        cam.GetHorizontalApertureAttr().Set(float(hap * 10.0))
        cam.GetVerticalApertureAttr().Set(float(vap * 10.0))
        print(
            f"[Pipeline] auto_locate_component: center=({cx:.3f}, {cy:.3f}) "
            f"bbox=({sx:.3f} x {sy:.3f}) z={z_val:.2f} "
            f"hap={hap:.3f} vap={vap:.3f} aspect={aspect:.4f} "
            f"(>=1/2 padding both dims, viewport-aspect locked)"
        )

    # Camera diagnostic — print effective intrinsics + derived FOV.
    # USD apertures / focal length are stored in tenths of scene units
    # (older "film-format" convention).
    _hap_val = cam.GetHorizontalApertureAttr().Get()
    _vap_val = cam.GetVerticalApertureAttr().Get()
    _fl_val = cam.GetFocalLengthAttr().Get()
    if _fl_val:
        import math as _math

        _fov_h = _math.degrees(2.0 * _math.atan(float(_hap_val) / (2.0 * float(_fl_val))))
        _fov_v = _math.degrees(2.0 * _math.atan(float(_vap_val) / (2.0 * float(_fl_val))))
        print(
            f"[Pipeline] Camera: fl={_fl_val} hap={_hap_val} vap={_vap_val} "
            f"fov_h={_fov_h:.2f}° fov_v={_fov_v:.2f}° (USD tenths-of-scene-unit)"
        )
    else:
        print(
            f"[Pipeline] Camera: fl={_fl_val} hap={_hap_val} vap={_vap_val} "
            f"(orthographic; USD tenths-of-scene-unit)"
        )

    if not CFG.get("use_scene_lights", False):
        disable_all_lights(stage)
    _rig = CFG.get("rig", {})
    _light_cfg = CFG.get("lighting", {})
    _cone_a_range = _light_cfg.get("cone_angle_range", [120.0, 120.0])
    _cone_s_range = _light_cfg.get("cone_softness_range", [1.0, 1.0])
    if _light_cfg.get("ring_light", True):
        build_ring_lights(
            stage, CFG["ring_light_root"],
            dome_radius=float(_rig.get("dome_radius", 100.0)),
            light_radius=float(_rig.get("light_radius_world", 4.0)),
            light_intensity=float(_rig.get("light_intensity", 5000.0)),
            cone_angle=float(np.mean(_cone_a_range)),
            cone_softness=float(np.mean(_cone_s_range)),
        )

    # Component material override — set up once; re-applied per trigger if randomize_material present
    _mat_state = None
    _mat_defs = {}
    if _MAT_AVAILABLE:
        _mat_state_obj = _MatOverrideState()
        _mat_ok, _mat_defs_result = _mat_state_obj.setup(
            stage,
            component_types=CFG.get("component_types"),
            vantablack_body=bool(CFG.get("vantablack_components", True)),
        )
        if _mat_ok:
            _mat_state = _mat_state_obj
            _mat_defs = _mat_defs_result
            if CFG.get("component_material"):
                _cm = CFG["component_material"]
                _body_color = _cm.get("body_color")
                if _body_color:
                    _body_rough = float(_cm.get("body_roughness", _mat_defs.get("body_rough", 0.3)))
                    _body_metallic = float(
                        _cm.get("body_metallic", _mat_defs.get("body_metallic", 0.0))
                    )
                    _tin_color = _cm.get("tin_color")
                    _tin_metallic = _cm.get("tin_metallic")
                    _pad_color = _cm.get("pad_color")
                    _pad_metallic = _cm.get("pad_metallic")
                    _mat_state.apply(
                        Gf.Vec3f(*_body_color),
                        _body_rough,
                        _body_metallic,
                        float(_cm.get("tin_roughness", _mat_defs.get("tin_rough", 0.20))),
                        float(_cm.get("pad_roughness", _mat_defs.get("pad_rough", 0.20))),
                        tin_color=Gf.Vec3f(*_tin_color) if _tin_color else None,
                        tin_metallic=float(_tin_metallic) if _tin_metallic is not None else None,
                        pad_color=Gf.Vec3f(*_pad_color) if _pad_color else None,
                        pad_metallic=float(_pad_metallic) if _pad_metallic is not None else None,
                    )
                    print(
                        f"[Pipeline] Material override: body_color={_body_color}, "
                        f"rough={_body_rough}, metallic={_body_metallic}; "
                        f"tin_color={_tin_color} tin_metallic={_tin_metallic}; "
                        f"pad_color={_pad_color} pad_metallic={_pad_metallic}"
                    )
                    # Diagnostic: walk every Material prim (including
                    # those in instance prototypes) and report any whose
                    # live composed diffuse_color_constant is still
                    # non-black after our override. Anything left over
                    # is what's painting orange-yellow in the render.
                    from pxr import Usd as _UsdDiag

                    n_mats = 0
                    leftover = []
                    for mp in stage.TraverseAll():
                        if mp.GetTypeName() != "Material":
                            continue
                        n_mats += 1
                    for mp in stage.TraverseAll():
                        if not mp.IsInstance():
                            continue
                        proto = mp.GetPrototype()
                        if not (proto and proto.IsValid()):
                            continue
                        for p in _UsdDiag.PrimRange(proto):
                            if p.GetTypeName() != "Material":
                                continue
                            n_mats += 1
                            for c in p.GetAllChildren():
                                if c.GetTypeName() != "Shader":
                                    continue
                                color = None
                                for an in (
                                    "inputs:diffuse_color_constant",
                                    "inputs:diffuse_reflection_color",
                                    "inputs:base_color",
                                ):
                                    a = c.GetAttribute(an)
                                    if a and a.HasAuthoredValue():
                                        v = a.Get()
                                        try:
                                            r, g, b = float(v[0]), float(v[1]), float(v[2])
                                            if max(r, g, b) > 0.05:
                                                leftover.append((str(p.GetPath()), an, (r, g, b)))
                                        except Exception:
                                            pass
                                        break
                                break
                    print(
                        f"[Pipeline/MatDiag] {n_mats} materials walked, "
                        f"{len(leftover)} still non-black:"
                    )
                    for path, an, c in leftover[:20]:
                        print(f"  {path}  [{an}] = {c}")
        else:
            print(f"[Pipeline] Material override setup failed: {_mat_defs_result}")

    # Tin perlin normal-map: bake once after material override is ready,
    # stamp the same PNG on every tin shader. Texture path is cache-keyed
    # on (amp, scale, octaves, size); per-trigger randomization re-bakes
    # only when ``randomize_tin_noise`` is set in yaml.
    if _TIN_ENABLED and _mat_state is not None and _mat_state.ready:
        try:
            _tin_png = _do_make_tin_normalmap(TIN_PARAMS, force_rebuild=True)
            _tin_bump = float(_tn_cfg.get("bump_factor", 0.3))
            _tin_uvs = float(_tn_cfg.get("texture_scale", 5.0))
            _mat_state.set_tin_normalmap(
                _tin_png, bump_factor=_tin_bump, texture_scale=_tin_uvs, project_uvw=True
            )
            print(
                f"[Pipeline] Tin normal-map → {_tin_png} "
                f"(bump={_tin_bump:.3f}, uv_scale={_tin_uvs:.2f}, "
                f"applied to {len(_mat_state.tin_paths)} tin shaders)"
            )
        except Exception as _e:  # noqa: BLE001
            print(f"[Pipeline] Tin normal-map setup failed: {_e!r}")

    # Board material override — same Sdf-sublayer pattern, applied to the
    # 6 board substrate materials at /World/pcba_main_s_detail/Looks/.
    # Both pipeline (yaml) and the editor UI sliders feed
    # apply_params_from_dict, so the two paths cannot drift.
    _board_state = None
    if _BOARD_MAT_AVAILABLE:
        _bs = _BoardMatOverrideState()
        _eb_paths = CFG.get("extra_blackout_prims")  # None → module default
        _eb_enabled = bool(CFG.get("extra_blackout_enabled", True))
        _bok, _binfo = _bs.setup(
            stage, extra_blackout_prim_paths=_eb_paths, extra_blackout_enabled=_eb_enabled
        )
        if _bok:
            _board_state = _bs
            print(f"[Pipeline] Board material override ready: {len(_binfo)} materials wired")
            _bm_cfg = CFG.get("board_material") or {}
            if _bm_cfg:
                _apply_board_params(_board_state, _bm_cfg)
                print(f"[Pipeline] Board material applied: keys={sorted(_bm_cfg.keys())}")
                # Diagnostic: read back the LIVE shader's diffuse_color_constant
                # for each board material so we can see whether Hydra sees the
                # override (USD-side authoring is verified in setup() — this
                # confirms the renderer side too).
                from pxr import UsdShade

                for _nm in ("SolderMask", "OuterConductor", "Dielectric"):
                    _sp = stage.GetPrimAtPath(f"/World/pcba_main_s_detail/Looks/{_nm}/Shader")
                    if _sp and _sp.IsValid():
                        sh = UsdShade.Shader(_sp)
                        for _attr in (
                            "inputs:diffuse_color_constant",
                            "inputs:diffuse_reflection_color",
                            "inputs:metallic_constant",
                            "inputs:metalness",
                            "info:mdl:sourceAsset",
                        ):
                            v = _sp.GetAttribute(_attr).Get()
                            print(f"  [Live/{_nm}] {_attr} = {v}")
        else:
            print(f"[Pipeline] Board material setup failed: {_binfo}")

    render_product = rep.create.render_product(camera_path, resolution)
    if bench is not None:
        bench["phase_once_s"]["scene_render_setup_s"] = time.perf_counter() - _tp

    # Warmup (initialize Fabric + annotators for deterministic rendering).
    #
    # WORKAROUND for OMPE-90559 / NVBug-5986841 ("Fix unexpected orchestrator
    # init", omni.replicator.core ≥ 1.13.16): step_async short-circuits to a
    # no-render fast path when AnnotatorRegistry has no attached annotators,
    # and _initialize_async skips the renderer-prep sequence under the same
    # condition. Without an annotator attached during warmup the orchestrator
    # ends up partially initialized — when writers later attach for a real
    # pass, ~50% of labelled prims drop out of the rasterized instance buffer
    # non-deterministically (transparent (0,0,0,0) pixels in
    # semantic_segmentation, while bbox/labels JSON still report them).
    #
    # Attaching a throwaway rgb annotator on the real render product across
    # warmup forces has_attached_annotators() == True so the full init/step
    # path runs. Detach + delete it before the real passes so writer outputs
    # are unaffected. (Note: pass render_product.path (str) to .detach(); a
    # Replicator 1.13.16 bug in annotators.py:1457 only unwraps HydraTexture
    # for single-object inputs, so detach([render_product]) raises
    # `'HydraTexture' object has no attribute 'split'` from
    # SyntheticData._get_node_path.)
    if True:
        print("[Pipeline] Warmup steps to initialize Fabric + annotators...")
        _tp = time.perf_counter()
        _warmup_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        _rp_path = render_product.path
        _warmup_annot.attach(_rp_path)
        try:
            for _ in range(5):
                await _orchestrator_step(4, 0.0, bench=None, record_output_frame=False)
        finally:
            _warmup_annot.detach(_rp_path)
            del _warmup_annot
        if bench is not None:
            bench["phase_once_s"]["warmup_orchestrator_s"] = time.perf_counter() - _tp
        print("[Pipeline] Warmup done (5 steps)")

    # Component pool (defect and missing need it)
    component_pool = None
    if PIPELINE_TYPE in ("defect", "missing"):
        _tp = time.perf_counter()
        component_pool = build_component_pool(stage, CFG["pcba_root"], CFG["component_types"])
        if bench is not None:
            bench["phase_once_s"]["component_pool_s"] = time.perf_counter() - _tp
        if not component_pool:
            print("[Error] No components found. Check pcba_root and component_types in config.")
            if bench is not None:
                _print_and_save_benchmark(bench)
            _app.shutdown()
            return

    # Good pipeline: apply class semantics.
    # Two independent layers can run, both written at init via direct USD
    # attribute writes (rep_modify.semantics under the hood):
    #   1. Component-level baseline (apply_semantics) — writes
    #      ``class:capacitor`` on every Xform instance under the listed
    #      Scope names. Required by the existing SDG-frames classification
    #      / detection workflow.
    #   2. Mesh-level rich rules (apply_semantic_rules) — opt-in via the
    #      ``semantics:`` yaml field; glob-match per-mesh classes for the
    #      Day-0 ROI extraction workflow. Mesh prims are deeper than Xform
    #      instances, so where a rule matches it overrides layer 1 for
    #      that mesh; uncovered prims still inherit the component baseline.
    # Either layer is independently optional — yamls can use one, both,
    # or neither.
    if PIPELINE_TYPE == "good":
        _tp = time.perf_counter()

        # Layer 1: component baseline
        if "pcba_root" in CFG and "component_types" in CFG:
            apply_semantics(stage, CFG["pcba_root"], CFG["component_types"])

        # Layer 2: mesh-level rules. Lazy-import the helper from the
        # sibling scripts/usd2roi/ package only when the yaml asks for it.
        if "semantics" in CFG and CFG["semantics"]:
            _sr_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "usd2roi"
            )
            if _sr_path not in sys.path:
                sys.path.insert(0, _sr_path)
            from semantic_rules import apply_semantic_rules

            _stats = apply_semantic_rules(stage, CFG["semantics"])
            _n_uninst = _stats.get('n_uninstanced_ancestors', 0) or 0
            print(
                f"[Pipeline] mesh semantics: {_stats.get('n_rules', '?')} rule(s) -> "
                f"{_stats.get('n_prims_affected', '?')} prim(s) affected, "
                f"{_stats.get('n_label_keys_total', '?')} label key(s), "
                f"{_n_uninst} uninstance op(s)"
            )

            # Drain Kit's event loop after the uninstance loop. Each
            # uninstance copies a LibRef prototype into a unique prim,
            # which schedules a fresh material binding + MDL JIT compile
            # for that prim's body shader. The scan loop's step_async
            # frame-capture barrier doesn't wait for MDL compile, so the
            # next render races against in-flight compiles — on slow /
            # cold hosts this drops capacitor / solder / ic from the seg
            # labels across all 100 cells while `pad` (PASTEMASK direct
            # mesh, no uninstance) still appears. Mirrors the fix in
            # usd2roi_render.py.
            if _n_uninst > 0:
                _ctx = omni.usd.get_context()
                for _ in range(60):
                    await _app.next_update_async()
                while _ctx.get_stage_loading_status()[2] > 0:
                    await _app.next_update_async()
                print(
                    f"[Pipeline] Post-uninstance settle done "
                    f"(pending stage ops drained)"
                )

        if bench is not None:
            bench["phase_once_s"]["apply_semantics_s"] = time.perf_counter() - _tp

    # Defect pipeline: prepare defects once (NumPy RNG keyed by random_seed, not output image `seed`)
    # Unless `randomize_defects_per_trigger: true` is set at top level — in that case
    # prepare_defects is deferred to inside the trigger loop so each trigger
    # samples a fresh selection (with the previous trigger's selection restored first).
    # In that path we ALSO pre-tag every defect-eligible prim with its defect
    # semantic at init, so the colorize semseg annotator bakes the prim->class
    # map before the first render. Without that pre-tag the annotator only
    # sees prims tagged at setup time and per-trigger rep_modify.semantics
    # calls don't show up in the colorize PNG.
    defect_records = None
    _randomize_defects_per_trigger = bool(CFG.get("randomize_defects_per_trigger", False))
    if PIPELINE_TYPE == "defect":
        if _randomize_defects_per_trigger:
            _tp = time.perf_counter()
            pretag_defect_eligible(component_pool, CFG["defects"])
            if bench is not None:
                bench["phase_once_s"]["pretag_defect_eligible_s"] = time.perf_counter() - _tp
        else:
            _tp = time.perf_counter()
            np.random.seed(RANDOM_SEED)
            defect_records = prepare_defects(component_pool, CFG["defects"])
            if bench is not None:
                bench["phase_once_s"]["prepare_defects_s"] = time.perf_counter() - _tp

    _tp = time.perf_counter()
    # camera_position: { translate: [x, y, z] } → render at one fixed pose,
    # skip scan_grid entirely. Useful for single-shot debugging /
    # registration captures where the grid is irrelevant.
    if "camera_position" in CFG:
        _cp = CFG["camera_position"]
        positions_full = [tuple(float(v) for v in _cp["translate"])]
        print(f"[Pipeline] Fixed camera_position: {positions_full[0]}; {num_triggers} triggers")
    else:
        # Auto scan_grid (x_num / y_num) → fill x_centers / y_centers / z from stage.
        # No-op for older configs (x_start / x_end / step / z).
        resolve_scan_grid_from_stage(stage, CFG)
        positions_full = build_scan_positions(CFG["scan_grid"])
        _cap_note = "unlimited" if MAX_IMAGE_COUNT < 0 else f"cap {MAX_IMAGE_COUNT} writer frames"
        print(
            f"[Pipeline] scan_grid: {len(positions_full)} cells; trigger_0 skips first {IMAGE_SEED}; "
            f"{_cap_note}; {num_triggers} triggers"
        )
    print(f"[Pipeline] Output: {output_root}")

    translate_op = find_translate_op(stage, camera_xform_path)

    # Setup camera randomizer
    camera = rep.get.prim_at_path(camera_path)
    cam_rot = CFG["camera_rotation"]
    if bench is not None:
        bench["phase_once_s"]["scan_grid_and_camera_bind_s"] = time.perf_counter() - _tp

    output_cursor = {"next": 0}
    _flat_output = bool(CFG.get("flat_output", False))

    for trigger_idx in range(num_triggers):
        if _flat_output:
            trigger_dir = output_root
            meta_filename = f"metadata_{trigger_idx:04d}.json"
        else:
            trigger_dir = os.path.join(output_root, f"trigger_{trigger_idx:04d}")
            meta_filename = "metadata.json"
        os.makedirs(trigger_dir, exist_ok=True)

        # Shared: seed NumPy, then optional per-trigger randomization
        _tp = time.perf_counter()
        seed_trigger_numpy(RANDOM_SEED, trigger_idx)

        # Per-trigger defect re-randomization (opt-in via
        # `randomize_defects_per_trigger: true`). Restores the previous
        # trigger's flipped prims to their original xforms before sampling
        # a fresh selection, so flips don't accumulate across triggers.
        if PIPELINE_TYPE == "defect" and _randomize_defects_per_trigger:
            if defect_records:
                restore_defects(defect_records)
            defect_records = prepare_defects(component_pool, CFG["defects"])

        if "randomize_rig" in CFG:
            _randomize_rig(stage, CFG)
        if _mat_state is not None and "randomize_material" in CFG:
            _randomize_material_params(_mat_state, _mat_defs, CFG)
        if _FILLET_ENABLED and "randomize_fillet" in CFG:
            _randomize_fillet_shape(FILLET_PARAMS, CFG)

        light_meta = randomize_lighting(stage, CFG)
        _bench_pt_append(bench, "trigger_lighting_s", time.perf_counter() - _tp)

        # Shared: randomize camera rotation (Replicator; same trigger_idx => same pose if random_seed matches)
        _tp = time.perf_counter()
        set_replicator_seed_for_output_frame(RANDOM_SEED + trigger_idx)
        with camera:
            rep.modify.pose(
                rotation=rep.distribution.uniform(
                    (cam_rot["x_range"][0], cam_rot["y_range"][0], cam_rot["z_fixed"]),
                    (cam_rot["x_range"][1], cam_rot["y_range"][1], cam_rot["z_fixed"]),
                )
            )
        _bench_pt_append(bench, "trigger_camera_pose_s", time.perf_counter() - _tp)
        print("[Pipeline] Camera rotation randomized")

        # Shared: augmentation
        _tp = time.perf_counter()
        aug_meta = setup_augmentation(render_product, CFG)
        _bench_pt_append(bench, "trigger_augmentation_s", time.perf_counter() - _tp)

        positions_this = scan_positions_for_trigger(positions_full, trigger_idx, IMAGE_SEED)
        positions_this = cap_positions_for_max_outputs(
            positions_this, PIPELINE_TYPE, MAX_IMAGE_COUNT, output_cursor["next"]
        )
        if not positions_this:
            if MAX_IMAGE_COUNT >= 0 and output_cursor["next"] >= MAX_IMAGE_COUNT:
                print(f"[Pipeline] max_image_count={MAX_IMAGE_COUNT} reached; stopping triggers.")
                break
            print(
                f"[Pipeline] Trigger {trigger_idx + 1}: no scan positions "
                f"(grid empty after seed skip or zero quota); skipping"
            )
            continue

        print(
            f"[Pipeline] Trigger {trigger_idx + 1}: {len(positions_this)} positions "
            f"(outputs so far {output_cursor['next']})"
        )

        # ``samples_per_position``: render N variants of every scan-grid
        # cell with full re-randomization between samples (lighting +
        # material + rig + fillet shape). Use this for fixed-camera
        # multi-sample runs (1 trigger, N images) so we don't pay the
        # ring-light rebuild cost per sample.
        _samples_per_position = int(CFG.get("samples_per_position", 1))

        def _per_sample_randomize():
            # ``_randomize_rig`` rebuilds 900 DiskLights — ~5 s per call.
            # For samples_per_position>1 that's a big chunk of total
            # render time. Skip it per-sample (rig dome / light_radius
            # stay at the trigger-level random values from line above);
            # ``randomize_lighting`` still re-jitters per-layer
            # intensity + color + exposure + cone for variety.
            if _mat_state is not None and "randomize_material" in CFG:
                _randomize_material_params(_mat_state, _mat_defs, CFG)
            if _FILLET_ENABLED and "randomize_fillet" in CFG:
                _randomize_fillet_shape(FILLET_PARAMS, CFG)
            if _TIN_ENABLED and _mat_state is not None and "randomize_tin_noise" in CFG:
                _randomize_tin_noise_shape(_mat_state, TIN_PARAMS, CFG, _do_make_tin_normalmap)
            return randomize_lighting(stage, CFG)

        # Dispatch scan based on pipeline type
        scan_args = dict(
            trigger_dir=trigger_dir,
            render_product=render_product,
            translate_op=translate_op,
            positions=positions_this,
            pt_total_spp=pt_total_spp,
            trigger_idx=trigger_idx,
            num_triggers=num_triggers,
            light_meta=light_meta,
            aug_meta=aug_meta,
            image_seed=IMAGE_SEED,
            output_cursor=output_cursor,
            meta_filename=meta_filename,
        )

        scan_args["bench"] = bench

        if PIPELINE_TYPE == "good":
            scan_args["samples_per_position"] = _samples_per_position
            scan_args["on_per_sample_randomize"] = _per_sample_randomize
            await _scan_good(**scan_args)
        elif PIPELINE_TYPE == "defect":
            await _scan_defect(**scan_args, defect_records=defect_records)
        elif PIPELINE_TYPE == "missing":
            await _scan_missing(**scan_args, stage=stage, component_pool=component_pool)

        print(f"[Pipeline] Trigger {trigger_idx + 1}/{num_triggers} done -> {trigger_dir}")

        if MAX_IMAGE_COUNT >= 0 and output_cursor["next"] >= MAX_IMAGE_COUNT:
            print(f"[Pipeline] max_image_count={MAX_IMAGE_COUNT} reached; stopping triggers.")
            break

    render_product.destroy()
    print(f"[Pipeline] All done! {output_cursor['next']} output frame(s) written.")
    if bench is not None:
        _print_and_save_benchmark(bench)
    _app.shutdown()


# === Schedule ===
asyncio.ensure_future(run_pipeline())
