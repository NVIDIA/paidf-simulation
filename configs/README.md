# `configs/`

Top-level config layout for the Simulation SDG pipelines.

Every run takes two YAMLs:

- **Flow YAML** — pipeline / render / writer / defect / lighting settings (under `<flow>/`).
- **PCBA YAML** — USD-bound settings: scene path, prim paths, `component_types` (the shared `pcba_target.yaml` at this level, or a per-board override under `usd_roi_examples/<component>_day<n>/`).

Pass both at the CLI. The canonical invocation is Docker — repo bind-mounted on top of a baked image so the current branch's code is used:

```bash
export PCB_USD_PATH=/path/to/board.usd          # scene usd file; expanded into pcba_target.yaml
export PAIDF_SIM_ROOT=$(pwd)                      # repo root; expanded into flow YAML's `output:`

docker run --gpus all --rm --network host \
  -e ACCEPT_EULA=Y -e PYTHONUNBUFFERED=1 \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PAIDF_SIM_ROOT \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
  -v $PAIDF_SIM_ROOT/sdg_test_output:$PAIDF_SIM_ROOT/sdg_test_output \
  nvcr.io/nv-metropolis-dev/metropolis-sdg/paidf-simulation:<TAG> \
  "scripts/sdg/standalone/sdg_pipeline.py \
    --config /workspace/paidf-simulation/configs/<flow>/<flow>.yaml \
    --pcba-config /workspace/paidf-simulation/configs/pcba_target.yaml"
```

- `<TAG>` — published image tag, e.g. `1.0.0-84fff6a8.main`.
- The image's entrypoint already wraps `/isaac-sim/kit/kit ... --no-window --exec`; the final string is just the script path + its args.
- The pipeline enforces a strict key boundary — any key appearing in both `--config` and `--pcba-config` raises at startup.

## Contents

- [Examples](#examples)
- [`cad2roi/` (skill-shipped worked examples)](#cad2roi-skill-shipped-worked-examples)
- [`pcba_target.yaml`](#pcba_targetyaml)
- [Components](#components)
  - [Subsets currently defined](#subsets-currently-defined)

## Examples

Concrete board-bound configs you can copy and run. Each subdirectory is a worked example with its own README documenting expected outputs.

| Path | Pipeline | Frames | Notes |
|---|---|---|---|
| `flow1_good_image/good_image.yaml` | `good` | 100 (10×10) | Baseline. Path-traced 1920×1080, dome lighting (`use_scene_lights: true`). |
| `flow1b_good_fixed/good_fixed.yaml` | `good` | 300 samples | Component-targeted close-up via `auto_locate_component` (default `_0603_H100`): pick a random instance, auto-frame the camera. 1920×1080 ortho, RGB ring rig, per-sample fillet + tin-noise randomization. |
| `flow2_defect_image/defect_image.yaml` | `defect` | 100 (10×10) | Pose defects: `shift` / `tombstone` / `sideflip`. `reverse_polarity` is opt-in. |
| `flow2_defect_image/missing_image.yaml` | `missing` | 200 (10×10 × 2) | Two-pass per cell: reference (all visible) + defective (50 % hidden). |
| `lighting_example/good_image_scene_lights.yaml` | `good` | 100 (10×10) | Reuse USD-authored lights, whitened. |
| `lighting_example/good_image_preserve_color.yaml` | `good` | 100 (10×10) | Same + keep authored RGB on scene lights. |
| `lighting_example/good_image_ring_light.yaml` | `good` | 100 (10×10) | Build per-trigger RGB ring rig from `lighting.layers`. |
| `lighting_example/good_image_dome_light.yaml` | `good` | 100 (10×10) | Drive dome via `lighting.white_light` (no per-layer RGB). |
| `usd_roi_examples/0603_H100_day0/{day0_image,day0_crop,pcba_target}.yaml` | cad2roi day-0 | (per workflow) | Day-0 USD-to-ROI: pure-synthetic crops anchored on 0603 capacitor. Templates with `__SCENE__`, `__OUTPUT__`, `__MAX_IMAGE_COUNT__` patched at runtime. |
| `usd_roi_examples/0603_H100_day1/usd2roi_nvpcb.yaml` | cad2roi day-1 | (per workflow) | Day-1: synth + real triples aligned to a real photo. |
| `usd_roi_examples/115_2819_000_day0/` | cad2roi day-0 | (per workflow) | IC board (115_2819_000) day-0 equivalent. |
| `usd_roi_examples/115_2819_000_day1/usd2roi_nvpcb.yaml` | cad2roi day-1 | (per workflow) | IC board day-1 equivalent. |

The `flow*` / `lighting_example` configs are runnable directly via `sdg_pipeline.py`. The `usd_roi_examples` are templates consumed by the OSMO / roi-track-of-`/simulation` workflow — they carry `__TOKEN__` placeholders patched at job submit time.

## `cad2roi/` (skill-shipped worked examples)

Author-once, run-as-is templates the `/simulation` skill exposes (it carries a symlink `assets/configs` → `cad2roi/`). Unlike `usd_roi_examples/`, these have **no `__TOKEN__` placeholders**; copy a yaml into a host `$CONFIG` dir and edit per board.

| Path | Purpose |
|---|---|
| `cad2roi/day0/sdg/day0_image.yaml` | Day-0 ROI render template |
| `cad2roi/day0/usd2roi/day0_crop.yaml` | Day-0 ROI crop template |
| `cad2roi/day1/replicator/usd2roi_target.yaml` | Day-1 single-config template |
| `cad2roi/day1/osmo/workflow.yaml` | Day-1 OSMO submission workflow |
| `cad2roi/spark/pcba_target.yaml` | Spark worked-example PCBA target |
| `cad2roi/spark/semantics.yaml` | Spark worked-example semantics block |

## `pcba_target.yaml`

Shared placeholder bound to the Spark BASE A04 PCBA. It holds USD-side settings the flow YAMLs intentionally don't carry:

| Key | What it points at |
|---|---|
| `scene` | USD stage path. Expanded from `${PCB_USD_PATH}` at load time. |
| `pcba_root` | Prim path of the PCBA inside the stage. |
| `camera_path` / `camera_xform_path` | Camera and its parent Xform, this two can be same. |
| `ring_light_root` | Root prim for the per-trigger RGB ring rig, can be normal lighting path. |
| `component_types` | List of components selected for labelling / defect injection (see below). |

For non-Spark boards, pass `--pcba-config configs/usd_roi_examples/<board>_day<n>/pcba_target.yaml` instead.

## Components

`components.yaml` is the master registry. `pcba_target.yaml` can either inline a `component_types` list (preferred for per-board overrides) or reference the registry by a keyword:

| Keyword | Resolves to |
|---|---|
| `ALL` | The top-level `all:` list (23 originals + PM-supplied additions). |
| `0` | `[]` — no components labelled, no defects applied. |
| any key in `subsets:` | That subset's list. |

### Subsets currently defined

| Subset | Coverage |
|---|---|
| `cap_small` | 0201 / 0402 chip capacitors. |
| `cap_large` | 0603 / 0805 / 1206 / 1210 chip capacitors. |
| `chip_passive` | All chip capacitors + resistors (everything except inductors). |
| `inductor` | Surface-mount inductors (`IND_SMD_*`). |
| `tantalum_capacitor` | Tantalum caps. |
| `small_discrete` | Small discrete components. |
| `sot5_sot6` | SOT-5 / SOT-6 packages. |
| `leadless_qfn_lga` | Leadless QFN / LGA. |
| `soic_dfn` | SOIC / DFN. |
| `bga` | BGA / large IC packages. |
| `optoelectronic` | Optoelectronic components. |
| `molded_inductor` | Molded power inductors. |

Add new subsets directly in `components.yaml` — no code change required.

