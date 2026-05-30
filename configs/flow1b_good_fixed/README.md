# `flow1b_good_fixed/`

Spark BASE A04 worked example for the **good_fixed** variant: component-targeted close-up via `auto_locate_component`. Pairs with `configs/pcba_target.yaml` (Spark default).

Default behaviour: pick a random `_0603_H100` capacitor instance under `pcba_root`, auto-frame the camera so its bbox plus 1/2 padding fills the viewport, then render `samples_per_position` frames with per-sample fillet + tin-noise + lighting randomization (board substrate forced to vantablack so only the component is visible).

## Contents

- [How to run](#how-to-run)
- [Expected output](#expected-output)
- [Sanity checks](#sanity-checks)
- [Switching the target component](#switching-the-target-component)

## How to run

Same Docker pattern as the other examples — repo bind-mounted on top of the baked image:

```bash
export PCB_USD_PATH=<path/to/your/board.usd>
export PAIDF_SIM_ROOT=$(pwd)

docker run --gpus all --rm --network host \
  -e ACCEPT_EULA=Y -e PYTHONUNBUFFERED=1 \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PAIDF_SIM_ROOT \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
  -v $PAIDF_SIM_ROOT/sdg_test_output:$PAIDF_SIM_ROOT/sdg_test_output \
  nvcr.io/nv-metropolis-dev/metropolis-sdg/paidf-simulation:<TAG> \
  "scripts/sdg/standalone/sdg_pipeline.py \
    --config /workspace/paidf-simulation/configs/flow1b_good_fixed/good_fixed.yaml \
    --pcba-config /workspace/paidf-simulation/configs/pcba_target.yaml"
```

## Expected output

Output dir (auto-created): `${PAIDF_SIM_ROOT}/sdg_test_output/flow1b_good_fixed/`. With the example's `samples_per_position: 300, max_image_count: 300, num_triggers: 1`, you get 300 frames at one component instance:

```
sdg_test_output/flow1b_good_fixed/
├── run.log
└── trigger_0000/
    ├── rgb_0000.png … rgb_0299.png                          # 300 close-ups
    ├── semantic_segmentation_*                              # only if writer.semantic_segmentation: true
    ├── bounding_box_2d_tight_*                              # only if writer.bounding_box_2d_tight: true
    └── metadata.{json,txt}
```

Each `rgb_NNNN.png` is a 1920×1080 ortho close-up of the same `_0603_H100` instance, but with per-sample variation in:
- ring-rig lighting (Inner_Red / Middle_Green / Outer_Blue intensity / cone / colour)
- fillet shape (`solder_fillet.smoothness_y`, `bump`, `delta_x_decay`)
- tin-noise pattern (`tin_noise.noise_amp`, `noise_scale`, `noise_octaves`, …)

For a full Spark-board scan (100 frames) instead of single-component sampling, see `flow1_good_image/`.

## Sanity checks

- `run.log` ends with `[Pipeline] All done! 300 output frame(s) written.` (or whatever `max_image_count` you set).
- `ls trigger_0000/rgb_*.png | wc -l` matches `max_image_count`.
- Each `rgb_NNNN.png` shows a single capacitor against a fully-black board substrate, lit by per-trigger-randomised ring colours; the silver solder fillet should differ subtly frame-to-frame.

## Switching the target component

Two override keys:

- `auto_locate_component: _0402_H060` — single substring; pick a random instance whose path contains that string.
- `component_list_override: [_0402_H060, _0603_H100, _0805U_H150]` — list of substrings; pick a random instance from any match. Overrides `auto_locate_component` when set.

The camera auto-frames the chosen instance's bbox + padding, so no manual `scan_grid.x/y` tuning is needed.
