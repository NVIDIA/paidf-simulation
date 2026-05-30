# `flow1_good_image/`

Spark BASE worked example for the **good** pipeline (single-pass scan, no defects). Pairs with `configs/pcba_target.yaml` (Spark default; 23-component inline list).

## Contents

- [How to run (Docker, repo bind-mounted)](#how-to-run-docker-repo-bind-mounted)
- [Expected output](#expected-output)
  - [Per-frame artefacts at a glance](#per-frame-artefacts-at-a-glance)
- [Sanity checks after a run](#sanity-checks-after-a-run)
- [Common deviations](#common-deviations)

## How to run (Docker, repo bind-mounted)

```bash
export PCB_USD_PATH=/path/to/spark_lighting.usd
export PAIDF_SIM_ROOT=$(pwd)            # repo root

docker run --gpus all --rm --network host \
  -e ACCEPT_EULA=Y -e PYTHONUNBUFFERED=1 \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PAIDF_SIM_ROOT \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
  -v $PAIDF_SIM_ROOT/sdg_test_output:$PAIDF_SIM_ROOT/sdg_test_output \
  nvcr.io/nv-metropolis-dev/metropolis-sdg/paidf-simulation:<TAG> \
  "scripts/sdg/standalone/sdg_pipeline.py \
    --config /workspace/paidf-simulation/configs/flow1_good_image/good_image.yaml \
    --pcba-config /workspace/paidf-simulation/configs/pcba_target.yaml"
```

`sdg_test_output/` must be writable by the in-container `isaac-sim` user (uid 1234). On a fresh checkout: `chmod -R o+w sdg_test_output/`.

## Expected output

Output dir (auto-created): `${PAIDF_SIM_ROOT}/sdg_test_output/flow1_good_image/`

```
sdg_test_output/flow1_good_image/
├── run.log                       # tee'd Kit + pipeline stdout
└── trigger_0000/
    ├── rgb_0000.png … rgb_0099.png                          # 100 frames
    ├── semantic_segmentation_0000.png … _0099.png           # 100 colorized seg
    ├── semantic_segmentation_labels_0000.json … _0099.json  # 100 colour→class maps
    ├── bounding_box_2d_tight_0000.npy … _0099.npy           # 100 raw bbox arrays
    ├── bounding_box_2d_tight_labels_0000.json … _0099.json  # 100 id→class maps
    ├── bounding_box_2d_tight_prim_paths_0000.json … _0099.json  # 100 id→USD prim paths
    ├── metadata.json
    └── metadata.txt
```

**Total file count**: 1 log + 800 frame artefacts + 2 metadata = **803 files** for a 100-frame run.

### Per-frame artefacts at a glance

| Stem | Type | Size (typical) | What it carries |
|---|---|---|---|
| `rgb_NNNN.png` | RGBA | ~270 KB | Path-traced render at `resolution` (default `[1920, 1080]`). |
| `semantic_segmentation_NNNN.png` | RGBA | ~40 KB | Per-pixel class colour map. |
| `semantic_segmentation_labels_NNNN.json` | JSON | small | `"(r,g,b,a)" → {"class": "<name>"}` — e.g. `{"(0,0,0,0)": "BACKGROUND", "(0,0,0,255)": "UNLABELLED", "(33,243,3,255)": "capacitor"}`. |
| `bounding_box_2d_tight_NNNN.npy` | NumPy | ~4 KB | Structured array: per-instance `(semantic_id, x_min, y_min, x_max, y_max, occlusion_ratio)`. |
| `bounding_box_2d_tight_labels_NNNN.json` | JSON | small | `"<id>" → {"class": "<name>"}` — e.g. `{"0": {"class": "capacitor"}}`. |
| `bounding_box_2d_tight_prim_paths_NNNN.json` | JSON | small | `"<id>" → "<USD prim path>"` — lets you back-reference each bbox to its source prim. |

`NNNN` runs from `0000` to `0099` for the canonical 10×10 scan grid.

## Sanity checks after a run

- `run.log` ends with `[Pipeline] All done! 100 output frame(s) written.`
- `ls trigger_0000/rgb_*.png | wc -l` → `100`
- A spot-checked `rgb_NNNN.png` shows the Spark board lit with the dome (default `use_scene_lights: true, ring_light: false`), components visible against the board substrate.
- A spot-checked `semantic_segmentation_NNNN.png` highlights only the `capacitor`-classed pixels (single class because the writer's `semantic_types: [class]` paired with the 23-component selection collapses everything in `pcba_target.yaml`'s `component_types:` under one label).
- `bounding_box_2d_tight_labels_*.json` should contain `{"0": {"class": "capacitor"}}` — same single class.

## Common deviations

- **Empty `trigger_0000/`** → check `chmod` (see above) and grep `run.log` for `PermissionError`.
- **Fewer than 100 frames** → check `run.log` for `[Pipeline] All done!` line; an early crash on a per-frame Kit step prints a Python traceback before the line is reached. Most often: GPU OOM (drop `total_spp`) or a missing prim path (override in `pcba_target.yaml`).
- **All-black RGB frames** → `vantablack_components: true` is on, or `lighting.white_light.intensity` is too low (Spark default `[160, 440]` is the post-MR-!10 dome at 20 % output).
