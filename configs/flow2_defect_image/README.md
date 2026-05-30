# `flow2_defect_image/`

Spark BASE worked examples for the **defect** family:

| File | Pipeline | What it produces |
|---|---|---|
| `defect_image.yaml` | `defect` | Single-pass scan, chosen components get pose defects (`shift` / `tombstone` / `sideflip` / `reverse_polarity`). |
| `missing_image.yaml` | `missing` | Two-pass per cell: reference (all visible) + defective (a fraction of components hidden). |

Both pair with `configs/pcba_target.yaml` (Spark default; 23-component inline list).

## Contents

- [How to run (Docker, repo bind-mounted)](#how-to-run-docker-repo-bind-mounted)
- [`defect_image.yaml`](#defect_imageyaml)
  - [Expected output](#expected-output)
  - [Per-frame artefacts](#per-frame-artefacts)
  - [Sanity checks](#sanity-checks)
  - [Common deviations](#common-deviations)
- [`missing_image.yaml`](#missing_imageyaml)
  - [Expected output](#expected-output-1)
  - [Per-frame artefacts](#per-frame-artefacts-1)
  - [Sanity checks](#sanity-checks-1)
  - [Common deviations](#common-deviations-1)

## How to run (Docker, repo bind-mounted)

```bash
export PCB_USD_PATH=/path/to/spark_lighting.usd
export PAIDF_SIM_ROOT=$(pwd)            # repo root

# Pick one
FLOW=defect_image.yaml                # or: missing_image.yaml

docker run --gpus all --rm --network host \
  -e ACCEPT_EULA=Y -e PYTHONUNBUFFERED=1 \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PAIDF_SIM_ROOT \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
  -v $PAIDF_SIM_ROOT/sdg_test_output:$PAIDF_SIM_ROOT/sdg_test_output \
  nvcr.io/nv-metropolis-dev/metropolis-sdg/paidf-simulation:<TAG> \
  "scripts/sdg/standalone/sdg_pipeline.py \
    --config /workspace/paidf-simulation/configs/flow2_defect_image/$FLOW \
    --pcba-config /workspace/paidf-simulation/configs/pcba_target.yaml"
```

`sdg_test_output/` must be writable by the in-container `isaac-sim` user (uid 1234). On a fresh checkout: `chmod -R o+w sdg_test_output/`.

## `defect_image.yaml`

Single-pass 10×10 scan with three pose defects active (`shift` / `tombstone` / `sideflip`); `reverse_polarity` is opt-in (`enabled: false` by default).

### Expected output

Output dir (auto-created): `${PAIDF_SIM_ROOT}/sdg_test_output/flow2_defect_image/`

```
sdg_test_output/flow2_defect_image/
├── run.log                       # tee'd Kit + pipeline stdout
└── trigger_0000/
    ├── rgb_0000.png … rgb_0099.png                          # 100 frames
    ├── semantic_segmentation_0000.png … _0099.png           # 100 colorized seg
    ├── semantic_segmentation_labels_0000.json … _0099.json  # 100 colour→defect maps
    ├── bounding_box_2d_tight_0000.npy … _0099.npy           # 100 raw bbox arrays
    ├── bounding_box_2d_tight_labels_0000.json … _0099.json  # 100 id→defect maps
    ├── bounding_box_2d_tight_prim_paths_0000.json … _0099.json  # 100 id→USD prim paths
    ├── metadata.json
    └── metadata.txt
```

**Total file count**: 1 log + 800 frame artefacts + 2 metadata = **803 files** for the canonical 100-frame run.

### Per-frame artefacts

| Stem | Type | Size (typical) | What it carries |
|---|---|---|---|
| `rgb_NNNN.png` | RGBA | ~270 KB | Path-traced render at `resolution` (default `[1920, 1080]`) with defects applied. |
| `semantic_segmentation_NNNN.png` | RGBA | ~40 KB | Per-pixel **defect** colour map (no `class` labels; the writer's `semantic_types: [defect]` only captures defect-tagged components). |
| `semantic_segmentation_labels_NNNN.json` | JSON | small | `"(r,g,b,a)" → {"defect": "<kind>"}` — e.g. `{"(0,0,0,0)": "BACKGROUND", "(0,0,0,255)": "UNLABELLED", "(33,243,3,255)": "sideflip", "(240,4,111,255)": "shift", "(27,186,239,255)": "tombstone"}`. |
| `bounding_box_2d_tight_NNNN.npy` | NumPy | ~4 KB | Structured array: per-instance `(semantic_id, x_min, y_min, x_max, y_max, occlusion_ratio)`. |
| `bounding_box_2d_tight_labels_NNNN.json` | JSON | small | `"<id>" → {"defect": "<kind>"}` — e.g. `{"0": {"defect": "shift"}, "1": {"defect": "tombstone"}, "2": {"defect": "sideflip"}}`. |
| `bounding_box_2d_tight_prim_paths_NNNN.json` | JSON | small | `"<id>" → "<USD prim path>"` — back-references each bbox to the defective prim. |

`NNNN` runs from `0000` to `0099` for the canonical 10×10 scan grid.

### Sanity checks

- `run.log` ends with `[Pipeline] All done! 100 output frame(s) written.`
- `ls trigger_0000/rgb_*.png | wc -l` → `100`
- Each `bounding_box_2d_tight_labels_*.json` contains entries keyed by the three defect kinds (`shift` / `tombstone` / `sideflip`); some frames may have only one or two if the defect ratios + non-overlapping selection happened to leave one type empty for that cell.
- A spot-checked `rgb_NNNN.png` shows the Spark board with at least one visibly-mispositioned, tilted, or flipped component vs. the matching frame from `flow1_good_image/`.
- The seg PNG only highlights the defective components (the rest of the board stays unlabelled), distinct from the `flow1_good_image` seg which colours every component.

### Common deviations

- **Empty `trigger_0000/`** → check `chmod` (see top of this README) and grep `run.log` for `PermissionError`.
- **`[CUDA LAUNCH ERROR] Transform ops kernel launch failed: invalid resource handle` lines** → known noise from `pose_ops`. Harmless if `[Pipeline] All done!` still prints and frame counts match.
- **No defects visible in any frame** → check `defects.<kind>.enabled: true` and `defects.<kind>.ratio > 0`; if all four are off you'll get clean Spark frames identical to `flow1_good_image`.
- **`reverse_polarity` not firing** → `enabled: false` by default in the canonical example. To turn it on, set `defects.reverse_polarity.enabled: true` AND verify `defects.reverse_polarity.component_types:` substrings actually match prims under `pcba_root` (default `[_032_0831]` is QFN033-family on Spark).

## `missing_image.yaml`

Two-pass 10×10 scan. Per cell the pipeline writes one **reference** frame (labels only — where the components *will be* hidden) and one **defective** frame (RGB only — the actual image with those components hidden). With the canonical `missing.ratio: 0.5`, half the components are hidden per trigger.

### Expected output

Output dir (auto-created): `${PAIDF_SIM_ROOT}/sdg_test_output/flow2_missing_image/`. The two passes land in separate subdirectories under `trigger_0000/`:

```
sdg_test_output/flow2_missing_image/
├── run.log                       # tee'd Kit + pipeline stdout
└── trigger_0000/
    ├── metadata.json
    ├── reference/                # Pass 1 — labels only (rgb: false)
    │   ├── semantic_segmentation_0000.png … _0099.png           # 100 colorized seg
    │   ├── semantic_segmentation_labels_0000.json … _0099.json  # 100 colour→defect maps
    │   ├── bounding_box_2d_tight_0000.npy … _0099.npy           # 100 raw bbox arrays
    │   ├── bounding_box_2d_tight_labels_0000.json … _0099.json  # 100 id→defect maps
    │   ├── bounding_box_2d_tight_prim_paths_0000.json … _0099.json
    │   └── metadata.txt
    └── defective/                # Pass 2 — rgb only
        ├── rgb_0000.png … rgb_0099.png                          # 100 frames
        └── metadata.txt
```

**Total file count**: 1 log + 100 defective RGB + 500 reference label artefacts (5×100) + 3 metadata = **604 files** for the canonical 100-cell run (200 writer frames total).

### Per-frame artefacts

`reference/` — labels only, no RGB:

| Stem | Type | Size (typical) | What it carries |
|---|---|---|---|
| `semantic_segmentation_NNNN.png` | RGBA | ~40 KB | Per-pixel **defect** colour map. Only the to-be-hidden components are coloured. |
| `semantic_segmentation_labels_NNNN.json` | JSON | small | `"(r,g,b,a)" → {"defect": "<kind>"}` — e.g. `{"(0,0,0,0)": "BACKGROUND", "(0,0,0,255)": "UNLABELLED", "(33,243,3,255)": "missing"}`. |
| `bounding_box_2d_tight_NNNN.npy` | NumPy | ~4 KB | Structured array: per-instance `(semantic_id, x_min, y_min, x_max, y_max, occlusion_ratio)` for the to-be-hidden components. |
| `bounding_box_2d_tight_labels_NNNN.json` | JSON | small | `"<id>" → {"defect": "missing"}` — e.g. `{"0": {"defect": "missing"}}`. |
| `bounding_box_2d_tight_prim_paths_NNNN.json` | JSON | small | `"<id>" → "<USD prim path>"` of the to-be-hidden component. |

`defective/` — RGB only:

| Stem | Type | Size (typical) | What it carries |
|---|---|---|---|
| `rgb_NNNN.png` | RGBA | ~480 KB | Path-traced render with the chosen components **hidden** (visibility off). Frame index matches `reference/` so the pair is `(defective/rgb_NNNN.png, reference/semantic_segmentation_NNNN.png)`. |

`NNNN` runs from `0000` to `0099` for the canonical 10×10 scan grid.

### Sanity checks

- `run.log` ends with `[Pipeline] All done! 200 output frame(s) written.` (reference + defective counted separately).
- `ls trigger_0000/defective/rgb_*.png | wc -l` → `100`; `ls trigger_0000/reference/semantic_segmentation_*.png | wc -l` → `100`.
- Pair the same `NNNN` across the two subdirs: `defective/rgb_NNNN.png` should be the same camera view as `reference/semantic_segmentation_NNNN.png`, with components masked in the seg showing *missing* in the RGB.
- Every `reference/bounding_box_2d_tight_labels_*.json` has at least one entry keyed `{"defect": "missing"}`.
- The `reference/` seg PNG masks should consistently mark only the components hidden in the matching defective frame — count of seg-coloured components ≈ `missing.ratio × |pool|` per cell (≈ 0.5 × 23 = ~11 instances on Spark).

### Common deviations

- **`defective/rgb_*.png` indistinguishable from `flow1_good_image/rgb_*.png`** → `missing.ratio` is too low to be visible at this view, or the hidden components are off-frame. Increase `missing.ratio` or check the seg mask for visibility.
- **`reference/` empty** → `writer.reference` block missing or all-false; double-check the yaml against the canonical example.
- **Mismatched pair counts (`reference/*` != `defective/*`)** → an early Kit crash; grep `run.log` for the first Traceback to find the cell where it stopped.
- **`defective/` includes seg / bbox files** → writer mis-configured (`writer.defective.semantic_segmentation: true` etc.); the canonical example only emits RGB in the defective pass.
