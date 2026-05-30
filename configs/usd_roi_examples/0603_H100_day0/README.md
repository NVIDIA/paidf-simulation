# `usd_roi_examples/0603_H100_day0/`

Day-0 CAD-to-ROI for the Spark BASE A04 board, anchored on `_0603_H100` capacitors. Pure-synthetic — no real photo. Two yamls plus a per-board `pcba_target.yaml`.

## Contents

- [Files](#files)
- [Tokens to patch](#tokens-to-patch)
- [How to run](#how-to-run)
- [Expected output](#expected-output)
- [Sanity checks](#sanity-checks)

## Files

| File | Purpose |
|---|---|
| `day0_image.yaml` | Stage 1: full-board good-flow scan (`pipeline_type: good`, `scan_grid: 10×10`, `rename_to_grid_index: true`). Fed to `sdg_pipeline.py`. |
| `day0_crop.yaml` | Stage 2: class-anchored crop over Stage 1's output (`crop.classes: [capacitor]`). Fed to `usd2roi_crop.py` (host python). |
| `pcba_target.yaml` | USD-bound side: `scene`, `camera_path`, `pcba_root`. |

## Tokens to patch

| Token | In file | Replace with |
|---|---|---|
| `__SCENE__` | `pcba_target.yaml`, `day0_image.yaml` | Absolute path to the board USD (in-repo Spark: `assets/spark_lighting.usd`). |
| `__OUTPUT__` | `day0_image.yaml`, `day0_crop.yaml` | Absolute path to the per-run output dir. |
| `__MAX_IMAGE_COUNT__` | `day0_image.yaml` | `-1` for unlimited (100 cells), or an integer to cap. |

## How to run

`day0_image.yaml` and `day0_crop.yaml` share the same `__OUTPUT__` — Stage 2 reads from Stage 1's `trigger_0000/`.

```bash
export PCB_USD_PATH=<path/to/your/board.usd>
export PAIDF_SIM_ROOT=<path/to/paidf-simulation>
IMG=nvcr.io/nv-metropolis-dev/metropolis-sdg/paidf-simulation:<TAG>

OUT=$PAIDF_SIM_ROOT/sdg_test_output/usd_roi_0603_H100_day0
RUN_DIR=$PAIDF_SIM_ROOT/configs/runs/usd_roi_0603_H100_day0
SRC=$PAIDF_SIM_ROOT/configs/usd_roi_examples/0603_H100_day0

mkdir -p "$RUN_DIR" "$OUT" && chmod -R o+w "$OUT"

# Patch placeholders
sed "s|__SCENE__|$PCB_USD_PATH|" "$SRC/pcba_target.yaml" > "$RUN_DIR/pcba_target.yaml"
sed -e "s|__SCENE__|$PCB_USD_PATH|" \
    -e "s|__OUTPUT__|$OUT|" \
    -e "s|__MAX_IMAGE_COUNT__|100|" \
    "$SRC/day0_image.yaml" > "$RUN_DIR/day0_image.yaml"
sed "s|__OUTPUT__|$OUT|" "$SRC/day0_crop.yaml" > "$RUN_DIR/day0_crop.yaml"

# Stage 1 — render (Kit)
docker run --gpus all --rm --network host \
  -e ACCEPT_EULA=Y -e PYTHONUNBUFFERED=1 \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PAIDF_SIM_ROOT \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
  -v $PAIDF_SIM_ROOT/sdg_test_output:$PAIDF_SIM_ROOT/sdg_test_output \
  $IMG \
  "scripts/sdg/standalone/sdg_pipeline.py \
    --config /workspace/paidf-simulation/configs/runs/usd_roi_0603_H100_day0/day0_image.yaml \
    --pcba-config /workspace/paidf-simulation/configs/runs/usd_roi_0603_H100_day0/pcba_target.yaml"

# Stage 2 — crop (host python; override entrypoint)
docker run --rm \
  -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
  -v $PAIDF_SIM_ROOT/sdg_test_output:$PAIDF_SIM_ROOT/sdg_test_output \
  --entrypoint python3 $IMG \
  scripts/usd2roi/usd2roi_crop.py \
    --config /workspace/paidf-simulation/configs/runs/usd_roi_0603_H100_day0/day0_crop.yaml
```

## Expected output

Stage 1 (`sdg_pipeline.py`) writes to `$OUT/trigger_0000/`. Filenames use the `xN_yM` grid index (not sequential) because `day0_image.yaml` sets `rename_to_grid_index: true`:

```
sdg_test_output/usd_roi_0603_H100_day0/
├── run.log
└── trigger_0000/
    ├── rgb_x0_y0.png … rgb_x9_y9.png                          # 100 RGB frames
    ├── semantic_segmentation_x0_y0.png … _x9_y9.png           # 100 colorized seg
    ├── semantic_segmentation_labels_x0_y0.json … _x9_y9.json  # 100 colour→class maps
    ├── metadata.json
    └── metadata.txt
```

No bbox files — the writer in `day0_image.yaml` emits only RGB + semantic segmentation.

Sample label JSON:

```json
{"(0, 0, 0, 0)": {"class": "BACKGROUND"},
 "(0, 0, 0, 255)": {"class": "UNLABELLED"},
 "(33, 243, 3, 255)": {"class": "pad"},
 "(240, 4, 111, 255)": {"class": "solder"},
 "(27, 186, 239, 255)": {"class": "capacitor"}}
```

Three real classes: `pad`, `solder`, `capacitor` (driven by `semantics:` rules under `day0_image.yaml`, not by `pcba_target.yaml`'s `component_types`).

Stage 2 (`usd2roi_crop.py`) writes per-class crops under `$OUT/crop/<class>/...` (see Day-1 README for shape; output structure is identical).

## Sanity checks

- `run.log` ends with `[Pipeline] All done! 100 output frame(s) written.`
- `ls trigger_0000/rgb_*.png | wc -l` → `100`
- Sample frames: `rgb_x5_y5.png` should land near the board centre; corners (`x0_y0`, `x9_y9`) at PCBA bbox extremes.
- Seg PNGs should colour-mask `pad` / `solder` / `capacitor` regions only — UNLABELLED everywhere else.
