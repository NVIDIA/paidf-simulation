# `usd_roi_examples/115_2819_000_day0/`

Day-0 CAD-to-ROI for the IC board, anchored on the `_115_2819_000` SOT5/6 package. Pure-synthetic — no real photo. Structure is identical to [`0603_H100_day0/`](../0603_H100_day0/) — only the target component differs.

## Contents

- [Files](#files)
- [Tokens to patch](#tokens-to-patch)
- [How to run](#how-to-run)
- [Expected output](#expected-output)
- [Sanity checks](#sanity-checks)
- [In-repo Spark caveat](#in-repo-spark-caveat)

## Files

| File | Purpose |
|---|---|
| `day0_image.yaml` | Stage 1: full-board good-flow scan (`pipeline_type: good`, `scan_grid: 10×10`, `rename_to_grid_index: true`). |
| `day0_crop.yaml` | Stage 2: class-anchored crop on `ic` / `solder` / `pad`. |
| `pcba_target.yaml` | USD-bound side. `assets/spark_lighting.usd`. |

## Tokens to patch

| Token | In file | Replace with |
|---|---|---|
| `__SCENE__` | `pcba_target.yaml`, `day0_image.yaml` | Absolute path to the IC board USD (in-repo: reuse `assets/spark_lighting.usd` for a "pipeline-alive" smoke test — see caveat below). |
| `__OUTPUT__` | `day0_image.yaml`, `day0_crop.yaml` | Absolute path to the per-run output dir. |
| `__MAX_IMAGE_COUNT__` | `day0_image.yaml` | `-1` for unlimited, or an integer to cap. |

## How to run

Same shape as `0603_H100_day0` — only the source dir name and output dir name differ:

```bash
export PCB_USD_PATH=<path/to/your/board.usd>
export PAIDF_SIM_ROOT=<path/to/paidf-simulation>
IMG=nvcr.io/nv-metropolis-dev/metropolis-sdg/paidf-simulation:<TAG>

OUT=$PAIDF_SIM_ROOT/sdg_test_output/usd_roi_115_2819_000_day0
RUN_DIR=$PAIDF_SIM_ROOT/configs/runs/usd_roi_115_2819_000_day0
SRC=$PAIDF_SIM_ROOT/configs/usd_roi_examples/115_2819_000_day0

mkdir -p "$RUN_DIR" "$OUT" && chmod -R o+w "$OUT"

sed "s|__SCENE__|$PCB_USD_PATH|" "$SRC/pcba_target.yaml" > "$RUN_DIR/pcba_target.yaml"
sed -e "s|__SCENE__|$PCB_USD_PATH|" \
    -e "s|__OUTPUT__|$OUT|" \
    -e "s|__MAX_IMAGE_COUNT__|100|" \
    "$SRC/day0_image.yaml" > "$RUN_DIR/day0_image.yaml"

docker run --gpus all --rm --network host \
  -e ACCEPT_EULA=Y -e PYTHONUNBUFFERED=1 \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PAIDF_SIM_ROOT \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
  -v $PAIDF_SIM_ROOT/sdg_test_output:$PAIDF_SIM_ROOT/sdg_test_output \
  $IMG \
  "scripts/sdg/standalone/sdg_pipeline.py \
    --config /workspace/paidf-simulation/configs/runs/usd_roi_115_2819_000_day0/day0_image.yaml \
    --pcba-config /workspace/paidf-simulation/configs/runs/usd_roi_115_2819_000_day0/pcba_target.yaml"
```

## Expected output

Identical shape to `0603_H100_day0` — 100 cells, filenames `rgb_xN_yM.png`, seg + label JSONs, no bbox:

```
sdg_test_output/usd_roi_115_2819_000_day0/
├── run.log
└── trigger_0000/
    ├── rgb_x0_y0.png … rgb_x9_y9.png                          # 100 RGB frames
    ├── semantic_segmentation_x0_y0.png … _x9_y9.png           # 100 seg
    ├── semantic_segmentation_labels_x0_y0.json … _x9_y9.json  # 100 colour→class maps
    ├── metadata.json
    └── metadata.txt
```

Sample label JSON:

```json
{"(0, 0, 0, 0)": {"class": "BACKGROUND"},
 "(0, 0, 0, 255)": {"class": "UNLABELLED"},
 "(33, 243, 3, 255)": {"class": "pad"},
 "(240, 4, 111, 255)": {"class": "solder"},
 "(27, 186, 239, 255)": {"class": "ic"}}
```

The `ic` class replaces `capacitor` from the 0603 board's rules — the `semantics:` block in `day0_image.yaml` is per-board.

## Sanity checks

- `run.log` ends with `[Pipeline] All done! 100 output frame(s) written.`
- `ls trigger_0000/rgb_*.png | wc -l` → `100`
- Seg PNGs colour-mask `pad` / `solder` / `ic` only.

## In-repo Spark caveat

The pcba_target's `pcba_root` matches the Spark BASE path, so running with `assets/spark_lighting.usd` produces 100 valid frames. But the `semantics:` rules in `day0_image.yaml` were authored against the IC board's per-board uninstanced scene — on Spark, the `ic` class may match no prims (zero-coloured seg) or different prims than intended. Use this for pipeline plumbing checks, not as training data.
