# `usd_roi_examples/115_2819_000_day1/`

Day-1 CAD-to-ROI for the IC board, anchored on the `_115_2819_000` SOT5/6 package. Real photo supplied → registered synth↔real ROI triples. Structure is identical to [`0603_H100_day1/`](../0603_H100_day1/); only the target component, `semantics:` rules, and camera framing differ.

## Contents

- [Files](#files)
- [Tokens to patch](#tokens-to-patch)
- [How to run (3 stages)](#how-to-run-3-stages)
- [Expected output](#expected-output)
- [Sanity checks](#sanity-checks)

## Files

| File | Purpose |
|---|---|
| `usd2roi_nvpcb.yaml` | Single config used by all three day-1 tools (render → register → crop). Camera framing is the IC close-up `translate=[-103.5, -77.601], horizontal_aperture=48.85`. |

## Tokens to patch

| Token | In file | Replace with |
|---|---|---|
| `__SCENE__` | `usd2roi_nvpcb.yaml` | Absolute path to the board USD (in-repo: `assets/spark_lighting.usd`). |
| `__REAL_IMAGE__` | `usd2roi_nvpcb.yaml` | Absolute path to the real PCB photo, e.g. `assets/input_real_image/115_2819_000.jpg`. |
| `__OUTPUT__` | `usd2roi_nvpcb.yaml` | Absolute path to the per-run output dir. |

## How to run (3 stages)

Identical to `0603_H100_day1`, just swap board name everywhere:

```bash
export PCB_USD_PATH=<path/to/your/board.usd>
export PAIDF_SIM_ROOT=<path/to/paidf-simulation>
IMG=nvcr.io/nv-metropolis-dev/metropolis-sdg/paidf-simulation:<TAG>

REAL=$PAIDF_SIM_ROOT/assets/input_real_image/115_2819_000.jpg
OUT=$PAIDF_SIM_ROOT/sdg_test_output/usd_roi_115_2819_000_day1
RUN_DIR=$PAIDF_SIM_ROOT/configs/runs/usd_roi_115_2819_000_day1
SRC=$PAIDF_SIM_ROOT/configs/usd_roi_examples/115_2819_000_day1

mkdir -p "$RUN_DIR" "$OUT" && chmod -R o+w "$OUT"

sed -e "s|__SCENE__|$PCB_USD_PATH|" \
    -e "s|__REAL_IMAGE__|$REAL|" \
    -e "s|__OUTPUT__|$OUT|" \
    "$SRC/usd2roi_nvpcb.yaml" > "$RUN_DIR/usd2roi_nvpcb.yaml"

YAML_IN=/workspace/paidf-simulation/configs/runs/usd_roi_115_2819_000_day1/usd2roi_nvpcb.yaml

MOUNT="-v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
       -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
       -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
       -v $PAIDF_SIM_ROOT/sdg_test_output:$PAIDF_SIM_ROOT/sdg_test_output"
ENVS="-e ACCEPT_EULA=Y -e PYTHONUNBUFFERED=1 -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PAIDF_SIM_ROOT"

# Stage 1 — render
docker run --gpus all --rm --network host $ENVS $MOUNT $IMG \
  "scripts/usd2roi/usd2roi_render.py --config $YAML_IN"

# Stage 2 — register
docker run --gpus all --rm --network host $ENVS $MOUNT --entrypoint python3 $IMG \
  scripts/usd2roi/usd2roi_register.py --config $YAML_IN

# Stage 3 — crop
docker run --rm $ENVS $MOUNT --entrypoint python3 $IMG \
  scripts/usd2roi/usd2roi_crop.py --config $YAML_IN
```

## Expected output

Same three-subdir layout as `0603_H100_day1`. The crop subdirectory is named after the IC target class:

```
sdg_test_output/usd_roi_115_2819_000_day1/
├── stage1_render.log
├── stage2_register.log
├── stage3_crop.log
├── sdg/                                # Stage 1
│   ├── rgb_0000.png … rgb_0001.png     # 2 captures
│   ├── semantic_segmentation_000{0,1}.png
│   ├── semantic_segmentation_labels_000{0,1}.json
│   ├── semantic_stats.json
│   └── metadata.txt
├── aligned/                            # Stage 2 (registration)
│   ├── ref_crop.png  aligned_crop.png  blink.gif  params.json
│   ├── semantic_segmentation_0000.png  semantic_segmentation_labels_0000.json
│   ├── sdg_crop_stats.json  metadata.txt
└── crop/                               # Stage 3
    └── IC/
        ├── cad_mask/
        └── normal_img/
```

`semantic_stats.json` shows 4 rules (vs. 8 on the 0603 board): `pad` + 2× `ic` + `solder`, one prim per rule.

## Sanity checks

- All three stage logs end with their own success line.
- `crop/IC/{cad_mask,normal_img}/` have matching file counts.
