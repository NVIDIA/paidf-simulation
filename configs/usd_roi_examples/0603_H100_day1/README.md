# `usd_roi_examples/0603_H100_day1/`

Day-1 CAD-to-ROI for the Spark BASE A04 board, anchored on `_0603_H100` capacitors. Real photo supplied → registered synth↔real ROI triples.

## Contents

- [Files](#files)
- [Tokens to patch](#tokens-to-patch)
- [How to run (3 stages)](#how-to-run-3-stages)
- [Expected output](#expected-output)
- [Sanity checks](#sanity-checks)

## Files

| File | Purpose |
|---|---|
| `usd2roi_nvpcb.yaml` | Single config used by all three day-1 tools (render → register → crop). Holds `scene`, `semantics`, `camera`, `resolution`, `writer`, `real_image`, `registration`, `crop`, `output`. |

## Tokens to patch

| Token | In file | Replace with |
|---|---|---|
| `__SCENE__` | `usd2roi_nvpcb.yaml` | Absolute path to the board USD (in-repo: `assets/spark_lighting.usd`). |
| `__REAL_IMAGE__` | `usd2roi_nvpcb.yaml` | Absolute path to the real PCB photo, e.g. `assets/input_real_image/0603_H100.jpg`. |
| `__OUTPUT__` | `usd2roi_nvpcb.yaml` | Absolute path to the per-run output dir. |

## How to run (3 stages)

The same `--config` is fed to three different scripts. Stage 1 boots Kit (default entrypoint); stages 2 and 3 override the entrypoint to plain `python3`.

```bash
export PCB_USD_PATH=<path/to/your/board.usd>
export PAIDF_SIM_ROOT=<path/to/paidf-simulation>
IMG=nvcr.io/nv-metropolis-dev/metropolis-sdg/paidf-simulation:<TAG>

REAL=$PAIDF_SIM_ROOT/assets/input_real_image/0603_H100.jpg
OUT=$PAIDF_SIM_ROOT/sdg_test_output/usd_roi_0603_H100_day1
RUN_DIR=$PAIDF_SIM_ROOT/configs/runs/usd_roi_0603_H100_day1
SRC=$PAIDF_SIM_ROOT/configs/usd_roi_examples/0603_H100_day1

mkdir -p "$RUN_DIR" "$OUT" && chmod -R o+w "$OUT"

sed -e "s|__SCENE__|$PCB_USD_PATH|" \
    -e "s|__REAL_IMAGE__|$REAL|" \
    -e "s|__OUTPUT__|$OUT|" \
    "$SRC/usd2roi_nvpcb.yaml" > "$RUN_DIR/usd2roi_nvpcb.yaml"

YAML_IN=/workspace/paidf-simulation/configs/runs/usd_roi_0603_H100_day1/usd2roi_nvpcb.yaml

MOUNT="-v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
       -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
       -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
       -v $PAIDF_SIM_ROOT/sdg_test_output:$PAIDF_SIM_ROOT/sdg_test_output"
ENVS="-e ACCEPT_EULA=Y -e PYTHONUNBUFFERED=1 -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PAIDF_SIM_ROOT"

# Stage 1 — render (Kit, ~5-7 min cold boot)
docker run --gpus all --rm --network host $ENVS $MOUNT $IMG \
  "scripts/usd2roi/usd2roi_render.py --config $YAML_IN"

# Stage 2 — register (host python+cupy on GPU; ~15-30 s)
docker run --gpus all --rm --network host $ENVS $MOUNT --entrypoint python3 $IMG \
  scripts/usd2roi/usd2roi_register.py --config $YAML_IN

# Stage 3 — crop (host python; seconds)
docker run --rm $ENVS $MOUNT --entrypoint python3 $IMG \
  scripts/usd2roi/usd2roi_crop.py --config $YAML_IN
```

Re-run any stage independently — there are no `--skip-*` flags; each stage overwrites its own output dir.

## Expected output

Three subdirectories under `$OUT/`, one per stage, plus a per-stage log:

```
sdg_test_output/usd_roi_0603_H100_day1/
├── stage1_render.log
├── stage2_register.log
├── stage3_crop.log
├── sdg/                                # Stage 1
│   ├── rgb_0000.png                    # 1 ortho capture
│   ├── semantic_segmentation_0000.png
│   ├── semantic_segmentation_labels_0000.json
│   ├── semantic_stats.json             # how many prims per semantic rule
│   └── metadata.txt
├── aligned/                            # Stage 2 (registration)
│   ├── ref_crop.png                    # real photo crop (registration target)
│   ├── aligned_crop.png                # synth warped onto real
│   ├── blink.gif                       # ref ↔ aligned animation for visual check
│   ├── params.json                     # registration transform parameters
│   ├── semantic_segmentation_0000.png
│   ├── semantic_segmentation_labels_0000.json
│   ├── sdg_crop_stats.json
│   └── metadata.txt
└── crop/                               # Stage 3 (per-class ROI crops)
    └── passive_component/
        ├── cad_mask/                   # crop's CAD-derived mask
        └── normal_img/                 # crop's RGB
```

`semantic_stats.json` dumps how many prims matched each `semantics:` rule (one block per rule, with sample paths). Each of the 8 rules in this example matches exactly one prim:

```json
{
  "n_rules": 8,
  "groups": [
    {"labels": {"class": "pad"},       "n_prims": 1, "sample_paths": [...]},
    {"labels": {"class": "solder"},    "n_prims": 1, "sample_paths": [...]},
    {"labels": {"class": "capacitor"}, "n_prims": 1, "sample_paths": [...]},
    ...
  ]
}
```

## Sanity checks

- All three stage logs end with their own success line. Stage 1: `[Pipeline] All done!` Stage 3: `[usd2roi_crop] emitted=N, skipped_*=…`.
- `aligned/blink.gif` opens in any animated-GIF viewer and lets you eyeball whether the synth is on top of the real photo.
- `crop/passive_component/` has matching counts in `cad_mask/` and `normal_img/`.