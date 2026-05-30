# Physical AI Data Factory - Simulation

Synthetic data generation for physical-AI use cases. Includes Defect Image Generation (DIG) on printed circuit boards.

Container: `${SDG_IMAGE}`

---

## Prerequisites

### Clone tested repo

```bash
cd ~
git clone https://github.com/NVIDIA/paidf-simulation.git
cd paidf-simulation
git checkout main
```

### Download USD assets

Obtain the USD asset bundle from your project channel. Extract anywhere convenient, then point `PCB_USD_PATH` at the main scene USD:

```bash
unzip <usd-assets>.zip -d <path/to/usd-assets>
export PCB_USD_PATH=<path/to/usd-assets>/spark_lighting.usd
```

The flow run commands below mount `$(dirname $PCB_USD_PATH)` read-only into the container; USDs don't need to live inside the repo.

### Pull image

```bash
IMAGE=${SDG_IMAGE}
docker login nvcr.io
docker pull $IMAGE
```

---

## Run the pipelines

### Flow 1: Good Image Pipeline

#### Prepare the configuration

**Config Path**: `configs/flow1_good_image/good_image.yaml`

| Key | Description |
|---|---|
| `scan_grid.x_num`, `y_num` | Number of cells along each axis. Default 10 by 10 produces 100 frames per trigger |
| `resolution` | Render resolution `[W, H]`. Default `[1920, 1080]` |
| `pathtracing.total_spp` | Accumulated samples per pixel. Higher values produce cleaner images and longer render times. Default: 32 |
| `lighting.ring_light` | `true`: per-layer RGB ring light (soldering light). `false`: white light only |
| `writer.{rgb, bounding_box_2d_tight, semantic_segmentation, ...}` | Per-annotator on/off switches |
| `rename_to_grid_index` | Default `false` keeps `_NNNN.png` naming. Do not change for standard SDG flows |

**Config Path**: `configs/pcba_target.yaml`

| Key | Description |
|---|---|
| `component_types` | Component scope names under `pcba_root`. Only listed scopes receive semantic labels. The good-image pipeline does not use this file |

#### Run the pipeline

```bash
# Flow 1 — Good Image Pipeline
PCB=~/paidf-simulation
IMAGE=${SDG_IMAGE}
OUTPUT=$PCB/sdg_test_output/flow1_good_image

# Pre-flight
mkdir -p $OUTPUT && chmod 777 $OUTPUT
ls /usr/share/nvidia/nvoptix.bin

# Run (Stage 1: Kit render)
docker run --rm --gpus all --network host \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PCB \
  -v $PCB:/workspace/paidf-simulation \
  $IMAGE \
  "scripts/sdg/standalone/sdg_pipeline.py \
    --config configs/flow1_good_image/good_image.yaml \
    --pcba-config configs/pcba_target.yaml"
```

#### Verify the output

```bash
$ ls $OUTPUT/trigger_0000/ | sed 's/.*\.//' | sort | uniq -c
    301 json    # 100 bbox_labels + 100 bbox_prim_paths + 100 semseg_labels + metadata.json
    100 npy     # bbox_2d_tight arrays
    200 png     # 100 rgb + 100 semantic_segmentation
      1 txt     # metadata.txt

$ ls $OUTPUT/trigger_0000/rgb_*.png | head -3
rgb_0000.png  rgb_0001.png  rgb_0002.png    # _NNNN naming, no _x*_y*

$ cat $OUTPUT/trigger_0000/semantic_segmentation_labels_0000.json
{"(0, 0, 0, 0)":   {"class": "BACKGROUND"},
 "(0, 0, 0, 255)": {"class": "UNLABELLED"},
 "(33, 243, 3, 255)": {"class": "capacitor"}}
# RGBA values may change; assignment is randomized per run.
```

---

### Flow 2: Defect Image Pipeline (Missing, Shift, Sideflip, Tombstone)

#### Prepare the configuration

**Config Path**: `configs/flow2_defect_image/defect_image.yaml`

| Key | Description |
|---|---|
| `defects.shift.{enabled, ratio, translate_range, rotate_z_range}` | XY translation and Z-axis rotation defects |
| `defects.tombstone.{enabled, ratio, angle_min, angle_max}` | Tilt around Y axis (tombstone) |
| `defects.sideflip.{enabled, ratio, angle_min, angle_max}` | Flip around X axis |
| `writer.semantic_types` | Must include `defect` for defect labels to appear in semantic segmentation output |
| `scan_grid.x_num/y_num`, `resolution`, `lighting`, `pathtracing` | Same as Flow 1 |

**Config Path**: `configs/flow2_defect_image/missing_image.yaml`

| Key | Description |
|---|---|
| `missing.ratio` | Fraction of component pool to hide per trigger (0–1) |
| `writer.reference.{rgb, semantic_segmentation, ...}` | Pass 1: all components visible. Segmentation labels mark hidden components with `defect=missing` |
| `writer.defective.rgb` | Pass 2: selected components hidden; RGB output only |

#### Run the pipeline

```bash
# Flow 2 — Defect Image Pipeline
PCB=~/paidf-simulation
IMAGE=${SDG_IMAGE}

# Pre-flight
mkdir -p $PCB/sdg_test_output/flow2_defect_image \
         $PCB/sdg_test_output/flow2_missing_image
chmod 777 $PCB/sdg_test_output/flow2_defect_image \
          $PCB/sdg_test_output/flow2_missing_image

# Run (a) pose defects: shift / tombstone / sideflip
docker run --rm --gpus all --network host \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PCB \
  -v $PCB:/workspace/paidf-simulation \
  $IMAGE \
  "scripts/sdg/standalone/sdg_pipeline.py \
   --config configs/flow2_defect_image/defect_image.yaml \
   --pcba-config configs/pcba_target.yaml"

# Run (b) missing components
docker run --rm --gpus all --network host \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PCB \
  -v $PCB:/workspace/paidf-simulation \
  $IMAGE \
  "scripts/sdg/standalone/sdg_pipeline.py \
   --config configs/flow2_defect_image/missing_image.yaml \
   --pcba-config configs/pcba_target.yaml"
```

#### Verify the output

**(a) Pose defects:** output under `flow2_defect_image/trigger_0000/`

```bash
$ ls $OUTPUT/flow2_defect_image/trigger_0000/ | sed 's/.*\.//' | sort | uniq -c
    301 json
    100 npy
    200 png
      1 txt
# 100 of each: rgb / semseg / semseg_labels / bbox_npy / bbox_labels / bbox_prim_paths

$ cat $OUTPUT/flow2_defect_image/trigger_0000/semantic_segmentation_labels_0000.json
# defect classes appear (cells without that defect won't show all three keys):
{"(0, 0, 0, 0)":   {"class": "BACKGROUND"},
 "(0, 0, 0, 255)": {"class": "UNLABELLED"},
 "(33, 243, 3, 255)":  {"defect": "sideflip"},
 "(240, 4, 111, 255)": {"defect": "shift"},
 "(27, 186, 239, 255)":{"defect": "tombstone"}}
```

**(b) Missing components:** output under `flow2_missing_image/trigger_0000/`

```bash
$ ls $OUTPUT/flow2_missing_image/trigger_0000/reference/ | sed 's/.*\.//' | sort | uniq -c
    300 json    # 100 bbox_labels + 100 bbox_prim_paths + 100 semseg_labels
    100 npy     # bbox arrays
    100 png     # 100 colorized semantic_segmentation (rgb off in reference)
      1 txt

$ ls $OUTPUT/flow2_missing_image/trigger_0000/defective/ | sed 's/.*\.//' | sort | uniq -c
    100 png     # rgb only
      1 txt

$ cat $OUTPUT/flow2_missing_image/trigger_0000/reference/semantic_segmentation_labels_0000.json
# Hidden components labeled with defect=missing
{"(0, 0, 0, 0)":   {"class": "BACKGROUND"},
 "(0, 0, 0, 255)": {"class": "UNLABELLED"},
 "(33, 243, 3, 255)": {"defect": "missing"}}
```

Between Pass 1 and Pass 2, the pipeline log includes `[Pipeline] Hiding N components`. For example, with `missing.ratio: 0.5`, about 1138 of ~2276 components in the pool may be hidden.

---

### Flow 3: Good and Defect Pairs Dataset

Paired golden / defect data for ChangeNet-style training is handled by
the simulation skill's **paired sub-mode** of the single-flow track.
See [`.agents/skills/simulation/SKILL.md`](.agents/skills/simulation/SKILL.md)
for routing detail; the implementation runs the `good` flow and the
`defect` flow with the same `random_seed` and post-processes via
`scripts/postprocess/build_pair_dataset.py`.

---

### Flow 4: USD2ROI Day-1

#### Prepare the configuration

**Config Path**: `configs/cad2roi/day1/replicator/usd2roi_target.yaml`

| Key | Description |
|---|---|
| `scene` | CAD-derived USD path relative to repo root |
| `real_image` | Real PCB photo path relative to repo root |
| `semantics: [{match, labels}, ...]` | Prim-path glob to label rules (author per board) |
| `camera.translate` | `[x, y]` ortho camera center in mm (z fixed at 5000) |
| `resolution` | Match the real photo aspect ratio so post-MI scale factors `sX` and `sY` are close to 1.0 |
| `registration.sx_range / sy_range / rot_range_deg / shift_range` | MI search ranges. Tighten if you have priors |
| `registration.min_mi` | Stage 2 exits with code 2 if `mi_after` < this. Default 0.5 |
| `crop.classes` | Class labels to extract ROIs for, e.g. `[capacitor, solder, pad, ic]` |
| `crop.bridge / bridge_dis / bridge_classes` | Enable bridge crops, pixel distance threshold, and class pairs to bridge |
| `output.dir` | Container-absolute path, e.g. `/workspace/paidf-simulation/sdg_test_output/flow4_day1_rois` |

#### Prepare the real-image input

Download the sample image (screenshot from an AOI machine): <https://drive.google.com/file/d/18rzCtpPgn7paNGv8AtN-xu9ZEcEKyk5c/view?usp=share_link>

```bash
cd ~/paidf-simulation
mv ~/Downloads/real.png ./scripts/usd2roi/input
```

#### Run the scripts

**Step 1: Pre-flight**

```bash
PCB=~/paidf-simulation
IMAGE=${SDG_IMAGE}
OUTPUT=$PCB/sdg_test_output/flow4_day1_rois
YAML=configs/cad2roi/day1/replicator/usd2roi_target.yaml

mkdir -p $OUTPUT && chmod 777 $OUTPUT
ls /usr/share/nvidia/nvoptix.bin
```

**Step 2: Stage 1 render**

```bash
docker run --rm --gpus all --network host \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PCB \
  -v $PCB:/workspace/paidf-simulation \
  $IMAGE \
  "scripts/usd2roi/usd2roi_render.py --config $YAML"
```

**Step 3: Stage 2 register**

```bash
docker run --rm --gpus all --network host \
  -v $PCB:/workspace/paidf-simulation \
  --entrypoint python3 \
  $IMAGE \
  scripts/usd2roi/usd2roi_register.py --config $YAML
```

**Step 4: Stage 3 crop**

```bash
docker run --rm \
  -v $PCB:/workspace/paidf-simulation \
  --entrypoint python3 \
  $IMAGE \
  scripts/usd2roi/usd2roi_crop.py --config $YAML
```

#### Verify the output

```bash
$ ls $OUTPUT/
sdg/  aligned/  crop/

$ ls $OUTPUT/sdg/
rgb_0000.png  semantic_segmentation_0000.png  semantic_segmentation_labels_0000.json
metadata.txt  semantic_stats.json

$ python3 -m json.tool $OUTPUT/aligned/params.json
{
    "scaleX":       n,
    "scaleY":       n,
    "rotation_deg": n,
    "tx":           n,
    "ty":           n,
    "mi_before":    n,
    "mi_after":     n
}
# n values change per run.
# Open $OUTPUT/aligned/blink.gif to visually QA ref ↔ aligned alternation.

$ ls $OUTPUT/aligned/
ref_crop.png  aligned_crop.png  blink.gif  params.json
semantic_segmentation_0000.png  semantic_segmentation_labels_0000.json
sdg_crop_stats.json  metadata.txt

$ echo "ROIs:    $(ls $OUTPUT/crop/component/normal_img/*.png | wc -l)"
$ echo "Bridges: $(ls $OUTPUT/crop/bridge/normal_img/*.png  | wc -l)"
ROIs:    24
Bridges: 2
```

---

### Flow 5: USD2ROI Day-0

#### Prepare the configuration

**Config Path**: `configs/cad2roi/day0/sdg/day0_image.yaml`

#### Run the scripts

**Step 1: Pre-flight**

```bash
PCB=~/paidf-simulation
IMAGE=${SDG_IMAGE}
OUTPUT=$PCB/sdg_test_output/flow5_day0_rois

mkdir -p $OUTPUT && chmod 777 $OUTPUT
ls /usr/share/nvidia/nvoptix.bin
```

**Step 2: Stage 1 render**

```bash
docker run --rm --gpus all --network host \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PCB \
  -v $PCB:/workspace/paidf-simulation \
  $IMAGE \
  "scripts/sdg/standalone/sdg_pipeline.py \
    --config configs/cad2roi/day0/sdg/day0_image.yaml \
    --pcba-config configs/pcba_target.yaml"
```

**Step 3: Set permissions between stages**

```bash
docker run --rm \
  -v $PCB:/workspace/paidf-simulation \
  --entrypoint chmod \
  $IMAGE 777 /workspace/paidf-simulation/sdg_test_output/flow5_day0_rois
```

**Step 4: Anchor crop (Stage 2)**

```bash
docker run --rm \
  -v $PCB:/workspace/paidf-simulation \
  --entrypoint python3 \
  $IMAGE \
  scripts/usd2roi/usd2roi_crop.py --config configs/cad2roi/day0/usd2roi/day0_crop.yaml
```

#### Verify the output

```bash
$ ls $OUTPUT/
trigger_0000/  crop/

# Stage 1 — labelled scan_grid render (rename_to_grid_index: true)
$ ls $OUTPUT/trigger_0000/ | head -3
rgb_x0_y0.png  rgb_x0_y1.png  rgb_x0_y2.png   # _x*_y* spatial naming

$ ls $OUTPUT/trigger_0000/ | sed 's/.*\.//' | sort | uniq -c
    101 json    # 100 semseg_labels + 1 metadata.json
    200 png     # 100 rgb_x*_y* + 100 semantic_segmentation_x*_y*
      1 txt     # metadata.txt

# Stage 2 — multi-cell anchor crop
$ ls $OUTPUT/crop/component/ | head -5
x0_y0  x0_y1  x0_y2  x0_y3  x0_y4   # 100 cell directories total

$ echo "Total ROIs: $(find $OUTPUT/crop/component -path '*/normal_img/*.png' | wc -l)"
Total ROIs: 2150

$ ls $OUTPUT/crop/component/x0_y0/
normal_img/  cad_mask/  semantic_segmentation_labels.json

$ cat $OUTPUT/crop/component/x0_y0/semantic_segmentation_labels.json
{"(33, 243, 3, 255)":  {"class": "pad"},
 "(240, 4, 111, 255)": {"class": "capacitor"},
 "(27, 186, 239, 255)":{"class": "solder"}}
# RGBA values may change per run.
```

---

## Wall-clock timings (one L40 GPU)

Approximate run times on a single NVIDIA L40 GPU:

| Flow | scan_grid / capture | Time |
|---|---|---|
| 1: Good | 10×10 = 100 frames | ~12 min |
| 2a: Defect (pose) | 10×10 = 100 frames | ~19 min |
| 2b: Missing | 10×10 × 2 pass | ~18 min |
| 3: Pairs, Mode A (sequential) | 10×10 + post-process | ~24 min |
| 4: Day-1 ROIs | 1 frame + register + crop | ~6 min |
| 5: Day-0 ROIs | 10×10 + crop | ~10 min |

---

## Contributing

External contributions are welcome. All commits must be signed off under
the Developer Certificate of Origin (DCO); see
[CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Source code in this repository is licensed under the Apache License,
Version 2.0; see [LICENSE](LICENSE). Third-party runtime dependencies and
their licenses are documented in [third_party/](third_party/) (see
[`third_party/licenses.txt`](third_party/licenses.txt) for the auto-generated
inventory and [`third_party/README.md`](third_party/README.md) for notes
on specific entries).
