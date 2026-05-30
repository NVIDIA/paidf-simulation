# cad2roi / usd2roi pipeline

USD render → mutual-information registration → label-based ROI crop.

Goal: take a CAD-derived USD scene and a real PCB photo, align them, and emit
per-ROI triples ``(synth crop, real crop, semantic seg crop)`` plus optional
pairwise "bridge" crops covering nearby ROIs together.

---

## Pipeline (3 stages, 1 YAML config)

```
[scene.usd] ── usd2roi_render.py ── output/sdg/        (Kit needed: Isaac-Sim with bundled Replicator)
                       ↓                               rgb_0000.png + seg_*.png + labels.json
[real.png]  ── usd2roi_register.py ─→ output/aligned/  (host python; uses cupy if available)
                       ↓                               ref_crop.png + aligned_crop.png + seg_cropped + params.json
                  usd2roi_crop.py ── output/components/ (host python)
                                                       rois/roi_NNNN_<class>/...
                                                       bridges/bridge_NNNN_<a>-<b>/...
```

All three scripts read the **same** YAML — section by section:

| Stage | Reads from yaml | Writes to disk |
|---|---|---|
| `usd2roi_render.py`   | `scene`, `camera`, `resolution`, `semantics`, `writer`, `output` | `<output.dir>/sdg/` |
| `usd2roi_register.py` | `real_image`, `registration`, `output`, (optional) `crop.xform_depth` | `<output.dir>/aligned/` |
| `usd2roi_crop.py`     | `crop`, `output` | `<output.dir>/components/` |

No `--skip-*` flags. Re-run any stage independently.

---

## Files

### Entry points

| File | Runtime | Purpose |
|---|---|---|
| `usd2roi_render.py`   | Kit App (`isaacsim.exp.base.kit --exec`) | Render synth + write seg / bbox / metadata |
| `usd2roi_register.py` | Host python | MI registration + crop SDG annotations to overlap bbox |
| `usd2roi_crop.py`     | Host python | Per-ROI label-based crop + optional bridges |

### Library modules (importable)

| File | What it does |
|---|---|
| `registration.py`     | MI image pyramid + grid + coordinate descent (CPU / cupy) |
| `sdg_crop.py`         | Crop Replicator annotations (rgb / semseg / bbox) to a given bbox |
| `semantic_rules.py`   | glob → regex; uninstance proxy ancestors; CLI dry-run |
| `component_crop.py`   | `crop_rois_by_label` + `crop_bridge_pairs` |

### CLI helpers

| File | Purpose |
|---|---|
| `semantic_rules.py --scene X --rules Y` | Dry-run rules against a USD without launching Kit (only needs `usd-core` + `pyyaml`) |

---

## Run

### 1. Render (Kit)

```bash
/isaac-sim/kit/kit /isaac-sim/apps/isaacsim.exp.base.kit --no-window --exec \
  "scripts/usd2roi/usd2roi_render.py --config <path/to/usd2roi_target.yaml>"
```

Produces `<output.dir>/sdg/{rgb_0000.png, semantic_segmentation_*, semantic_stats.json, ...}`.

### 2. Register (host python)

```bash
python3 scripts/usd2roi/usd2roi_register.py --config <path/to/yaml>
```

Reads `<output>/sdg/rgb_0000.png` + `yaml.real_image`, runs MI registration,
crops both into the valid-overlap bbox; also crops the seg + bbox annotators
to the same bbox. Writes `<output.dir>/aligned/`.

### 3. Crop (host python)

```bash
python3 scripts/usd2roi/usd2roi_crop.py --config <path/to/yaml>
```

Connected components on the post-aligned seg → per-ROI triples. Optional
bridge crops for ROI pairs within `bridge_dis` px. Writes
`<output.dir>/components/{rois, bridges}/`.

### Dry-run semantic rules without Kit

Useful when iterating on `semantics:` glob patterns:
```bash
python3 scripts/usd2roi/semantic_rules.py \
  --scene <scene.usd> --rules <yaml> --show 10
```

---

## YAML config (full schema)

```yaml
# === Scene ===
scene: /path/to/pcba.usd

# === Step 1: Semantic rules (applied at render time) ===
# Glob `**` = any chars including '/', `*` = within one path segment.
# Top-down; same key in later rule overwrites; different keys merge.
semantics:
  - match: "**/_0805U_H150_*/LibRef/.../surface_0"
    labels: {class: solder}
  - match: "**/_0805U_H150_*"
    labels: {class: whole_component}
  # multi-class on same prim: write two rules to the same path
  - match: "**/IC_/tn__1151581000_2*"
    labels: {class: ic}
  - match: "**/IC_/tn__1151581000_2*"
    labels: {class: whole_component}

# === Step 2: Camera (orthographic, generated at /World/cad2roi_camera) ===
# Fixed by spec: focal_length=50, z=5000, rotation_xyz=(0,0,90) deg.
# `horizontal_aperture` is in 0.1 × world units (USD/Replicator convention).
camera:
  translate: [-55, 1.5]                # [x, y] world coords (mm); z fixed
  horizontal_aperture: 130             # ortho extent ÷10 = mm captured (here 13mm wide)
resolution: [640, 480]

# === Step 3: Render (writer + render settings) ===
writer:
  rgb: true
  semantic_segmentation: true
  colorize_semantic_segmentation: true
  bounding_box_2d_tight: false         # the new crop logic doesn't need this
  semantic_types: [class]
  frame_padding: 4
  image_output_format: png

# === Step 4: Registration (real → synth alignment) ===
real_image: /path/to/real.png
registration:
  sx_range: [0.95, 1.05, 0.01]         # [min, max, step]
  sy_range: [0.95, 1.05, 0.01]
  rot_range_deg: [-0.1, 0.1, 0.05]     # near-zero rotation typical
  shift_range: 100                     # ± px for tx/ty
  shift_step: 1
  pyr_levels: 3
  bins: 64
  no_resize: true
  gpu: auto                            # auto / true / false
  min_mi: null                         # exit 2 if MI after < this; null disables

# === Step 5: Per-ROI label-based crop ===
crop:
  classes: [capacitor, solder, pad, ic]   # which classes to extract
  morph_kernel: 2                          # px close radius (merge adjacent labelled meshes)
  min_area: 50                             # px² filter for noise
  max_area: null                           # px² filter for huge merges (null = no cap)
  offset: 10                               # pixel padding around each ROI bbox

  # Optional pairwise crops covering two nearby ROIs in one bbox.
  bridge: false
  bridge_dis: 30                           # px; bbox edge-to-edge distance threshold
  bridge_classes: [solder, pad]            # only ROIs with dominant_class in this list pair

# === Output ===
output:
  dir: /path/to/run_001
```

---

## Output layout

```
<output.dir>/
├── sdg/                                       (from usd2roi_render.py)
│   ├── rgb_0000.png
│   ├── semantic_segmentation_0000.png
│   ├── semantic_segmentation_labels_0000.json
│   ├── metadata.txt
│   └── semantic_stats.json
│
├── aligned/                                   (from usd2roi_register.py)
│   ├── ref_crop.png                           # synth, cropped to MI valid bbox
│   ├── aligned_crop.png                       # real warped + cropped to same bbox
│   ├── semantic_segmentation_0000.png         # seg cropped, crop-local coords
│   ├── semantic_segmentation_labels_0000.json
│   ├── params.json                            # {scaleX, scaleY, rotation_deg, tx, ty, mi_before, mi_after}
│   ├── blink.gif                              # 2-frame ref↔aligned blink for visual QA
│   └── sdg_crop_stats.json
│
└── components/                                (from usd2roi_crop.py)
    ├── semantic_segmentation_labels.json      # color → class map (shared)
    ├── rois/
    │   └── roi_NNNN_<dominant_class>/
    │       ├── ov_rgb.png                     # synth crop (RGBA)
    │       ├── ov_seg.png                     # seg crop (RGBA)
    │       ├── real_rgb.png                   # real crop (RGBA)
    │       └── bbox.json                      # {bbox_local, area_pixels, dominant_class, pixel_count_per_class}
    └── bridges/                               (only if crop.bridge: true)
        └── bridge_NNNN_<classA>-<classB>/
            ├── ov_rgb.png
            ├── ov_seg.png
            ├── real_rgb.png
            └── bbox.json                      # {pair: [roi_A, roi_B], distance_px, ...}
```

---

## Algorithm notes

### Semantic application (Stage 1)

* Rules are matched against the full prim path with a glob → regex translation
  in `semantic_rules.glob_to_regex`. Each match adds `(class, value)` tuples
  to that prim, applied via `rep.modify.semantics(...)` inside a Replicator
  graph (no `with rep.new_layer()` — it's not needed for modifying existing
  prims and was actually causing RGB drops on some configurations).
* USD instances need special handling: `stage.Traverse()` doesn't enter
  instances by default. We use `Usd.PrimRange(..., TraverseInstanceProxies(...))`
  to find them, and call `SetInstanceable(False)` on the topmost ancestor of
  any matched instance proxy so the descendants become writable. This expands
  the prototype in place and costs USD instancing optimization — fine for one
  capture; can be expensive when thousands of instances are involved.
* Render warmup: 5 `step_async` calls before `writer.attach` so all annotator
  buffers (RGB / LdrColor in particular) are initialized. Without this, RGB is
  dropped intermittently while seg still writes — same buffer initialization
  race seen in the SDG defect pipeline.

### Registration (Stage 2)

* 5-DOF affine: `(scaleX, scaleY, rotation, tx, ty)`. Real photo is the test;
  synth render is the reference. The output `warped` image is the real photo
  warped into the synth pixel grid.
* Search algorithm: image pyramid (coarse → fine) → grid search at coarsest
  level → coordinate descent at every level. Implemented in `registration.py`.
* `crop_to_valid_bbox` then crops both ref + warped to the bounding box of
  pixels where the warp is fully defined (the mask), so the two crops are
  exactly the same size and pixels-aligned.
* `sdg_crop` translates SDG annotations (semseg / bbox) into the same
  crop-local coordinates so downstream crops can reference them.

### Per-ROI crop (Stage 3)

* For each color in the seg labels JSON whose `class` field contains any of
  `crop.classes`, mark its pixels in a binary mask.
* Apply ``cv2.MORPH_CLOSE`` with kernel radius
  ``morph_kernel`` to merge labelled
  meshes that are physically close (e.g. capacitor body + its solder pads sit
  a few px apart but should be the same ROI).
* ``cv2.connectedComponents`` → one ROI per blob. Filter by ``min_area`` /
  `max_area`, compute bbox + `offset` padding, crop the three images, save.
* Folder name carries the dominant class (by pixel-count vote) for quick
  triage: `roi_0001_capacitor`, `roi_0007_solder`, etc.

### Bridge crops (optional, Stage 3)

* Pair every two ROIs whose dominant_class is in `bridge_classes`.
* Pixel distance = bbox edge-to-edge Euclidean (0 if rectangles touch / overlap).
* Pairs with distance ≤ `bridge_dis` get a union-bbox crop. Useful for
  inspecting "are these two pads bridged?".

---

## Dependencies

### Stage 1 (Kit App)

* Isaac-Sim standalone install (bundles the required Kit App + Replicator extensions)
* Run via the Kit's bundled python; no pip install needed at this stage
* Required Replicator version: ``omni.replicator.core >= 1.13.3``

### Stage 2 + 3 (host python)

```bash
pip install numpy opencv-python-headless pyyaml pillow
# optional GPU registration via cupy:
pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
```

The `semantic_rules.py` dry-run CLI additionally needs:
```bash
pip install usd-core
```

---

## Common knobs

| Symptom | Knob |
|---|---|
| RGB occasionally missing in `sdg/` | (already mitigated — 5 warmup steps + retry) |
| Registration too slow | drop `pyr_levels` to 2; widen `shift_step`; narrow `sx_range` / `sy_range` |
| Real & synth scales very different | match `resolution` to real photo dims (so sX, sY ~1.0 with `no_resize: true`) |
| Synth FOV doesn't match real | adjust `camera.translate` and `camera.horizontal_aperture` (USD aperture is in 0.1 × world units, so `130` ⇒ 13mm at metersPerUnit=0.001) |
| Too few ROIs | lower `crop.min_area`; expand `crop.classes`; increase `crop.morph_kernel` to merge more |
| ROIs too big (board background mistakenly merged) | shrink `crop.morph_kernel`; set `crop.max_area` |
| Bridge mode emits too many pairs | lower `crop.bridge_dis`; tighten `crop.bridge_classes` |

---

## Status

* All three stages working end-to-end
* GPU registration: ~14s on L40s for 640×480 with `pyr_levels: 3, shift_step: 1`
* MI typical: 0.55–0.65 (above docs' "same modality" range of 0.3–0.6)
