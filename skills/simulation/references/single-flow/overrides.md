# Overrides — per-flow knob cheatsheet

All overrides are applied via `deep_set(cfg, "a.b.c", value)` on the base YAML loaded from `configs/<flow>/...`. Boolean/int/list values pass through `yaml.safe_dump` cleanly.

## Common to every flow

| Knob (dotted) | Type | Default | Effect |
|---|---|---|---|
| `max_image_count` | int (-1 = unlimited) | varies | Total RGB frames cap. -1 = unlimited (writer runs every grid cell × num_triggers). |
| `num_triggers` | int | 1 | Number of full grid scans (each trigger re-rolls randomization). |
| `random_seed` | int | 0 | Base seed for lighting / camera / defect / aug NumPy streams. Reproducibility anchor. |
| `seed` | int | 0 | Skip first N scan cells on trigger 0 only; also the first frame index. Don't confuse with random_seed. |
| `resolution` | [int, int] | varies | Render resolution `[width, height]`. |
| `output` | string | `${PAIDF_SIM_ROOT}/sdg_test_output/<flow>` | Per-trigger frames write under `<output>/trigger_NNNN/`. |
| `pathtracing.spp` | int | 1 | Samples per pixel per frame. |
| `pathtracing.total_spp` | int | 32 (good/defect/missing) / 64 (good_fixed) | Total accumulated SPP. Higher = less noise, slower. |
| `pathtracing.max_bounces` | int | 4 | Max path bounces. |
| `writer.rgb` | bool | true | Write `rgb_NNNN.png`. |
| `writer.semantic_segmentation` | bool | true | Write seg + label JSONs. |
| `writer.bounding_box_2d_tight` | bool | true | Write 2D tight bbox `.npy` + label JSONs. |

## `good` flow (scan-grid PCB board renders)

| Knob | Type | Default | Effect |
|---|---|---|---|
| `scan_grid.x_num` | int | 3 | Auto-mode: cells along X (aperture grows to bbox/x_num). |
| `scan_grid.y_num` | int | 3 | Cells along Y. |
| `lighting.ring_light` | bool | false (dome) | true → build per-trigger RGB ring rig; false → dome via `lighting.white_light`. |
| `lighting.white_light.intensity` | [min,max] | [800, 2200] | Dome-calibrated. |
| `use_scene_lights` | bool (top-level) | **true (default)** | true → skip rig build, reuse lights authored in the scene USD (`spark_lighting.usd` etc.). Set false to build rig per `lighting.ring_light`. |
| `preserve_scene_light_color` | bool (top-level) | **true (default)** | with `use_scene_lights: true`, keep authored RGB on scene lights. Set false to whiten the scene lights. |
| `camera_rotation.{x,y,z}_range` | [min,max] | [0,0] / fixed | Per-trigger camera rotation randomization. |

## `good_fixed` flow (single-component close-up)

| Knob | Type | Default | Effect |
|---|---|---|---|
| `samples_per_position` | int | 300 | Frames per camera position (single position by default). |
| `max_image_count` | int | 300 | Match `samples_per_position` for fixed-cam runs. |
| `scan_grid.x_{start,end}` | float | -5.388 | Camera XY (mm). Single cell → start==end. |
| `scan_grid.y_{start,end}` | float | -0.007 | Same. |
| `scan_grid.z` | float | 10 | Camera height. |
| `horizontal_aperture` | float | 22.756 | Ortho zoom. |
| `randomize_rig.{dome_radius,light_radius,intensity}` | [min,max] | ±20% of base | Per-trigger rig randomization. |
| `component_material.{body_color,roughness,metallic}` | (color / float) | vantablack defaults | Per-sample material override on body. |
| `tin_normal_map.*` | object | enabled | Procedural tin perlin normals on solder pads. |

## `defect` flow (pose defects on scan grid)

| Knob | Type | Default | Effect |
|---|---|---|---|
| `scan_grid.x_num` / `y_num` | int | 10 | 10×10=100-cell grid by default. |
| `defects.shift.enabled` | bool | true | XY translate + Z rotate. |
| `defects.shift.ratio` | float | 0.4 | Fraction of eligible parts hit per trigger. |
| `defects.shift.translate_range` | float (mm) | 0.2 | Max +/- XY shift. |
| `defects.shift.rotate_z_range` | float (deg) | 15 | Max +/- Z rotation. |
| `defects.tombstone.enabled` | bool | true | Tilt around Y axis. |
| `defects.tombstone.ratio` | float | 0.3 | Per-trigger fraction. |
| `defects.tombstone.angle_{min,max}` | float (deg) | [70, 90] | Tilt angle range. |
| `defects.sideflip.enabled` | bool | true | Flip around X axis. |
| `defects.sideflip.angle_{min,max}` | float (deg) | [70, 90] | Flip range. |
| `defects.reverse_polarity.enabled` | bool | false (opt-in) | 180° Z rotation on polarity-sensitive parts. |
| `defects.reverse_polarity.ratio` | float | 0.5 | Coin-flip per trigger. |
| `defects.reverse_polarity.component_types` | [str] | `[_032_0831]` | Substring match against scope names. |

## `missing` flow (two-pass: reference + defective)

| Knob | Type | Default | Effect |
|---|---|---|---|
| `scan_grid.x_num` / `y_num` | int | 10 | 10×10=100 cells. Each cell renders twice (reference + defective). |
| `missing.ratio` | float | 0.5 | Fraction of components hidden in the defective pass. |
| `missing.component_types` | [str] | (inherits from pcba_target) | Restrict pool. |
| `writer.reference.*` | object | full annotations | Annotations on the reference pass. |
| `writer.defective.*` | object | full annotations | Annotations on the defective pass. |

## `lighting` flow (variants of `good`)

No extra knobs vs `good`. Each variant file already has the appropriate top-level flags set. To override, edit the lighting block directly.

## `crop` block (skill-consumed, NOT sdg_pipeline.py)

The skill reads the `crop:` block on the derived YAML to decide whether to
emit a second docker invocation of `scripts/postprocess/crop_components.py`
after the render. The block is invisible to `sdg_pipeline.py` (which has no
`crop:` handling) — it lives in the same YAML purely for ergonomics.

| Knob | Type | Canonical default | Effect |
|---|---|---|---|
| `crop.enabled` | bool | `true` for good/defect/missing; `false` for good_fixed/lighting | Master switch. If false, the skill skips the crop step. |
| `crop.mode` | string | omitted (single-pass) for good/defect; `missing` for the missing flow | `missing` tells the skill to use `--input <input_subdir>` + `--reference <reference_subdir>` shape. |
| `crop.types` | [str] | `[rgb, semantic_segmentation, component_instance]` | Passed as `--crops`. Each requested type writes a parallel folder under each label bucket. |
| `crop.offset` | int (px) | `10` | Passed as `--offset`. Padding around each component bbox. |
| `crop.class_filter` | [str] or null | null (no filter) | Passed as `--class-filter`. Only crop components whose class / defect label contains any of these substrings. |
| `crop.xform_depth` | int | `7` | Passed as `--xform-depth`. USD hierarchy depth at which to truncate the prim path to get the component Xform. |
| `crop.input_subdir` | string or null | `defective` for the missing flow; null elsewhere | Subdirectory of `trigger_NNNN/` that holds the RGB source. Only meaningful when `mode: missing`. |
| `crop.reference_subdir` | string or null | `reference` for the missing flow; null elsewhere | Subdirectory that holds bbox / seg / prim_paths. Only meaningful when `mode: missing`. |
| `crop.output_subdir` | string | `cropped` | Folder name (relative to the flow's `output:`) where crops are written. |

### Natural-language overrides

| User phrasing | YAML override | Notes |
|---|---|---|
| `no crop` / `skip crop` / `render only` / `without cropping` | `crop.enabled: false` | Disable the second docker step. |
| `with crop` / `also crop` / `include crops` / `crop too` | `crop.enabled: true` | Force on for flows where canonical default is false (good_fixed / lighting). |
| `crop offset N` / `crop padding N` / `N-px crop margin` | `crop.offset: N` | |
| `only crop X` / `crop only X components` | `crop.class_filter: [X]` | Pass-through to the crop step's class filter; does NOT affect rendering. |
| `crop just rgb` / `rgb crops only` | `crop.types: [rgb]` | Drops semantic and instance crops. |

The approval gate (Stage 5) MUST list the crop step alongside the render
command when `crop.enabled: true`, including the estimated extra duration
(~20 s per 100 frames).

## Component-list keywords (pcba_target.yaml only)

These apply when overriding the `component_types` key (in pcba_target.yaml). Resolved at load time by `sdg_pipeline.py`.

| Value | Result |
|---|---|
| `ALL` | Full 23-component list from `configs/components.yaml` `all:` |
| `0` | Empty list (no defects fire, no semantics labeled) |
| (named subset) | If `configs/components.yaml` `subsets:` has the key, expands to that list |
| (literal list, e.g. `[_0805U_H150, _0402_H060]`) | Used as-is |
