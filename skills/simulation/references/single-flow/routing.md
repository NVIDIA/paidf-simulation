# Routing — natural language → flow (single-flow track)

This table is the single-flow track's flow-selection logic, applied AFTER the Stage-0 router (in `../../SKILL.md`) has classified the request as single-flow. Rules are first-match-wins.

## Cross-track gate (handled by the Stage-0 router, not here)

| Phrase contains | Hand off to |
|---|---|
| `paired` / `golden+defect` / `golden / defect pair` / `ChangeNet` / `siamese` | single-flow track, paired sub-mode (good + defect with shared seed; post-process). |
| `ROI crop` / `per-component crop` / `usd2roi` / `cad2roi` / `align to real photo` / `MI registration` / `bridge crops` | **roi track** — see `../roi/stages.md`. The Stage-0 router gates this on explicit keywords; a bare photo path or component mention does NOT route to the roi track. |

## In-scope flows

| Keywords (any in the prompt) | Flow | Base config |
|---|---|---|
| `missing` (when paired with images / components) / `hidden components` / `two-pass` / `reference + defective` | `missing` | `configs/flow2_defect_image/missing_image.yaml` |
| `defect` / `defects` / `pose defect` / `shift` / `tombstone` / `sideflip` / `polarity` / `reverse polarity` | `defect` | `configs/flow2_defect_image/defect_image.yaml` |
| `zoom in` / `close-up` / `single capacitor` / `fixed camera` / `solder fillet randomization` / `tin pad` / `tin noise` / `vantablack` / `per-sample randomization` | `good_fixed` | `configs/flow1b_good_fixed/good_fixed.yaml` |
| `lighting demo` / `lighting variant` (EXPLICIT demo intent) | `lighting` | `configs/lighting_example/good_image_<variant>.yaml` |
| `good` / `clean` / `defect-free` / `no defects` / `scan grid` / `auto scan` / `full board` / (anything else) | `good` | `configs/flow1_good_image/good_image.yaml` |

## Lighting variant disambiguation

Pick the most specific match. If user says multiple, ask.

| Phrase | File |
|---|---|
| `ring light` / `RGB ring` / `per-layer color` | `good_image_ring_light.yaml` |
| `dome light` / `dome` / `single white dome` | `good_image_dome_light.yaml` |
| `scene lights` / `use authored lights` / `skip rig build` | `good_image_scene_lights.yaml` |
| `preserve color` / `keep authored RGB` | `good_image_preserve_color.yaml` |

## Defect-mode disambiguation

If the user says **"only X"** or **"just X"** for X ∈ {shift, tombstone, sideflip, reverse_polarity / polarity}:

- Set `defects.X.enabled: true`
- Set the other three `defects.*.enabled: false`
- Keep their `ratio` values as-is (won't fire since disabled)

If the user lists multiple (`"shift and tombstone"`): enable those, disable the rest.

If the user gives a `ratio` (`"50% tombstone"`): set `defects.tombstone.ratio: 0.5`.

## Board disambiguation

| Phrase | Board | pcba_target.yaml to pass |
|---|---|---|
| `Spark` / `BASE A04` / `60014242` / no mention | Spark | `configs/pcba_target.yaml` (default) |
| `IC` / `115_2819_000` / `2819` / `the IC board` | IC | `configs/usd_roi_examples/115_2819_000_day0/pcba_target.yaml` |
| `0603 H100` / `H100 capacitor` | 0603 H100 | `configs/usd_roi_examples/0603_H100_day0/pcba_target.yaml` |
| Other | unclear | ASK the user which board + which `pcba_target.yaml` |

## Image-count → grid sizing

Auto-scan defaults are 3×3=9 (good), 10×10=100 (defect/missing). When the user requests N:

- N ≤ 9: keep base grid (3×3 or 10×10), cap with `max_image_count: N`.
- N > grid size: still cap with `max_image_count: N`; bump `num_triggers` so the cap is achievable.
  - For `defect` / `missing` at base 10×10=100: `num_triggers = ceil(N / 100)`.
  - For `good` at base 3×3=9: bump `scan_grid.{x_num,y_num}` to the smallest k×k ≥ N, OR keep 3×3 and bump `num_triggers`. Prefer the grid bump (less randomness reset per trigger).
- `good_fixed` is fixed-camera, so `max_image_count = samples_per_position = N` (single cell, all variation from per-sample randomization).


## Lighting parameter (orthogonal to flow)

Lighting choice is independent of the flow — it's an override applied on top of whichever flow was selected. The skill applies the **scene-lights default** unless the user names a different lighting mode in the prompt.

| Phrase | Lighting mode | Overrides to inject |
|---|---|---|
| (nothing about lighting) | scene (default) | `use_scene_lights: true`, `preserve_scene_light_color: true`, `lighting.ring_light: false` |
| `scene lights` / `usd lights` / `authored lights` / `use the lights in the usd` | scene (explicit) | same as default |
| `ring light` / `rgb ring` / `per-layer color` | ring | `use_scene_lights: false`, `lighting.ring_light: true` |
| `dome light` / `dome` / `white dome` / `dome-lit` | dome | `use_scene_lights: false`, `lighting.ring_light: false` |
| `whitened scene lights` | scene + whiten | `use_scene_lights: true`, `preserve_scene_light_color: false` |

The 4 lighting demo YAMLs under `configs/lighting_example/` exist for reference / direct invocation — they're not the only way to get those lighting modes. Any flow can be combined with any lighting via these overrides.


## Default routing

If the prompt does NOT EXPLICITLY name any cad2roi/usd2roi/ROI/MI/crop/bridge keyword, the Stage-0 router classifies the request as the single-flow track. The roi track is only triggered by explicit ROI-extraction intent. A "real PCB photo" path alone does NOT auto-route to the roi track -- the user must also say what they want to DO with it (e.g. "register", "align", "crop ROIs"). This discipline carries forward the `74eef40` rule.
