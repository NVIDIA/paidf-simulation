# Single-flow track — stage detail

This file holds the Stage 1-6 pipeline that the `simulation` router
hands off to whenever a request resolves to the `single-flow` track.

```
"generate 5 defect images, only tombstone"
        |     parse intent
flow = defect  |  base = examples/flow2_defect_image/defect_image.yaml
overrides = {max_image_count: 5, defects.shift.enabled: false,
             defects.sideflip.enabled: false,
             defects.reverse_polarity.enabled: false}
        |     derive YAML
configs/runs/<slug>.yaml
        |     docker run  (or local Isaac-Sim if `local` keyword)
sdg_test_output/<slug>/trigger_0000/rgb_0000.png .. rgb_0004.png
```

## Stage 1 — Parse intent, pick the flow

Decision rules (apply in order, first match wins):

1. **`paired` / `golden+defect` / `ChangeNet` / `siamese`** → handled in-skill. Run the `good` flow + the `defect` flow with the same `random_seed`; emit Pair-dataset under `${PAIDF_SIM_ROOT}/sdg_test_output/pairs_<slug>/`. (Implementation: TODO — wire two run_render calls + post-process step.)
2. **`missing` / `hidden components` / `two-pass`** → flow = `missing`, base = `configs/flow2_defect_image/missing_image.yaml`
3. **`defect` / `pose defect` / `shift` / `tombstone` / `sideflip` / `polarity`** → flow = `defect`, base = `configs/flow2_defect_image/defect_image.yaml`
4. **`zoom` / `close-up` / `single capacitor` / `fixed camera` / `solder fillet randomization` / `tin pad` / `vantablack`** → flow = `good_fixed`, base = `configs/flow1b_good_fixed/good_fixed.yaml`
5. **`lighting demo` / `lighting variant`** (explicit demo intent) → flow = `lighting`, base = `configs/lighting_example/good_image_<variant>.yaml` (variant = `ring_light` / `dome_light` / `scene_lights` / `preserve_color`)
6. **`good` / `clean` / `defect-free` / `no defects` (and anything else)** → flow = `good`, base = `configs/flow1_good_image/good_image.yaml`

Cross-track gate (handled by the router, not this file): ROI / cad2roi / usd2roi / MI / bridge-crop keywords route to the **roi** track, not here. See `../../SKILL.md` Stage 0 routing rules.

### Lighting parameter (orthogonal — applies to any flow)

The lighting choice is independent of the flow. If the user names a lighting mode anywhere in the prompt, apply it as an override on top of the flow's base config. **Default when nothing is specified: scene-authored lights (`use_scene_lights: true`, `preserve_scene_light_color: true`).**

| Phrase contains | Lighting mode | Overrides |
|---|---|---|
| (nothing about lighting) | scene (default) | `use_scene_lights: true`, `preserve_scene_light_color: true`, `lighting.ring_light: false` |
| `scene lights` / `usd lights` / `authored lights` / `use the lights in the usd` | scene (explicit) | same as default |
| `ring light` / `rgb ring` / `per-layer color` | ring | `use_scene_lights: false`, `lighting.ring_light: true` (per-trigger RGB ring rig) |
| `dome light` / `dome` / `white dome` | dome | `use_scene_lights: false`, `lighting.ring_light: false` (white_light block drives the dome) |
| `preserve color` / `keep authored colors` (with scene lights) | scene + preserve | `use_scene_lights: true`, `preserve_scene_light_color: true` (this IS the default, so this phrase is redundant but accepted) |
| `whitened scene lights` / `scene lights but white` | scene + whiten | `use_scene_lights: true`, `preserve_scene_light_color: false` |

This means **a single prompt can carry both a flow choice and a lighting choice independently**:

- `generate 5 defect images, ring light` → flow=`defect` + lighting=`ring`
- `zoom in to a capacitor with dome light` → flow=`good_fixed` + lighting=`dome`
- `generate 9 good PCB images` → flow=`good` + lighting=`scene` (default)


See `routing.md` for the full natural-language → flow map.

## Normalized Intent (structured output of Stage 1)

Every prompt parses into a structured intent record. This is the CANONICAL schema — anyone implementing or auditing the track should produce / consume this exact shape:

```json
{
  "flow":            "good | good_fixed | defect | missing | lighting | paired",
  "lighting":        "scene | ring | dome | scene_whitened",
  "count":           int | null,        // null = no count in prompt → use canonical YAML default
  "board":           "Spark | IC | 0603_H100 | <custom>",
  "components":      [str] | "ALL" | "0" | "chip_passive" | "cap_small" | "cap_large" | "inductor" | null,
  "defect_modes":    ["shift", "tombstone", "sideflip", "reverse_polarity"] | null,   // defect/missing only
  "defect_component_filter": [str] | null,    // e.g. ["_032_0667"] for polarity targets
  "execution_mode":  "docker | local",
  "usd_path":        "<absolute path>",
  "output_path":     "<absolute path> | null",        // null → ${PAIDF_SIM_ROOT}/sdg_test_output/<auto-slug>/
  "seed":            int | null,
  "resolution":      [int, int] | null,
  "crop": {                                            // skill-consumed; canonical default per flow
    "enabled":           bool,                         // true for good/defect/missing; false for good_fixed/lighting
    "mode":              "missing" | null,             // null = single-pass; "missing" = two-pass with --reference
    "types":             [str],                        // e.g. ["rgb", "semantic_segmentation", "component_instance"]
    "offset":            int,                          // px padding (default 10)
    "class_filter":      [str] | null,                 // optional --class-filter
    "xform_depth":       int,                          // default 7
    "input_subdir":      str | null,                   // "defective" for missing flow; null elsewhere
    "reference_subdir":  str | null,                   // "reference" for missing flow; null elsewhere
    "output_subdir":     str                           // default "cropped"
  }
}
```

**Count semantics — the rule that prevents over-engineering:**

- `count == null` (user did not name a number) → **emit NO `max_image_count` override**. The canonical YAML at `configs/<flow>/...` defines the "consistent default" via its `scan_grid` + `max_image_count`. Run it as-is.
  - `good` canonical = `scan_grid: {3, 3}`, `max_image_count: -1` → 9 frames in 1 trigger.
  - `defect` canonical = `scan_grid: {10, 10}`, `max_image_count: -1` → 100 frames.
  - `missing` canonical = `scan_grid: {10, 10}` → 100 cells × 2 passes = 200 writer frames.
  - `good_fixed` canonical = `samples_per_position: 300`, `max_image_count: 300` → 300 frames.
- `count == N` (user named an integer) → balanced-grid factorization:
  - `good`: pick `(a, b)` with `a × b = N` and `b/a` closest to 1. Use `scan_grid: {a, b}`, `max_image_count: N`, `num_triggers: 1`. For primes, **ASK** before rounding.
  - `defect`: `max_image_count: N`, keep `scan_grid: {10, 10}`.
  - `missing`: `max_image_count: 2 * N` (each cell writes ref + defective; `N defective frames` = user intent).
  - `good_fixed`: `max_image_count: N`, `samples_per_position: N`.
- Lighting / USD / output_path / seed overrides are applied regardless of count.

This is the contract Stage 2 implements.

## Stage 2 — Extract overrides

Walk the user's prompt for these knobs (full list in `overrides.md`):

| User phrasing | YAML key | Example |
|---|---|---|
| "N images" / "N frames" | `max_image_count: N` | "5 images" → `5` |
| (no count specified — "create a dataset" / "render images") | **no count override** — run the canonical YAML as-is. The YAML's baked-in `scan_grid` is the "consistent default": `good`=3×3=9, `defect`=10×10=100, `missing`=10×10=100 cells × 2 passes=200 writer frames, `good_fixed`=whatever `samples_per_position` is set to (default 300). | tell the user the count that'll result; offer to bump |
| count N where N has a balanced factorization a×b (b/a ≤ 2) | `scan_grid.{x_num,y_num}: {a, b}`, `max_image_count: N`, `num_triggers: 1` | grid covers exactly N cells in one trigger, no waste, no per-trigger randomization reset. Examples: 9→3×3, 12→3×4, 20→4×5, 25→5×5, 100→10×10. |
| count N skinny (a×b with b/a > 2, e.g. 10=2×5, 14=2×7) | same — accept the skinny grid | mention "grid is 2×N — accept or round to nearest balanced N±k?" |
| count N prime or degenerate (7, 11, 13, 17, 19, 23, …) | `scan_grid.{x_num,y_num}` of nearest balanced N±1, OR keep N via `num_triggers + cap` | **ASK the user**: "N=7 is prime — render 8 (2×4) or 9 (3×3) instead, or keep 7 via num_triggers + cap?" |
| "N x N grid" / "NxN" / "N per axis" | `scan_grid.{x_num,y_num}` | "5x5" → `{x_num: 5, y_num: 5}` |
| "only tombstone" / "tombstone only" | `defects.{shift,sideflip,reverse_polarity}.enabled: false`, `defects.tombstone.enabled: true` | — |
| "no defects" | `defects.*.enabled: false` (or use `good` flow instead) | — |

**Missing-pipeline count gotcha.** The `missing` flow writes a *reference* pass and a *defective* pass per cell. `max_image_count` caps the TOTAL writer frames (ref + defective). When the user says "N missing defects", they mean N defective frames. So for `flow=missing`, set `max_image_count = 2 * N` so the cap accommodates both passes, and the user gets N defective + N reference frames.
| "1024 / 2048 / 1920x1080" | `resolution: [w, h]` | "1024" → `[1024, 1024]` |
| "seed N" / "reproduce N" | `random_seed: N` | "seed 42857" → `42857` |
| "high quality" / "HQ" | `pathtracing.total_spp: 128` | bump from 32 |
| "scene lights" / "usd lights" / (default) | top-level `use_scene_lights: true`, `preserve_scene_light_color: true` | also flips `lighting.ring_light: false` |
| "ring light" / "rgb ring" | top-level `use_scene_lights: false`, `lighting.ring_light: true` | builds per-trigger RGB ring rig |
| "dome light" / "white dome" | top-level `use_scene_lights: false`, `lighting.ring_light: false` | white_light block drives dome |
| "for the IC board" / "115_2819_000" | switch base + pcba_target to the IC variants | see Stage 4 |
| "use `path/to/x.usd`" / "with the USD at X" / "from X.usd" | `PCB_USD_PATH=X` for the run (overrides default) | quote the path back in approval |
| "to `path/to/out`" / "into `path/to/out`" / "output to X" | `output: X` (overrides the default `${PAIDF_SIM_ROOT}/sdg_test_output/<slug>`) | host path; skill bind-mounts parent for Docker mode. Quote it back in approval. |
| "of component X" / "for X" / "zoom on X" (in good_fixed flow) | `auto_locate_component: <X-substr>` | pipeline finds a random instance of the named component under pcba_root, computes its world-bbox, frames camera at bbox center with >=1/2 padding on both dims (viewport aspect locked). Overrides any `horizontal_aperture` / `scan_grid` from the YAML. |
| "of X or Y or Z" / "either A or B" / multi-target single run | `component_list_override: [X, Y, Z]` | list-form alternative. Pipeline merges all matching prims across the list into one candidate pool, then picks one randomly. Overrides `auto_locate_component` when both are set. |
| pin to an exact USD prim path | `auto_locate_prim_path: /World/.../tn__0603_H100_27_` | pinpoint; takes precedence over `auto_locate_component` and `component_list_override`. No randomness. Useful when re-rendering a specific instance the user picked in the editor. |
| component_type != _0603_H100 (auto_locate_component or component_list_override resolves to anything else) | `solder_fillet.enabled: false` | the solder-fillet generator is geometrically calibrated to 0603 chip-cap dimensions; running it on any other type (inductors, larger caps, ICs, ...) places fillets at wrong positions / sizes. MUST be forced off whenever the target is not _0603_H100. |
| "local" / "fast" / "host run" / "skip docker" | use the local Isaac-Sim install | see Stage 5 |
| "no crop" / "skip crop" / "render only" / "without cropping" | `crop.enabled: false` | Disables the post-render crop step. |
| "with crop" / "also crop" / "include crops" / "crop too" | `crop.enabled: true` | Forces the crop step ON for flows whose canonical default is off (good_fixed / lighting). |
| "crop offset N" / "N-px crop margin" / "crop padding N" | `crop.offset: N` | Passed to `crop_components.py --offset`. |
| "only crop X" / "crop only X components" | `crop.class_filter: [X]` | Filters which components get cropped (does NOT affect rendering). |

Full per-knob detail for the crop block is in `overrides.md` §`crop` block.

## Stage 3 — Validate prereqs

Before generating, check these. **If any is missing, ASK THE USER. Do not guess.**

| Prereq | How to check | What to ask |
|---|---|---|
| `PCB_USD_PATH` set OR named in prompt | `echo $PCB_USD_PATH`; also scan prompt for `usd=...`, `from <path>.usd`, `use <name>.usd` | If prompt names a `.usd`, use it. Otherwise if `$PCB_USD_PATH` is set, use it. If neither is provided, **ASK the user**: "`PCB_USD_PATH` isn't set and no USD was named in the prompt. Where's the USD?" Always confirm the path back in the approval gate. |
| `PAIDF_SIM_ROOT` set | `echo $PAIDF_SIM_ROOT` | Default to repo root (`$HOME/paidf-simulation`) and tell the user. |
| (Container mode = default) Docker image available | `docker images paidf-simulation:sqa` (preferred — rolling stable from main); else `paidf-simulation:local-sqa-test` (fallback) | If `:sqa` is present, use it (no source bind-mount needed — released code is baked in). If only `:local-sqa-test` is present, use it **with the host source bind-mounted** over `/workspace/paidf-simulation` so the local code path is exercised. If **neither** is present, **ASK the user**: "No paidf-simulation image found. Build it (~15-25 min, needs NGC creds), pull from registry, or switch to `, local` mode?" |
| (Local mode only) launcher present | `test -x "$ISAAC_SIM_PATH"` | If `$ISAAC_SIM_PATH` is set and executable, use it and tell the user. If unset or missing, **ASK the user**: "Local mode requested but `ISAAC_SIM_PATH` isn't set. Provide a path to your Isaac-Sim install's `isaac-sim.sh`, or shall I fall back to Docker?" |
| Board-specific pcba_target | check user's board mention | Spark → `configs/pcba_target.yaml`; IC → `configs/usd_roi_examples/115_2819_000_day<N>/pcba_target.yaml` |

### Explicit defaults & what the track ALWAYS asks vs. assumes

This track is designed to work on machines other than the maintainer's. On a fresh machine, walk through these defaults explicitly:

| User input | Behaviour |
|---|---|
| `/simulation <prompt>` (no `, local`) | **Container mode** is the default. Expects `paidf-simulation:sqa` (or the `:local-sqa-test` fallback). If neither image is present, **ASK** whether to build, pull, or switch to local. Do not silently fall back. |
| `/simulation <prompt>, local` | **Host mode**. Use `$ISAAC_SIM_PATH` (must point at an Isaac-Sim standalone install's `isaac-sim.sh`). If the env var is unset or the file is missing, **ASK** for a launcher path. |
| Prompt names a `.usd` (`use /path/x.usd`) | Use that path verbatim. Quote it back in the approval gate. |
| Prompt does NOT name a `.usd` | If `$PCB_USD_PATH` is set, use it (tell the user). Otherwise **ASK**. |
| Prompt says `to /path/to/out` | Use `/path/to/out` as `output:`. In Docker mode, bind-mount its parent for write access. |
| Prompt does NOT name an output path | Default to `${PAIDF_SIM_ROOT}/sdg_test_output/<auto-slug>/`. Tell the user the default was used and offer to redirect. |
| Prompt names a non-Spark board ("IC board" / "115_2819_000") | **ASK** before assuming `configs/pcba_target.yaml` (the Spark default). Offer the per-board override under `configs/usd_roi_examples/<board>_day<N>/pcba_target.yaml` (verify path exists). |
| Prompt names a lighting mode (`ring light` / `dome light` / `scene lights`) | Apply orthogonally (overrides any flow-baked default). |
| Prompt names a `seed N` | Set `random_seed: N`. |

**Three things this track will NEVER silently assume:**

1. **PCB_USD_PATH** when unset and no USD is named in the prompt — always ask.
2. **Local launcher** when `$ISAAC_SIM_PATH` is unset or missing — always ask.
3. **Docker image** when neither `:sqa` nor `:local-sqa-test` is present — always ask (build / pull / local fallback).

**Three things this track WILL silently default** (but tell the user what it picked):

1. `OUTPUT_PATH` when not named — uses `${PAIDF_SIM_ROOT}/sdg_test_output/<auto-slug>/`.
2. Lighting — defaults to scene-authored.
3. `PAIDF_SIM_ROOT` — defaults to the detected repo root (`pwd` ancestor with `.agents/skills/simulation/`); on a fresh machine outside a repo, **ASK**.

## Stage 4 — Generate the derived YAML

Write to `configs/runs/<timestamp>_<slug>.yaml`. Steps:

1. `mkdir -p configs/runs/` (gitignored — safe scratchpad).
2. Pick the slug from the flow + key knobs: `defect_tombstone_5`, `good_image_10`, `good_fixed_300_samples`, `lighting_ring_light`.
3. Resolve the timestamp: `date +%Y%m%d_%H%M%S`.
4. Apply overrides via this minimal Python (run once per request):

```python
import yaml, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
overrides = yaml.safe_load(sys.argv[3])  # YAML string from the skill
cfg = yaml.safe_load(src.read_text())
def deep_set(d, dotted, v):
    keys = dotted.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = v
for k, v in overrides.items():
    deep_set(cfg, k, v)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(dst)
```

5. Show the user the derived YAML diff (max 30 lines, highlight the changed keys).

## Stage 5 — Approve and execute

**[APPROVAL GATE]** Show the user:

- The derived YAML path
- A summary of the changes vs. the base (e.g. "max_image_count: -1 → 5; defects.shift.enabled: true → false; ...")
- The exact command that will run (BOTH the render docker and, when `crop.enabled: true`, the crop docker right after)
- Estimated cost: "~3 min cold-start + ~1 s/frame on the RTX (+ ~20 s/100 frames if crop is enabled)"

The skill reads `crop.enabled` from the merged YAML to decide whether to
chain the crop step. If true, surface it in the approval gate as:

```
+ crop step: scripts/postprocess/crop_components.py
    --input  <output>/trigger_0000[/<input_subdir>]
    --output <output>/<crop.output_subdir>
    --crops  <crop.types ...>
    --offset <crop.offset>
    [--reference <output>/trigger_0000/<reference_subdir>]   # missing mode only
    [--class-filter <crop.class_filter ...>]                 # if set
```

Wait for explicit confirmation. Then execute:

### Before execution — write `prompt_metadata.json`

Drop a sidecar JSON into the output dir so the run is traceable back to the natural prompt + intent + configs:

```bash
$PAIDF_SIM_ROOT/scripts/sdg/write_prompt_metadata.py   --output-dir "$OUT"   --prompt "<user's natural prompt verbatim>"   --flow <good|good_fixed|defect|missing|lighting>   --lighting <scene|ring|dome|scene_whitened>   --count <N or omit>   --board <Spark|IC|0603_H100|...>   --base configs/<flow>/<yaml>   --derived configs/runs/<slug>.yaml   --pcba-target configs/pcba_target.yaml   --pcb-usd-path "$PCB_USD_PATH"   --mode <docker|local>   --overrides "<yaml-or-json string of the overrides applied>"
```

This writes `<output_dir>/prompt_metadata.json` capturing prompt, normalized intent, config paths, execution mode, and (post-run) frame counts. Always written — it makes every run reproducible from English alone.

### Default — Docker

```bash
docker run --gpus all --rm --network host \
  -e ACCEPT_EULA=Y -e PYTHONUNBUFFERED=1 \
  -e PCB_USD_PATH=$PCB_USD_PATH -e PAIDF_SIM_ROOT=$PAIDF_SIM_ROOT \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v $(dirname $PCB_USD_PATH):$(dirname $PCB_USD_PATH):ro \
  -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
  -v $PAIDF_SIM_ROOT/sdg_test_output:$PAIDF_SIM_ROOT/sdg_test_output \
  paidf-simulation:local-sqa-test \
  "scripts/sdg/standalone/sdg_pipeline.py \
    --config /workspace/paidf-simulation/configs/runs/<slug>.yaml \
    --pcba-config /workspace/paidf-simulation/configs/pcba_target.yaml"
```

### Local — when user says `local` / `fast` / `host run`

See `local-mode.md`. TL;DR:

```bash
"$ISAAC_SIM_PATH" --no-window --exec \
  "$PAIDF_SIM_ROOT/scripts/sdg/standalone/sdg_pipeline.py \
    --config $PAIDF_SIM_ROOT/configs/runs/<slug>.yaml \
    --pcba-config $PAIDF_SIM_ROOT/configs/pcba_target.yaml"
```

Default output dir: `${PAIDF_SIM_ROOT}/sdg_test_output/<flow>/trigger_NNNN/` (each flow YAML's `output:` key controls this; override per-run for history). `sdg_test_output/` is gitignored.

Fall back to Docker if `$ISAAC_SIM_PATH` is unset or the launcher is absent.

### Stage 5a — Crop (only when `crop.enabled: true`)

Chained after the render. Only runs if the render exits 0 and the merged
YAML has `crop.enabled: true`. The crop step is **always docker** even in
`, local` mode — it's pure Python (no Kit), so the container's bundled
NumPy / Pillow are fine and the host doesn't need them installed.

#### Single-pass (good / defect)

```bash
docker run --rm \
  -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
  --entrypoint python3 \
  paidf-simulation:local-sqa-test \
  scripts/postprocess/crop_components.py \
    --input  /workspace/paidf-simulation/sdg_test_output/<flow>/trigger_0000 \
    --output /workspace/paidf-simulation/sdg_test_output/<flow>/<crop.output_subdir> \
    --crops  <crop.types ...> \
    --offset <crop.offset>
```

#### Two-pass (missing flow)

```bash
docker run --rm \
  -v $PAIDF_SIM_ROOT:/workspace/paidf-simulation \
  --entrypoint python3 \
  paidf-simulation:local-sqa-test \
  scripts/postprocess/crop_components.py \
    --input     /workspace/paidf-simulation/sdg_test_output/<flow>/trigger_0000/<crop.input_subdir> \
    --reference /workspace/paidf-simulation/sdg_test_output/<flow>/trigger_0000/<crop.reference_subdir> \
    --output    /workspace/paidf-simulation/sdg_test_output/<flow>/<crop.output_subdir> \
    --crops     <crop.types ...> \
    --offset    <crop.offset>
```

If `crop.class_filter` is non-null, append `--class-filter <values...>`.
If `crop.xform_depth != 7`, append `--xform-depth <N>`.

Output layout (single-pass):

```
sdg_test_output/<flow>/<crop.output_subdir>/
├── <label_or_defect>/
│   ├── rgb/                 frame<NNNN>_<comp_name>.png
│   ├── semantic_segmentation/
│   └── component_instance/
```

Output layout (missing mode, two-pass):

```
sdg_test_output/<flow>/<crop.output_subdir>/
├── missing/                 defective-pass RGB + matching seg/instance
│   └── {rgb,semantic_segmentation,component_instance}/
└── ok/                      reference-pass RGB (same bbox; clean)
    └── {rgb,semantic_segmentation,component_instance}/
```

The crop step deletes its own crops whose final semantic mask has <20%
coverage (`crop_components.py:264-288`). It does NOT touch the raw frames
under `trigger_NNNN/` — the source data is preserved untouched.

## Stage 6 — Report

After execution, report (concise):

- Exit code of the render (assert 0; if non-zero, show last 20 lines of run.log; skip the crop step)
- Output dir (absolute path)
- Count of `rgb_*.png` produced vs requested
- If `crop.enabled: true`: exit code of the crop step + total crop count + per-bucket breakdown (e.g. `tombstone: 14, shift: 12, ok: 0`)
- Path to the derived config (so the user can re-run / tweak)
- Path to the run log
- Path to `prompt_metadata.json` (sidecar — re-runnable record of prompt + intent + configs)

Example (render + crop, defect flow):

```
Render ran in 2m 47s, exit 0
  Output:        ~/paidf-simulation/sdg_test_output/defect_tombstone_5/trigger_0000/
  Frames:        5 rgb + 5 semantic_segmentation + 5 bbox_2d_tight (matches max_image_count=5)

Crop ran in 4s, exit 0
  Output:        ~/paidf-simulation/sdg_test_output/defect_tombstone_5/cropped/
  Buckets:       tombstone: 14  (capacitor body-only crops with the defect label)
                 capacitor:  6  (clean components in the same frames)

Derived YAML:  configs/runs/<slug>.yaml
Log:           sdg_test_output/defect_tombstone_5/run.log
```

## Concrete end-to-end example

User: **"generate 5 defect images, only tombstone"**

```
Stage 1  → flow=defect, base=configs/flow2_defect_image/defect_image.yaml
           (base carries crop.enabled: true by default)
Stage 2  → overrides = {
    max_image_count: 5,
    defects.shift.enabled: false,
    defects.tombstone.enabled: true,
    defects.sideflip.enabled: false,
    defects.reverse_polarity.enabled: false,
}
Stage 3  → PCB_USD_PATH unset → ASK user
Stage 4  → write configs/runs/<slug>.yaml
Stage 5  → show diff + render docker cmd + crop docker cmd; await OK
Stage 5a → render exits 0 → run crop_components.py docker
Stage 6  → 5 frames at sdg_test_output/defect_tombstone_5/trigger_0000/
           + per-bucket crops at sdg_test_output/defect_tombstone_5/cropped/
```

User: **"generate 5 defect images, only tombstone, no crop"** (override the canonical default)

```
Stage 1  → flow=defect, base=configs/flow2_defect_image/defect_image.yaml
Stage 2  → overrides = { ..., crop.enabled: false }
Stage 5  → show diff + render docker cmd only (no crop step in approval gate)
Stage 6  → 5 frames at sdg_test_output/defect_tombstone_5/trigger_0000/
```

## Sibling references

- `routing.md` — full natural-language → flow mapping
- `overrides.md` — per-flow knob cheatsheet
- `local-mode.md` — when/how to use the Isaac-Sim host install
- `troubleshooting.md` — common errors and fixes
