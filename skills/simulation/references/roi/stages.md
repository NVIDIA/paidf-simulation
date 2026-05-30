# ROI track — stage detail

This file holds the 5-stage scaffold that the `simulation` router
hands off to whenever a request resolves to the `roi` track.
Variant-specific docker invocations and yaml fields live in
`day0.md` / `day1.md` — this file is the shared scaffold they plug into.

```
Day-0 (no real photo):
  [scene.usd] + semantics:
    ── sdg_pipeline.py     (Kit, GPU)  ── <out>/trigger_NNNN/{rgb,seg,labels}_x*_y*
    ── usd2roi_crop.py     (host py)   ── <out>/crop/component/<x*_y*>/{normal_img,cad_mask}/

Day-1 (real photo supplied):
  [scene.usd]   ── usd2roi_render.py    (Kit, GPU)         ── <out>/sdg/
  [real.png]    ── usd2roi_register.py  (host python+cupy) ── <out>/aligned/
                  usd2roi_crop.py       (host python)      ── <out>/crop/
```

All stages run in their own container — they share state only through
the host output dir. Scripts (`scripts/sdg/standalone/*`,
`scripts/usd2roi/*`) are baked into the image; no `git clone` required.

## Overview

Pick the variant in **Validate** (Stage 1 below) before doing anything
else:

| User has… | Variant | Read for stage detail |
|---|---|---|
| CAD-derived USD, **no real photo** | Day-0 | `day0.md` |
| CAD-derived USD, **real PCB photo** | Day-1 | `day1.md` |

Cross-track gate (handled by the router, not this file): if the request
is for a paired golden/defect ChangeNet dataset, that's a `single-flow`
track concern — see `../single-flow/stages.md`. The ROI track only
handles per-component crops from a CAD USD.

## Prerequisites

Shared base (both variants):

| Resource | How to satisfy |
|---|---|
| Container image pulled | `docker pull $SDG_IMAGE`. User supplies the tag (e.g. `<your-registry>/paidf-simulation:1.0.0-<sha>`). The track does NOT assume a default — ask the user if `$SDG_IMAGE` is unset. |
| GPU with OptiX | Driver ≥ 525; `ls /usr/share/nvidia/nvoptix.bin` succeeds on host |
| `<usd-assets>/` slim USD pack | Download + unzip per repo README; one host dir mounted into the container |
| Output dir writable to uid 1234 | `mkdir -p $OUTPUT && chmod 777 $OUTPUT` |

Variant-specific prerequisites (real PCB photo, OSMO submission, repo
clone vs. baked scripts) live in `day0.md` and `day1.md`.

## Usage

Five stages: **Validate → Prepare → Execute → Verify → Report**. The
shared scaffold is below; per-variant commands, yaml summary contents,
and verify checks live in the variant's reference file.

### 1. Validate

a) **Pick the variant** from `$ARGUMENTS`:

- No real photo supplied → **Day-0**.
- A real photo supplied (`--photo-path`, "real.png", "AOI screenshot",
  etc.) → **Day-1**.

If ambiguous, ask: "Do you have a captured real PCB photo to align
against, or only the CAD-derived USD?"

b) **Collect host paths + container image tag** for the chosen
variant. Both variants follow the same shape: user authors the
yaml(s) ahead of time, track reads + summarises + runs.

- Day-0: `$ASSETS`, `$CONFIG` (contains `day0_image.yaml`,
  `day0_crop.yaml`, `pcba_target.yaml`), `$OUTPUT`, `$SDG_IMAGE`. See
  `day0.md` §Validate.
- Day-1: `$ASSETS`, `$CONFIG` (contains `usd2roi_target.yaml`),
  `$PHOTO_DIR` (contains `real.png`), `$OUTPUT`, `$SDG_IMAGE`. See
  `day1.md` §Validate.

`$SDG_IMAGE` is the full image reference (`nvcr.io/.../paidf-simulation:1.0.0-<sha>.<channel>`).
The track does NOT assume a default tag — ask the user explicitly if
it's missing.

Ask for any missing path. Do not invent semantic rules — point the
user at `semantic-rules.md` and `../../../../../configs/cad2roi/spark/semantics.yaml`
as the reference pattern when the user's yaml has an empty `semantics:`
block.

c) **Input sanitisation** (both variants): reject any path containing
`;`, `|`, `` ` ``, `$()`, or `..`. Outputs live only under
`/workspace/paidf-simulation/sdg_test_output/`.

Idempotency: no side effects. Safe to retry.

### 2. Prepare

a) Read the user-authored yaml(s) from `$CONFIG`. Surface a one-screen
summary covering the key fields (scene path, semantics rule count, crop
classes, resolution, output path, plus variant extras). Do NOT rewrite
the yaml — Day-0 / Day-1 templates ship runnable on the spark scene
out-of-the-box, and the user is expected to edit them per board before
invoking this track. Exact summary blocks and per-variant `cp` snippets
are in the variant's reference file.

**Count override.** If the prompt names a target ROI count (e.g.
"generate 10 components", "~50 ROIs"), write a derived crop yaml
under `configs/runs/<ts>_*` with `crop.max_emit: N` and pass that to
the crop stage instead of the user's file. `max_emit` is a **global**
cap across all cells, not per-cell — on a 10×10 grid, `max_emit: 10`
typically emits all 10 from the first non-empty cell and skips the
rest. For ~N per cell, leave `max_emit: null` and tighten `min_area`,
`min_coverage`, or `edge_skip` instead.

b) **Dry-run the semantic rules** before paying for a Kit boot. This is
the same helper for both variants:

```bash
docker run --rm \
  -v $ASSETS:/workspace/paidf-simulation/<usd-assets>:ro \
  -v $CONFIG:/workspace/paidf-simulation/_config:ro \
  --entrypoint python3 ${SDG_IMAGE} \
  scripts/usd2roi/semantic_rules.py \
    --scene <container-relative scene.usd> \
    --rules <container-relative resolved.yaml> \
    --show 5
```

If any rule reports `0 prim(s)`, the glob is wrong or the prim doesn't
exist — fix before Stage 1. See `semantic-rules.md` for glob syntax and
the multi-class-on-same-prim pattern.

c) Pre-create the output dir host-owned writable:

```bash
mkdir -p $OUTPUT && chmod 777 $OUTPUT
chmod -R o+rX $ASSETS $CONFIG          # plus $PHOTO_DIR for Day-1
```

d) Show a one-screen summary covering: scene, semantics rule count,
anchor / crop classes, resolution, output path — plus
camera.translate + bridge config for Day-1, scan_grid dimensions for
Day-0.

**[APPROVAL GATE]** — Pause. Tell the user which variant was selected,
how many containers will run (Day-0: 2 plus a chmod helper; Day-1: 3),
the expected duration of the longest stage (Day-0 Stage 1 ~9-12 min for
100 cells; Day-1 Stage 1 ~5-7 min cold boot), and the host output path.
Wait for explicit confirmation before Execute.

Idempotency: regenerating resolved yamls overwrites the previous files.
Safe to retry.

### 3. Execute

Run the variant-specific docker commands. Full invocations (mounts,
entrypoint overrides, the `--exec` single-quoted-string shape for the
Kit stage) live in:

- Day-0: `day0.md` §Execute — Stage 1 `sdg_pipeline.py` (Kit) → chmod
  helper → Stage 2 `usd2roi_crop.py`.
- Day-1: `day1.md` §Execute — Stage 1 `usd2roi_render.py` (Kit) →
  Stage 2 `usd2roi_register.py` → Stage 3 `usd2roi_crop.py`.

Common shape across variants: Stage 1 uses the Kit default entrypoint
and passes `"<script> --config X ..."` as **one quoted argument**;
later stages override `--entrypoint python3` and pass args directly.

Idempotency: each stage overwrites its own output subtree
(`trigger_NNNN/`, `sdg/`, `aligned/`, `crop/`). Re-run any stage
independently. Day-1 Stage 2 exits with code 2 and writes nothing if
`mi_after < registration.min_mi` (default 0.5) — see `day1.md` §Execute
and `troubleshooting.md`.

### 4. Verify

Variant-specific output-tree inspection. Common checks:

- Day-0: every cell has `rgb_x*_y*.png` + matching seg + labels.json;
  `crop/component/<x*_y*>/normal_img/*.png` counts.
- Day-1: `sdg/` + `aligned/` + `crop/` populated; `aligned/params.json`
  carries `{scaleX, scaleY, rotation_deg, tx, ty, mi_before, mi_after}`;
  `aligned/blink.gif` visualises overlap.

Exact `ls` / `find` commands and expected counts are in each variant's
reference file §Verify.

Idempotency: read-only inspection.

### 5. Report

Surface:

- Variant selected (Day-0 or Day-1) and a one-line verdict
  (Day-0: ROI count + skip breakdown; Day-1: `mi_after` quality band +
  ROI / bridge counts)
- Host path to outputs (and to `blink.gif` for Day-1 visual QA)
- Sample paths to spot-check (`crop/component/x5_y5/normal_img/0001.png`
  for Day-0; `crop/component/normal_img/0001.png` for Day-1)

If any stage exited non-zero, surface the last 30 lines of its log and
link to `troubleshooting.md`.

## Reference

| Resource | Path |
|---|---|
| Day-0 stage detail (render yaml, crop yaml, granular mounts) | `day0.md` |
| Day-1 stage detail (single config, MI registration, OSMO submission) | `day1.md` |
| Semantic-rule glob syntax + multi-class-on-same-prim | `semantic-rules.md` |
| Per-stage failure modes (both variants, tagged) | `troubleshooting.md` |
| Day-0 SDG render template (author-once, spark-default) | `../../../../../configs/cad2roi/day0/sdg/day0_image.yaml` |
| Day-0 crop template (author-once, spark-default) | `../../../../../configs/cad2roi/day0/usd2roi/day0_crop.yaml` |
| Day-1 single-config template | `../../../../../configs/cad2roi/day1/replicator/usd2roi_target.yaml` |
| Day-1 OSMO workflow | `../../../../../configs/cad2roi/day1/osmo/workflow.yaml` |
| Spark worked example — semantics block | `../../../../../configs/cad2roi/spark/semantics.yaml` |
| Spark worked example — PCBA target | `../../../../../configs/cad2roi/spark/pcba_target.yaml` |
| Container image | `${SDG_IMAGE}` |
| Pipeline schema + MI algorithm notes | `scripts/usd2roi/README.md` (in repo, not duplicated here) |

## Error Handling

Symptom → cause → fix tables (both variants, rows tagged `[day0]` /
`[day1]`) live in `troubleshooting.md`. The two most common
cross-variant failures:

| Symptom | Likely cause | Action |
|---|---|---|
| Dry-run reports `0 prim(s) affected` for a rule | Glob doesn't match the scene's prim hierarchy | Open the USD; instance-proxy paths often look right but actual paths use `LibRef/` segments. See `semantic-rules.md`. |
| `PermissionError` / `Cannot open USD` on any stage | Container uid 1234 cannot read host file | `chmod -R o+rX $ASSETS $CONFIG $PHOTO_DIR`; for the output dir, the chmod-helper docker step. |

## Examples

### Example 1 — Day-0 spark smoke

User prompt:
```
Run Day-0 ROI extraction.

Host paths:
  ASSETS = ~/sdg-day0/<usd-assets>
  CONFIG = ~/sdg-day0/<day0-config>   (contains day0_image.yaml, day0_crop.yaml, pcba_target.yaml)
  OUTPUT = ~/sdg-day0/output
```

Track: detects Day-0 (no photo path supplied), follows `day0.md`. Reads
`$CONFIG/day0_image.yaml` + `$CONFIG/day0_crop.yaml` and prints a
summary (21 semantic rules, 10×10 scan_grid, `crop.classes: [capacitor,
ic]`). Runs the semantic-rules dry-run; pauses for approval; Stage 1
(~10 min) → Stage 2 (~20 s). Reports ROI count + skip breakdown at
`$OUTPUT/crop/component/`.

### Example 2 — Day-1 spark smoke with bridge

User prompt:
```
Run Day-1 ROI extraction.

Host paths:
  ASSETS    = ~/sdg-day1/<usd-assets>
  CONFIG    = ~/sdg-day1/<day1-config>   (contains usd2roi_target.yaml)
  PHOTO_DIR = ~/sdg-day1/input          (contains real.png)
  OUTPUT    = ~/sdg-day1/output
```

Track: detects Day-1 (PHOTO_DIR supplied), follows `day1.md`. Reads
`$CONFIG/usd2roi_target.yaml` and prints a summary (scene
`<usd-assets>/temp_scene.usd`, `camera.translate=(-55.4, 2)`,
`horizontal_aperture=97`, `resolution=[640, 480]`,
`crop.classes=[capacitor, solder, pad, ic]`, `bridge=true,
bridge_dis=30, bridge_classes=[capacitor]`). Stage 1 render → Stage 2
register (`mi_after ≈ 0.51`) → Stage 3 crop (~24 ROIs + 2 bridges).
Reports MI band + ROI count + path to `$OUTPUT/aligned/blink.gif`.

### Example 3 — Ambiguous request

User: "Crop ROIs from a board."

Track: asks **first** whether a real PCB photo exists (routes to Day-0
vs. Day-1), then asks for the variant's required host paths (Day-0:
`$ASSETS`, `$CONFIG`, `$OUTPUT`; Day-1: also `$PHOTO_DIR`). Does NOT
emit any docker run or write any yaml in this turn.
