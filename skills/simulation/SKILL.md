---
name: simulation
description: Build Docker-backed PAIDF Simulation commands for PCBA synthetic-data renders (good/defect/missing/lighting/ChangeNet pairs) and ROI crops with optional MI registration. Do NOT use for unrelated Docker tasks.
license: Apache-2.0
owner: NVIDIA
service: paidf-simulation
version: 1.0.0
reviewed: '2026-05-30'
metadata:
  author: NVIDIA
  tags:
  - physical-ai
  - synthetic-data-generation
  - simulation
  - pcba
  - aoi
---

# simulation

PAIDF (Physical-AI Data Factory) Simulation. One skill, two tracks
selected by a Stage-0 router. All per-track stage detail is in
`references/<track>/`.

```
user prompt
    |
    v
Stage 0 — Router
    |
    +── single-flow ──► references/single-flow/stages.md
    |     flows: good / good_fixed / defect / missing / lighting / paired
    |
    +── roi ─────────► references/roi/stages.md
          variants: Day-0 (synth only) / Day-1 (synth + real + MI)
```

## Instructions

Apply the Stage-0 router to the user prompt first, then hand off to the matching reference doc under `references/<track>/`. The router emits a small JSON intent record; downstream stages parse the user yaml(s), surface a summary, gate on approval, and emit docker run commands. The skill never executes docker on its own — it only emits the commands and waits for the user.

### Stage 0 — Router

Apply these rules to the user's prompt **in order, first match wins**.
Rule 1 is the safety-critical gate that prevents silent over-delegation
to the ROI track (the failure mode that reverted commit 18cd8d5).

1. **ROI track gate — explicit keywords only.** Trigger if the prompt
   contains any of:
   - `cad2roi`, `usd2roi`, `usd-to-real`, `usd to real`
   - `ROI` (case-insensitive; standalone token), `per-ROI`, `per-component crop`
   - `MI registration`, `mutual information`, `align to real photo`, `register to <photo>`
   - `bridge crops`, `bridge between pads / solders`
   - `anchor on <component class>`, `crop <class> ROIs`
   - `scan_grid plus per-cell crop`, `pure-synthetic ROI`
   - `mesh-level semantic rules`
   → **track = roi.** Then sub-route by photo:
     - prompt names a real photo (`--photo-path`, `real.png`, "AOI screenshot", "PHOTO_DIR=...") → **Day-1**
     - otherwise → **Day-0**
     - ambiguous → ASK "Do you have a captured real PCB photo to align against, or only the CAD-derived USD?"
   → Hand off to `references/roi/stages.md`.

   **NOT triggers for the ROI track** (these stay single-flow):
   - A bare photo path mentioned without any crop / register / align verb
     (e.g. `"generate 9 good images for /tmp/photo.png"`).
   - A component class mentioned without a crop verb
     (e.g. `"render images of capacitors"`).
   The user must say what to DO with the photo or class. This is the
   `74eef40` rule, preserved verbatim.

2. **Paired / ChangeNet** — prompt contains `paired`, `golden+defect`,
   `golden / defect`, `ChangeNet`, `siamese`, `pair dataset` →
   **track = single-flow**, sub-mode = paired. Run the `good` flow + the
   `defect` flow with the same `random_seed`; post-process via
   `build_pair_dataset.py`. Hand off to `references/single-flow/stages.md`
   (Stage 1 Rule 1).

3. **Defect intent** — prompt contains `defect`, `pose defect`, `shift`,
   `tombstone`, `sideflip`, `polarity`, `reverse polarity`, `missing`,
   `hidden component`, `two-pass` → **track = single-flow**, flow =
   `defect` (or `missing` for the missing/hidden keywords).

4. **Zoom / close-up / fixed-camera** — prompt contains `zoom`,
   `close-up`, `single <component>`, `fixed camera`,
   `solder fillet randomization`, `tin pad`, `vantablack` →
   **track = single-flow**, flow = `good_fixed`.

5. **Lighting demo / variant** (explicit demo intent) — prompt contains
   `lighting demo`, `lighting variant` plus a named variant (`ring light`,
   `dome light`, `scene lights`, `preserve color`) → **track = single-flow**,
   flow = `lighting`.

6. **Default catch-all** — `good`, `clean`, `defect-free`, `no defects`,
   or anything else (including bare `generate` / `render` /
   `make images`) → **track = single-flow**, flow = `good`.

7. **Ambiguous** — if no rule matches confidently OR Rule 1 / 5 needs
   disambiguation (variant unclear, lighting variant unclear), ASK ONE
   question before any further work. Do NOT emit docker commands or
   write YAML in the disambiguation turn.

Lighting parameter is **orthogonal to the flow** in the single-flow
track — if the user names a lighting mode anywhere in the prompt, apply
it as an override on top of the chosen flow's base config. Default when
nothing is named: scene-authored lights. Full table in
`references/single-flow/stages.md` §Stage 1.

### Normalized router output

The router emits this record before handing off:

```json
{
  "track":   "single-flow | roi",
  "flow":    "good | good_fixed | defect | missing | lighting | paired",   // single-flow only
  "variant": "day0 | day1",                                                // roi only
  "raw_intent": { "...": "rest of the canonical intent schema for the chosen track" }
}
```

Then load `references/<track>/stages.md` and execute the rest of the
pipeline from Stage 1 onward.

### Track summary

| Track | Pipeline | Scripts | Output |
|---|---|---|---|
| single-flow | parse → overrides → derive YAML → docker run `sdg_pipeline.py` → (when `crop.enabled` in the YAML) docker run `crop_components.py` | `scripts/sdg/standalone/sdg_pipeline.py` + `scripts/postprocess/crop_components.py` | `sdg_test_output/<slug>/trigger_NNNN/rgb_NNNN.png` (+ seg / bbox), plus `sdg_test_output/<slug>/cropped/<label>/rgb/...` when crop ran |
| roi | validate → prepare → render → (register) → crop | Day-0: `sdg_pipeline.py` + `usd2roi_crop.py`; Day-1: `usd2roi_render.py` + `usd2roi_register.py` + `usd2roi_crop.py` | `<out>/crop/component/.../normal_img/*.png` (+ seg / aligned / bridge for Day-1) |

Single-flow `crop.enabled` defaults: `true` for good / defect / missing
(crops are usually the training target); `false` for good_fixed (camera is
already on one component) and lighting (full-board visual QA). Override
via prompt: "no crop" / "with crop". Full knob list in
`references/single-flow/overrides.md` §`crop` block.

Both tracks share the same Docker image (`paidf-simulation:<tag>`), the same `pcba_target.yaml` shape, the same
`<usd-assets>/` mount convention, and the same approval-gate discipline.

## Examples

| User prompt | Router output |
|---|---|
| "generate 5 defect images, only tombstone" | track=single-flow, flow=defect |
| "render missing-component frames" | track=single-flow, flow=missing |
| "zoom in to a capacitor with dome light" | track=single-flow, flow=good_fixed, lighting=dome |
| "generate 50 paired golden/defect images for ChangeNet" | track=single-flow, flow=paired |
| "lighting demo with ring light" | track=single-flow, flow=lighting, variant=ring_light |
| "crop ROIs from the spark board USD" | track=roi, variant=day0 (no photo) |
| "register the synth render to /tmp/real.png and emit per-ROI crops" | track=roi, variant=day1 |
| "cad2roi Day-1 with bridge crops" | track=roi, variant=day1 |
| **"generate 9 good images for /tmp/photo.png"** | **track=single-flow, flow=good** (photo path alone is NOT an ROI trigger — Rule 1's "NOT triggers" clause) |
| "render images of capacitors on the spark board" | track=single-flow, flow=good (component mention without crop verb is NOT an ROI trigger) |
| "Crop ROIs from a board." | ambiguous within roi track → ASK photo-or-not |

## References

| File | Purpose |
|---|---|
| `references/single-flow/stages.md` | Stage 1-6 detail for single-flow renders (parse intent, extract overrides, validate prereqs, derive YAML, approve+execute, report) |
| `references/single-flow/routing.md` | Full natural-language → flow mapping |
| `references/single-flow/overrides.md` | Per-flow knob cheatsheet |
| `references/single-flow/local-mode.md` | When/how to use the Isaac-Sim host install instead of Docker |
| `references/single-flow/troubleshooting.md` | Single-flow error → fix table |
| `references/roi/stages.md` | 5-stage scaffold (Validate → Prepare → Execute → Verify → Report) shared by Day-0 and Day-1 |
| `references/roi/day0.md` | Day-0 variant detail (scan_grid + per-cell crop, synth only) |
| `references/roi/day1.md` | Day-1 variant detail (single config, MI registration, OSMO submission) |
| `references/roi/semantic-rules.md` | Glob syntax + multi-class-on-same-prim + dry-run helper |
| `references/roi/troubleshooting.md` | Per-stage failure modes for both variants |
| `<repo>/configs/cad2roi/day0/` | Day-0 yaml templates (in-tree under the repo root) |
| `<repo>/configs/cad2roi/day1/` | Day-1 yaml templates + OSMO workflow (in-tree under the repo root) |
| `<repo>/configs/cad2roi/spark/` | Spark worked example (in-tree under the repo root) |
| `evals/evals.json` | Merged eval set; each case carries an explicit `track:` tag |
