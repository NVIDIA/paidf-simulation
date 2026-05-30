> **Human-facing overview.** Agents should load [`SKILL.md`](SKILL.md) (the Stage-0 router + instructions). This README orients humans browsing the folder.

# simulation

PAIDF (Physical-AI Data Factory) synthetic-data generation. **One skill,
two tracks**, selected by a Stage-0 router in `SKILL.md`:

- **single-flow** — full-PCB image renders. Flows: `good`, `good_fixed`,
  `defect`, `missing`, `lighting`. Plus paired golden/defect for
  ChangeNet. Maps an English prompt onto one of
  `configs/{flow1*,flow2*,lighting_example}/...yaml`, applies
  overrides, writes the derived YAML to `configs/runs/`, then runs
  `sdg_pipeline.py` in Docker (default) or via the local Isaac-Sim
  install when the user says `local` / `fast`.
- **roi** — per-component crops from a CAD-derived USD. Day-0 (no real
  photo, synth-only) and Day-1 (real photo + MI alignment + synth /
  real / seg ROI triples + optional bridge crops). User authors the
  yaml(s) ahead of time; track reads, summarises, and runs.

## Triggers

The router routes based on the user's prompt:

| Prompt looks like… | Track |
|---|---|
| "generate N good / defect / missing images" | single-flow |
| "make 10 defect images with only tombstone" | single-flow |
| "render missing-component frames" | single-flow |
| "zoom in to a capacitor with solder fillet randomization" | single-flow |
| "lighting demo with ring light" | single-flow |
| "synth data for the IC board" | single-flow |
| "paired golden/defect for ChangeNet" | single-flow (paired sub-mode) |
| "cad2roi" / "usd2roi" / "crop ROIs" / "MI registration" / "bridge crops" | roi |

**The router does NOT silently delegate to the ROI track.** A bare
photo path or component class mention is not enough — the user must
explicitly say what to DO with it (crop / register / align). This is
the `74eef40` discipline, preserved as Rule 1 in `SKILL.md`.

## Files

| Path | Purpose |
|---|---|
| `SKILL.md` | Stage-0 router (loaded into context when the skill is invoked) |
| `references/single-flow/stages.md` | Stage 1-6 detail for the single-flow track |
| `references/single-flow/{routing,overrides,local-mode,troubleshooting}.md` | Per-knob detail for single-flow |
| `references/roi/stages.md` | Five-stage scaffold shared by Day-0 and Day-1 |
| `references/roi/{day0,day1,semantic-rules,troubleshooting}.md` | ROI variant + helper detail |
| `configs/cad2roi/{day0,day1,spark}/...` | Day-0 / Day-1 yaml templates + spark worked example. `<repo>/configs/cad2roi/` is the source of truth for ROI yaml templates; the skill body references those paths directly (no symlink indirection from inside the skill folder, so the nvcarps harbor sandbox can copy the skill cleanly for Tier 3 eval). |
| `evals/evals.json` | Merged eval set; each case tagged `track: single-flow | roi` |

## Why this exists

The skill is the natural-language entry point for any SDG work in this
repo. Without it, end-users would need to: choose the right canonical
YAML by hand, remember which `defects.*.enabled` flag corresponds to
"only tombstone", know whether their request is a `flow2_defect_image`
or a `usd2roi` problem, and assemble a multi-mount `docker run` line.
This skill turns those English requests into the right canonical + diff
+ command without making the user know any of that.


End-to-end repo guide: [README.md](../../../README.md) at repo root.
