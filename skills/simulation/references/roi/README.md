# references/roi/

Long-form docs the roi track loads on demand. Keep these specific (one
topic per file); don't duplicate `stages.md` (the shared five-stage
scaffold) or the top-level `SKILL.md` router.

| File | Purpose |
|---|---|
| [stages.md](stages.md) | Five-stage scaffold (Validate → Prepare → Execute → Verify → Report) shared by Day-0 and Day-1. |
| [day0.md](day0.md) | Day-0 variant stage detail: user-authored yaml inventory, granular host-mount contract, two-stage docker pipeline (`sdg_pipeline.py` → `usd2roi_crop.py`), output tree. Read when no real photo is supplied. |
| [day1.md](day1.md) | Day-1 variant stage detail: single config, three-stage docker pipeline (render → register → crop), MI quality bands, OSMO submission walkthrough. Read when a real photo is supplied. |
| [semantic-rules.md](semantic-rules.md) | Glob syntax for `match:` patterns, multi-class-on-same-prim pattern, `semantic_rules.py` dry-run invocation for both variants. Read whenever the user is authoring or debugging `semantics:` rules. |
| [troubleshooting.md](troubleshooting.md) | Per-stage failure modes for both variants. Rows tagged `[day0]`, `[day1]`, `[day1-osmo]`. Read on any non-zero exit or unexpected output. |

Pipeline schema (YAML field types + defaults) and crop algorithm
details live at `scripts/usd2roi/component_crop.py` (docstring) and
`scripts/usd2roi/README.md` in the repo, not duplicated here.

When adding a new doc, link it from `stages.md`'s Reference table so
the track knows it exists.
