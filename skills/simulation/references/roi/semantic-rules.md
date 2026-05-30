# Semantic rule syntax

Shared by both Day-0 and Day-1. `semantics:` is a list of rules; each
rule has a `match:` glob and a `labels:` map.

## Glob syntax (`match:` patterns)

`match` patterns translate to a regex anchored at the full prim path:

- `*` — any chars within one path segment (no `/`)
- `**` — any chars including `/` (zero or more)
- `?` — single char within a segment

Common idioms:

- `**/_0402_*` — every prim whose last segment starts with `_0402_`,
  anywhere in the scene
- `**/IC_/tn__1151581000_2*` — every IC instance of that part number
- `**/_0805U_H150_*/LibRef/.../surface_0` — `surface_0` mesh of every
  0805U_H150 instance

USD instances are crossed transparently: when a match lands inside a
USD-instanced subtree, the topmost `IsInstance() == True` ancestor is
implicitly uninstanced (`SetInstanceable(False)`) so the descendant
becomes labellable. Cost on a large board: a few seconds plus a few
hundred MB of RAM; rendered pixels are unchanged.

## Multi-class on the same prim

Replicator's `mode='add'` keeps multiple `(type, value)` tuples on the
same prim. To tag a prim with both `class: ic` AND
`class: whole_component`, write two rules with the same match path:

```yaml
semantics:
  - match: "**/IC_/tn__1151581000_2*"
    labels: {class: ic}
  - match: "**/IC_/tn__1151581000_2*"
    labels: {class: whole_component}
```

The rendered seg label appears as `ic,whole_component`; the per-color
mapping is recorded in `semantic_segmentation_labels_*.json`.

## Dry-run helper

`scripts/usd2roi/semantic_rules.py` reports how many prims each rule
matches without booting Kit. Always run it before paying for a Stage 1
render — a rule that matches `0 prim(s)` means the glob is wrong or
the prim doesn't exist in this scene.

Day-0 invocation:

```bash
docker run --rm -v $PCB:/workspace/paidf-simulation --entrypoint python3 \
  ${SDG_IMAGE} \
  scripts/usd2roi/semantic_rules.py \
    --scene <scene-path> \
    --rules <resolved-yaml> \
    --show 5
```

Day-1 invocation (granular mounts, scene under `<usd-assets>/`):

```bash
docker run --rm \
  -v $ASSETS:/workspace/paidf-simulation/<usd-assets>:ro \
  -v $CONFIG:/workspace/paidf-simulation/<day1-config>:ro \
  --entrypoint python3 ${SDG_IMAGE} \
  scripts/usd2roi/semantic_rules.py \
    --scene <usd-assets>/temp_scene.usd \
    --rules <day1-config>/usd2roi_target.yaml \
    --show 5
```

## Worked example — spark

[`configs/cad2roi/spark/semantics.yaml`](../../../../configs/cad2roi/spark/semantics.yaml)
ships ~21 rules covering capacitor / solder / pad / ic on the spark
board (~5800 prims affected). Paste it verbatim for the spark scene;
adapt the same pattern (one rule per surface variant; multi-class via
repeated match path) when authoring rules for a new board.

When the user asks for a new board, **do not invent rules from
guesswork** — ask them to enumerate which prim-path globs map to which
class labels first, then iterate via the dry-run helper.
