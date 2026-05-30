# `examples/usd_roi_examples/`

Per-board CAD-to-ROI examples consumed by the roi track of the `/simulation` skill (and the OSMO workflow). Two boards × two flow modes:

| Folder | Board | Real photo? | Tooling |
|---|---|---|---|
| [`0603_H100_day0/`](0603_H100_day0/) | Spark , 0603 capacitor anchor | no | `sdg_pipeline.py` (good flow) + `usd2roi_crop.py` |
| [`0603_H100_day1/`](0603_H100_day1/) | Spark , 0603 capacitor anchor | yes | `usd2roi_render.py` → `usd2roi_register.py` → `usd2roi_crop.py` |
| [`115_2819_000_day0/`](115_2819_000_day0/) | Spark, IC, 115_2819_000 | no | same as 0603 day0 |
| [`115_2819_000_day1/`](115_2819_000_day1/) | Spark, IC, 115_2819_000| yes | same as 0603 day1 |

These yamls are **templates**: every absolute path is a `__TOKEN__` placeholder (`__SCENE__`, `__OUTPUT__`, `__MAX_IMAGE_COUNT__`, `__REAL_IMAGE__`) patched at job submit time. They are not directly runnable via `sdg_pipeline.py` until you `sed` the placeholders.

## Day-0 vs Day-1 in one glance

**Day-0** (no real photo): one `good`-flow scan over the PCBA producing a label-grid of `(rgb, semantic_seg)`, then a class-anchored crop per cell. Use when you only have a CAD-derived USD and you'll style-transfer the synth to look like real later.

**Day-1** (real photo supplied): single ortho capture, register the synth onto the real photo, then per-ROI triples `(synth crop, real crop, semantic seg crop)`. Use when you have a matching AOI machine photo and want a registered pair.

See each board's own README for token list, the command we used, and the expected output tree.

## Real images

For Day-1 runs, the assets live under `assets/input_real_image/`:

```
0603_H100.jpg
115_2819_000.jpg
```

Each Day-1 example's `__REAL_IMAGE__` placeholder maps to one of these JPGs (board → matching file).
