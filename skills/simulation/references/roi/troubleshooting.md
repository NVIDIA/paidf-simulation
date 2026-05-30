# Troubleshooting — roi track

Per-stage failure modes for both variants. Rows are tagged `[day0]`,
`[day1]`, or `[day1-osmo]`. Stage labels match each variant's pipeline:

- Day-0: Stage 1 = `sdg_pipeline.py` (Kit); Stage 2 = `usd2roi_crop.py`.
- Day-1: Stage 1 = `usd2roi_render.py` (Kit); Stage 2 = `usd2roi_register.py`; Stage 3 = `usd2roi_crop.py`.

## Pre-flight (both variants)

### `ls /usr/share/nvidia/nvoptix.bin` returns no such file `[day0][day1]`

GPU driver missing OptiX. Stage 1 silently produces blank frames
without it.

Fix: install NVIDIA driver ≥ 525 with OptiX.

### `docker pull` returns 401 / 403 `[day0][day1]`

Not logged in to NGC.

Fix: `docker login nvcr.io` with the NGC API key from build.nvidia.com.

### Container uid 1234 cannot read host file `[day0][day1]`

Default uid mismatch between host user and container user.

Fix:
```bash
chmod -R o+rX $ASSETS $CONFIG $PHOTO_DIR   # all read-only host dirs you bind-mount
```

For the output dir, use the in-container chmod helper (Day-0) or
pre-`chmod 777 $OUTPUT` on the host (Day-1).

## Validate / Prepare

### Dry-run reports `0 prim(s) affected` for a rule `[day0][day1]`

Glob doesn't match the scene's real prim hierarchy.

Fix: open the USD and confirm the prim path; common gotcha is
instance-proxy paths look right but actual paths use `LibRef/`
segments. Adjust the `match:` glob. See
[semantic-rules.md](semantic-rules.md).

### `[Pipeline] mesh semantics: 0 rule(s)` `[day0]`

`semantics:` block empty in `$CONFIG/day0_image.yaml` (or rule globs
match zero prims).

Fix: edit `day0_image.yaml` and add rules per the spark example +
`semantic-rules.md`; dry-run via `semantic_rules.py --show 5` to
confirm each rule matches before paying for a Kit boot.

## Stage 1 — render

### `argparse: error: the following arguments are required: --config` `[day0]`

Stage 1 args passed as separate argv to docker instead of one quoted
`--exec` string.

Fix: pass `"<script> --config X --pcba-config Y"` as **one** quoted
argument. See [day0.md](day0.md) §Execute.

### `Cannot open USD: <path>` `[day1]`

The scene file (`scene:` in the YAML) is unreachable from the
container.

Fixes:
1. Confirm the path is correct relative to the container WORKDIR
   (`/workspace/paidf-simulation`). With granular mounts, `<usd-assets>/foo.usd` on
   host appears at `/workspace/paidf-simulation/<usd-assets>/foo.usd`.
2. `chmod -R o+rX $ASSETS`.

### Stage 1 trigger writes `rgb_0000.png` instead of `rgb_x0_y0.png` `[day0]`

`rename_to_grid_index: true` not set in the resolved SDG yaml. The
Day-0 crop template's `pattern: 'x*_y*'` requires the spatial naming.

Fix: set `rename_to_grid_index: true`.

### Stage 1 writes seg + bbox but no `rgb_0000.png` `[day1]`

Known annotator init race at higher resolutions. The render script
already retries once, but a fully cold container can still miss.

Fix: mount the Kit cache so a second invocation runs warm:
```bash
mkdir -p ~/.cache/ov ~/.local/share/ov
docker run ... \
  -v ~/.cache/ov:/home/isaac-sim/.cache/ov \
  -v ~/.local/share/ov:/home/isaac-sim/.local/share/ov \
  ...
```
Re-run the stage; warm boot reduces the race window.

### Render is on a pure black background (no PCB substrate) `[day1]`

`scene` points at a USD that references components but not the PCB
substrate (typical mistake: `<usd-assets>/pcba_main_s_detail.usd` without
the composition over `<usd-assets>/pcb.usd`).

Fix: switch `scene` to the composition file that includes both, e.g.
`<usd-assets>/temp_scene.usd`. Re-render and re-register.

## Stage 2 — register (Day-1 only)

### `Stage 2 exit=2` and `aligned/` is empty `[day1]`

`mi_after < registration.min_mi` (default 0.5). The register script
exits before writing any output when MI is below threshold.

Diagnosis:
1. Open `sdg/rgb_0000.png` and compare to `real_image`. Same physical
   region of the board?
2. If not, adjust `camera.translate` / `camera.horizontal_aperture` /
   `resolution` so the synth covers the same region.
3. If yes but MI is still low, the slim `<usd-assets>/` may be missing the
   IC mesh; the visual gap pushes MI to ~0.45 on the spark scene. Use
   the full `assets/` pack OR lower `registration.min_mi: 0.4` (the
   alignment is still usable; the threshold is a safety net, not a
   hard quality gate).

### Stage 2 reports `gpu=False (cupy available: False)` `[day1]`

Container image lacks cupy. CPU registration works but takes 3-5 min
instead of ~20 s.

Fix: post-MR-10 main-line images (any tag containing `.main` built after the `shangru/sqa-container-fix` merge) bundle
cupy. If still false on a main-line image, check the host driver is
≥ 525 and that `--gpus all` is passed. If GPU path can't be fixed,
accept the CPU runtime.

### `sX/sY` converges to the grid edge `[day1]`

Search range too narrow.

Fix: widen `registration.sx_range` / `sy_range`. Typical good range
is `0.9 → 1.1 (step 0.02)`; tighten to `0.95 → 1.05 (step 0.01)` with a good
prior.

### MI plateau is < 0.1 even after fixing scene + camera `[day1]`

Synth and real really don't overlap. Most common with the wrong scene
file (e.g. rendering the wrong board), or a camera centre far outside
the photographed region.

Fix: visually compare `sdg/rgb_0000.png` and `real_image`; iterate on
camera parameters until you see roughly the same components in both.

## Stage 2 / 3 — crop

### Stage 2 emits 0 ROIs everywhere `[day0]`

`crop.classes` doesn't intersect any class authored by `semantics:`.

Fix: open `semantic_segmentation_labels_*.json`; align `crop.classes`
with the anchor classes you actually authored.

### `emitted=0 n_components_total=0` `[day1]`

Either Stage 1 produced no seg labels (annotator off) or Stage 2
dropped the seg crop.

Fix: open `aligned/semantic_segmentation_labels_0000.json`. If
present, check `crop.classes` intersects ≥1 label value. If absent,
re-run Stage 1 with the seg annotator on.

### `emitted=0` but `n_components_total > 0` `[day1]`

All components filtered by `edge_skip` (bbox touches image border) or
`min_coverage` (< 20% labelled pixels in the crop box).

Fix:
- For edge_skip: widen `camera.horizontal_aperture` or move
  `camera.translate` so targets are not at the frame edge.
- For min_coverage: lower the threshold to 0.1 (or 0) for sparser
  crops.

### Stage 2 emits noise ROIs over blank board `[day0]`

`min_coverage` too low / off.

Fix: raise `min_coverage` (try `0.3`); confirm with per-frame skip
counts.

### Stage 2 emits cut-in-half components at cell borders `[day0]`

`edge_skip: false`, or `scan_grid` cells too small so components
straddle boundaries.

Fix: keep `edge_skip: true`; lower `x_num` / `y_num` so each cell
covers more of the board.

### Way too many tiny ROIs `[day1]`

`morph_kernel` too small (labelled pixels don't merge into one bbox
per component) or `min_area` too low.

Fix: bump `morph_kernel` to 3-4 px and `min_area` to 200+ px².

### Way too few ROIs vs labelled prims `[day0]`

`morph_kernel` too aggressive, merging anchors into one giant CC.

Fix: lower `morph_kernel` to `1`. Adjacent anchors should be
separated, not merged.

### Bridge mode emits implausible pairs `[day1]`

`bridge_classes` too broad or `bridge_dis` too generous.

Fix: restrict `bridge_classes` to the meaningful subset (typical
`[solder, pad]`) and lower `bridge_dis` to 20-30 px.

### `PermissionError: ...crop` in Stage 2 `[day0]`

Host user can't write into uid-1234-owned `$OUTPUT`.

Fix: run the chmod-helper docker step between Stage 1 and Stage 2
(already in [day0.md](day0.md) §Execute).

## Cleanup

### Host user cannot delete `<output>/sdg/` or `<output>/crop/` `[day1]`

Files owned by container uid 1234. Either chmod files or wipe via the
container:

```bash
find ~/paidf-simulation/sdg_test_output/day1_<run_name> -type f \
  -exec chmod 666 {} + 2>/dev/null

# or wipe everything as the container uid
docker run --rm -v ~/paidf-simulation:/workspace/paidf-simulation --entrypoint rm \
  ${SDG_IMAGE} \
  -rf /workspace/paidf-simulation/sdg_test_output/day1_<run_name>
```

## OSMO submission

### `render` task exits immediately with no output `[day1-osmo]`

OptiX init failure or no physical GPU available on the pool.

Fix: `osmo workflow logs cad2roi-<N> --task render`; confirm pool has
physical GPUs free (`osmo pool list --mode free`).

### `nvoptix.bin: No such file or directory` in render logs `[day1-osmo]`

OptiX binary missing from the node. **Pool-level fix**, not workflow
YAML. See [day1.md](day1.md) §Running on OSMO → `nvoptix.bin and the
pod template`.

### `cad2roi.yaml: dir:` sed patch has no effect `[day1-osmo]`

`output` block in `config.yaml` uses a different indent style.

Fix: confirm `dir:` is indented with exactly 2 spaces under `output:`
— `yaml.dump` from the configure task always produces 2-space indent.

### MI score `mi_after` < 0.05 `[day1-osmo]`

Camera `translate` or `aperture` is far from the real board region.

Fix: adjust `camera.translate` and `camera.horizontal_aperture` in
`config.yaml`; re-upload and resubmit. The `aligned/blink.gif` in the
register output shows the alignment visually.

### Workflow stuck in PENDING `[day1-osmo]`

Insufficient GPU quota or pool capacity.

Fix: `osmo workflow events cad2roi-<N>` to see the scheduling reason.
Submit with `--priority LOW` to use idle GPUs outside quota.

## Filing a new case

Capture:
1. Variant (day0 / day1 / day1-osmo) and the YAML(s) passed to all
   stages.
2. Full pipeline log for the failing stage (last 200 lines).
3. `aligned/params.json` if Stage 2 ran at all (Day-1).
4. `aligned/blink.gif` for visual context (Day-1).
5. Image tag in use.

Add a new heading under the matching stage section above with the
appropriate `[day0]` / `[day1]` / `[day1-osmo]` tag.
