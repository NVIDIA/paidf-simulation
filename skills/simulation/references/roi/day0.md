# Day-0 variant — stage detail

Day-0 is the **no real photo** path. Two independent stages, each driven
by one author-once yaml:

```
[scene.usd] + semantics:
  ── sdg_pipeline.py     (Kit, GPU)  ── <out>/trigger_NNNN/{rgb,seg,labels}_x*_y*
  ── usd2roi_crop.py     (host py)   ── <out>/crop/component/<x*_y*>/{normal_img,cad_mask}/
```

Use when the user has a CAD-derived USD but **no captured real photo**,
and wants per-ROI crops anchored on labelled components for training or
for validating semantic-rule coverage.

## Prerequisites (Day-0 specifics)

All scripts (`scripts/sdg/standalone/*.py`, `scripts/usd2roi/*.py`) are
baked into the container image — **no `git clone` required**. The user
provides assets, two yamls, and an output dir. Each is bind-mounted to a
fixed container path; the yamls refer only to container paths.

| Resource | How to satisfy |
|---|---|
| USD scene folder | Download per repo README; unzip to any host dir, e.g. `~/sdg-day0/<usd-assets>/`. The whole folder is mounted; the render yaml's `scene:` (via `--pcba-config`) references one USD inside it as `<usd-assets>/temp_scene.usd`. |
| Day-0 config dir | Copy [`configs/cad2roi/day0/sdg/day0_image.yaml`](../../../../configs/cad2roi/day0/sdg/day0_image.yaml) and [`configs/cad2roi/day0/usd2roi/day0_crop.yaml`](../../../../configs/cad2roi/day0/usd2roi/day0_crop.yaml) into a host dir, e.g. `~/sdg-day0/<day0-config>/`. The whole dir is mounted into `/workspace/paidf-simulation/<day0-config>/`. |
| PCBA target yaml | Copy [`configs/cad2roi/spark/pcba_target.yaml`](../../../../configs/cad2roi/spark/pcba_target.yaml) into the same `$CONFIG` dir (or hand-author one for a non-spark board). |

The shared base (container image, GPU + OptiX, output dir chmod) is in
SKILL.md §Prerequisites.

## Validate (Day-0 fields)

Day-0 expects the user to have authored the two yamls already. Skill's
job: **read them, surface a summary, ask the user to confirm before
running**. Pull from `$ARGUMENTS`:

- `$ASSETS` — host dir containing the USD scene + supporting USDs
- `$CONFIG` — host dir containing `day0_image.yaml`, `day0_crop.yaml`, and the `--pcba-config` yaml (e.g. `pcba_target.yaml`)
- `$OUTPUT` — writable host dir for the run's render + crop outputs
- Optional run name (used only for log lines, not for paths)

If any of the above is missing, ask. Do NOT auto-generate semantics
rules — point the user at [semantic-rules.md](semantic-rules.md) and the
spark example if their `day0_image.yaml`'s `semantics:` block is empty.

## Prepare (Day-0)

> **Note — env-var expansion is asymmetric across Day-0's two stages.**
> Stage 1 (`sdg_pipeline.py`) expands `${VAR}` in YAML at load, so
> `day0_image.yaml` can use `${PCB_USD_PATH}` / `${PAIDF_SIM_ROOT}` /
> `${OUTPUT}` placeholders. Stage 2 (`usd2roi_crop.py`) does **not** expand
> env vars — `day0_crop.yaml`'s `output.dir` (and any other path field)
> must be a **literal string**. Hand-edit it for each run, or pre-substitute
> with `envsubst < day0_crop.yaml > /tmp/derived.yaml` before invoking
> the crop script.

1. Read `$CONFIG/day0_image.yaml` and `$CONFIG/day0_crop.yaml`. Surface
   their key fields:

   ```
   Scene (via pcba_target.yaml):  <usd-assets>/temp_scene.usd
   Scan grid:                     10 × 10  (≈ 100 cells)
   Resolution:                    [1024, 1024]
   Semantics:                     21 rules (capacitor / solder / pad / ic)
   Anchor classes (crop.classes): [capacitor, ic]
   Output:                        $OUTPUT  (mounts to /workspace/paidf-simulation/sdg_test_output)
   ```

2. **Dry-run the semantic rules** before paying for a Kit boot:

   ```bash
   docker run --rm \
     -v $ASSETS:/workspace/paidf-simulation/<usd-assets>:ro \
     -v $CONFIG:/workspace/paidf-simulation/<day0-config>:ro \
     --entrypoint python3 ${SDG_IMAGE} \
     scripts/usd2roi/semantic_rules.py \
       --scene <usd-assets>/temp_scene.usd \
       --rules <day0-config>/day0_image.yaml \
       --show 5
   ```

   Each rule prints how many prims matched. If any rule reports
   `0 prim(s)`, the glob is wrong or the prim doesn't exist — fix
   `day0_image.yaml` before Stage 1.

3. Pre-create output dir on host:

   ```bash
   mkdir -p $OUTPUT && chmod 777 $OUTPUT
   ```

Then return to SKILL.md §Prepare for the shared **APPROVAL GATE**.

## Execute (Day-0 docker commands)

```bash
# Host paths (user supplies)
ASSETS=~/sdg-day0/<usd-assets>
CONFIG=~/sdg-day0/<day0-config>
OUTPUT=~/sdg-day0/output

IMAGE=${SDG_IMAGE}
IMG_YAML=<day0-config>/day0_image.yaml      # container-relative
CROP_YAML=<day0-config>/day0_crop.yaml      # container-relative
PCBA_YAML=<day0-config>/pcba_target.yaml    # container-relative

# Granular mounts — scripts come from the image, only user-supplied
# paths are bind-mounted. <usd-assets> / _config are read-only; output is rw.
MOUNTS="-v $ASSETS:/workspace/paidf-simulation/<usd-assets>:ro \
        -v $CONFIG:/workspace/paidf-simulation/<day0-config>:ro \
        -v $OUTPUT:/workspace/paidf-simulation/sdg_test_output"

# Stage 1 — render labelled scan_grid (Kit; ~9-12 min cold boot + render)
docker run --rm --gpus all --network host \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  $MOUNTS $IMAGE \
  "scripts/sdg/standalone/sdg_pipeline.py --config $IMG_YAML --pcba-config $PCBA_YAML"

# Stage 2 — multi-cell anchor crop (host python; seconds)
docker run --rm \
  $MOUNTS --entrypoint python3 $IMAGE \
  scripts/usd2roi/usd2roi_crop.py --config $CROP_YAML
```

Stage 1's Kit entrypoint expects script + `--config` + `--pcba-config`
as **one** quoted string (the `bash -c "kit ... --exec \"$@\""` shape).
Stage 2 overrides the entrypoint with `python3`; args go directly.

Idempotency:
- Stage 1 overwrites `trigger_0000/` outputs under `output:`. The rename
  step at trigger end is also idempotent.
- Stage 2 overwrites `crop/component/` and `crop/bridge/`.

## Verify (Day-0 output inspection)

```bash
# Stage 1: every cell has rgb + seg + labels
ls $OUTPUT/trigger_0000/ | head
# expected per cell: rgb_x*_y*.png, semantic_segmentation_x*_y*.png,
#                    semantic_segmentation_labels_x*_y*.json

# Stage 2: per-cell ROI subdirs
ls $OUTPUT/crop/component/ | wc -l           # ≈ scan_grid cells (off-board cells empty)
find $OUTPUT/crop/component -path '*/normal_img/*.png' | wc -l   # total ROIs
ls $OUTPUT/crop/component/x0_y0/normal_img/ | head
```

Stage 2 logs a summary line:
`emitted=N skipped_min=A skipped_max=B skipped_edge=C skipped_low_coverage=D n_components_total=M`.

Typical ROI counts vary with `scan_grid` and asset pack. Verified
samples on spark scene:
- 10 × 10 grid + slim `<usd-assets>/`:   ~2150 ROIs
- 10 × 10 grid + full `__assets/`:  ~2100 ROIs (cleaner: BoardMatOverride hits all 3 substrate prims)
- 10 × 1 grid (smoke test):         ~170 ROIs

## Report (Day-0)

Surface:
- total ROIs emitted (and how many cells were empty / off-board)
- skip counts from the crop log line (`skipped_min`, `skipped_edge`,
  `skipped_low_coverage`)
- sample paths to spot-check (e.g.
  `crop/component/x5_y5/normal_img/0001.png` + the matching
  `cad_mask/0001_cad_mask.png`)

If `emitted` is 0 across the board, the most common root cause is the
`semantics:` rules didn't actually match any prims — re-run the dry-run
in Prepare to find rules with `0 prim(s)`. See
[troubleshooting.md](troubleshooting.md) for the full failure-mode
table.
