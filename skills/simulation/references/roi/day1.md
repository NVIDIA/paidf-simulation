# Day-1 variant — stage detail

Day-1 is the **real photo supplied** path. Three independent stages
share one YAML config:

```
[scene.usd]   ── usd2roi_render.py    (Kit, GPU)         ── <out>/sdg/
[real.png]    ── usd2roi_register.py  (host python+cupy) ── <out>/aligned/
                usd2roi_crop.py       (host python)      ── <out>/crop/
```

Stage 1 runs Kit (default container entrypoint). Stages 2 and 3
override the entrypoint to plain `python3`. Re-run any stage
independently — no `--skip-*` flags.

Use when the user has a real PCB photo and wants per-ROI crops aligned
to it via mutual-information registration.

## Prerequisites (Day-1 specifics)

All scripts (`scripts/usd2roi/*.py`, `scripts/postprocess/*.py`) are
baked into the container image — **no `git clone` required**. The user
only provides assets, config, photo, and an output dir. Each is mounted
into the container at a fixed path; the yaml refers to those container
paths, not host paths.

| Resource | How to satisfy |
|---|---|
| USD scene folder | Download `<usd-assets>.zip` per the repo README; unzip to any host dir, e.g. `~/sdg-day1/<usd-assets>/`. The whole folder is mounted; the yaml's `scene:` references one USD inside it as `<usd-assets>/temp_scene.usd`. |
| Day-1 config YAML | Author from [`configs/cad2roi/day1/replicator/usd2roi_target.yaml`](../../../../configs/cad2roi/day1/replicator/usd2roi_target.yaml); save to a host dir, e.g. `~/sdg-day1/<day1-config>/usd2roi_target.yaml`. The whole dir is mounted into `/workspace/paidf-simulation/<day1-config>/`. |
| Real PCB photo dir | Save the AOI machine screenshot as `real.png` in a host dir, e.g. `~/sdg-day1/input/real.png`. Aspect should roughly match the planned `resolution` (keeps post-MI `sX, sY ≈ 1.0`). |

The shared base (container image, GPU + OptiX, output dir chmod) is in
SKILL.md §Prerequisites.

## Validate (Day-1 fields)

Pull from `$ARGUMENTS`:
- scene USD path (the CAD-derived USD; the verified spark scene is
  `<usd-assets>/temp_scene.usd`)
- real photo path
- camera ortho centre `(x, y)` in mm and `horizontal_aperture`
  (0.1 × world units, e.g. `97` ≈ 9.7 mm wide window)
- resolution that matches the real photo's aspect
- which semantic classes to crop (typical: `[capacitor, solder, pad, ic]`)
- bridge mode + `bridge_classes` + `bridge_dis` if needed
- output run name

If any of the above is missing, ask.

## Prepare (Day-1 substitution)

> **Note — env-var expansion does NOT apply here.** Day-1 uses
> `usd2roi_render.py` and `usd2roi_crop.py`, which read the YAML as-is and
> do **not** expand `${VAR}` placeholders the way `sdg_pipeline.py` does
> (single-flow track). Every field that holds a path (`scene`, `real_image`,
> `output.dir`, mount targets, etc.) must be a **literal string**. Either
> hand-edit the YAML for each run, or pre-substitute with `envsubst < tmpl.yaml > derived.yaml`
> and pass the derived file to `--config`. Setting `PCB_USD_PATH` in the
> shell alone is **not enough** for Day-1.

1. Copy [`configs/cad2roi/day1/replicator/usd2roi_target.yaml`](../../../../configs/cad2roi/day1/replicator/usd2roi_target.yaml)
   to `$CONFIG/usd2roi_target.yaml` (host) and edit:
   - `scene` — container-relative, e.g. `<usd-assets>/temp_scene.usd`
   - `semantics:` — `**/` portable glob style (see
     [semantic-rules.md](semantic-rules.md))
   - `camera.translate`, `camera.horizontal_aperture`
   - `resolution`
   - `real_image` — fixed at `scripts/usd2roi/input/real.png` (the mount
     places the photo there)
   - `crop.classes`, `crop.bridge*`
   - `output.dir` — fixed at `/workspace/paidf-simulation/sdg_test_output/` (the
     output mount lands there)

2. **Dry-run the semantic rules** before paying for a Kit boot:

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

Each rule prints how many prims matched. If any rule shows `0 prim(s)`,
the glob is wrong or the prim doesn't exist — fix before Stage 1.

3. Pre-create output dir on host (only writable mount target):

```bash
mkdir -p $OUTPUT && chmod 777 $OUTPUT
```

4. Show summary:

```
Scene:        <usd-assets>/temp_scene.usd
Real photo:   scripts/usd2roi/input/real.png
Camera:       translate x=-55.4 mm; y=2.0 mm; aperture=97 (≈ 9.7 mm window)
Resolution:   [640, 480]
Semantics:    21 rules (capacitor, solder, pad, ic, …)
ROI classes:  [capacitor, solder, pad, ic]
Bridge:       on (dis≤30 px, classes=[capacitor])
Output:       ~/paidf-simulation/sdg_test_output/day1_<run_name>
```

Then return to SKILL.md §Prepare for the shared **APPROVAL GATE**.

## Execute (Day-1 docker commands)

```bash
# Host paths (user supplies)
ASSETS=~/sdg-day1/<usd-assets>
CONFIG=~/sdg-day1/<day1-config>
PHOTO_DIR=~/sdg-day1/input         # must contain real.png
OUTPUT=~/sdg-day1/output

IMAGE=${SDG_IMAGE}
YAML=<day1-config>/usd2roi_target.yaml   # container-relative

# Granular mounts — scripts come from the image, only user-supplied
# paths are bind-mounted. <usd-assets> / _config / photo are read-only;
# output is read-write.
MOUNTS="-v $ASSETS:/workspace/paidf-simulation/<usd-assets>:ro \
        -v $CONFIG:/workspace/paidf-simulation/<day1-config>:ro \
        -v $PHOTO_DIR:/workspace/paidf-simulation/scripts/usd2roi/input:ro \
        -v $OUTPUT:/workspace/paidf-simulation/sdg_test_output"

# Stage 1 — render (Kit; ~5-7 min cold boot)
docker run --rm --gpus all --network host \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  $MOUNTS $IMAGE \
  "scripts/usd2roi/usd2roi_render.py --config $YAML"

# Stage 2 — register (host python+cupy on GPU; ~15-30 s)
docker run --rm --gpus all --network host \
  $MOUNTS --entrypoint python3 $IMAGE \
  scripts/usd2roi/usd2roi_register.py --config $YAML

# Stage 3 — crop (host python; seconds)
docker run --rm \
  $MOUNTS --entrypoint python3 $IMAGE \
  scripts/usd2roi/usd2roi_crop.py --config $YAML
```

Stage 1 expects script + `--config` as one quoted string (Kit `--exec`
shape). Stages 2 and 3 pass args directly because the entrypoint is now
plain `python3`.

Idempotency:
- Stage 1 overwrites `sdg/` outputs at the configured `output.dir`.
- Stage 2 overwrites `aligned/`. **However**, if the configured
  `registration.min_mi` (default `0.5`) is not met, Stage 2 exits with
  code 2 and writes nothing — Stage 3 is then skipped. Surface this
  clearly: either lower `min_mi`, fix the scene reference (most common
  with `<usd-assets>/` slim pack is a missing IC or PCB substrate), or
  re-author `camera.*` so the synth view actually overlaps the real
  photo.
- Stage 3 overwrites `crop/component/` and `crop/bridge/`.

## Verify (Day-1 output inspection)

```bash
ls $OUTPUT/                          # expect: sdg/  aligned/  crop/
ls $OUTPUT/sdg/                      # rgb_0000.png + seg + labels JSON + metadata
ls $OUTPUT/aligned/                  # ref_crop + aligned_crop + blink.gif + params.json + cropped seg
python3 -m json.tool $OUTPUT/aligned/params.json
echo "ROIs:    $(ls $OUTPUT/crop/component/normal_img/*.png 2>/dev/null | wc -l)"
echo "Bridges: $(ls $OUTPUT/crop/bridge/normal_img/*.png    2>/dev/null | wc -l)"
```

`params.json` carries `{scaleX, scaleY, rotation_deg, tx, ty, mi_before,
mi_after}`. Read `mi_after`:
- `> 0.5` reasonable on the slim `<usd-assets>/` (which misses the IC mesh)
- `> 0.7` excellent (typically only achievable with the full `assets/`
  pack)
- below `min_mi` threshold (default 0.5) → Stage 2 exited; see
  [troubleshooting.md](troubleshooting.md).

Open `$OUTPUT/aligned/blink.gif` in any image viewer that supports
animated GIFs to visually confirm `ref ↔ aligned` overlap.

## Report (Day-1)

Surface:
- `mi_after` from `params.json` and a one-line verdict (excellent /
  reasonable / poor)
- Number of ROIs and bridges emitted (`Stage 3 emitted=N skipped_min=X
  skipped_edge=Y skipped_low_coverage=Z`)
- Host path to `blink.gif` for visual QA

If any stage exited non-zero, surface the last 30 lines of its log and
link to [troubleshooting.md](troubleshooting.md).

## Running on OSMO

The workflow at [`configs/cad2roi/day1/osmo/workflow.yaml`](../../../../configs/cad2roi/day1/osmo/workflow.yaml)
maps the 3-stage Day-1 pipeline onto 4 serial OSMO tasks: `configure →
render → register → crop`.

OSMO automatically appends an incremental counter to the workflow name
on each submission (e.g. `cad2roi-1`, `cad2roi-2`). Output dataset
names are fixed (`cad2roi-config`, `cad2roi-render`, `cad2roi-aligned`,
`cad2roi-components`) — OSMO handles versioning server-side.

### 1. Upload input assets

Prepare a local directory with exactly these three files:

```
assets-dir/
├── scene.usd     # CAD-derived USD file
├── real.png      # real PCB photo to register against
└── config.yaml   # cad2roi config (based on configs/cad2roi/day1/replicator/usd2roi_target.yaml)
```

**`config.yaml` notes:**
- Leave `scene`, `real_image`, and `output.dir` as placeholder values
  — the workflow patches these at runtime with OSMO dataset paths.
- All other fields (`semantics`, `camera`, `resolution`,
  `registration`, `crop`, etc.) must be filled in correctly before
  uploading.

Upload the directory as a single OSMO dataset:

```bash
# The trailing /. avoids adding an extra directory level inside the dataset
osmo dataset upload cad2roi-assets /path/to/assets-dir/.
```

Verify:

```bash
osmo dataset list | grep cad2roi-assets
osmo dataset download cad2roi-assets /tmp/verify-assets && ls /tmp/verify-assets/
# Expected: scene.usd  real.png  config.yaml
```

### 2. Submit the workflow

```bash
osmo workflow submit configs/cad2roi/day1/osmo/workflow.yaml --pool <pool-name>
```

Find a pool with GPU capacity:

```bash
osmo pool list --mode free
```

Choose a pool where `Effective = min(Quota Free, Total Free) ≥ 1`.

Submit multiple runs in parallel by uploading a separate dataset per
board, then `osmo workflow submit` once per board. The workflow always
reads from the `cad2roi-assets` dataset name; parameterise via `--set`
if you need to point at a different name.

### 3. Monitor progress

```bash
# List recent submissions to get the auto-assigned workflow name (e.g. cad2roi-3)
osmo workflow list --format-type json

# Overall status
osmo workflow query cad2roi-<N> --format-type json

# Live logs for a specific task (render is usually the longest)
osmo workflow logs cad2roi-<N> --task render -n 5000
```

Expected durations per task (1× L40S GPU):

| Task | Duration |
|------|----------|
| configure | < 1 min |
| render | 2–7 min (cold Kit boot: ~7 min; warm: ~2 min) |
| register | 15–30 s |
| crop | < 1 min |

### 4. Download results

```bash
osmo dataset download cad2roi-components ./output/
# Intermediate datasets, if needed:
osmo dataset download cad2roi-render  ./output/render/
osmo dataset download cad2roi-aligned ./output/aligned/
```

Expected output structure under `components/`:

```
components/
  rois/
    <roi_id>/
      ov_rgb.png       # synthetic crop
      real_rgb.png     # real photo crop (same region)
      ov_seg.png       # semantic segmentation crop
      bbox.json
  bridges/             # only if crop.bridge: true in config
```

### nvoptix.bin and the pod template

If the render task logs show anything related to OptiX or
`nvoptix.bin` not found, the fix is at the **pool level**, not in the
workflow YAML. Ask the pool admin to verify the pool's pod template
includes:

```yaml
volumes:
- name: nvoptix
  hostPath:
    path: /usr/share/nvidia/nvoptix.bin
    type: File
volumeMounts:
- name: nvoptix
  mountPath: /usr/share/nvidia/nvoptix.bin
  readOnly: true
```

One-time pool configuration — once in place, all cad2roi (and SDG)
workflows on that pool have OptiX available automatically.

OSMO-side failure modes are in
[troubleshooting.md](troubleshooting.md) under the `[day1-osmo]` tag.
