# Troubleshooting

Symptoms first; fixes second.

## `Config boundary violation: keys [...] appear in both --config and --pcba-config`

`sdg_pipeline.py` enforces a strict split: USD/scene-bound fields go in the pcba_target YAML (passed via `--pcba-config`); pipeline/render/lighting/defect settings go in the main config (passed via `--config`). Overlap raises.

**Fix:** move each listed key to exactly one file. Usually the `--config` side wins — delete the duplicate from the pcba_target.

## `KeyError: 'horizontal_aperture'`

You hit team's pre-aperture-optional sdg_pipeline.py (or an older container image). The fix landed in `fb3ce35`:

```python
if "horizontal_aperture" in CFG:
    cam.GetHorizontalApertureAttr().Set(float(CFG["horizontal_aperture"]))
```

**Fix:** rebuild the docker image from the current branch, or set `horizontal_aperture: 200.0` explicitly in the run YAML.

## `Unknown component_types keyword: 'X'`

`component_types: X` in pcba_target.yaml was neither `ALL` nor `0` nor a key in `configs/components.yaml` `subsets:`.

**Fix:** use `ALL` for the full list, `0` for empty, or write a literal list. Or add `X` under `subsets:` in components.yaml.

## `omni.warp.core` startup error / `module 'warp' has no attribute 'int32'`

Older Isaac-Sim images had a warp circular-import race. The shim at `scripts/sdg/standalone/run_sdg_standalone.py` pre-imported warp to dodge it. The new code path (`sdg_pipeline.py` invoked directly via Kit's `--exec`) avoids the race entirely.

**Fix:** make sure you invoke through the image's entrypoint (`paidf-simulation:<tag>` with `"<script> <args>"` as the trailing arg), NOT `--entrypoint /isaac-sim/python.sh ... run_sdg_standalone.py`.

## Docker: `unable to find user isaac-sim: no matching entries in passwd file`

Some recent images (`paidf-simulation:fix-test`) were built without the `isaac-sim` user entry in `/etc/passwd` despite the Dockerfile setting `USER isaac-sim`. Bug in the image build.

**Fix:** use `paidf-simulation:local-sqa-test` (the validated tag) instead, or pass `--user $(id -u):$(id -g)` to docker run.

## Pipeline runs but `output/` is empty

Output landed somewhere else. Check the log for `[Pipeline] Output: <path>`. With our placeholder convention, the path is `${PAIDF_SIM_ROOT}/sdg_test_output/<flow>` — not the run dir directly.

```bash
find $PAIDF_SIM_ROOT/sdg_test_output/<slug>/trigger_0000/ -name "rgb_*.png" | wc -l
```

## Only `rgb_0000..rgb_0003.png`, last frame is `rgb_0004.png` and 0 bytes

Semantic-segmentation segfault workaround interaction — the writer reports N frames but the last gets truncated on close. Pre-existing; commit `a95a633` documented the workaround.

**Fix:** pad your `max_image_count` by 1 ("ask for 6, get 5"), OR set `writer.semantic_segmentation: false`.

## "Loaded pcba target" prints but `component_types` is still a literal string

`--pcba-config` merged the pcba_target, then the keyword resolver should have expanded `ALL` / `0`. If you see `Resolved component_types keyword 'X'` missing from the log:

- Check the host's `scripts/sdg/standalone/sdg_pipeline.py` has the resolver block (look for `Resolved component_types keyword`).
- Make sure `configs/components.yaml` exists at the expected path (`$PAIDF_SIM_ROOT/configs/components.yaml`).
- Inside Docker, the path resolution uses `Path(__file__).resolve().parent.parent.parent.parent / "configs" / "components.yaml"` which means `/workspace/paidf-simulation/configs/components.yaml`. Confirm the bind-mount put `configs/` there.

## Asset references break — `pcba_main_s_detail.usd not found`

`spark_lighting.usd` references peer USDs (`pcba_main_s_detail.usd`, `pcb.usd`, `aoi_ring_light.usda`) via relative paths. Moving the scene file alone leaves the references dangling.

**Fix:** point `PCB_USD_PATH` at your `spark_lighting.usd` from the canonical asset bundle (which includes all peer USDs), OR copy the entire `assets/` dir as a unit.
