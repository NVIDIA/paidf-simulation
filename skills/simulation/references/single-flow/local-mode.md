# Local mode — host Isaac-Sim install

When the user says `local`, `fast`, `host run`, `skip docker`, or `--local`, run on the host's Isaac-Sim standalone install instead of in Docker. Faster startup (~10 s vs ~30 s container init), easier file access, brittler to host env drift.

## Prerequisites

Set `ISAAC_SIM_PATH` to your Isaac-Sim standalone install's `isaac-sim.sh` script. The skill ASKs if the env var is unset — it does not probe well-known paths.

```bash
export ISAAC_SIM_PATH=<install>/isaac-sim.sh
test -x "$ISAAC_SIM_PATH" && echo OK
```

If the launcher is absent, fall back to Docker with a one-line warning to the user.

## Invocation

`isaac-sim.sh` is a Kit wrapper — pass `--no-window --exec "<script> <args>"` and it launches Kit with the script.

```bash
"$ISAAC_SIM_PATH" --no-window --exec   "$PAIDF_SIM_ROOT/scripts/sdg/standalone/sdg_pipeline.py     --config $PAIDF_SIM_ROOT/configs/runs/<slug>.yaml     --pcba-config $PAIDF_SIM_ROOT/configs/pcba_target.yaml"
```

### Env vars

`PCB_USD_PATH` and `PAIDF_SIM_ROOT` must be set in the calling shell — `sdg_pipeline.py` reads them via `os.path.expandvars` after loading the YAML.

```bash
export PCB_USD_PATH=<path/to/your/board.usd>
export PAIDF_SIM_ROOT=$HOME/paidf-simulation
```

### Output dir

Default output: `${PAIDF_SIM_ROOT}/sdg_test_output/<flow>/` (e.g. `flow1_good_image`, `flow2_defect_image`, `lighting_ring_light`). The flow YAML's `output:` key controls this; override it per-run to keep history (e.g. `${PAIDF_SIM_ROOT}/sdg_test_output/<flow>_<timestamp>`).

Frames land under `<output>/trigger_NNNN/rgb_XXXX.png` + `semantic_segmentation_XXXX.png` + `bounding_box_2d_tight_XXXX.{npy,json}`.

`sdg_test_output/` is gitignored (under the top-level `/sdg_*/` rule in `.gitignore`).

### Streaming the run log

Tee to file + stdout so the user sees progress live while the log persists:

```bash
OUT_DIR=$PAIDF_SIM_ROOT/sdg_test_output/<flow>_<timestamp>
mkdir -p "$OUT_DIR"
"$ISAAC_SIM_PATH" --no-window --exec "..." 2>&1 | tee "$OUT_DIR/run.log"
```

## When to prefer local

- Iterating on a config (re-running 10×); cold-start savings dominate.
- The user explicitly asks for `fast` / `host` / `local`.
- Debugging — `print()` statements show up immediately in the live terminal, not in the docker-buffered stderr.

## When NOT to use local

- The Isaac-Sim install is missing or its bundled `omni.replicator.core` extension is stale (the SDG pipeline imports it).
- The user is on a host without an Isaac-Sim standalone install.
- The user explicitly wants the reproducible Docker environment for a release artifact.

## Caveats

- **No container isolation** — your run interacts with the user's Kit user-config and extscache directly. Crashes can leave temp files / locks behind under `~/.local/share/ov/`. Don't `rm -rf` anything there without asking.
- **No warp pre-import shim** is needed — Kit's `--exec` path loads extensions before invoking the script, so `omni.warp.core` is wired up by the time `sdg_pipeline.py` imports `omni.replicator.core`.
- **No path translation** — `${PCB_USD_PATH}` / `${PAIDF_SIM_ROOT}` resolve against the host file system. No bind-mount mapping. The same YAML the docker path uses works unchanged.
- **GPU contention** — local mode shares the host's compositor / Kit cache. Don't run two local renders simultaneously.
