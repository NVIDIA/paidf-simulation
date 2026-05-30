ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0-dev3
FROM ${BASE_IMAGE}

USER root

ARG AIOHTTP_VERSION=3.13.5
ARG PILLOW_VERSION=12.2.0

# Snapshot the base image's installed-package set before our first apt-get
# install. The final OSS-sources layer diffs against this to report the full
# new-package set (including transitive deps), so we can widen the source
# collection if OSS later asks for the full closure.
RUN mkdir -p /tmp/oss && \
    dpkg-query -W -f='${Package}\n' | sort > /tmp/oss/pkgs-base && \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Host-python deps for usd2roi_register.py + usd2roi_crop.py + semantic_rules.py.
# Pin Pillow / aiohttp for reproducible builds (PyPI latest at build time otherwise).
RUN pip3 install --no-cache-dir --break-system-packages \
        numpy Pillow==${PILLOW_VERSION} aiohttp==${AIOHTTP_VERSION} pyyaml usd-core 2>/dev/null \
    || pip3 install --no-cache-dir \
        numpy Pillow==${PILLOW_VERSION} aiohttp==${AIOHTTP_VERSION} pyyaml usd-core

# OSS compliance: ship source code of every Debian package the final image
# has on top of the base container at /usr/share/oss-sources/debian/. Python
# packages already carry source (pip installs .py for pure-Python) so they
# are out of scope per OSS guidance.
#
# We compute the new-package set by diffing dpkg-query output against the
# baseline snapshot from the first RUN layer. NEW_PACKAGES.txt records the
# full diff (including transitive deps apt pulled in). For each new binary we
# resolve its source package via `dpkg-query -f '${source:Package}'`, dedupe,
# and fetch the .dsc + tarballs with `apt-get source`. No hand-curated list,
# so any future transitive-dep change in the base picks up source coverage
# automatically.
RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's/^# *deb-src/deb-src/' /etc/apt/sources.list; \
    fi; \
    for f in /etc/apt/sources.list.d/*.sources; do \
        [ -f "$f" ] || continue; \
        if ! grep -q '^Types:.*deb-src' "$f"; then \
            sed -i 's/^Types: deb$/Types: deb deb-src/' "$f"; \
        fi; \
    done; \
    apt-get update; \
    mkdir -p /usr/share/oss-sources/debian; \
    dpkg-query -W -f='${Package}\n' | sort > /tmp/oss/pkgs-after; \
    comm -13 /tmp/oss/pkgs-base /tmp/oss/pkgs-after \
        > /usr/share/oss-sources/debian/NEW_PACKAGES.txt; \
    echo "=== OSS: new Debian packages in final image (vs base) ==="; \
    cat /usr/share/oss-sources/debian/NEW_PACKAGES.txt; \
    src_pkgs=$(while read -r p; do \
        [ -z "$p" ] && continue; \
        dpkg-query -W -f='${source:Package}\n' "$p"; \
    done < /usr/share/oss-sources/debian/NEW_PACKAGES.txt | sort -u); \
    echo "=== OSS: unique source packages to collect ==="; \
    echo "$src_pkgs" | tee /usr/share/oss-sources/debian/SOURCE_PACKAGES.txt; \
    cd /usr/share/oss-sources/debian; \
    for src in $src_pkgs; do \
        apt-get source --download-only "$src"; \
    done; \
    cd /; \
    rm -rf /tmp/oss; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# cupy is optional: registration falls back to CPU if missing.
# nvidia-cuda-nvrtc-cu12 + cuda-runtime-cu12 supply libnvrtc.so / libcudart.so
# that cupy needs at runtime (the omni base image doesn't bundle them).
RUN pip3 install --no-cache-dir --break-system-packages \
        cupy-cuda12x nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 2>/dev/null \
    || pip3 install --no-cache-dir \
        cupy-cuda12x nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
    || echo "[cad2roi] cupy install failed — registration will run on CPU"

# Stage repo into /workspace owned by the base image's default user (uid 1234).
# /home/isaac-sim doesn't exist in this base — isaac-sim's HOME is /isaac-sim
# (the install dir itself), so we keep the repo under a clean /workspace path.
COPY --chown=isaac-sim:isaac-sim . /workspace/paidf-simulation

# Drop stale outputs / caches that might have come along with the build context
RUN rm -rf /workspace/paidf-simulation/scripts/usd2roi/output \
           /workspace/paidf-simulation/scripts/usd2roi/__pycache__ \
           /workspace/paidf-simulation/scripts/sdg/standalone/__pycache__ \
           /workspace/paidf-simulation/sdg_test_output \
           /workspace/paidf-simulation/output

# Surface base-image LICENSE and a VERSION marker at the WORKDIR 
# VERSION is derived from pyproject.toml
RUN cp /isaac-sim/LICENSE.txt /workspace/paidf-simulation/BASE_IMAGE_LICENSE.txt && \
    awk -F'"' '/^version *=/ {print $2; exit}' /workspace/paidf-simulation/pyproject.toml \
        > /workspace/paidf-simulation/VERSION && \
    chown isaac-sim:isaac-sim \
        /workspace/paidf-simulation/BASE_IMAGE_LICENSE.txt \
        /workspace/paidf-simulation/VERSION

USER isaac-sim

# Share Kit's cv2 with host-python for --entrypoint python3 stages.
ENV PYTHONPATH=/isaac-sim/exts/omni.pip.compute/pip_prebundle:${PYTHONPATH}

WORKDIR /workspace/paidf-simulation

# Default entrypoint boots Kit directly with the lean `isaacsim.exp.base.kit`
# instead of `isaac-sim.sh` (which forces `isaacsim.exp.full.kit` ~120 ext +
# ROS2 chain that fails in container). base.kit ships only ~96 ext including
# omni.replicator.core / replicator_yaml / syntheticdata / hydra.rtx — enough
# for headless SDG / cad2roi without the full Isaac Sim app's listener tax.
# Measured: cad2roi Stage 1 ~4.4 min (base.kit) vs ~20 min (full.kit).
# For host-python stages (register / crop) override with `--entrypoint python3`.
ENTRYPOINT ["/bin/bash", "-c", \
            "/isaac-sim/kit/kit /isaac-sim/apps/isaacsim.exp.base.kit --no-window --/rtx/hydra/supportMultiTickRate=false --exec \"$@\"", \
            "--"]
