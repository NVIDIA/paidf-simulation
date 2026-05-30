# Third-party notices

This directory contains attribution and license information for third-party
open-source software consumed by paidf-simulation at runtime.

## Contents

- [`LICENSE-3rd-party.txt`](LICENSE-3rd-party.txt) — concatenated full
  license text and copyright notices for every direct and transitive
  Python runtime dependency, fetched verbatim from each project's
  upstream LICENSE file. Each entry includes the package name, version,
  SPDX identifier, and upstream LICENSE URL. This is the authoritative
  attribution file for redistribution (the format used by NVIDIA AI
  Blueprint releases).

## Upstream LICENSE references

Direct dependencies from [`requirements.txt`](../requirements.txt):

| Package | Upstream LICENSE |
|---|---|
| numpy | <https://github.com/numpy/numpy/blob/main/LICENSE.txt> |
| Pillow | <https://github.com/python-pillow/Pillow/blob/main/LICENSE> |
| aiohttp | <https://github.com/aio-libs/aiohttp/blob/master/LICENSE.txt> |
| PyYAML | <https://github.com/yaml/pyyaml/blob/main/LICENSE> |
| usd-core | <https://github.com/PixarAnimationStudios/OpenUSD/blob/release/LICENSE.txt> |

Transitive dependencies (aiohappyeyeballs, aiosignal, attrs, frozenlist,
idna, multidict, propcache, typing-extensions, yarl) are pulled in by
`aiohttp` and licensed under permissive licenses (Apache-2.0 / BSD-3 /
MIT / PSF). Full license text for all 14 packages is included in
[`LICENSE-3rd-party.txt`](LICENSE-3rd-party.txt).

### `usd-core` license note

`usd-core` (OpenUSD) ships under Pixar's **Modified Apache 2.0 License**
(SPDX `LicenseRef-TOST-1.0`), which is functionally identical to
Apache-2.0 with the addition of a trademark-use restriction. It is
compatible with Apache-2.0-licensed projects.

## Regenerating `LICENSE-3rd-party.txt`

When dependency versions change in [`requirements.txt`](../requirements.txt):

1. Resolve the actual installed package set inside the runtime container
   (Python 3.12) so PEP 508 environment markers are applied correctly —
   running `pip install` or `licensecheck` on a host with a different
   Python version will resolve a different (wrong) set. For example,
   `aiohttp` declares `async-timeout` only under Python <3.11; running
   on Python 3.10 would incorrectly include it.

   ```bash
   docker run --rm --entrypoint bash "$IMG" -lc \
     'pip3 install --quiet --break-system-packages licensecheck uv && \
      /isaac-sim/.local/bin/licensecheck --requirements-paths requirements.txt'
   ```

2. For each package in the resolved set, fetch the upstream LICENSE file
   from its GitHub source and concatenate into `LICENSE-3rd-party.txt`
   using the section format documented at the top of that file (matches
   the NVIDIA AI Blueprint convention).

## Runtime container base image

The pipeline executes inside the NVIDIA Isaac Sim / Omniverse Kit SDK
container, referenced as `${SDG_IMAGE}` in [`README.md`](../README.md).
The container image is distributed separately under NVIDIA's container
licensing terms and is not redistributed by this repository. Consumers
obtain the container directly from NVIDIA's official distribution
channels and are responsible for complying with that image's license
terms. OSS source-code compliance for content inside the base container
is covered by the Isaac Sim base container's own OSRB ticket.

The base container's `LICENSE.txt` is also surfaced inside the runtime
image at `/workspace/paidf-simulation/BASE_IMAGE_LICENSE.txt` for
in-container reference.
