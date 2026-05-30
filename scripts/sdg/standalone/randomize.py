# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-trigger randomization helpers — extracted from sdg_pipeline.py.

These read the user yaml `randomize_*` blocks and apply them to the live
stage / fillet params / tin material state. Each helper is a pure
function over its inputs; the caller (sdg_pipeline) holds mutable
references to FILLET_PARAMS / TIN_PARAMS / mat_state and passes them in.

Splitting these out keeps sdg_pipeline.py focused on the async render
loop while the per-trigger sampling logic stays grep-able and unit-
testable in isolation.
"""

from typing import Any, Callable

import numpy as np
from common import build_ring_lights, disable_all_lights
from pxr import Gf


def sample(lo_or_val, hi=None) -> float:
    """Scalar → float; ``[lo, hi]`` / ``(lo, hi)`` → uniform sample on the closed interval."""
    if hi is None:
        if isinstance(lo_or_val, (list, tuple)):
            lo, hi = float(lo_or_val[0]), float(lo_or_val[1])
            return float(np.random.uniform(lo, hi))
        return float(lo_or_val)
    return float(np.random.uniform(lo_or_val, hi))


def randomize_rig(stage, cfg: dict[str, Any]) -> None:
    """Rebuild ring-light rig with per-trigger sampled dome / radius / intensity / cone."""
    r = cfg.get("randomize_rig", {})
    base = cfg.get("rig", {})
    light = cfg.get("lighting", {})
    dome_r = sample(r.get("dome_radius", base.get("dome_radius", 184.0)))
    l_rad = sample(r.get("light_radius", base.get("light_radius_world", 18.0)))
    intens = sample(r.get("intensity", base.get("light_intensity", 6204.0)))
    cone_a = sample(light.get("cone_angle_range", [120.0, 120.0]))
    cone_s = sample(light.get("cone_softness_range", [1.0, 1.0]))
    disable_all_lights(stage)
    build_ring_lights(
        stage,
        cfg["ring_light_root"],
        dome_radius=dome_r,
        light_radius=l_rad,
        light_intensity=intens,
        cone_angle=cone_a,
        cone_softness=cone_s,
    )
    print(
        f"[Pipeline] Rig: dome={dome_r:.1f} l_rad={l_rad:.1f} intensity={intens:.0f} "
        f"cone={cone_a:.1f} softness={cone_s:.2f}"
    )


def randomize_material_params(mat_state, mat_defs: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Apply per-trigger material randomization. ``mat_state`` is the editor
    ``component_material_override._State`` (carries ``apply()``)."""
    rm = cfg["randomize_material"]
    body_r = sample(rm.get("body_r", [0.0, 0.1]))
    body_g = sample(rm.get("body_g", [0.0, 0.1]))
    body_b = sample(rm.get("body_b", [0.0, 0.1]))
    body_rough = sample(rm.get("body_roughness", 1.0))
    body_metal = sample(rm.get("body_metallic", [0.9, 1.0]))
    tin_rough = sample(rm.get("tin_roughness", mat_defs.get("tin_rough", 0.2)))
    pad_rough = sample(rm.get("pad_roughness", mat_defs.get("pad_rough", 0.2)))
    # Tin / pad colour + metallic — fall back to ``component_material``
    # block scalars when ``randomize_material`` doesn't override; never
    # leave them as None (else apply()'s silver fallback paints tin /
    # pad bright grey, which is the editor↔pipeline mismatch we hit).
    cm = cfg.get("component_material", {})

    def _rgb(name, default):
        v = rm.get(name) if name in rm else cm.get(name, default)
        return (
            [sample(c) for c in v]
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple))
            else ([float(c) for c in v] if v is not None else default)
        )

    tin_rgb = _rgb("tin_color", [0.0, 0.0, 0.0])
    pad_rgb = _rgb("pad_color", [0.0, 0.0, 0.0])
    tin_metal = sample(rm.get("tin_metallic", cm.get("tin_metallic", 0.89)))
    pad_metal = sample(rm.get("pad_metallic", cm.get("pad_metallic", 0.0)))
    mat_state.apply(
        Gf.Vec3f(body_r, body_g, body_b),
        body_rough,
        body_metal,
        tin_rough,
        pad_rough,
        tin_color=Gf.Vec3f(*tin_rgb),
        tin_metallic=float(tin_metal),
        pad_color=Gf.Vec3f(*pad_rgb),
        pad_metallic=float(pad_metal),
    )
    print(
        f"[Pipeline] Material: body=({body_r:.3f},{body_g:.3f},{body_b:.3f}) "
        f"rough={body_rough:.2f} metal={body_metal:.2f}  "
        f"tin=({tin_rgb[0]:.2f},{tin_rgb[1]:.2f},{tin_rgb[2]:.2f}) rough={tin_rough:.2f} metal={tin_metal:.2f}  "
        f"pad=({pad_rgb[0]:.2f},{pad_rgb[1]:.2f},{pad_rgb[2]:.2f}) rough={pad_rough:.2f} metal={pad_metal:.2f}"
    )


def randomize_fillet_shape(fillet_params, cfg: dict[str, Any]) -> None:
    """Sample per-trigger fillet shape into ``fillet_params`` (mutated in place)."""
    rf = cfg["randomize_fillet"]
    fp = fillet_params
    fp.plat_start = sample(rf.get("platform_start", fp.plat_start))
    fp.plat_end = sample(rf.get("platform_end", fp.plat_end))
    fp.smooth_y = sample(rf.get("smoothness_y", fp.smooth_y))
    fp.bump = sample(rf.get("bump", fp.bump))
    fp.decay = sample(rf.get("delta_x_decay", fp.decay))
    fp.noise_scale = sample(rf.get("noise_scale", fp.noise_scale))
    fp.noise_amp = sample(rf.get("noise_amp", fp.noise_amp))
    if "fillet_sx" in rf:
        fp.fillet_sx = sample(rf["fillet_sx"])  # also syncs fillet_sy via __setattr__
    if "fillet_sz" in rf:
        fp.fillet_sz = sample(rf["fillet_sz"])
    print(
        f"[Pipeline] Fillet: plat=[{fp.plat_start:.2f},{fp.plat_end:.2f}] "
        f"smooth_y={fp.smooth_y:.1f} bump={fp.bump:.3f} "
        f"decay={fp.decay:.1f} noise_scale={fp.noise_scale:.2f} "
        f"noise_amp={fp.noise_amp:.4f}"
    )


def randomize_tin_noise_shape(
    mat_state, tin_params, cfg: dict[str, Any], make_tin_normalmap: Callable
) -> None:
    """Sample fresh tin-noise params, re-bake the perlin normal map, re-stamp on
    every tin shader. ``make_tin_normalmap`` is the callable from
    ``tin_noise_patch`` (passed in to avoid coupling this module to the
    editor-script import path)."""
    rt = cfg["randomize_tin_noise"]
    tp = tin_params
    tp.noise_amp = sample(rt.get("noise_amp", tp.noise_amp))
    tp.noise_scale = sample(rt.get("noise_scale", tp.noise_scale))
    tp.noise_octaves = sample(rt.get("noise_octaves", tp.noise_octaves))
    tp.resolution = sample(rt.get("resolution", tp.resolution))
    bump = sample(rt.get("bump_factor", cfg.get("tin_noise", {}).get("bump_factor", 0.24)))
    uvs = sample(rt.get("texture_scale", cfg.get("tin_noise", {}).get("texture_scale", 17.62)))
    # Per-sample seed for the underlying value-noise grid so the noise
    # PATTERN (not just amplitude / freq quantization) varies frame-to-
    # frame. Without this, neighboring samples whose ``int(noise_scale)``
    # rounds to the same base octave produce visually identical bumps.
    seed = int(np.random.randint(0, 2**31 - 1))
    smooth_sigma = float(cfg.get("tin_noise", {}).get("smooth_sigma", 1.0))
    try:
        png = make_tin_normalmap(tp, force_rebuild=True, seed=seed, smooth_sigma=smooth_sigma)
        mat_state.set_tin_normalmap(png, bump_factor=bump, texture_scale=uvs, project_uvw=True)
        print(
            f"[Pipeline] Tin: amp={tp.noise_amp:.3f} scale={tp.noise_scale:.2f} "
            f"oct={tp.noise_octaves:.2f} res={tp.resolution:.0f} "
            f"bump={bump:.3f} uv={uvs:.2f} seed={seed} sm={smooth_sigma:.2f}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Pipeline] Tin re-bake failed: {exc!r}")
