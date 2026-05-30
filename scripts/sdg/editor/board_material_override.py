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

"""
board_material_override.py — runtime override of the PCB-substrate
materials at ``/World/pcba_main_s_detail/Looks/{SolderMask, Silkscreen,
OuterConductor, SolderPaste, InnerConductor, Dielectric}``.

Mirrors the pattern in ``component_material_override.py``:

  1. Find the ``pcba_main_s_detail`` layer in ``stage.GetUsedLayers()``.
  2. Create an **anonymous Sdf sublayer**, pre-create over-prim specs at
     each board shader path, and insert it as the strongest sublayer
     (index 0) of the pcba layer.
  3. ``apply(...)`` writes ``inputs:diffuse_reflection_color`` /
     ``inputs:specular_reflection_roughness`` / ``inputs:metalness``
     etc. into each over-prim. Composition pulls the override on top
     of the original Material prim — every consumer sees the new
     values instantly.
  4. ``teardown()`` removes the anonymous sublayer from the pcba
     layer's ``subLayerPaths``, restoring the original shaders.

Two non-obvious correctness requirements (each cost a debug session):

* The original board shaders use **OmniSurface** MDLs
  (``solder_mask_black.mdl``, ``osp_brass.mdl``,
  ``Plastic_Thick_Translucent_Flakes_Mod.mdl``), NOT OmniPBR. Their
  inputs are named ``diffuse_reflection_color`` / ``metalness`` /
  ``specular_reflection_roughness`` etc. — **not** the OmniPBR
  ``diffuse_color_constant`` / ``metallic_constant`` style. Stamping
  the OmniPBR names alone is a no-op. We stamp both naming
  conventions so anything downstream that reads either still finds
  values.

* The anon sublayer is a sublayer of ``pcba_main_s_detail.usd``,
  NOT of the parent stage. The pcba layer's internal namespace is
  ``/World/Looks/<MatName>/Shader`` — the parent stage references
  the layer's ``/World`` subtree at ``/World/pcba_main_s_detail/...``
  which is why the LIVE-stage path has the extra prefix. Override
  prim specs in our anon must use the layer-internal path
  (``/World/Looks/...``); writing them at ``/World/pcba_main_s_detail/Looks/...``
  composes into nothing.

To make a "vantablack" board that stays dark even under intense
ring-light illumination, set ``specular_weight = 0`` AND
``diffuse_weight = 0`` along with a near-black ``color`` and high
``roughness``. The strong red ring lights then have no specular
lobe to bounce off and the diffuse lobe absorbs all incoming
energy. (See the "vantablack" demo overrides in
``configs/flow1b_good_fixed/good_fixed.yaml``.)
"""

from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

_BASE = "/World/pcba_main_s_detail/Looks"

# Inside ``pcba_main_s_detail.usd`` (the layer that defines the substrate
# materials), the shader prims live at ``/World/Looks/<name>/Shader``.
# The parent stage references that subtree at
# ``/World/pcba_main_s_detail/...`` which is why the LIVE-stage path has
# the extra prefix. When we author override prims into a sublayer of
# pcba_main_s_detail.usd we must use the layer-internal path
# (``_LAYER_BASE``), NOT the live-stage path (``_BASE``); composition
# only picks up specs that align with the host layer's namespace.
_LAYER_BASE = "/World/Looks"

_MATERIALS = (
    "SolderMask",
    "Silkscreen",
    "OuterConductor",
    "SolderPaste",
    "InnerConductor",
    "Dielectric",
)

# Defaults used when a yaml block / UI slider doesn't specify a key.
# Tuned to a typical green PCB; ``diffuse_weight`` and ``specular_weight``
# default to ``None`` → leave the MDL's authored defaults in place.
_FALLBACK = {
    "SolderMask": {
        "color": Gf.Vec3f(0.05, 0.30, 0.10),
        "roughness": 0.35,
        "metallic": 0.0,
        "diffuse_weight": None,
        "specular_weight": None,
    },
    "Silkscreen": {
        "color": Gf.Vec3f(0.95, 0.95, 0.92),
        "roughness": 0.55,
        "metallic": 0.0,
        "diffuse_weight": None,
        "specular_weight": None,
    },
    "OuterConductor": {
        "color": Gf.Vec3f(0.85, 0.65, 0.40),
        "roughness": 0.25,
        "metallic": 0.85,
        "diffuse_weight": None,
        "specular_weight": None,
    },
    "SolderPaste": {
        "color": Gf.Vec3f(0.70, 0.70, 0.72),
        "roughness": 0.40,
        "metallic": 0.60,
        "diffuse_weight": None,
        "specular_weight": None,
    },
    "InnerConductor": {
        "color": Gf.Vec3f(0.85, 0.55, 0.30),
        "roughness": 0.25,
        "metallic": 0.85,
        "diffuse_weight": None,
        "specular_weight": None,
    },
    "Dielectric": {
        "color": Gf.Vec3f(0.0, 0.0, 0.0),
        "roughness": 1.0,
        "metallic": 0.0,
        "diffuse_weight": None,
        "specular_weight": None,
    },
}


# Substrate prims that aren't part of the standard 6 board materials
# but the user wants to look exactly like the Dielectric (pure black).
# We rebind their ``material:binding`` to the Dielectric Material so
# the existing Diel R/G/B/rough/metal sliders drive them too.
EXTRA_BLACKOUT_PRIMS = (
    "/World/pcba_main_s_detail/PCB/TOP/TOP/Geometry/TOP__5820_bodies__0/TOP_Model_1546_Inst_7/Mesh_6",
    "/World/pcba_main_s_detail/PCB/BOARD/DIELECTRIC/Geometry/DIELECTRIC__1_bodies__0/DIELECTRIC_Model_1_Inst_1/Mesh_0",
    "/World/pcba_main_s_detail/PCB/TOP/TOP/Geometry/TOP__5820_bodies__0/TOP_Model_1529_Inst_24/Mesh_23",
)

_PCBA_LIVE_PREFIX = "/World/pcba_main_s_detail"
_PCBA_LAYER_PREFIX = "/World"  # pcba_main_s_detail.usd's namespace strips the asset xform


def _live_to_layer_path(live_path: str) -> str:
    """Strip the live-stage ``/World/pcba_main_s_detail`` prefix so the
    path lines up with the pcba layer's internal namespace, where our
    anon sublayer authors overrides."""
    if live_path.startswith(_PCBA_LIVE_PREFIX):
        return _PCBA_LAYER_PREFIX + live_path[len(_PCBA_LIVE_PREFIX) :]
    return live_path


def apply_params_from_dict(state: _State, d: dict) -> None:
    """Translate a yaml-shaped or UI-slider-shaped dict into a
    ``state.apply(**per_material)`` call.

    Two input shapes are accepted:

    * **Per-material nested dicts** (yaml convention)::

        {"SolderMask": {"color": [r, g, b], "roughness": 0.4, ...},
         "Silkscreen": {"color": [r, g, b]}}

    * **Flat slider keys** (UI convention)::

        {"soldermask_color_r": 0.1, "soldermask_color_g": 0.05,
         "soldermask_color_b": 0.05, "soldermask_roughness": 0.99,
         "soldermask_specular_weight": 0.0,
         "silkscreen_color_r": 0.05, ...}

    The flat form is recognised by the ``<material_lower>_<field>``
    prefix; ``<material_lower>`` matches one of :data:`_MATERIALS`
    (lowercased; underscores allowed for camelCase names).
    """
    # Build a per-material accumulator; both input shapes feed it.
    per_mat: dict[str, dict] = {}
    for nm in _MATERIALS:
        sub = d.get(nm)
        if isinstance(sub, dict):
            per_mat[nm] = dict(sub)

    # Map material name → its slider-prefix.
    flat_prefix = {
        "SolderMask": "soldermask",
        "Silkscreen": "silkscreen",
        "OuterConductor": "outer_conductor",
        "SolderPaste": "solder_paste",
        "InnerConductor": "inner_conductor",
        "Dielectric": "dielectric",
    }
    for nm, pref in flat_prefix.items():
        bucket = per_mat.setdefault(nm, {})
        r = d.get(f"{pref}_color_r")
        g = d.get(f"{pref}_color_g")
        b = d.get(f"{pref}_color_b")
        if r is not None and g is not None and b is not None:
            bucket["color"] = [float(r), float(g), float(b)]
        for f in ("roughness", "metallic", "diffuse_weight", "specular_weight"):
            key = f"{pref}_{f}"
            if key in d:
                bucket[f] = float(d[key])
        if not bucket:
            per_mat.pop(nm, None)
    state.apply(**per_mat)


def state_to_flat_dict(state: _State) -> dict:
    """Inverse of :func:`apply_params_from_dict` flat form. Used by
    the equivalence test to compare UI vs pipeline state without
    inspecting Sdf attribute specs directly."""
    out = {}
    flat_prefix = {
        "SolderMask": "soldermask",
        "Silkscreen": "silkscreen",
        "OuterConductor": "outer_conductor",
        "SolderPaste": "solder_paste",
        "InnerConductor": "inner_conductor",
        "Dielectric": "dielectric",
    }
    last = state._last_per_material  # populated by apply()
    for nm, pref in flat_prefix.items():
        spec = last.get(nm) or {}
        fb = _FALLBACK[nm]
        color = spec.get("color", fb["color"])
        if not isinstance(color, Gf.Vec3f):
            color = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        out[f"{pref}_color_r"] = float(color[0])
        out[f"{pref}_color_g"] = float(color[1])
        out[f"{pref}_color_b"] = float(color[2])
        out[f"{pref}_roughness"] = float(spec.get("roughness", fb["roughness"]))
        out[f"{pref}_metallic"] = float(spec.get("metallic", fb["metallic"]))
        d_w = spec.get("diffuse_weight", fb.get("diffuse_weight"))
        s_w = spec.get("specular_weight", fb.get("specular_weight"))
        if d_w is not None:
            out[f"{pref}_diffuse_weight"] = float(d_w)
        if s_w is not None:
            out[f"{pref}_specular_weight"] = float(s_w)
    return out


_UNSET = object()


class _State:
    """Holds the anonymous override sublayer + helpers for ``apply()``."""

    def __init__(self) -> None:
        self.pcba_layer: Sdf.Layer | None = None
        self.anon: Sdf.Layer | None = None
        self.ready: bool = False
        # Resolved layer-internal shader prim path per material
        # (filled by setup()).
        self._shader_path: dict[str, Sdf.Path] = {}
        # path → {attr_name: original_value_or_UNSET}, populated when we
        # mutate pcba_layer directly. Used by teardown() to restore.
        self._pcba_restore: dict[str, dict] = {}
        # Snapshot of the most recent per-material apply() inputs;
        # used by :func:`state_to_flat_dict`.
        self._last_per_material: dict[str, dict] = {}

    def _find_shader_path(
        self, stage: Usd.Stage, mat_name: str
    ) -> tuple[Sdf.Path, Sdf.Path] | None:
        """Return ``(stage_path, layer_path)``:
        * ``stage_path`` — composed-stage path used to look up the
          prim during ``setup()`` validation
          (``/World/pcba_main_s_detail/Looks/<m>/<sh>``).
        * ``layer_path`` — pcba_main_s_detail.usd's internal path
          for the same prim (``/World/Looks/<m>/<sh>``); this is
          the path we write override specs at.
        """
        mat_prim = stage.GetPrimAtPath(f"{_BASE}/{mat_name}")
        if not mat_prim or not mat_prim.IsValid():
            return None
        for c in mat_prim.GetAllChildren():
            if c.GetTypeName() == "Shader":
                shader_name = c.GetName()
                stage_path = c.GetPath()
                layer_path = Sdf.Path(f"{_LAYER_BASE}/{mat_name}/{shader_name}")
                return (stage_path, layer_path)
        return None

    def setup(
        self,
        stage: Usd.Stage,
        extra_blackout_prim_paths: list[str] | tuple[str, ...] | None = None,
        extra_blackout_enabled: bool = True,
    ) -> tuple[bool, dict | str]:
        """Wire the override sublayer + (optionally) hide a list of
        substrate prims that can't be rebound through normal layer
        composition. Defaults to module-level :data:`EXTRA_BLACKOUT_PRIMS`
        when ``extra_blackout_prim_paths`` is None; pass an explicit list
        to override per-yaml. Set ``extra_blackout_enabled=False`` to
        skip the MakeInvisible step (e.g. when the user toggles the UI
        slider off).

        Returns ``(ok, info)``."""
        self.pcba_layer = None
        for layer in stage.GetUsedLayers():
            if "pcba_main_s_detail" in layer.identifier:
                self.pcba_layer = layer
                break
        if not self.pcba_layer:
            return False, "pcba_main_s_detail layer not found in GetUsedLayers()"

        info: dict = {}
        self._shader_path.clear()
        for nm in _MATERIALS:
            sp = self._find_shader_path(stage, nm)
            if sp is None:
                info[nm] = "Material or Shader not found at expected path; skipping"
                continue
            stage_path, layer_path = sp
            self._shader_path[nm] = layer_path
            info[nm] = f"{stage_path} ↔ {layer_path}"

        if not self._shader_path:
            return False, "no board materials found at /World/pcba_main_s_detail/Looks/<...>/Shader"

        # Anon layer with one over-prim per resolved shader. Paths
        # MUST be the layer-internal paths (no /World/pcba_main_s_detail
        # prefix); see the module docstring for the why.
        self.anon = Sdf.Layer.CreateAnonymous(".usda")
        for sp in self._shader_path.values():
            if not sp:
                continue  # defensive: skip empty paths from prototypes with no recoverable shader
            Sdf.CreatePrimInLayer(self.anon, sp)

        # The "extra blackout" prims come from deeper reference chains
        # in the pcba asset that even pcba_layer's own opinions can't
        # override (their material:binding is authored on a stronger
        # composition arc than we can write to without breaking the
        # asset hierarchy). Pragmatic fix: ``MakeInvisible`` on each.
        # With the substrate underneath already vantablack, hidden prims
        # look identical to "fully black bound" — same rendered effect.
        # The path list comes from yaml (``extra_blackout_prims:``) /
        # editor UI; falls back to module default ``EXTRA_BLACKOUT_PRIMS``.
        self._extra_blackout_paths = list(
            extra_blackout_prim_paths
            if extra_blackout_prim_paths is not None
            else EXTRA_BLACKOUT_PRIMS
        )
        rebind_log: list[str] = []
        if extra_blackout_enabled:
            for live in self._extra_blackout_paths:
                live_prim = stage.GetPrimAtPath(live)
                if not (live_prim and live_prim.IsValid()):
                    rebind_log.append(f"  ✗ MISSING: {live}")
                    continue
                try:
                    UsdGeom.Imageable(live_prim).MakeInvisible()
                    rebind_log.append(f"  HIDDEN  {live}")
                except Exception as exc:  # noqa: BLE001
                    rebind_log.append(f"  ✗ MakeInvisible failed: {live}  {exc!r}")
        else:
            rebind_log.append(
                f"  (disabled — keeping {len(self._extra_blackout_paths)} prim(s) visible)"
            )

        self.pcba_layer.subLayerPaths.insert(0, self.anon.identifier)
        self.ready = True

        if rebind_log:
            info["EXTRA_BLACKOUT"] = "\n".join(rebind_log)
            print(
                f"[BoardMatOverride] EXTRA_BLACKOUT: "
                f"{'enabled' if extra_blackout_enabled else 'disabled'} — "
                f"{len(self._extra_blackout_paths)} prim(s) listed"
            )
            for line in rebind_log:
                print(line)

        return True, info

    def _set(self, prim_path: Sdf.Path, attr_name: str, value, sdf_type) -> None:
        spec = self.anon.GetPrimAtPath(prim_path)
        if not spec:
            return
        if attr_name in spec.attributes:
            spec.attributes[attr_name].default = value
        else:
            Sdf.AttributeSpec(
                spec,
                attr_name,
                sdf_type,
                Sdf.VariabilityVarying,
            ).default = value

    def _set_pcba(self, prim_path: Sdf.Path, attr_name: str, value, sdf_type) -> None:
        """Mutate pcba_layer directly. The pcba layer's own opinions
        (e.g. the original ``info:mdl:sourceAsset``) are stronger than
        anything in our anon sublayer, so to swap the MDL we have to
        author into pcba itself. Snapshot the original on first touch
        so ``teardown()`` can restore."""
        if not prim_path:
            return  # defensive: skip empty paths from prototypes with no recoverable shader
        spec = self.pcba_layer.GetPrimAtPath(prim_path)
        if not spec:
            spec = Sdf.CreatePrimInLayer(self.pcba_layer, prim_path)
            spec.specifier = Sdf.SpecifierOver
        per_path = self._pcba_restore.setdefault(str(prim_path), {})
        if attr_name not in per_path:
            existing = spec.attributes[attr_name] if attr_name in spec.attributes else None
            per_path[attr_name] = existing.default if existing is not None else _UNSET
        if attr_name in spec.attributes:
            spec.attributes[attr_name].default = value
        else:
            Sdf.AttributeSpec(
                spec,
                attr_name,
                sdf_type,
                Sdf.VariabilityVarying,
            ).default = value

    def _set_pcba_binding(self, prim_path: Sdf.Path, target_material_path: Sdf.Path) -> None:
        """Force-bind the prim at ``prim_path`` to ``target_material_path``
        by writing ``material:binding`` (and per-purpose variants) directly
        into pcba_layer. pcba's own opinions beat its sublayers + any
        binding inherited via referenced LibParts USDs. Per-purpose
        names (``:render``, ``:full``, ``:preview``) are stamped too so
        Hydra's purpose-aware resolution picks our override regardless
        of which purpose the original asset used."""
        if not prim_path:
            return  # defensive: skip empty paths from prototypes with no recoverable shader
        spec = self.pcba_layer.GetPrimAtPath(prim_path)
        if not spec:
            spec = Sdf.CreatePrimInLayer(self.pcba_layer, prim_path)
            spec.specifier = Sdf.SpecifierOver
        per_path = self._pcba_restore.setdefault(str(prim_path), {})
        for rel_name in (
            "material:binding",
            "material:binding:render",
            "material:binding:full",
            "material:binding:preview",
            "material:binding:allPurpose",
        ):
            rel_key = f"REL:{rel_name}"
            if rel_key not in per_path:
                existing_rel = spec.relationships.get(rel_name)
                if existing_rel is not None:
                    per_path[rel_key] = list(existing_rel.targetPathList.explicitItems)
                else:
                    per_path[rel_key] = _UNSET
            rel = spec.relationships.get(rel_name)
            if rel is None:
                rel = Sdf.RelationshipSpec(spec, rel_name)
            rel.targetPathList.explicitItems = [target_material_path]

    def apply(self, **per_material) -> None:
        """Stamp inputs onto each material's override prim.

        Each kwarg name is one of :data:`_MATERIALS` and its value is
        ``{ "color": [r, g, b] | Gf.Vec3f, "roughness": float,
        "metallic": float, "diffuse_weight": float|None,
        "specular_weight": float|None }``. Missing fields fall back
        to :data:`_FALLBACK`.

        Both OmniSurface-native input names and the OmniPBR-style
        names are stamped; whichever the runtime shader reads will
        find a value, so the override is robust to MDL swaps.
        """
        if not self.ready:
            return
        # Snapshot for state_to_flat_dict.
        self._last_per_material = {
            nm: dict(spec) for nm, spec in per_material.items() if isinstance(spec, dict)
        }
        with Sdf.ChangeBlock():
            for nm in self._shader_path:
                spec = per_material.get(nm) or {}
                fb = _FALLBACK[nm]
                color = spec.get("color", fb["color"])
                if not isinstance(color, Gf.Vec3f):
                    color = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
                rough = float(spec.get("roughness", fb["roughness"]))
                metal = float(spec.get("metallic", fb["metallic"]))
                d_w = spec.get("diffuse_weight", fb.get("diffuse_weight"))
                s_w = spec.get("specular_weight", fb.get("specular_weight"))
                shader_path = self._shader_path[nm]
                # Force-swap MDL to OmniPBR. Original substrate shaders
                # use OmniSurface (osp_brass.mdl, solder_mask_black.mdl,
                # …) which often ignore our colour/weight overrides
                # when the user asks for vantablack — the MDL clamps to
                # its built-in defaults. OmniPBR with explicit
                # ``diffuse_color_constant`` etc. honours the override
                # 1:1, including pure (0,0,0). The MDL swap MUST go to
                # pcba_layer directly because the original opinion is
                # in pcba and beats our anon sublayer.
                self._set_pcba(
                    shader_path,
                    "info:implementationSource",
                    "sourceAsset",
                    Sdf.ValueTypeNames.Token,
                )
                self._set_pcba(
                    shader_path,
                    "info:mdl:sourceAsset",
                    Sdf.AssetPath("OmniPBR.mdl"),
                    Sdf.ValueTypeNames.Asset,
                )
                self._set_pcba(
                    shader_path,
                    "info:mdl:sourceAsset:subIdentifier",
                    "OmniPBR",
                    Sdf.ValueTypeNames.Token,
                )
                # OmniPBR-native inputs — these are the ones the swapped
                # MDL reads. Setting both ``_constant`` and the older
                # OmniSurface names so a cached MDL layer still sees
                # values either way.
                self._set(
                    shader_path, "inputs:diffuse_color_constant", color, Sdf.ValueTypeNames.Color3f
                )
                self._set(
                    shader_path,
                    "inputs:reflection_roughness_constant",
                    rough,
                    Sdf.ValueTypeNames.Float,
                )
                self._set(shader_path, "inputs:metallic_constant", metal, Sdf.ValueTypeNames.Float)
                # Kill OmniPBR's residual Fresnel: F0 = mix(0.04, color,
                # metallic) is non-zero even at metallic=0 + color=(0,0,0)
                # because the dielectric baseline is 0.04. Multiplying
                # F0 by 0 → no specular at any angle → fully matte
                # vantablack.
                self._set(shader_path, "inputs:specular_level", 0.0, Sdf.ValueTypeNames.Float)
                # OmniSurface-native (older) — left in for
                # back-compat with consumers that don't follow the MDL swap.
                self._set(
                    shader_path,
                    "inputs:diffuse_reflection_color",
                    color,
                    Sdf.ValueTypeNames.Color3f,
                )
                self._set(
                    shader_path,
                    "inputs:diffuse_reflection_roughness",
                    rough,
                    Sdf.ValueTypeNames.Float,
                )
                self._set(
                    shader_path,
                    "inputs:specular_reflection_roughness",
                    rough,
                    Sdf.ValueTypeNames.Float,
                )
                self._set(
                    shader_path,
                    "inputs:specular_reflection_color",
                    color,
                    Sdf.ValueTypeNames.Color3f,
                )
                self._set(shader_path, "inputs:metalness", metal, Sdf.ValueTypeNames.Float)
                if d_w is not None:
                    self._set(
                        shader_path,
                        "inputs:diffuse_reflection_weight",
                        float(d_w),
                        Sdf.ValueTypeNames.Float,
                    )
                if s_w is not None:
                    self._set(
                        shader_path,
                        "inputs:specular_reflection_weight",
                        float(s_w),
                        Sdf.ValueTypeNames.Float,
                    )

    def apply_extra_blackout(self, hide: bool) -> None:
        """Runtime toggle: hide (True) or show (False) the extra blackout
        prims. Editor UI calls this when its checkbox flips. No-op if
        ``setup()`` hasn't been called or the path list is empty."""
        import omni.usd as _ousd

        try:
            stage = _ousd.get_context().get_stage()
        except Exception:  # noqa: BLE001
            return
        if stage is None:
            return
        for live in getattr(self, "_extra_blackout_paths", ()):
            p = stage.GetPrimAtPath(live)
            if not (p and p.IsValid()):
                continue
            try:
                im = UsdGeom.Imageable(p)
                if hide:
                    im.MakeInvisible()
                else:
                    im.MakeVisible()
            except Exception:  # noqa: BLE001
                pass

    def teardown(self) -> None:
        # 1) Restore instanceable flag on any prim we broke for the
        #    EXTRA_BLACKOUT rebind. Has to happen via the LIVE stage
        #    (instanceable is a stage-level meta), so do it before we
        #    drop sublayers.
        import omni.usd as _ousd

        try:
            stage_now = _ousd.get_context().get_stage()
        except Exception:  # noqa: BLE001
            stage_now = None
        if stage_now is not None:
            for path_str, items in self._pcba_restore.items():
                if "INSTANCEABLE" not in items:
                    continue
                p = stage_now.GetPrimAtPath(path_str)
                if p and p.IsValid():
                    try:
                        p.SetInstanceable(bool(items["INSTANCEABLE"]))
                    except Exception:  # noqa: BLE001
                        pass

        # 2) Restore pcba-layer attrs + rels we mutated directly (MDL swap
        #    and material:binding rebinds for EXTRA_BLACKOUT_PRIMS).
        if self.pcba_layer and self._pcba_restore:
            try:
                with Sdf.ChangeBlock():
                    for path_str, items in self._pcba_restore.items():
                        spec = self.pcba_layer.GetPrimAtPath(path_str)
                        if not spec:
                            continue
                        for key, original in items.items():
                            if key == "INSTANCEABLE":
                                continue  # handled above on the live stage
                            if key.startswith("REL:"):
                                rel_name = key[len("REL:") :]
                                rel = spec.relationships.get(rel_name)
                                if rel is None:
                                    continue
                                if original is _UNSET:
                                    del spec.relationships[rel_name]
                                else:
                                    rel.targetPathList.explicitItems = list(original)
                                continue
                            attr_name = key
                            if attr_name not in spec.attributes:
                                continue
                            if original is _UNSET:
                                del spec.attributes[attr_name]
                            else:
                                spec.attributes[attr_name].default = original
            except Exception:  # noqa: BLE001
                pass
        self._pcba_restore.clear()

        if self.pcba_layer and self.anon:
            try:
                paths = list(self.pcba_layer.subLayerPaths)
                if self.anon.identifier in paths:
                    paths.remove(self.anon.identifier)
                    self.pcba_layer.subLayerPaths[:] = paths
            except Exception:  # noqa: BLE001
                pass
        self.ready = False
        self.anon = None
        self.pcba_layer = None
        self._shader_path.clear()
        self._last_per_material.clear()


# Module-level singleton, mirroring the component override pattern.
_state = _State()
