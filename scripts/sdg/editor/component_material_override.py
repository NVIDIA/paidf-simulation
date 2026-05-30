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
Component Material Override — all component types
Controls body color + roughness for every component type listed in the
pipeline yaml's ``component_types:`` block (or a custom list passed to
``setup``). Up to 23 types in the production yaml.

Approach:
  • Find pcba_main_s_detail.usd in stage.GetUsedLayers().
  • Walk every prototype's Shader prims, classify each as body / tin / pad
    by material name (heuristic: "tin_plating" / "solder_pad" / "pad" /
    "Solder_Paste" → tin/pad; everything else → body).
  • Build the corresponding pcba-layer namespace path for each Shader by
    swapping the live ``/__Prototype_N/...`` prefix for ``/LibParts/<ct>``.
  • Create one anonymous Sdf sublayer with `over` specs at all those paths,
    insert as sublayer[0] of pcba layer → wins over component USD opinions.
  • Update attribute defaults in-place on each slider change (Sdf.ChangeBlock).

Entry points:
  headless verify:
    "$ISAAC_SIM_PATH" --no-window \
      --exec "${PAIDF_SIM_ROOT}/scripts/component_material_override.py --verify"
    → writes /tmp/matctrl_verify.txt, then shuts down

  ui (default):
    Paste into Script Editor while temp_scene.usd is open
    → floating window with sliders; applies to current stage immediately
"""

import asyncio
import os
import sys

import omni.kit.app
import omni.ui
import omni.usd
from pxr import Gf, Sdf, Usd

_app = omni.kit.app.get_app()


def _resolve_scene():
    """Lazy-resolve the scene USD path. Required only when this module is
    used as a standalone Kit script (--verify mode or editor open_stage).
    Importing the module without PCB_USD_PATH set is intentionally OK so
    the pipeline can still pull the _State class for material overrides.
    """
    val = os.environ.get("PCB_USD_PATH")
    if not val:
        raise RuntimeError("PCB_USD_PATH not set; export PCB_USD_PATH=/path/to/board.usd")
    return os.path.expandvars(val)


VERIFY_OUT = "/tmp/matctrl_verify.txt"
HEADLESS = "--verify" in sys.argv

# Default fallback list of component types — matches the production yaml's
# ``component_types:`` block. Override by passing ``component_types`` to setup().
_DEFAULT_COMPONENT_TYPES = (
    "_0201_H030",
    "_0201_H040",
    "_0201_LARGE_H040",
    "_0402_H040",
    "_0402_H060",
    "_0402_LARGE_H070",
    "_0603_H060",
    "_0603_H070",
    "_0603_H100",
    "_0805U_H070",
    "_0805U_H150",
    "_0805U_H160",
    "_1206U_H080",
    "_1206U_H180",
    "_1210U_H280",
    "_2010_D_H110",
    "_2512_C_H090",
    "_2512_C_H220",
    "IND_SMD_0603_B_H100",
    "IND_SMD_0603_H100",
    "IND_SMD_0805_H110",
    "IND_SMD_2520_D_H120",
    "IND_SMD_2520_G_H120",
)

# Body: switched to OmniPBR.mdl (replaces carpaint painted_yellow.mdl via sublayer override)
# tin/pad remain their original MDL; roughness attr name = reflection_roughness
_FALLBACK = {
    "body_r": 0.0,
    "body_g": 0.0,
    "body_b": 0.0,
    "body_rough": 0.97,
    "body_metallic": 0.89,
    "tin_r": 0.0,
    "tin_g": 0.0,
    "tin_b": 0.0,
    "tin_rough": 0.36,
    "tin_metallic": 0.89,
    "pad_r": 0.0,
    "pad_g": 0.0,
    "pad_b": 0.0,
    "pad_rough": 1.0,
    "pad_metallic": 0.0,
}


_UNSET = object()  # sentinel for "attribute didn't exist before override"


def _classify_material(mat_name: str) -> str:
    """body / tin / pad classification by material name (case-sensitive)."""
    if "tin_plating" in mat_name:
        return "tin"
    if ("solder_pad" in mat_name) or mat_name in ("pad", "Solder_Paste"):
        return "pad"
    return "body"


# Tin noise slider defaults — must match the production yaml-driven
# defaults in tin_noise_patch.TinNoiseParams so the editor opens with
# the same look as the pipeline. Slider ranges are wider in the UI so
# the user can crank up the bump amplitude past the production default
# (production amp=0.05 mm; UI goes up to 0.5).
_DEFAULT_TIN_NOISE = {
    "noise_amp": 0.05,
    "noise_scale": 0.7,
    "noise_octaves": 4.0,
    "resolution": 24.0,
    "patch_x_frac": 0.30,
    "patch_y_frac": 1.0,
}


# ════════════════════════════════════════════════════════════════════════════
#  Override state — singleton, kept alive while UI is open
# ════════════════════════════════════════════════════════════════════════════


class _State:
    def __init__(self):
        self.pcba_layer = None
        self.anon = None
        self.ready = False
        # Layer-internal Sdf paths discovered at setup() — three buckets.
        # Each is a list of Sdf.Path pointing at the Shader prim spec.
        self.body_paths = []  # type: list[Sdf.Path]
        self.tin_paths = []
        self.pad_paths = []
        # Subsets of body/tin/pad paths whose authoritative ``def`` lives
        # in pcba_main_s_detail itself (not a referenced LibParts asset).
        # For these, the anon sublayer can author NEW inputs (anon wins
        # because pcba has no opinion), but it CANNOT override pcba's
        # existing ``info:mdl:sourceAsset`` — the parent layer is stronger
        # than its sublayers. So we mutate those attrs directly in
        # pcba_layer and restore the originals on teardown().
        self.pcba_native_body_paths = []  # subset of body_paths
        self.pcba_native_tin_paths = []
        self.pcba_native_pad_paths = []
        # path → {attr_name: original_value_or_None}, populated when we
        # mutate pcba_layer directly. Used by teardown() to restore.
        self._pcba_restore = {}
        # Set of component types covered (for diagnostics)
        self.types = ()

    # ── setup ─────────────────────────────────────────────────────────────

    def setup(self, stage, component_types=None, vantablack_body=True):
        """
        Discover all body/tin/pad Shader prim paths across the requested
        component types, create one anonymous override sublayer with `over`
        specs at those paths, insert as sublayer[0] of pcba layer.

        ``vantablack_body=True`` (default) keeps the AOI vantablack look:
        immediately apply ``_FALLBACK`` body_color=(0,0,0)+metallic=0.89
        plus prototype-read tin/pad roughness. Set ``False`` to surface the
        scene's authored body colors instead; the anon sublayer + path
        discovery still run so ``randomize_material`` / tin_noise paths
        keep working.

        Returns (ok: bool, info: dict|str). On success, info has the same
        keys as ``_FALLBACK`` (body_r/g/b/rough/metallic + tin/pad rough)
        with prototype-read defaults where available.
        """
        self.types = tuple(component_types) if component_types else _DEFAULT_COMPONENT_TYPES

        self.pcba_layer = None
        for layer in stage.GetUsedLayers():
            if "pcba_main_s_detail" in layer.identifier:
                self.pcba_layer = layer
                break
        if not self.pcba_layer:
            return False, "pcba_main_s_detail layer not found in GetUsedLayers()"

        # Walk live prototypes once; bucket every Shader prim into body/tin/pad
        # and compute the layer-internal pcba namespace path for each.
        (
            self.body_paths,
            self.tin_paths,
            self.pad_paths,
            self.pcba_native_body_paths,
            self.pcba_native_tin_paths,
            self.pcba_native_pad_paths,
        ) = self._discover_paths(stage)
        if not (self.body_paths or self.tin_paths or self.pad_paths):
            return False, (
                "no body/tin/pad shader paths discovered for types "
                f"{self.types}; check pcba prototypes"
            )

        self.anon = Sdf.Layer.CreateAnonymous(".usda")
        for p in self.body_paths + self.tin_paths + self.pad_paths:
            if not p:
                continue  # _discover_paths can yield "" for prototypes whose Shader prim has no recoverable pcba-layer path
            Sdf.CreatePrimInLayer(self.anon, p)

        self.pcba_layer.subLayerPaths.insert(0, self.anon.identifier)
        self.ready = True

        defs = self._read_proto_defaults(stage)
        if vantablack_body:
            self.apply(
                Gf.Vec3f(defs["body_r"], defs["body_g"], defs["body_b"]),
                defs["body_rough"],
                defs["body_metallic"],
                defs["tin_rough"],
                defs["pad_rough"],
            )
        info = dict(defs)
        info["vantablack_body"] = bool(vantablack_body)
        info["body_path_count"] = len(self.body_paths)
        info["tin_path_count"] = len(self.tin_paths)
        info["pad_path_count"] = len(self.pad_paths)
        info["types"] = list(self.types)
        return True, info

    def _discover_paths(self, stage):
        """For each component type in ``self.types``, find one instance,
        walk its prototype for Shader prims, and translate every Shader's
        live stage path into the corresponding pcba-layer namespace path
        ``/LibParts/<ct>/<proto-relative-path>``."""
        # Walk EVERY instance prim and collect EVERY distinct prototype,
        # plus a representative comp_type for each. The comp_types list
        # is used to compute the per-prototype lib_path, but its
        # membership is not gating — prototypes whose instance path
        # doesn't match any listed ct still get walked, classified
        # heuristically (last path segment that matches the regex
        # ``_\w+_H\d+`` or starts with ``IND_SMD_`` or ``tn__``); when
        # no comp_type can be derived we fall back to the prototype's
        # synthetic name. This catches material variants the original
        # ``first-prototype-per-listed-ct`` discovery missed (e.g. the
        # 7 extra ``_033_painted_yellow`` and the lone
        # ``Metal_Ceramic_Brakes_Golden`` shaders that stayed
        # un-overridden and rendered orange-yellow).
        import re as _re

        _CT_RE = _re.compile(r"_(?:\w+?_H\d+|IND_SMD_\w+)\b")
        seen_types: dict[str, list[str]] = {ct: [] for ct in self.types}
        proto_to_ct: dict[str, str] = {}
        for prim in stage.TraverseAll():
            if not prim.IsInstance():
                continue
            proto = prim.GetPrototype()
            if not (proto and proto.IsValid()):
                continue
            proto_str = str(proto.GetPath())
            inst_path = str(prim.GetPath())

            # Derive a comp_type label: prefer one of self.types if it
            # appears in the instance path; else extract from path; else
            # use the prototype's synthetic name as a unique key.
            matched_ct = None
            for ct in self.types:
                if f"/{ct}/" in inst_path or inst_path.endswith(f"/{ct}"):
                    matched_ct = ct
                    break
            if matched_ct is None:
                m = _CT_RE.search(inst_path)
                matched_ct = m.group(0).lstrip("_") if m else proto_str.lstrip("/")

            if matched_ct not in seen_types:
                seen_types[matched_ct] = []
            if proto_str not in seen_types[matched_ct]:
                seen_types[matched_ct].append(proto_str)
                proto_to_ct[proto_str] = matched_ct

        body, tin, pad = [], [], []
        pcba_native = {"body": [], "tin": [], "pad": []}
        pcba_id = self.pcba_layer.identifier
        seen_lib_paths: set[str] = set()
        for ct, proto_paths in seen_types.items():
            for proto_path_str in proto_paths:
                proto_prim = stage.GetPrimAtPath(proto_path_str)
                if not proto_prim or not proto_prim.IsValid():
                    continue
                for p in Usd.PrimRange(proto_prim):
                    if not p.IsValid() or p.GetTypeName() != "Shader":
                        continue
                    mat_name = p.GetParent().GetName()
                    cls = _classify_material(mat_name)

                    # Two layouts coexist (see older comment); for (a)
                    # we author at ``/LibParts/<ct>/LibRef/<rel>`` in
                    # the anon sublayer; for (b) we mutate pcba_layer
                    # directly so the MDL swap wins over pcba's own
                    # opinion (sublayers are weaker than their parent).
                    stack = p.GetPrimStack()
                    authored_in_pcba = bool(stack) and (stack[0].layer.identifier == pcba_id)
                    if authored_in_pcba:
                        lib_path = Sdf.Path(str(stack[0].path))
                    else:
                        rel = str(p.GetPath())[len(proto_path_str) :].lstrip("/")
                        if not rel.startswith("LibRef/"):
                            rel = "LibRef/" + rel
                        lib_path = Sdf.Path(f"/LibParts/{ct}/{rel}")
                    if str(lib_path) in seen_lib_paths:
                        continue  # same shader reachable via several prototypes
                    seen_lib_paths.add(str(lib_path))
                    {"body": body, "tin": tin, "pad": pad}[cls].append(lib_path)
                    if authored_in_pcba:
                        pcba_native[cls].append(lib_path)

        # Second pass: catch shaders authored directly in the LIVE (non-
        # instanced) tree. ``assets_final/spark_lighting.usd`` authors many
        # capacitor types (_0201_*, _0402_*, _0603_*, _0805U_*, _1206U_*,
        # _1210U_*, _2010_*, _2512_*) as plain tn__-prefixed Xforms, NOT
        # USD-native instances. The prototype walk above only finds shaders
        # reachable through prim.IsInstance(), so live capacitor shaders
        # (notably _033_painted_yellow / _007_tin_plating / _001_solder_pad
        # under tn__-prefixed Xforms) were never discovered and the
        # material override silently no-op'd on the live body.
        #
        # spark_lighting.usd mounts pcba_main_s_detail's /World at
        # /World/pcba_main_s_detail/ via reference. Overrides authored in
        # pcba_layer must use pcba's INTERNAL namespace (e.g., /World/PCBA/
        # ...), NOT the live composed path (/World/pcba_main_s_detail/PCBA/
        # ...). For each live shader whose stack[0] is an external ref
        # layer, walk up to the nearest ancestor that has a spec in
        # pcba_layer and construct the pcba-internal path by appending the
        # suffix from that ancestor down to the shader.
        for p in stage.Traverse():
            if not p.IsValid() or p.GetTypeName() != "Shader":
                continue
            parent = p.GetParent()
            if not parent.IsValid() or parent.GetTypeName() != "Material":
                continue
            cls = _classify_material(parent.GetName())
            stack = p.GetPrimStack()
            if not stack:
                continue
            authored_in_pcba = stack[0].layer.identifier == pcba_id
            if authored_in_pcba:
                lib_path = Sdf.Path(str(stack[0].path))
            else:
                # Find the nearest pcba-authored ancestor and use its
                # pcba-internal path as the anchor.
                suffix_parts = [p.GetName()]
                anchor_internal = None
                cur = parent  # Material prim
                while cur and cur.IsValid():
                    for spec in cur.GetPrimStack():
                        if spec.layer.identifier == pcba_id:
                            anchor_internal = str(spec.path)
                            break
                    if anchor_internal is not None:
                        break
                    suffix_parts.insert(0, cur.GetName())
                    cur = cur.GetParent()
                if anchor_internal is None:
                    continue  # no pcba-side anchor; can't author override
                lib_path = Sdf.Path(anchor_internal + "/" + "/".join(suffix_parts))
            if str(lib_path) in seen_lib_paths:
                continue
            seen_lib_paths.add(str(lib_path))
            {"body": body, "tin": tin, "pad": pad}[cls].append(lib_path)
            # All live-tree discoveries get authored DIRECTLY in pcba_layer
            # via _set_pcba (the apply() codepath always uses _set_pcba
            # regardless of native flag, but we still mark these as native
            # for downstream consistency / for set_tin_normalmap which uses
            # native vs non-native to choose between anon sublayer and pcba
            # mutation).
            pcba_native[cls].append(lib_path)
        return body, tin, pad, pcba_native["body"], pcba_native["tin"], pcba_native["pad"]

    def _read_proto_defaults(self, stage):
        # Body defaults come from _FALLBACK; tin/pad rough read from any
        # prototype that has authored the attribute (typically `_0603_H100`).
        d = dict(_FALLBACK)
        for prim in stage.TraverseAll():
            if not prim.IsInstance():
                continue
            if not any(ct in str(prim.GetPath()) for ct in self.types):
                continue
            proto = prim.GetPrototype()
            if not (proto and proto.IsValid()):
                continue
            for p in Usd.PrimRange(proto):
                if not p.IsValid() or p.GetTypeName() != "Shader":
                    continue
                mat_name = p.GetParent().GetName()
                cls = _classify_material(mat_name)
                if cls not in ("tin", "pad"):
                    continue
                for attr in p.GetAttributes():
                    if attr.GetName() != "inputs:reflection_roughness":
                        continue
                    v = attr.Get()
                    if v is None:
                        continue
                    key = "tin_rough" if cls == "tin" else "pad_rough"
                    d[key] = float(v)
            # Once we've seen at least one instance with both, stop early.
            if (
                d["tin_rough"] != _FALLBACK["tin_rough"]
                and d["pad_rough"] != _FALLBACK["pad_rough"]
            ):
                break
        return d

    # ── apply ─────────────────────────────────────────────────────────────

    def _set(self, prim_path, attr_name, value, sdf_type):
        spec = self.anon.GetPrimAtPath(prim_path)
        if not spec:
            return
        if attr_name in spec.attributes:
            spec.attributes[attr_name].default = value
        else:
            Sdf.AttributeSpec(spec, attr_name, sdf_type, Sdf.VariabilityVarying).default = value

    def _set_pcba(self, prim_path, attr_name, value, sdf_type):
        """Mutate pcba_layer directly — used for ``info:mdl:*`` on
        pcba-native body shaders, where a sublayer can't override the
        parent's opinion. Snapshot the original on first touch so
        teardown() can restore. Creates the over-prim spec if pcba
        has no local opinion at the path (the LibParts case)."""
        if not prim_path:
            return  # _discover_paths can yield "" for prototypes with no recoverable shader path
        spec = self.pcba_layer.GetPrimAtPath(prim_path)
        if not spec:
            spec = Sdf.CreatePrimInLayer(self.pcba_layer, prim_path)
            spec.specifier = Sdf.SpecifierOver
        # Snapshot on first touch
        per_path = self._pcba_restore.setdefault(str(prim_path), {})
        if attr_name not in per_path:
            existing = spec.attributes[attr_name] if attr_name in spec.attributes else None
            per_path[attr_name] = existing.default if existing is not None else _UNSET
        if attr_name in spec.attributes:
            spec.attributes[attr_name].default = value
        else:
            Sdf.AttributeSpec(spec, attr_name, sdf_type, Sdf.VariabilityVarying).default = value

    def apply(
        self,
        body_color,
        body_rough,
        body_metallic,
        tin_rough,
        pad_rough,
        tin_color=None,
        tin_metallic=None,
        pad_color=None,
        pad_metallic=None,
    ):
        """Stamp the override values across every discovered body / tin / pad
        shader path. ALL three categories are force-switched to OmniPBR.mdl
        so ``inputs:diffuse_color_constant`` etc. take effect uniformly.
        ``tin_color`` / ``pad_color`` / ``*_metallic`` default to ``None``
        which preserves the older "tin & pad rough only" callers."""
        if not self.ready:
            return
        # Default tin/pad to a tin-like silver appearance when caller doesn't
        # pass them (so older 5-arg callers still get a sane look).
        tin_color = tin_color if tin_color is not None else Gf.Vec3f(0.85, 0.85, 0.88)
        tin_metallic = tin_metallic if tin_metallic is not None else 1.0
        pad_color = pad_color if pad_color is not None else Gf.Vec3f(0.78, 0.78, 0.80)
        pad_metallic = pad_metallic if pad_metallic is not None else 0.85

        # In-pcba sets — for each, MDL swap requires direct pcba mutation.
        body_native = set(str(p) for p in self.pcba_native_body_paths)
        tin_native = set(str(p) for p in getattr(self, "pcba_native_tin_paths", ()))
        pad_native = set(str(p) for p in getattr(self, "pcba_native_pad_paths", ()))

        def _stamp(path, color, rough, metal, native_set, kill_specular=False):
            # Always mutate pcba_layer directly (NOT the anon sublayer)
            # for both the MDL swap AND the colour/rough/metal inputs.
            # pcba_layer's local opinions beat any sublayer + every
            # reference arc, so this guarantees the override wins
            # regardless of where the LibParts asset re-authors the
            # shader. The earlier "anon for inputs, pcba for MDL" split
            # left orange-yellow ``_033_painted_yellow`` and
            # ``Metal_Ceramic_Brakes_Golden`` looking original because
            # Hydra cached the source MDL before our anon override
            # composed in. Writing inputs to pcba_layer too means the
            # values are present from layer-load time onward.
            self._set_pcba(
                path, "info:implementationSource", "sourceAsset", Sdf.ValueTypeNames.Token
            )
            self._set_pcba(
                path, "info:mdl:sourceAsset", Sdf.AssetPath("OmniPBR.mdl"), Sdf.ValueTypeNames.Asset
            )
            self._set_pcba(
                path, "info:mdl:sourceAsset:subIdentifier", "OmniPBR", Sdf.ValueTypeNames.Token
            )
            self._set_pcba(path, "inputs:diffuse_color_constant", color, Sdf.ValueTypeNames.Color3f)
            # ALSO zero inputs:base_color (carpaint MDL leftover). The MDL
            # warning says base_color is not in OmniPBR.mdl, but empirically
            # the renderer DOES read it as the metallic F0 base, producing the
            # leftover tan-brown body tint. Zero it for a true black body.
            self._set_pcba(path, "inputs:base_color", color, Sdf.ValueTypeNames.Color3f)
            self._set_pcba(path, "inputs:diffuse_reflection_color", color, Sdf.ValueTypeNames.Color3f)
            self._set_pcba(
                path, "inputs:reflection_roughness_constant", float(rough), Sdf.ValueTypeNames.Float
            )
            self._set_pcba(path, "inputs:metallic_constant", float(metal), Sdf.ValueTypeNames.Float)
            if kill_specular:
                # OmniPBR's F0 = mix(0.04, base_color, metallic). Even
                # at metallic=0 + base=(0,0,0) the dielectric Fresnel
                # baseline (~0.04) gives the surface white edge highlights
                # under bright lights. specular_level=0 multiplies F0 by
                # zero → completely matte black, no Fresnel.
                self._set_pcba(path, "inputs:specular_level", 0.0, Sdf.ValueTypeNames.Float)

        with Sdf.ChangeBlock():
            for path in self.body_paths:
                _stamp(path, body_color, body_rough, body_metallic, body_native)
            for path in self.tin_paths:
                _stamp(path, tin_color, tin_rough, tin_metallic, tin_native)
            for path in self.pad_paths:
                _stamp(path, pad_color, pad_rough, pad_metallic, pad_native, kill_specular=True)

    # ── tin normal map ────────────────────────────────────────────────────

    def set_tin_normalmap(self, png_path, bump_factor=0.5, texture_scale=1.0, project_uvw=True):
        """Stamp ``inputs:normalmap_texture`` + supporting OmniPBR knobs
        on every discovered tin shader. Independent from apply() so the
        normal map persists across body/tin slider color changes.

        Tin meshes from CAD typically lack UV coordinates, so we force
        ``project_uvw=True`` (triplanar/world-space projection) — without
        it OmniPBR samples at constant (0,0) and the entire tin face
        renders one uniform normal direction (the "all blue" artifact).
        ``texture_scale`` controls the world-space tile size of the
        projection; ``bump_factor`` is OmniPBR's normal-map intensity
        and is decoupled from the noise amplitude in the texture.
        """
        if not self.ready:
            return
        tin_native = set(str(p) for p in self.pcba_native_tin_paths)
        ts_v2 = Gf.Vec2f(float(texture_scale), float(texture_scale))
        with Sdf.ChangeBlock():
            for path in self.tin_paths:
                # Texture path is an asset attribute — anon sublayer wins
                # for the LibParts-ref case (no parent opinion at that
                # path) and ALSO wins for the in-pcba case here because
                # the original tin MDL never authors normalmap_texture
                # (it uses a procedural tin look, no texture parameter).
                self._set(
                    path,
                    "inputs:normalmap_texture",
                    Sdf.AssetPath(str(png_path)),
                    Sdf.ValueTypeNames.Asset,
                )
                self._set(path, "inputs:bump_factor", float(bump_factor), Sdf.ValueTypeNames.Float)
                # CRITICAL for UV-less tin meshes: world-space planar/
                # triplanar projection. Without it the normal map is
                # sampled at constant UV → entire face one normal → the
                # "all blue / uniform tint" artifact users see at any
                # nonzero amp.
                self._set(path, "inputs:project_uvw", bool(project_uvw), Sdf.ValueTypeNames.Bool)
                self._set(path, "inputs:texture_scale", ts_v2, Sdf.ValueTypeNames.Float2)
                # Belt-and-suspenders: explicitly enable normal map.
                self._set(path, "inputs:enable_normalmap_texture", True, Sdf.ValueTypeNames.Bool)
                # The MDL must be OmniPBR for normalmap_texture to mean
                # anything. We already swapped during apply(); for safety,
                # re-stamp the MDL switch here so set_tin_normalmap can be
                # called before the first apply().
                set_mdl = self._set_pcba if str(path) in tin_native else self._set
                set_mdl(path, "info:implementationSource", "sourceAsset", Sdf.ValueTypeNames.Token)
                set_mdl(
                    path,
                    "info:mdl:sourceAsset",
                    Sdf.AssetPath("OmniPBR.mdl"),
                    Sdf.ValueTypeNames.Asset,
                )
                set_mdl(
                    path, "info:mdl:sourceAsset:subIdentifier", "OmniPBR", Sdf.ValueTypeNames.Token
                )

    # ── teardown ──────────────────────────────────────────────────────────

    def teardown(self):
        """Remove the anonymous sublayer from pcba_layer AND restore any
        attributes we mutated directly in pcba_layer for in-pcba prims."""
        # Restore pcba-native attribute opinions in reverse order of
        # mutation. ``_pcba_restore[path][attr]`` holds either the
        # original default (any USD value) or the _UNSET sentinel
        # (the attribute didn't exist before — remove it).
        if self.pcba_layer and self._pcba_restore:
            try:
                with Sdf.ChangeBlock():
                    for path_str, attrs in self._pcba_restore.items():
                        spec = self.pcba_layer.GetPrimAtPath(path_str)
                        if not spec:
                            continue
                        for attr_name, original in attrs.items():
                            if attr_name not in spec.attributes:
                                continue
                            if original is _UNSET:
                                # Attribute didn't exist before — remove our spec.
                                del spec.attributes[attr_name]
                            else:
                                spec.attributes[attr_name].default = original
            except Exception:
                pass
        self._pcba_restore.clear()

        if self.pcba_layer and self.anon:
            try:
                paths = list(self.pcba_layer.subLayerPaths)
                if self.anon.identifier in paths:
                    paths.remove(self.anon.identifier)
                    self.pcba_layer.subLayerPaths[:] = paths
            except Exception:
                pass
        self.ready = False


_state = _State()


async def _wait_loaded():
    ctx = omni.usd.get_context()
    while not _app.is_app_ready() or not ctx.get_stage():
        await _app.next_update_async()
    while ctx.get_stage_loading_status()[2] > 0:
        await _app.next_update_async()
    # Payloads can take additional frames to resolve after the loading counter
    # hits 0.  Poll until pcba_main_s_detail.usd appears in GetUsedLayers(),
    # with a generous timeout (120 extra frames ≈ a few seconds headless).
    stage = ctx.get_stage()
    for _ in range(120):
        if any("pcba_main_s_detail" in l.identifier for l in stage.GetUsedLayers()):
            break
        await _app.next_update_async()


# ════════════════════════════════════════════════════════════════════════════
#  HEADLESS VERIFY ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════


async def _verify():
    lines = []
    ctx = omni.usd.get_context()

    # Always open the target scene explicitly — Kit starts with a default
    # template stage, so ctx.get_stage() is non-None from the start.
    result, err = await ctx.open_stage_async(_resolve_scene())
    if not result:
        open(VERIFY_OUT, "w").write(f"OPEN FAILED: {err}\n")
        _app.shutdown()
        return
    await _wait_loaded()
    stage = ctx.get_stage()

    ok, info = _state.setup(stage)
    lines.append(f"Setup OK : {ok}")
    lines.append(f"Info     : {info}")
    lines.append(f"pcba     : {_state.pcba_layer.identifier if _state.pcba_layer else 'None'}")
    lines.append(f"anon     : {_state.anon.identifier if _state.anon else 'None'}")

    if ok:
        # Test: bright red body, extreme roughness to verify all three shaders
        _state.apply(
            Gf.Vec3f(1.0, 0.0, 0.0),
            body_rough=0.99,
            body_metallic=0.5,
            tin_rough=0.01,
            pad_rough=0.5,
        )
        for _ in range(6):
            await _app.next_update_async()

        lines.append("\n=== Prototype shader values after override ===")
        for prim in stage.TraverseAll():
            if "_0603_H100" not in str(prim.GetPath()) or not prim.IsInstance():
                continue
            proto = prim.GetPrototype()
            if not (proto and proto.IsValid()):
                continue
            target_parents = {"_033_painted_yellow", "_007_tin_plating", "_001_solder_pad"}
            for p in Usd.PrimRange(proto):
                if not p.IsValid() or p.GetTypeName() != "Shader":
                    continue
                parent = p.GetParent().GetName()
                if parent not in target_parents:
                    continue
                for attr in p.GetAttributes():
                    n = attr.GetName()
                    if n.startswith("inputs:"):
                        lines.append(f"  [{parent}] {n} = {attr.Get()}")
                lines.append(
                    f"  [{parent}] winning layer: "
                    f"{p.GetPrimStack()[0].layer.identifier if p.GetPrimStack() else 'n/a'}"
                )
            break

        # Verify color match
        exp_r, exp_body_rough, exp_tin_rough, exp_pad_rough = 1.0, 0.99, 0.01, 0.5
        lines.append("\nExpected body_r=1.0  body_rough=0.99  tin_rough=0.01  pad_rough=0.5")
        lines.append("See values above to confirm.")

    with open(VERIFY_OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"[MatCtrl] Written to {VERIFY_OUT}")
    for l in lines:
        print(l)
    _app.shutdown()


# ════════════════════════════════════════════════════════════════════════════
#  UI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════


class _Window:
    TITLE = "0603_H100 Material Override"

    def __init__(self):
        self._win = None
        self._sliders = {}  # key → FloatSlider widget (access .model from it)
        self._status = None
        # Tin-noise sliders push to tin_noise_patch.PARAMS via the shared
        # apply_params_from_dict helper (so UI and pipeline cannot drift).
        # See _push_tin_noise() below.
        try:
            import tin_noise_patch as _tnp

            self._tnp = _tnp
        except Exception:  # noqa: BLE001
            self._tnp = None
        # Board-material sliders go through the same shared helper —
        # apply_params_from_dict reads either the yaml-style nested
        # dicts or the UI-style flat keys (``soldermask_color_r`` …),
        # so the editor and the pipeline cannot drift.
        try:
            import board_material_override as _bmo

            self._bmo = _bmo
            # Each Window gets its own _State so opening/closing the
            # window doesn't tear down the pipeline's anon sublayer.
            self._board_state = _bmo._State()
        except Exception:  # noqa: BLE001
            self._bmo = None
            self._board_state = None

    # ── construction ──────────────────────────────────────────────────────

    def build(self):
        # Don't use Workspace.get_window() — it returns a WindowHandle with no destroy().
        # Cleanup of the previous instance is handled by _run_ui() calling _win_instance.destroy().
        self._win = omni.ui.Window(self.TITLE, width=420, height=340)
        d = _FALLBACK

        with self._win.frame:
            with omni.ui.VStack(spacing=6):
                omni.ui.Label(
                    "Body  (OmniPBR override)", style={"font_size": 16, "font_style": "bold"}
                )
                self._row("Color  R", "body_r", 0.0, 1.0, d["body_r"])
                self._row("Color  G", "body_g", 0.0, 1.0, d["body_g"])
                self._row("Color  B", "body_b", 0.0, 1.0, d["body_b"])
                self._row("Roughness", "body_rough", 0.0, 1.0, d["body_rough"])
                self._row("Metallic", "body_metallic", 0.0, 1.0, d["body_metallic"])

                omni.ui.Spacer(height=4)
                omni.ui.Label(
                    "Tin plating  (_007_tin_plating)", style={"font_size": 16, "font_style": "bold"}
                )
                self._row("Roughness", "tin_rough", 0.0, 1.0, d["tin_rough"])

                omni.ui.Spacer(height=4)
                omni.ui.Label(
                    "Solder pad  (_001_solder_pad)", style={"font_size": 16, "font_style": "bold"}
                )
                self._row("Roughness", "pad_rough", 0.0, 1.0, d["pad_rough"])

                # Board materials — PCB substrate (SolderMask /
                # Silkscreen / OuterConductor) override via Sdf
                # sublayer. Each slider writes through
                # ``board_material_override.apply_params_from_dict``,
                # the same helper the pipeline uses for the yaml
                # ``board_material:`` block, so the two paths cannot
                # drift. Set ``Spec weight`` and ``Diffuse weight`` to
                # 0 with a near-black colour for the "vantablack" demo.
                omni.ui.Spacer(height=4)
                omni.ui.Label(
                    "Board: SolderMask  (PCB base colour)",
                    style={"font_size": 16, "font_style": "bold"},
                )
                self._row_board("Color  R", "soldermask_color_r", 0.0, 1.0, 0.05)
                self._row_board("Color  G", "soldermask_color_g", 0.0, 1.0, 0.30)
                self._row_board("Color  B", "soldermask_color_b", 0.0, 1.0, 0.10)
                self._row_board("Roughness", "soldermask_roughness", 0.0, 1.0, 0.35)
                self._row_board("Spec weight", "soldermask_specular_weight", 0.0, 1.0, 1.0)

                omni.ui.Spacer(height=4)
                omni.ui.Label(
                    "Board: Silkscreen  (printed text)",
                    style={"font_size": 16, "font_style": "bold"},
                )
                self._row_board("Color  R", "silkscreen_color_r", 0.0, 1.0, 0.95)
                self._row_board("Color  G", "silkscreen_color_g", 0.0, 1.0, 0.95)
                self._row_board("Color  B", "silkscreen_color_b", 0.0, 1.0, 0.92)

                omni.ui.Spacer(height=4)
                omni.ui.Label(
                    "Board: OuterConductor  (exposed copper)",
                    style={"font_size": 16, "font_style": "bold"},
                )
                self._row_board("Color  R", "outer_conductor_color_r", 0.0, 1.0, 0.85)
                self._row_board("Color  G", "outer_conductor_color_g", 0.0, 1.0, 0.65)
                self._row_board("Color  B", "outer_conductor_color_b", 0.0, 1.0, 0.40)

                # Tin noise patches — perlin bumps overlaid on tin
                # terminals. Defaults mirror tin_noise_patch.TinNoiseParams;
                # ranges are wider (especially noise_amp) so the editor can
                # crank up the bumpiness past the production default.
                omni.ui.Spacer(height=4)
                omni.ui.Label(
                    "Tin noise patches  (perlin bumps)",
                    style={"font_size": 16, "font_style": "bold"},
                )
                tin_def = _DEFAULT_TIN_NOISE
                self._row_tin(
                    "Noise amp (mm)", "noise_amp", 0.0, 0.5, tin_def["noise_amp"], fmt="{:.4f}"
                )
                self._row_tin(
                    "Noise scale", "noise_scale", 0.05, 5.0, tin_def["noise_scale"], fmt="{:.3f}"
                )
                self._row_tin(
                    "Noise octaves",
                    "noise_octaves",
                    1.0,
                    8.0,
                    tin_def["noise_octaves"],
                    fmt="{:.0f}",
                )
                self._row_tin(
                    "Resolution", "resolution", 8.0, 48.0, tin_def["resolution"], fmt="{:.0f}"
                )
                self._row_tin(
                    "Patch x frac", "patch_x_frac", 0.05, 0.6, tin_def["patch_x_frac"], fmt="{:.3f}"
                )
                self._row_tin(
                    "Patch y frac", "patch_y_frac", 0.5, 1.5, tin_def["patch_y_frac"], fmt="{:.3f}"
                )

                omni.ui.Spacer(height=6)
                self._status = omni.ui.Label("Setting up…", style={"font_size": 12})

        self._win.set_visibility_changed_fn(lambda v: _state.teardown() if not v else None)

    def _row(self, label, key, lo, hi, default):
        lbl_ref = [None]
        with omni.ui.HStack(height=24, spacing=6):
            omni.ui.Label(label, width=110)
            s = omni.ui.FloatSlider(min=lo, max=hi)
            s.model.set_value(default)
            lbl_ref[0] = omni.ui.Label(f"{default:.3f}", width=44)

        def _on_change(_model=None):
            v = s.model.get_value_as_float()
            lbl_ref[0].text = f"{v:.3f}"
            self._push()

        s.model.add_value_changed_fn(_on_change)
        self._sliders[key] = s

    def _row_board(self, label, key, lo, hi, default, fmt="{:.3f}"):
        """Slider row for a board-material param. Writes through
        ``board_material_override.apply_params_from_dict`` so the
        editor and pipeline always agree."""
        lbl_ref = [None]
        with omni.ui.HStack(height=24, spacing=6):
            omni.ui.Label(label, width=110)
            s = omni.ui.FloatSlider(min=lo, max=hi)
            s.model.set_value(default)
            lbl_ref[0] = omni.ui.Label(fmt.format(default), width=60)

        def _on_change(_model=None):
            v = s.model.get_value_as_float()
            lbl_ref[0].text = fmt.format(v)
            self._push_board()

        s.model.add_value_changed_fn(_on_change)
        self._sliders[f"board_{key}"] = s

    def _row_tin(self, label, key, lo, hi, default, fmt="{:.3f}"):
        """Slider row for a tin-noise param. Writes through
        ``apply_params_from_dict`` to ``tin_noise_patch.PARAMS`` and
        triggers clear+regen so the viewport updates live."""
        lbl_ref = [None]
        with omni.ui.HStack(height=24, spacing=6):
            omni.ui.Label(label, width=110)
            s = omni.ui.FloatSlider(min=lo, max=hi)
            s.model.set_value(default)
            lbl_ref[0] = omni.ui.Label(fmt.format(default), width=60)

        def _on_change(_model=None):
            v = s.model.get_value_as_float()
            lbl_ref[0].text = fmt.format(v)
            self._push_tin_noise()

        s.model.add_value_changed_fn(_on_change)
        self._sliders[f"tin_{key}"] = s

    # ── apply sliders → state ─────────────────────────────────────────────

    def _push(self):
        if not _state.ready:
            return
        sl = self._sliders
        _state.apply(
            Gf.Vec3f(
                sl["body_r"].model.get_value_as_float(),
                sl["body_g"].model.get_value_as_float(),
                sl["body_b"].model.get_value_as_float(),
            ),
            sl["body_rough"].model.get_value_as_float(),
            sl["body_metallic"].model.get_value_as_float(),
            sl["tin_rough"].model.get_value_as_float(),
            sl["pad_rough"].model.get_value_as_float(),
        )

    def _read_tin_slider_dict(self) -> dict:
        """Build a yaml-shaped dict from the tin slider models. Used
        directly by the equivalence test so it can compare UI vs
        pipeline state without instantiating omni.ui."""
        sl = self._sliders
        return {
            "noise_amp": sl["tin_noise_amp"].model.get_value_as_float(),
            "noise_scale": sl["tin_noise_scale"].model.get_value_as_float(),
            "noise_octaves": sl["tin_noise_octaves"].model.get_value_as_float(),
            "resolution": sl["tin_resolution"].model.get_value_as_float(),
            "patch_x_frac": sl["tin_patch_x_frac"].model.get_value_as_float(),
            "patch_y_frac": sl["tin_patch_y_frac"].model.get_value_as_float(),
        }

    def _push_tin_noise(self):
        """Slider → TIN_PARAMS via the same shared helper as the pipeline,
        then re-generate the patches so the viewport reflects the new
        bumpiness immediately. No-op if tin_noise_patch failed to import
        (e.g. running outside Kit)."""
        if self._tnp is None:
            return
        d = self._read_tin_slider_dict()
        # Same code path as pipeline's CFG['tin_noise']: identical TIN_PARAMS
        # state for identical input dicts.
        self._tnp.apply_params_from_dict(self._tnp.PARAMS, d)
        try:
            self._tnp.clear_tin_patches()
            self._tnp.generate_tin_patches()
        except Exception as exc:  # noqa: BLE001
            print(f"[MaterialUI] tin patch regen failed: {exc!r}")

    def _read_board_slider_dict(self) -> dict:
        """Build a flat dict of board-material slider values. Same
        shape as the yaml-style flat keys
        (``soldermask_color_r`` …) — used directly by the equivalence
        test to compare UI vs pipeline state without omni.ui."""
        sl = self._sliders
        out = {}
        for k, w in sl.items():
            if not k.startswith("board_"):
                continue
            out[k[len("board_") :]] = float(w.model.get_value_as_float())
        return out

    def _push_board(self):
        """Slider → board override anonymous Sdf sublayer via the
        shared helper. First call also runs setup() so the override
        layer gets inserted into composition. No-op if
        board_material_override failed to import."""
        if self._bmo is None or self._board_state is None:
            return
        if not self._board_state.ready:
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
            ok, _ = self._board_state.setup(stage)
            if not ok:
                return
        d = self._read_board_slider_dict()
        # Same code path as pipeline's CFG['board_material'].
        self._bmo.apply_params_from_dict(self._board_state, d)

    def sync(self, defs):
        """Update sliders to actual prototype-read defaults."""
        for key, val in defs.items():
            if key in self._sliders:
                self._sliders[key].model.set_value(float(val))

    def set_status(self, msg, color=0xFF88FF88):
        if self._status:
            self._status.text = msg
            self._status.style = {"font_size": 12, "color": color}

    def destroy(self):
        _state.teardown()
        if self._win:
            self._win.destroy()
            self._win = None


_win_instance = None


async def _run_ui():
    global _win_instance

    # Clean up any previous Script-Editor run
    if _win_instance is not None:
        _win_instance.destroy()
    _win_instance = _Window()
    _win_instance.build()

    ctx = omni.usd.get_context()
    stage = ctx.get_stage()

    # Check if the current stage already has pcba_main_s_detail loaded
    # (i.e. the user already has temp_scene.usd open in Composer/Script Editor).
    _has_pcba = stage and any("pcba_main_s_detail" in l.identifier for l in stage.GetUsedLayers())
    if not _has_pcba:
        _win_instance.set_status("Opening scene…", 0xFFFFCC44)
        result, err = await ctx.open_stage_async(_resolve_scene())
        if not result:
            _win_instance.set_status(f"Open failed: {err}", 0xFFFF4444)
            return
        await _wait_loaded()
        stage = ctx.get_stage()

    ok, info = _state.setup(stage)
    if not ok:
        _win_instance.set_status(f"Setup failed: {info}", 0xFFFF4444)
        return

    # info == defs dict; sync sliders to actual prototype values then push
    _win_instance.sync(info)
    _win_instance._push()
    pcba_short = _state.pcba_layer.identifier.split("/")[-1]
    _win_instance.set_status(f"Active — sublayer injected into {pcba_short}", 0xFF88FF88)


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

if not os.getenv("FILLET_PIPELINE_MODE"):
    if HEADLESS:
        asyncio.ensure_future(_verify())
    else:
        asyncio.ensure_future(_run_ui())
