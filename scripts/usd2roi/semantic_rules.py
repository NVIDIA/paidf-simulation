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

"""Apply semantic labels to USD prims by glob-pattern rules.

Used by the cad2roi (usd2roi) pipeline. The rule list is a YAML block like:

    semantics:
      - match: "**/_0402_*/**"
        labels: {class: capacitor, footprint: "0402"}
      - match: "**/IND_SMD_*/inst_*"
        labels: {class: inductor}
      - match: "/World/PCBA/.../inst_0042"      # explicit prim path also works
        labels: {marker: special}

Glob syntax (pattern is anchored to the full prim path):
    *   matches any chars within a single path segment (no '/')
    **  matches any chars including '/' (zero or more, greedy)
    ?   matches a single char within a segment (no '/')
    Other characters are literal.

Rule application is top-down: for the same prim, a later rule's same-key
entry **overwrites** the earlier value; different keys are **merged**. After
merging, every affected prim gets a single ``rep_modify.semantics(prim, value=...)``
call so all keys land in one shot.

Semantic inheritance: Replicator + USD propagate semantic labels from parent
prims to descendant meshes at render time. Labelling a component Xform is
usually enough — only label deeper meshes when you need to *override* a
descendant's value (e.g. a "lead" sub-mesh under a chip body).

Standalone dry-run (no Kit needed; just usd-core + pyyaml):

    python semantic_rules.py --scene scene.usd --rules cad2roi_target.yaml --show 10
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# === Glob translation ===

_GLOBSTAR = "\x00GS\x00"
_STAR = "\x00S\x00"
_QMARK = "\x00Q\x00"


def glob_to_regex(pattern: str) -> str:
    """Translate a glob pattern to an anchored regex string.

    Rules:
        ``**`` -> ``.*``         (any chars including ``/``)
        ``*``  -> ``[^/]*``      (any chars within one segment)
        ``?``  -> ``[^/]``       (single char within one segment)

    Other regex metacharacters are escaped via ``re.escape`` so the pattern
    only does what the glob spec promises.
    """
    if "\x00" in pattern:
        raise ValueError("pattern must not contain null bytes")

    p = pattern
    p = p.replace("**", _GLOBSTAR)
    p = p.replace("*", _STAR)
    p = p.replace("?", _QMARK)
    p = re.escape(p)  # null-byte placeholders survive escape
    p = p.replace(_GLOBSTAR, ".*")
    p = p.replace(_STAR, "[^/]*")
    p = p.replace(_QMARK, "[^/]")
    return "^" + p + "$"


def find_matching_prims(stage: Any, pattern: str) -> list[Any]:
    """Walk the stage (including instance proxies) and return prims whose full path matches.

    USD instances hide their descendants from the default ``stage.Traverse()``;
    we use ``Usd.TraverseInstanceProxies()`` so rules can target sub-mesh prims
    (e.g. ``surface_0`` / ``surface_1`` under a referenced component).
    """
    from pxr import Usd

    rx = re.compile(glob_to_regex(pattern))
    pred = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    return [p for p in Usd.PrimRange(stage.GetPseudoRoot(), pred) if rx.match(str(p.GetPath()))]


def _uninstance_ancestor(prim: Any) -> Any | None:
    """If ``prim`` is an instance proxy, walk up to find the first ancestor with
    ``IsInstance() == True`` and call ``SetInstanceable(False)`` on it.

    Returns the ancestor that was uninstanced, or None if no action was needed.
    Editing instance proxies directly is forbidden by USD; uninstancing the
    ancestor expands the prototype in place so children become writable.
    """
    if not prim.IsInstanceProxy():
        return None
    cur = prim
    while cur and cur.IsValid():
        if cur.IsInstance():
            cur.SetInstanceable(False)
            return cur
        cur = cur.GetParent()
    return None


# === Rule merging + application ===


def _normalize_labels(labels: dict[str, Any]) -> dict[str, str]:
    """Coerce label values to strings (semantics:labels:* are USD strings)."""
    return {str(k): str(v) for k, v in labels.items()}


def apply_semantic_rules(
    stage: Any,
    rules: list[dict[str, Any]],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply all rules in order; write merged labels via ``rep_modify.semantics``.

    Args:
        stage: an open ``Usd.Stage``.
        rules: list of ``{"match": <glob>, "labels": {<k>: <v>, ...}}``.
        dry_run: when True, skip the ``rep_modify.semantics`` write — only
            compute and return the merged label dict. Lets you validate a
            rule list with plain ``usd-core`` (no Kit needed).

    Returns:
        ``{
            "n_rules": int,
            "n_prims_affected": int,
            "n_label_keys_total": int,
            "merged": {prim_path: {key: value, ...}, ...},
            "per_rule": [{"match": ..., "labels": ..., "n_matched": int}, ...],
            "dry_run": bool,
        }``
    """
    rep_modify = None
    tag_cache = None
    if not dry_run:
        try:
            from omni.replicator.core.functional import modify as rep_modify  # type: ignore
        except ImportError:
            from omni.replicator.core.scripts.functional import modify as rep_modify  # type: ignore
        try:
            from omni.replicator.core.scripts.functional.modify import (
                TagCache as tag_cache,  # type: ignore
            )
        except ImportError:
            tag_cache = None

    merged: dict[str, dict[str, str]] = {}
    per_rule: list[dict[str, Any]] = []

    for rule in rules:
        pat = rule["match"]
        labels = _normalize_labels(rule.get("labels") or {})
        if not labels:
            logger.warning("Rule has no labels, skipping: match=%s", pat)
            per_rule.append({"match": pat, "labels": {}, "n_matched": 0})
            continue
        prims = find_matching_prims(stage, pat)
        for prim in prims:
            path = str(prim.GetPath())
            merged.setdefault(path, {}).update(labels)
        per_rule.append({"match": pat, "labels": labels, "n_matched": len(prims)})

    uninstanced_ancestors: set[str] = set()
    if not dry_run:
        # Phase A: uninstance instance ancestors so descendants become writable.
        for path in merged:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            anc = _uninstance_ancestor(prim)
            if anc is not None:
                uninstanced_ancestors.add(str(anc.GetPath()))

        # Phase B: write semantics via the functional API (matches what sdg
        # apply_semantics + defect_ops use).
        for path, labels in merged.items():
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                rep_modify.semantics(prim, value=labels, mode="replace")

        # Phase C: clear Replicator TagCache so subsequent renders rebuild
        # prim->tag mappings (mirrors the sdg defect_ops workaround).
        if tag_cache is not None:
            try:
                tag_cache._cache.clear()
                tag_cache._path_order_cache.clear()
            except AttributeError:
                logger.warning("TagCache private API changed — cache not cleared")

    stats = {
        "n_rules": len(rules),
        "n_prims_affected": len(merged),
        "n_label_keys_total": sum(len(v) for v in merged.values()),
        "n_uninstanced_ancestors": len(uninstanced_ancestors),
        "uninstanced_ancestors": sorted(uninstanced_ancestors),
        "merged": merged,
        "per_rule": per_rule,
        "dry_run": dry_run,
    }
    logger.info(
        "[semantic_rules] %d rule(s), %d prim(s) affected, %d label key(s)%s",
        stats["n_rules"],
        stats["n_prims_affected"],
        stats["n_label_keys_total"],
        " (dry-run)" if dry_run else "",
    )
    return stats


# === Standalone dry-run CLI ===


def _cli() -> None:
    import argparse

    import yaml
    from pxr import Usd

    ap = argparse.ArgumentParser(
        description="Dry-run semantic_rules against a USD (no Kit needed).",
    )
    ap.add_argument("--scene", required=True, help="USD file path")
    ap.add_argument(
        "--rules",
        required=True,
        help="YAML file with top-level 'semantics: [...]' list",
    )
    ap.add_argument(
        "--show",
        type=int,
        default=10,
        help="Print up to N matched prim paths per rule (default 10)",
    )
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        raise SystemExit(f"Cannot open USD: {args.scene}")
    with open(args.rules) as f:
        cfg = yaml.safe_load(f) or {}
    rules = cfg.get("semantics", [])
    if not rules:
        raise SystemExit(f"No 'semantics:' block in {args.rules}")

    stats = apply_semantic_rules(stage, rules, dry_run=True)

    for r in stats["per_rule"]:
        print(f"\nmatch={r['match']!r} -> {r['n_matched']} prim(s)")
        prims = find_matching_prims(stage, r["match"])[: args.show]
        for p in prims:
            print(f"  {p.GetPath()}")
        if r["n_matched"] > args.show:
            print(f"  ... and {r['n_matched'] - args.show} more")
        print(f"  labels = {r['labels']}")

    print(
        f"\nTotal: {stats['n_prims_affected']} prim(s) affected, "
        f"{stats['n_label_keys_total']} label key(s) (across all merged rules)"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _cli()
