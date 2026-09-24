#!/usr/bin/env python3
"""pool_common.py — the candidate pool, shared by every laziness analysis.

WHY SHARED.  Each F-family analysis (F2 window compression, F4 branch
shortcutting, ...) needs the same two primitives: the choice set the
planner was SHOWN, and a predicate test against it.  Duplicating them per
analysis invites silent divergence -- two modules disagreeing about
whether a pool entity qualifies would make their numbers incomparable for
no visible reason.  So both live here and every analysis imports them.

THE POOL IS ``reference_information``, NOT THE DB.  The HF
``reference_information`` column is the retrieved candidate set the
planner was actually shown.  That is the correct denominator for "could
it have done better": the TravelPlanner DB contains entities the model
never saw, so DB-derived availability would overstate its real options.

ATTRIBUTE RENAMING.  The retrieval schema and the preference-bank schema
differ (``categories`` vs ``category``, ``cuisines`` vs ``cuisine``,
``house_rules_list`` vs ``house_rules``).  Renaming here means the SAME
``AtomicPreference._check_one`` the evaluator uses can be applied to pool
entities -- reimplementing predicate logic would risk drifting from
evaluation semantics.

TRANSPORTATION.  Its ``mode`` is not a field in the retrieval blocks; it
is implied by the block Description (Flight / Self-driving / Taxi).  We
set it to the canonical LOWER-CASE value ``_parse_transportation`` in
``evaluate_preferences`` produces, so a pool entity and a plan entity are
comparable.  NOTE four preference-bank entries spell the literal as
``"Flight"`` (capital F) and therefore match nothing -- the same casefold
defect handled at the evaluation boundary by
``evaluate_preferences._casefold_literals``.  Callers that care about
mode predicates must apply the same fold or exclude those entries.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from preferences import AtomicPreference               # noqa: E402

# Retrieval block Description prefix -> (entity_type, attribute renames).
_BLOCKS: dict[str, tuple[str, dict[str, str]]] = {
    "Attractions":    ("Attraction",     {"categories": "category"}),
    "Restaurants":    ("Restaurant",     {"cuisines": "cuisine"}),
    "Accommodations": ("Accommodation",  {"house_rules_list": "house_rules"}),
    "Flight":         ("Transportation", {}),
    "Self-driving":   ("Transportation", {}),
    "Taxi":           ("Transportation", {}),
}
# Canonical lower-case modes, matching evaluate_preferences._MODE_TOKENS.
_MODE_OF = {"Flight": "flight", "Self-driving": "self-driving", "Taxi": "taxi"}


_DESC_TO = __import__("re").compile(r"\bto\s+(.+?)(?:\s+on\s|$)", __import__("re").I)


def _dest_from_desc(description: str) -> str | None:
    """Destination city from a block Description such as
    ``"Self-driving from Washington to Tampa"``."""
    m = _DESC_TO.search(description or "")
    return m.group(1).strip() if m else None


def loads(v: Any) -> Any:
    """Tolerant JSON load: HF ships these columns as strings."""
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (TypeError, ValueError):
            return None
    return None


def build_pools(reference_information: Any,
                entity_types: tuple[str, ...] | None = None,
                people_number: int | None = None
                ) -> dict[str, dict[str, list[dict]]]:
    """``{city -> {entity_type -> [entity, ...]}}`` from the shown choice set.

    ``entity_types`` restricts what is collected (default: everything).
    ``people_number``, when given, drops accommodations whose
    ``maximum_occupancy`` cannot seat the party -- such a listing can
    satisfy a room_type or rating predicate yet be unusable, and the
    hard-constraint evaluator enforces occupancy anyway, so counting it
    as available would inflate the denominator.
    """
    pools: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    for block in (loads(reference_information) or []):
        if not isinstance(block, dict):
            continue
        desc = str(block.get("Description") or "")
        kind = next((k for k in _BLOCKS if desc.startswith(k)), None)
        if kind is None:
            continue
        entity_type, renames = _BLOCKS[kind]
        if entity_types is not None and entity_type not in entity_types:
            continue
        content = block.get("Content")
        # Ground-transport blocks carry a single dict, not a list.
        rows = (content if isinstance(content, list)
                else [content] if isinstance(content, dict) else [])
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            e = dict(raw)
            for src, dst in renames.items():
                if src in e:
                    e[dst] = e.pop(src)
            if entity_type == "Transportation":
                e["mode"] = _MODE_OF.get(kind, kind.lower())
                # Ground-transport blocks are a single dict with no
                # dest_city/origin_city, so fall back to parsing the
                # Description ("Self-driving from A to B").  Bucketing
                # them under "?" instead would create a phantom city, and
                # dropping them (as candidate_pool.py does by skipping
                # non-list Content) loses every self-driving / taxi
                # option from the pool.
                city = str(e.get("dest_city") or _dest_from_desc(desc)
                           or e.get("origin_city") or "?")
            else:
                city = str(e.get("city") or "?")
            if entity_type == "Accommodation" and people_number:
                cap = e.get("maximum_occupancy")
                if isinstance(cap, (int, float)) and cap < people_number:
                    continue
            e["entity_type"] = entity_type
            pools[city][entity_type].append(e)
    return {c: dict(d) for c, d in pools.items()}


def qualify(entities: list[dict], leaf: dict | None) -> list[dict]:
    """Entities satisfying ``leaf``, via the EVALUATOR's own predicate.

    ``leaf`` is a preferences_json template block: ``entity_type``,
    ``attribute``, ``op``, ``value``, ``scope``.  ``None`` means "no
    predicate", so everything qualifies -- which is what makes a bare
    Numeric preference's share come out at 1.0.
    """
    if leaf is None:
        return list(entities)
    ap = AtomicPreference(leaf["entity_type"], leaf["attribute"],
                          leaf["op"], leaf["value"],
                          leaf.get("scope") or "any")
    return [e for e in entities if ap._check_one(e)]


def leaf_of(template_block: Any) -> dict | None:
    """Normalise a preferences_json node to a flat leaf dict, unwrapping
    the ``{"class": ..., "template": {...}}`` form nested children use."""
    if not isinstance(template_block, dict):
        return None
    t = template_block.get("template", template_block)
    if not isinstance(t, dict) or not t.get("entity_type"):
        return None
    return {k: t.get(k) for k in
            ("entity_type", "attribute", "op", "value", "scope")}


# --------------------------------------------------------------------------- #
# Gating [all] predicates — what a PARTNER preference forecloses               #
# --------------------------------------------------------------------------- #
# ~2/3 of records carry two preferences, and where both bind the same
# entity type, satisfying one shrinks the pool available to the other.
# Measured on the F2 in-scope set: 41 of 107 instances (test_large) have
# such a partner and 83 city-pools shrink, one from 24 candidates to 8.
# Ignoring this leaves n_max too LARGE and therefore WCI too SMALL --
# overstating shrinkage, the wrong direction for a conservative claim.
#
# Only predicates that actually GATE ``passed`` may filter the pool.  This
# is not "every [all] leaf":
#
#   Composite AND children   bind      (all must hold)
#   Composite OR / NOT       DO NOT    (a union; one child suffices)
#   Lexicographic tier 1     binds     (passed = sub_results[0].passed)
#   Lexicographic tiers 2+   DO NOT    (cannot change the verdict)
#   Compensatory margin_ap   binds     (tier-3 = fails MARGIN)
#   Compensatory primary_ap  DOES NOT  (tier-2 is rescuable)
#   Conditional branches     DO NOT    (endogenous: the planner chooses
#                                       whether to trigger the condition)
#   Temporal subject/reference  DO NOT (role selectors, not filters)
#   Scoped inner             DOES NOT  (binds only on scoped days)

_ENDOGENOUS_OR_SCOPED = ("ConditionalPreference", "TemporalPreference",
                         "ScopedPreference", "NumericPreference")


def gating_all_leaves(preferences_json: list, entity_type: str,
                      exclude: tuple | None = None) -> list[dict]:
    """``[all]``-scoped leaves on ``entity_type`` whose failure would make
    their own preference fail.

    ``exclude`` is a ``(paradigm, bank_id)`` pair skipped entirely -- pass
    the instance under analysis so it never filters itself.
    """
    out: list[dict] = []

    def walk(paradigm: str | None, tmpl: Any) -> None:
        if not isinstance(tmpl, dict):
            return
        if paradigm == "CompositePreference":
            if str(tmpl.get("op") or "").upper() != "AND":
                return
            for child in (tmpl.get("children") or []):
                if isinstance(child, dict):
                    walk(child.get("class") or child.get("paradigm"),
                         child.get("template", child))
            return
        if paradigm == "LexicographicPreference":
            tiers = tmpl.get("preferences") or []
            if tiers and isinstance(tiers[0], dict):
                walk(tiers[0].get("class") or tiers[0].get("paradigm"),
                     tiers[0].get("template", tiers[0]))
            return
        if paradigm == "CompensatoryPreference":
            margin = tmpl.get("margin_ap")
            if isinstance(margin, dict):
                walk(margin.get("class") or margin.get("paradigm"),
                     margin.get("template", margin))
            return
        if paradigm in _ENDOGENOUS_OR_SCOPED:
            return
        if tmpl.get("scope") == "all" and tmpl.get("entity_type") == entity_type:
            out.append({k: tmpl.get(k) for k in
                        ("entity_type", "attribute", "op", "value", "scope")})

    for entry in (preferences_json or []):
        if exclude is not None and (entry.get("paradigm"),
                                    entry.get("bank_id")) == exclude:
            continue
        walk(entry.get("paradigm"), entry.get("template") or {})
    return out
