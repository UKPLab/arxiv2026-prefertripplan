#!/usr/bin/env python3
r"""tier_completion.py — F4 branch-shortcutting on Lexicographic preferences.

Emits one row per Lexicographic instance carrying both tiers' scores, the
scope pairing, and whether the instance is scorable.  Two contrasts are
built from those rows in ``tier_completion_summary.py``:

    LEVEL 1  within a pairing        t1 - t2
    LEVEL 2  within a scope          score as tier 1  -  score as tier 2

Both are reported GRADED and BINARY, and always WITH THE LEVELS, never as
a bare difference -- a 0.000 must be readable as a ceiling (1.000-1.000)
or as a coincidence of equal means, and a difference alone hides which.

--------------------------------------------------------------------------
THE MECHANISM
--------------------------------------------------------------------------
``LexicographicPreference.evaluate`` takes the verdict from tier 1 alone:

    passed = sub_results[0].passed          # preferences.py:815

Lower tiers move the scalar score but can never flip ``passed``.  So a
planner that honours its top priority and abandons the rest passes
identically to one that honours the whole ordering.

--------------------------------------------------------------------------
WHY TWO LEVELS
--------------------------------------------------------------------------
LEVEL 1 alone cannot separate two explanations.  Comparing the
``[any]->[all]`` pairing against ``[all]->[any]`` mixes

    (a) POSITION  -- tier 2 does not gate the verdict, so it is skippable
    (b) SCOPE     -- ``[all]`` is simply harder than ``[any]``

because the two pairings put the hard scope in opposite tiers.  LEVEL 2
holds scope fixed and varies only position, which isolates (a).

    ``[any]``  appears as tier 1 in [any]->[all] and [any]->[any]
               appears as tier 2 in [any]->[any] and [all]->[any]
    ``[all]``  appears as tier 1 in [all]->[any] and [all]->[all]
               appears as tier 2 in [any]->[all] and [all]->[all]

Level 2 is therefore reported under TWO pairing sets, because the choice
changes the answer for one scope and not the other:

    VARIANT A   all pairings
    VARIANT B   only [any]->[all] and [all]->[any]

Measured: ``[all]`` is IDENTICAL under both -- structurally so, since
``[any]->[any]`` contains no ``[all]`` leaf and ``[all]->[all]`` has zero
scorable instances, so variant B removes nothing from it.  ``[any]``
swings (e.g. gemma +0.085 -> -0.003) because ``[any]->[any]``'s tier-2
scores are the low ones.  Hence the robust result is the ``[all]`` half;
the ``[any]`` half is at ceiling (0.93-1.00 in both positions) and should
not be claimed.

--------------------------------------------------------------------------
GRADED vs BINARY
--------------------------------------------------------------------------
GRADED is ``details.tiers[i].score``; BINARY is ``details.tiers[i].passed``.
For an Atomic ``[all]`` leaf the score is ``passed_count / n_entities``, a
real gradation.  For ``[any]`` it is ``float(passed)``, so graded and
binary are IDENTICAL there by construction -- not a duplication bug.

Binary runs ~3x graded on the ``[any]->[all]`` pairing because it counts a
tier-2 score of 0.94 as a total failure.  Of 44 tier-2 failures only 3
scored below 0.1 while 16 scored above 0.8, so binary alone overstates the
effect.  Both are reported; graded is the honest magnitude.

THE BIG-M SCALAR IS DELIBERATELY UNUSED.  ``details.scalar`` is exact and
invertible, but ``M = max(plan entity count) + 1`` varies per instance
(measured 2..40, median 8), so identical behaviour scores differently, and
tier 2 occupies only ``1/(M+1)`` of the range.  The per-tier scores are on
a common [0,1] scale and need no correction.

--------------------------------------------------------------------------
ATTAINABILITY — not satisfying the impossible is not shortcutting
--------------------------------------------------------------------------
An instance is SCORABLE only if BOTH tiers were JOINTLY attainable from
the shown choice set.  ``[all]`` binds PER CITY (every scheduled entity
must comply, and entities come from the city they sit in); ``[any]`` binds
TRIP-WIDE.  With A = {e : t1(e)}, B = {e : t2(e)} over the gated pool:

  all/all   forall c: |A n B|_c >= R_c
  all/any   forall c: |A|_c >= R_c       and  sum_c |A n B|_c >= 1
  any/all   forall c: |B|_c >= R_c       and  sum_c |A n B|_c >= 1
  any/any   forall c: |pool|_c >= R_c    and  ( sum_c |A n B|_c >= 1  or
                                                capacity >= 2 with a
                                                witness for each )

The all/all row subsumes the structural mutual-exclusivity check: bank 11
(``mode == flight [all]`` over ``mode in [self-driving, taxi] [all]``) has
|A n B| = 0 and drops out unaided.  A shape-based rule was considered and
rejected -- it would wrongly exclude a NESTED pair such as
``rating >= 3 [all]`` over ``rating >= 4 [all]``, which is compatible.

FLOOR FOR ``[all]``, CEILING FOR ``[any]``.  Attainability asks whether a
VALID plan satisfying both tiers exists, so each tier gets the most
permissive valid plan -- and permissive points in opposite directions:
``[all]`` is easier in a SMALLER plan (fewer entities must comply) so it
takes the FLOOR; ``[any]`` is easier in a LARGER plan (more slots host
more distinct witnesses) so its hosting capacity is the CEILING.  Using
the floor for both capped attraction capacity at 1 per city and wrongly
forced a single entity to satisfy both ``[any]`` tiers of bank 14.
Accommodation is where floor and ceiling coincide (one stay per city),
which is why bank 7's exclusions are correct.

  FLOOR    per city   Restaurant 3, Attraction 1, Accommodation 1,
                      Transportation 1
  CEILING  per trip   Restaurant 3/day, Attraction 4/day, Accommodation
                      and Transportation 1 per city

``is_not_absent`` exempts meals on transit days and attractions on
transit-or-travel days, so a minimum-viable plan needs entities only on
the REQUIRED days.  Measured: exactly ONE required day per city (99.7% /
90.6% / 90.6% of plans at 1 / 2 / 3 cities), which makes R an INTEGER and
removes the per-city day split from the filter.  ``plan_geography``'s
fractional ``days_c`` is right for an EXPECTATION but wrong for a
FEASIBILITY THRESHOLD -- "at least 7.5 qualifying candidates" has no
meaning as a count, and the rounding direction would change the verdict.

MODEL-INDEPENDENT BY CONSTRUCTION.  Requirement and geography come from
the RECORD, never the plan.  Using ``plan_geography`` would let a model
shrink its own denominator and move its own attainability -- the F2
denominator-control problem imported into F4's gate -- and would give each
model a different scorable set.  Verified: scorable counts are identical
across all five models.

--------------------------------------------------------------------------
TIER STATES — reported alongside, because 0.5 is ambiguous
--------------------------------------------------------------------------
``n_passed_tiers / n_tiers`` maps two OPPOSITE behaviours to 0.5:

    tier1 PASS, tier2 FAIL  ->  verdict PASS  ->  the shortcut
    tier1 FAIL, tier2 PASS  ->  verdict FAIL  ->  priority INVERSION

so prefix depth is kept instead:

    k = prefix_depth   largest k with tiers 1..k ALL passed
    m = n_passed       total tiers passed
    m > k   <=>   a satisfied tier below a failed one   <=>   INVERSION

Measured: gemma posts 7 inversions in 31 scorable ``[all]->[any]``
instances, so the collapse is real, not hypothetical.

WHAT F4 DOES NOT COVER
  Compensatory   compensation is a DESIGNED affordance; using it is not
                 prima facie lazy.
  Composite OR   ``details`` keeps ``passed_children`` as a COUNT, not
                 which child, so "cheapest branch" is untestable without
                 a preferences.py change.
  Conditional    its branch choice is the gate dodge, already measured in
                 the drift/gate analysis.  Would double-count.

USAGE
    python3 tier_completion.py --split test_large --model qwen3.8-27b \
        --eval-file ../qwen3.8-27b_test_large/eval_qwen3.8-27b.jsonl \
        --out runs/tier_qwen3.8-27b_test_large.txt \
        --json-out runs/tier_qwen3.8-27b_test_large.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pool_common import (build_pools, qualify as _qualify,        # noqa: E402
                         loads as _loads, gating_all_leaves)
from evaluate_preferences import _casefold_literals               # noqa: E402

REQUIRED_DAYS_PER_CITY = 1
FLOOR_PER_REQUIRED_DAY = {"Restaurant": 3, "Attraction": 1}
FLOOR_PER_CITY = {"Accommodation": 1, "Transportation": 1}
CEILING_PER_DAY = {"Restaurant": 3, "Attraction": 4}
CEILING_PER_CITY = {"Accommodation": 1, "Transportation": 1}

_LC_PREDICATE = {
    "cuisine":    ("Restaurant", "cuisine", "in"),
    "house rule": ("Accommodation", "house_rules", "not_in"),
    "room type":  ("Accommodation", "room_type", "=="),
}

# scope pair -> pairing label.  These four are exhaustive at 2 tiers.
PAIRINGS = {("any", "all"): "any->all", ("any", "any"): "any->any",
            ("all", "any"): "all->any", ("all", "all"): "all->all"}
PAIRING_ORDER = ("any->all", "any->any", "all->any", "all->all")
# Level-2 variant B keeps only the pairings whose two tiers differ in scope.
VARIANT_B = ("any->all", "all->any")


def floor_required(entity_type: str) -> int:
    if entity_type in FLOOR_PER_REQUIRED_DAY:
        return FLOOR_PER_REQUIRED_DAY[entity_type] * REQUIRED_DAYS_PER_CITY
    return FLOOR_PER_CITY.get(entity_type, 1)


def host_capacity(entity_type: str, days: int, n_cities: int) -> int:
    if entity_type in CEILING_PER_DAY:
        return CEILING_PER_DAY[entity_type] * max(days, 1)
    return CEILING_PER_CITY.get(entity_type, 1) * max(n_cities, 1)


def lc_leaf(local_constraint: Any, entity_type: str) -> dict | None:
    lc = _loads(local_constraint) or {}
    if not isinstance(lc, dict):
        return None
    for key, (et, attr, op) in _LC_PREDICATE.items():
        if et != entity_type:
            continue
        val = lc.get(key)
        if val in (None, "", [], {}):
            continue
        return {"entity_type": et, "attribute": attr, "op": op,
                "value": val, "scope": "all"}
    return None


def leaf_of(node: Any) -> dict:
    t = node.get("template", node) if isinstance(node, dict) else {}
    return {k: t.get(k) for k in
            ("entity_type", "attribute", "op", "value", "scope")}


# --------------------------------------------------------------------------- #
# Attainability                                                                #
# --------------------------------------------------------------------------- #

def _city_sets(pools: dict, et: str, cities: list[str], gating: list[dict],
               lc: dict | None, t1: dict | None, t2: dict | None) -> dict:
    out: dict[str, dict] = {}
    for c in cities:
        pool = (pools.get(c) or {}).get(et) or []
        for g in gating:
            pool = _qualify(pool, g)
        if lc is not None:
            pool = _qualify(pool, lc)
        A = _qualify(pool, t1) if t1 else list(pool)
        B = _qualify(pool, t2) if t2 else list(pool)
        out[c] = {"pool": len(pool), "A": len(A), "B": len(B),
                  "AB": len({id(x) for x in A} & {id(x) for x in B})}
    return out


def _attainable_single(cs: dict, R: dict, scope: str) -> tuple[bool, str]:
    slots = {c: R[c] for c in cs if R.get(c, 0) > 0}
    if not slots:
        return False, "no-slots"
    if scope == "all":
        ok = all(cs[c]["A"] >= R[c] for c in slots)
        return ok, "" if ok else "[all] short"
    ok = sum(cs[c]["A"] for c in slots) >= 1
    return ok, "" if ok else "no witness"


def _attainable_joint(cs: dict, R: dict, s1: str, s2: str,
                      capacity: int = 1) -> tuple[bool, str]:
    slots = {c: R[c] for c in cs if R.get(c, 0) > 0}
    if not slots:
        return False, "no-slots"
    if s1 == "all" and s2 == "all":
        ok = all(cs[c]["AB"] >= R[c] for c in slots)
        return ok, "" if ok else "t1&t2 disjoint / insufficient"
    if s1 == "all" and s2 == "any":
        if not all(cs[c]["A"] >= R[c] for c in slots):
            return False, "t1 [all] short"
        ok = sum(cs[c]["AB"] for c in slots) >= 1
        return ok, "" if ok else "no t2 witness inside t1"
    if s1 == "any" and s2 == "all":
        if not all(cs[c]["B"] >= R[c] for c in slots):
            return False, "t2 [all] short"
        ok = sum(cs[c]["AB"] for c in slots) >= 1
        return ok, "" if ok else "no t1 witness inside t2"
    if not all(cs[c]["pool"] >= R[c] for c in slots):
        return False, "pool short"
    if sum(cs[c]["AB"] for c in slots) >= 1:
        return True, ""
    if capacity >= 2 and sum(cs[c]["A"] for c in slots) >= 1 \
            and sum(cs[c]["B"] for c in slots) >= 1:
        return True, ""
    return False, "cannot host both [any] witnesses"


def attainability(rec: dict, prefs: list, entry: dict,
                  t1: dict, t2: dict) -> dict:
    pools = build_pools(rec.get("reference_information"),
                        people_number=int(rec.get("people_number") or 1))
    e1, e2 = t1["entity_type"], t2["entity_type"]
    exclude = (entry.get("paradigm"), entry.get("bank_id"))

    def cities_for(et: str) -> list[str]:
        # Only cities that CARRY this entity type get slots.  `pools` also
        # holds the origin city (Transportation blocks only); charging it a
        # restaurant/accommodation quota made every [all] tier vacuously
        # unattainable in an earlier draft.
        return sorted(c for c in pools if (pools.get(c) or {}).get(et))

    if e1 == e2:
        cl = cities_for(e1)
        R = {c: floor_required(e1) for c in cl}
        cs = _city_sets(pools, e1, cl, gating_all_leaves(prefs, e1, exclude),
                        lc_leaf(rec.get("local_constraint"), e1), t1, t2)
        ok1, r1 = _attainable_single(cs, R, t1["scope"])
        okj, rj = _attainable_joint(
            cs, R, t1["scope"], t2["scope"],
            capacity=host_capacity(e1, int(rec.get("days") or 0), len(cl)))
        audit = {e1: cs}
    else:
        c1, c2 = cities_for(e1), cities_for(e2)
        R1 = {c: floor_required(e1) for c in c1}
        R2 = {c: floor_required(e2) for c in c2}
        cs1 = _city_sets(pools, e1, c1, gating_all_leaves(prefs, e1, exclude),
                         lc_leaf(rec.get("local_constraint"), e1), t1, None)
        cs2 = _city_sets(pools, e2, c2, gating_all_leaves(prefs, e2, exclude),
                         lc_leaf(rec.get("local_constraint"), e2), t2, None)
        ok1, r1 = _attainable_single(cs1, R1, t1["scope"])
        ok2, r2 = _attainable_single(cs2, R2, t2["scope"])
        # Different entity types: tier 1 cannot constrain tier 2's pool.
        okj, rj = (ok1 and ok2), (r1 or r2)
        audit = {e1: cs1, e2: cs2}

    return {"scorable": bool(okj),
            "filter_state": ("SCORABLE" if okj else
                             "t1-unattainable" if not ok1 else
                             "t2-unattainable"),
            "exclusion_reason": (rj or r1) if not okj else "",
            "cross_entity": e1 != e2, "pool_audit": audit,
            "required_per_city": {e1: floor_required(e1),
                                  e2: floor_required(e2)}}


def tier_state(tiers: list[dict]) -> dict:
    """Prefix depth and passed count.  0.5 on n_passed/n_tiers is ambiguous
    between tier-1-only (a PASS, the shortcut) and tier-2-only (a FAIL,
    priority inversion); prefix depth separates them."""
    p = [bool(t.get("passed")) for t in tiers]
    n = len(p)
    k = 0
    for x in p:
        if not x:
            break
        k += 1
    m = sum(p)
    label = ("complete" if k == n else "inverted" if m > k
             else "prefix_stop" if k >= 1 else "none")
    return {"n_tiers": n, "prefix_depth": k, "n_passed": m,
            "prefix_completion": (k / n) if n else None,
            "inversion": m > k, "state": label, "gate_passed": k >= 1}


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def analyse(dataset: str, split: str, model: str,
            eval_file: Path) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, verification_mode="no_checks")
    records = {int(r["id"]): r for r in ds}
    rows: list[dict] = []
    diag = Counter()

    with eval_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            rid = int(ev["id"])
            rec = records.get(rid)
            if rec is None:
                diag["no_record"] += 1
                continue
            prefs = _casefold_literals(_loads(rec.get("preferences_json")) or [])
            by_bank = {e.get("bank_id"): e for e in prefs
                       if e.get("paradigm") == "LexicographicPreference"}
            for pref in (ev.get("preferences") or []):
                if pref.get("paradigm") != "LexicographicPreference":
                    continue
                det = pref.get("details") or {}
                tiers = det.get("tiers") or []
                if len(tiers) != 2:
                    diag["arity_ne_2"] += 1
                    continue
                entry = by_bank.get(pref.get("bank_id"))
                if entry is None:
                    diag["bank_unmatched"] += 1
                    continue
                tmpl = (entry.get("template") or {}).get("preferences") or []
                if len(tmpl) != 2:
                    diag["tier_count_mismatch"] += 1
                    continue
                t1, t2 = leaf_of(tmpl[0]), leaf_of(tmpl[1])
                st = tier_state(tiers)
                at = attainability(rec, prefs, entry, t1, t2)
                # The verdict IS tier 1's verdict; assert rather than assume,
                # since a mismatch would mean details' tier order does not
                # match preferences_json.
                if bool(pref.get("passed")) != st["gate_passed"]:
                    diag["verdict_mismatch"] += 1
                rows.append({
                    "id": rid, "split": split, "model": model,
                    "bank_id": pref.get("bank_id"),
                    "pairing": PAIRINGS[(t1["scope"], t2["scope"])],
                    "scope_1": t1["scope"], "scope_2": t2["scope"],
                    "entity_1": t1["entity_type"], "entity_2": t2["entity_type"],
                    "t1": t1, "t2": t2,
                    # the two contrasts are built from exactly these four
                    "graded_1": float(tiers[0]["score"]),
                    "graded_2": float(tiers[1]["score"]),
                    "binary_1": float(bool(tiers[0]["passed"])),
                    "binary_2": float(bool(tiers[1]["passed"])),
                    "entity_count_1": int(tiers[0].get("entity_count") or 0),
                    "entity_count_2": int(tiers[1].get("entity_count") or 0),
                    "evaluator_passed": bool(pref.get("passed")),
                    "big_m": det.get("M"), "scalar": det.get("scalar"),
                    **st,
                    **{k: v for k, v in at.items() if k != "pool_audit"},
                    "pool_audit": at["pool_audit"],
                })
    return rows, {"n_rows": len(rows), **dict(diag)}


def report(model: str, split: str, rows: list[dict], diag: dict) -> str:
    def m(v):
        return sum(v) / len(v) if v else None

    def f3(v):
        return " --  " if v is None else f"{v:.3f}"

    def fd(v):
        return "  --  " if v is None else f"{v:+.3f}"

    L = [f"TIER COMPLETION — {model} / {split}", "=" * 78,
         "  Lexicographic branch-shortcutting.  passed = tier 1 only, so",
         "  abandoning tier 2 is free to the verdict.",
         "  Levels are printed with every difference: a 0.000 must be",
         "  readable as a ceiling or as equal means.", ""]
    sc = [r for r in rows if r["scorable"]]
    L.append("  LEVEL 1 — within pairing (scorable)")
    L.append(f"    {'pairing':<10s}{'n':>4s}{'t1 g':>7s}{'t2 g':>7s}{'diff':>8s}"
             f"{'  ':>2s}{'t1 b':>7s}{'t2 b':>7s}{'diff':>8s}")
    for pr in PAIRING_ORDER:
        S = [r for r in sc if r["pairing"] == pr]
        if not S:
            L.append(f"    {pr:<10s}{0:>4d}" + "      --" * 6)
            continue
        g1, g2 = m([r["graded_1"] for r in S]), m([r["graded_2"] for r in S])
        b1, b2 = m([r["binary_1"] for r in S]), m([r["binary_2"] for r in S])
        L.append(f"    {pr:<10s}{len(S):>4d}{f3(g1):>7s}{f3(g2):>7s}"
                 f"{fd(g1-g2):>8s}{'  ':>2s}{f3(b1):>7s}{f3(b2):>7s}"
                 f"{fd(b1-b2):>8s}")
    L.append("")
    L.append("  LEVEL 2 — within scope, position varied (scorable)")
    for vlab, keep in (("A: all pairings", PAIRING_ORDER),
                       ("B: any->all + all->any", VARIANT_B)):
        S = [r for r in sc if r["pairing"] in keep]
        L.append(f"    variant {vlab}")
        L.append(f"      {'scope':<7s}{'as t1':>7s}{'as t2':>7s}{'diff':>8s}"
                 f"{'  n1/n2':>9s}{'  ':>2s}{'b t1':>7s}{'b t2':>7s}{'diff':>8s}")
        for scope in ("any", "all"):
            a1 = [r["graded_1"] for r in S if r["scope_1"] == scope]
            a2 = [r["graded_2"] for r in S if r["scope_2"] == scope]
            B1 = [r["binary_1"] for r in S if r["scope_1"] == scope]
            B2 = [r["binary_2"] for r in S if r["scope_2"] == scope]
            d = (m(a1) - m(a2)) if a1 and a2 else None
            db = (m(B1) - m(B2)) if B1 and B2 else None
            L.append(f"      [{scope}]{'':<3s}{f3(m(a1)):>7s}{f3(m(a2)):>7s}"
                     f"{fd(d):>8s}{f'{len(a1)}/{len(a2)}':>9s}{'  ':>2s}"
                     f"{f3(m(B1)):>7s}{f3(m(B2)):>7s}{fd(db):>8s}")
    L += ["", "  tier states (scorable): " + str(
        dict(Counter(r["state"] for r in sc))),
        f"  scorable {len(sc)} of {len(rows)}",
        f"  diagnostics: {diag}"]
    if diag.get("verdict_mismatch"):
        L.append("  !! verdict_mismatch > 0 — details tier order may not match "
                 "preferences_json")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="UKPLab/PreferTripPlan")
    ap.add_argument("--split", default="test_large")
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows, diag = analyse(args.dataset, args.split, args.model, args.eval_file)
    txt = report(args.model, args.split, rows, diag)
    print(txt)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"diagnostics": diag, "rows": rows}, indent=2, default=str))
        print(f"[out] {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
