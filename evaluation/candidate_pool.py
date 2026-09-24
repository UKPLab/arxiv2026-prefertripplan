#!/usr/bin/env python3
"""candidate_pool.py — availability baseline for preference satisfaction.

WHY THIS EXISTS
---------------
Every "the planner was lazy" claim needs a denominator.  A plan that fails
a universally-scoped cuisine preference in a city offering three
restaurants is not lazy -- it is constrained.  Without an availability
baseline, laziness measures (denominator-shrinking, satisficing,
branch-shortcutting) and the Suppression Index are all confounded by what
the choice set actually permitted.

THE POOL IS ``reference_information``, NOT THE DB
-------------------------------------------------
The HF ``reference_information`` column is the retrieved candidate set the
planner was SHOWN.  That is the correct denominator for "could it have
done better", and it is strictly better than querying the TravelPlanner DB
directly: the DB contains entities the model never saw, so DB-derived
availability would overstate the planner's real options.  Blocks scale
with the trip (10 / 16 / 22 for 1 / 2 / 3 cities) and carry every
attribute the preference bank references:

    Attractions      categories, rating, city
    Restaurants      cuisines, cost, rating, city
    Accommodations   house_rules_list, room_type, cost, rating,
                     maximum_occupancy, minimum_nights, city
    Flight / Self-driving / Taxi   price, distance, elapsed

PREFERENCES ARE NOT INDEPENDENT
-------------------------------
Marginal availability (one leaf at a time) is only an UPPER BOUND.  In
test_large, 638/1000 records carry two preferences and 449 of those bind
the SAME entity type, so satisfying one shrinks the pool for the other.
Sub-preferences interact too -- a ScopedPreference's ``scope_filters``
narrow what its ``inner`` ever sees, exactly as ``filtered_plan`` does in
``preferences.py``.  So availability is reported on a three-tier ladder:

  MARGINAL     per leaf, ignoring everything else.  Upper bound.
  CONDITIONAL  per leaf, on the pool left after every OTHER [all]-scoped
               predicate binding the same entity type (siblings AND the
               pair partner) has been honoured.  "How much room was left."
  JOINT        does a slot assignment exist at all?  Because every [all]
               predicate must hold on every selected entity, the eligible
               set is their intersection E, and [any] witnesses must come
               from inside E:

                   E = { e in pool : all [all] predicates hold on e }
                   feasible  <=>  |E| >= n_required
                                  and every [any] predicate has a witness in E

               Exact and cheap at these sizes; no search required.

FEASIBILITY IS THE GATE
-----------------------
Each (record, entity_type) is classified ``infeasible`` / ``tight`` /
``comfortable``.  Laziness measures only carry a laziness interpretation
in the ``comfortable`` stratum -- a failure under ``infeasible`` is
arithmetic, not behaviour.

KNOWN LIMITS, NOT PAPERED OVER
------------------------------
  * BUDGET is a global knapsack ACROSS entity types: entities can each be
    available yet jointly unaffordable.  This module is per-entity-type
    and does not see that; it is reported as a separate flag, never
    folded into feasibility.
  * TEMPORAL ORDERING and Compensatory cross-entity day-coupling need a
    day-level assignment, not set membership.  Entity availability is
    computed for them; the ordering dimension is explicitly UNCOVERED.
  * Days-per-city is approximated as ``days / visiting_city_number``;
    the dataset carries no day->city map.

USAGE
-----
  python3 evaluation/candidate_pool.py --split test_large
  python3 evaluation/candidate_pool.py --split test --json-out pools.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


DATASET = "UKPLab/PreferTripPlan"

# Pool construction, predicate testing and tolerant JSON loading come
# from the SHARED ``pool_common`` module.  This file previously carried
# its own copy; two implementations disagreeing about which pool entities
# qualify would make the F-family analyses incomparable, and they had in
# fact already diverged (this copy dropped every ground-transport option
# by skipping non-list Content, while the other bucketed them under a
# phantom "?" city).
from pool_common import (build_pools, qualify as _qualify,   # noqa: E402
                         loads as _loads)

# Slots the plan must fill per city per day, from the TravelPlanner
# commonsense constraints the benchmark already enforces (three meals a
# day, at least one attraction a day, accommodation per night).
_SLOTS_PER_DAY = {"Restaurant": 3, "Attraction": 1, "Accommodation": 1}
# A pool is "tight" when it barely covers the requirement: a failure
# there is nearly forced, so only ``comfortable`` supports a laziness
# reading.
_TIGHT_MULTIPLE = 1.5


# --------------------------------------------------------------------------- #
# Leaf extraction                                                              #
# --------------------------------------------------------------------------- #
_CHILD_KEYS = {
    "CompositePreference":     ("children",),
    "ConditionalPreference":   ("condition", "then_pref", "else_pref"),
    "LexicographicPreference": ("preferences",),
    "CompensatoryPreference":  ("primary_ap", "margin_ap", "secondary_ap"),
    "ScopedPreference":        ("scope_filters", "inner"),
    "TemporalPreference":      ("subject_ap", "reference_ap"),
}


def leaf_predicates(paradigm: str | None, template: Any,
                    path: tuple = (), depth: int = 0) -> list[dict]:
    """Every quantified leaf under ``template``, with its structural role.

    ``role`` records WHERE the leaf sits, because the dependency structure
    differs by position: a ScopedPreference ``scope_filter`` narrows the
    pool its ``inner`` sees, and a ConditionalPreference ``condition`` is
    ENDOGENOUS -- the planner chooses whether to make it true.
    """
    out: list[dict] = []
    if depth > 12 or not isinstance(template, dict):
        return out
    if paradigm in _CHILD_KEYS:
        for key in _CHILD_KEYS[paradigm]:
            value = template.get(key)
            if value is None:
                continue
            items = value if isinstance(value, list) else [value]
            for i, child in enumerate(items):
                if not isinstance(child, dict):
                    continue
                out.extend(leaf_predicates(
                    child.get("class") or child.get("paradigm"),
                    child.get("template", child),
                    path + ((key, i) if isinstance(value, list) else (key,)),
                    depth + 1))
        return out
    if paradigm == "NumericPreference":
        return out          # optimization, not a membership predicate
    scope = template.get("scope")
    if scope in ("all", "any") and template.get("entity_type"):
        out.append({
            "entity_type": template["entity_type"],
            "attribute":   template.get("attribute"),
            "op":          template.get("op"),
            "value":       template.get("value"),
            "scope":       scope,
            "path":        path,
            "role":        (path[0] if path else "root"),
        })
    return out


# --------------------------------------------------------------------------- #
# The availability ladder                                                      #
# --------------------------------------------------------------------------- #

def analyse_record(record: dict) -> dict[str, Any]:
    """Marginal / conditional / joint availability for one HF record."""
    prefs = _loads(record.get("preferences_json")) or []
    people = int(record.get("people_number") or 1)
    days = int(record.get("days") or 0)
    n_cities = max(int(record.get("visiting_city_number") or 1), 1)
    pools = build_pools(record.get("reference_information"),
                        people_number=people)

    # No day->city map exists in the dataset, so slots are split evenly.
    # Recorded on the output so the approximation stays visible.
    days_per_city = days / n_cities if n_cities else days

    leaves: list[dict] = []
    for entry in prefs:
        for lf in leaf_predicates(entry.get("paradigm"),
                                  entry.get("template") or {}):
            lf = dict(lf)
            lf["paradigm"] = entry.get("paradigm")
            lf["bank_id"] = entry.get("bank_id")
            leaves.append(lf)

    # Cross-preference binding is real: preference_pair.type == "overlapping"
    # means both preferences bind the same entity type.  Grouping leaves by
    # entity_type reproduces that label, and the reproduction is reported as
    # a cross-check rather than the label being trusted blindly.
    by_type: dict[str, list[dict]] = defaultdict(list)
    for lf in leaves:
        by_type[lf["entity_type"]].append(lf)

    out_leaves: list[dict] = []
    feasibility: dict[str, dict] = {}

    for etype, group in by_type.items():
        required_per_city = _SLOTS_PER_DAY.get(etype)
        n_required = (round(required_per_city * days_per_city)
                      if required_per_city else None)

        # [all]-scoped leaves bind EVERY selected entity and so shrink each
        # other's pool.  A Conditional ``condition`` is excluded from the
        # binding set because it is ENDOGENOUS -- the planner may choose not
        # to trigger it, so treating it as a hard filter understates room.
        binding = [lf for lf in group
                   if lf["scope"] == "all" and lf["role"] != "condition"]
        any_leaves = [lf for lf in group if lf["scope"] == "any"]

        per_city: dict[str, dict] = {}
        for city, pool_by_type in pools.items():
            pool = pool_by_type.get(etype) or []
            if not pool:
                continue
            # JOINT eligible set: intersection of all binding [all] leaves.
            E = list(pool)
            for lf in binding:
                E = _qualify(E, lf)
            witnesses_ok = all(bool(_qualify(E, lf)) for lf in any_leaves)
            enough = (n_required is None) or (len(E) >= n_required)
            per_city[city] = {
                "n_pool": len(pool), "n_eligible": len(E),
                "n_required": n_required,
                "witnesses_ok": witnesses_ok,
                "feasible": bool(enough and witnesses_ok),
            }
            for lf in group:
                marg = _qualify(pool, lf)
                residual = list(pool)
                for other in binding:
                    if other is lf:
                        continue
                    residual = _qualify(residual, other)
                cond = _qualify(residual, lf)
                out_leaves.append({
                    "city": city, "entity_type": etype,
                    "paradigm": lf["paradigm"], "bank_id": lf["bank_id"],
                    "attribute": lf["attribute"], "op": lf["op"],
                    "scope": lf["scope"], "role": lf["role"],
                    "n_pool": len(pool),
                    "n_marginal": len(marg),
                    "avail_marginal": len(marg) / len(pool),
                    "n_residual": len(residual),
                    "n_conditional": len(cond),
                    "avail_conditional": (len(cond) / len(residual)
                                          if residual else 0.0),
                    "n_required": n_required,
                    # Per-member availability for set predicates: this is
                    # what de-confounds the Suppression Index, since a member
                    # with zero candidates cannot have been "suppressed".
                    "per_member": (
                        {str(v): len(_qualify(pool, {**lf, "op": "in",
                                                     "value": [v]}))
                         for v in lf["value"]}
                        if isinstance(lf["value"], (list, tuple)) else None),
                })

        if per_city:
            n_elig_min = min(c["n_eligible"] for c in per_city.values())
            all_feasible = all(c["feasible"] for c in per_city.values())
            if n_required:
                ratio = n_elig_min / n_required
                klass = ("infeasible" if not all_feasible else
                         "tight" if ratio < _TIGHT_MULTIPLE else "comfortable")
            else:
                klass = "comfortable" if all_feasible else "infeasible"
            feasibility[etype] = {
                "class": klass, "n_required": n_required,
                "n_eligible_min_city": n_elig_min,
                "per_city": per_city,
                "n_binding_all": len(binding), "n_any": len(any_leaves),
            }

    return {
        "id": record.get("id"),
        "days": days, "n_cities": n_cities,
        "days_per_city": days_per_city, "people_number": people,
        "cities": sorted(pools),
        "pool_sizes": {c: {t: len(v) for t, v in d.items()}
                       for c, d in pools.items()},
        "leaves": out_leaves,
        "feasibility": feasibility,
        "entity_types_bound": sorted(by_type),
        "pair_type": ((_loads(record.get("preference_pair")) or {}).get("type")
                      or "single"),
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #

def _agg(xs: list[float]) -> dict[str, float]:
    n = len(xs)
    if not n:
        return {"n": 0, "mean": 0.0, "median": 0.0}
    mu = sum(xs) / n
    s = sorted(xs)
    return {"n": n, "mean": mu,
            "median": s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2}


def report(split: str, per_record: list[dict]) -> str:
    L = [f"CANDIDATE-POOL AVAILABILITY BASELINE — split = {split}",
         "=" * 92,
         "Pool = the retrieved choice set the planner was SHOWN",
         "(reference_information), not the full TravelPlanner DB.",
         ""]

    # [1] pool sizes
    sizes: dict[str, list[int]] = defaultdict(list)
    for r in per_record:
        for _city, per in r["pool_sizes"].items():
            for t, n in per.items():
                sizes[t].append(n)
    L += ["[1] Pool size per city, by entity type",
          f"  {'entity_type':<16s} {'city-pools':>11s} {'mean':>8s} {'median':>8s}",
          "  " + "-" * 46]
    for t, v in sorted(sizes.items(), key=lambda kv: -len(kv[1])):
        a = _agg([float(x) for x in v])
        L.append(f"  {t:<16s} {a['n']:>11d} {a['mean']:>8.1f} {a['median']:>8.1f}")
    L.append("")

    # [2] feasibility -- THE GATE
    feas = defaultdict(Counter)
    for r in per_record:
        for t, f in r["feasibility"].items():
            feas[t][f["class"]] += 1
    L += ["[2] FEASIBILITY CLASS per (record, entity_type)  <-- the gate",
          "  Laziness measures may only be read in the 'comfortable' stratum;",
          "  an 'infeasible' failure is arithmetic, not behaviour.",
          f"  {'entity_type':<16s} {'comfortable':>12s} {'tight':>8s} "
          f"{'infeasible':>11s} {'total':>7s}",
          "  " + "-" * 60]
    tot = Counter()
    for t, c in sorted(feas.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values()); tot.update(c)
        L.append(f"  {t:<16s} {c['comfortable']:>12d} {c['tight']:>8d} "
                 f"{c['infeasible']:>11d} {n:>7d}")
    n = sum(tot.values())
    L += ["  " + "-" * 60,
          f"  {'ALL':<16s} {tot['comfortable']:>12d} {tot['tight']:>8d} "
          f"{tot['infeasible']:>11d} {n:>7d}",
          f"  {'':16s} {100*tot['comfortable']/max(n,1):>11.1f}% "
          f"{100*tot['tight']/max(n,1):>7.1f}% "
          f"{100*tot['infeasible']/max(n,1):>10.1f}%", ""]

    # [3] the ladder: marginal vs conditional
    L += ["[3] AVAILABILITY LADDER — marginal (upper bound) vs conditional",
          "  RETENTION is the headline, not the rate.  avail_conditional is a",
          "  rate computed inside an ALREADY-SHRUNK residual pool, so it can",
          "  rise even as the planner's absolute options fall -- the rate and",
          "  the option count move independently.  Retention =",
          "  n_conditional / n_marginal is what actually says how many of the",
          "  leaf's qualifying candidates survive the other predicates.",
          f"  {'scope':<8s} {'leaves':>7s} {'n_marg':>8s} {'n_cond':>8s} "
          f"{'retention':>10s} {'rate_marg':>10s} {'rate_cond':>10s}",
          "  " + "-" * 62]
    for scope in ("any", "all"):
        sel = [l for r in per_record for l in r["leaves"] if l["scope"] == scope]
        if not sel:
            continue
        nm = _agg([float(l["n_marginal"]) for l in sel])
        nc = _agg([float(l["n_conditional"]) for l in sel])
        ret = _agg([(l["n_conditional"] / l["n_marginal"])
                    for l in sel if l["n_marginal"]])
        rm = _agg([l["avail_marginal"] for l in sel])
        rc = _agg([l["avail_conditional"] for l in sel])
        L.append(f"  {scope:<8s} {nm['n']:>7d} {nm['mean']:>8.2f} "
                 f"{nc['mean']:>8.2f} {ret['mean']:>10.3f} "
                 f"{rm['mean']:>10.3f} {rc['mean']:>10.3f}")
    L.append("")

    # [4] does the interaction actually bite?  by pair type
    L += ["[4] Conditional shrink by pairing type",
          "  'overlapping' pairs bind the same entity type, so their pools",
          "  interact; 'independent' pairs should show ~no shrink.  That",
          "  contrast validates both the label and this computation.",
          f"  {'pair_type':<14s} {'leaves':>7s} {'n_marg':>8s} {'n_cond':>8s} "
          f"{'retention':>10s}",
          "  " + "-" * 52]
    for pt in ("single", "independent", "overlapping"):
        sel = [l for r in per_record if r["pair_type"] == pt
               for l in r["leaves"]]
        if not sel:
            continue
        nm = _agg([float(l["n_marginal"]) for l in sel])
        nc = _agg([float(l["n_conditional"]) for l in sel])
        ret = _agg([(l["n_conditional"] / l["n_marginal"])
                    for l in sel if l["n_marginal"]])
        L.append(f"  {pt:<14s} {nm['n']:>7d} {nm['mean']:>8.2f} "
                 f"{nc['mean']:>8.2f} {ret['mean']:>10.3f}")
    L.append("")

    # [5] per-member availability -- de-confounds the Suppression Index
    zero = tot_m = 0
    for r in per_record:
        for l in r["leaves"]:
            pm = l.get("per_member")
            if not pm:
                continue
            for _v, n_av in pm.items():
                tot_m += 1
                zero += int(n_av == 0)
    L += ["[5] Set-predicate members with ZERO candidates",
          f"  {zero}/{tot_m} members ({100*zero/max(tot_m,1):.1f}%) have no",
          "  candidate in the shown pool.  A member that cannot be booked",
          "  cannot have been 'suppressed' -- so Suppression Index values",
          "  computed on raw realization counts are an UPPER BOUND on",
          "  suppression until divided by these availabilities.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--split", default="test_large")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Per-record availability, for downstream gating "
                         "of laziness measures and Suppression Index "
                         "normalisation.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split,
                      verification_mode="no_checks")
    per_record = []
    for i, r in enumerate(ds):
        if args.limit and i >= args.limit:
            break
        per_record.append(analyse_record(r))
    print(f"[in] {args.dataset}:{args.split}  {len(per_record)} records",
          file=sys.stderr)

    txt = report(args.split, per_record)
    print(txt)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
        print(f"\n[out] {args.out}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(per_record, indent=2))
        print(f"[out] {args.json_out}")


if __name__ == "__main__":
    main()
