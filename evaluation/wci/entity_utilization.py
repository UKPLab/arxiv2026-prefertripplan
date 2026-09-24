#!/usr/bin/env python3
r"""entity_utilization.py — entity utilization, decomposed.

Companion to ``window_compression.py``.  That module scores a plan against
a normative CEILING alone; this one adds an EMPIRICAL EXPECTATION,
estimated from records under no window pressure, and reports the two
together as an exact identity rather than as competing metrics.

--------------------------------------------------------------------------
THE IDENTITY
--------------------------------------------------------------------------
        n_used            n_used          n_expected
      ---------   =   ------------   x   ------------
        n_max          n_expected           n_max
       REALIZED         RELATIVE            EXPECTED
      <------------ ENTITY UTILIZATION ------------->

  RELATIVE   did the preference change the window relative to what THIS
             model does when unpressured?  Model style cancels, so this
             is the causal contrast -- the preference effect proper.
  EXPECTED   how much of the structurally possible window does the
             unpressured baseline itself use?  No preference is involved;
             it is a model x record property.
  REALIZED   fraction of the physically possible that was realised.

The decomposition matters because a low REALIZED ENTITY UTILIZATION has
two very different explanations.  gemma schedules 1.13 restaurants/day
against a 3/day ceiling: its realized value is low because of EXPECTED
ENTITY UTILIZATION, not because any preference changed its window.  A
single number conflates the two.

--------------------------------------------------------------------------
WHY O/E, AND WHY NOT SUBTRACT THE FLOOR
--------------------------------------------------------------------------
This is indirect standardisation: reference rates applied to the study
population's own structure (days per city) to form an expected count, then
observed/expected.  Same construction as a standardised mortality ratio in
epidemiology or expected species richness in ecology -- a known estimator
with known properties, not an invention.

The floor-subtracted form (n_used - n_min) / (n_expected - n_min) was
tested and REJECTED: gemma's baseline restaurant rate (1.13/day) is below
the commonsense floor rate, so the denominator goes NEGATIVE (e.g. 7 days
/ 4 required: n_min = 12, n_expected = 7.91, denominator = -4.09) and the
ratio flips sign.  Plain O/E is always positive and symmetric about 1.
``subfloor`` remains a separate flag, which is what it should have been.

--------------------------------------------------------------------------
BASELINE ESTIMATION  (lambda-hat)
--------------------------------------------------------------------------
A record contributes to lambda-hat for entity ``et`` iff NO COMPLEX
preference touches ``et``.  Complex = anything other than Atomic or
Composite, i.e. Numeric / Conditional / Lexicographic / Compensatory /
Scoped / Temporal.  Atomic and Composite are excluded from "complex"
because neither creates a window incentive: an Atomic [all] predicate must
hold on every entity however many there are.  Complex preferences on OTHER
entity types are fine -- they do not touch this window.

Because the rule excludes every complex preference on ``et`` regardless of
which preference slot it occupies, the paired-preference channel is absent
by construction.

NO POOL FILTERING IN lambda-hat.  Partner gating and local constraints
are deliberately NOT applied when estimating lambda-hat.  The quantity it
will be compared against, ``n_used``, is itself unfiltered -- for bare
Numeric it counts every plan entity of the type regardless of compliance
-- so filtering only the denominator would make the ratio compare two
different populations.  Measured before removal: applying them shifted
lambda-hat by +0.7% (local constraint) and +2.0% (simple preference), and
their inclusion filter dropped 0 of 289 records, so nothing is lost.

They ARE still applied in SCORING, where the pool feeds share_c and the
no-repeat cap.  For bare Numeric that has almost no effect (filt is None,
so share_c == 1 regardless, and the cap binds on 9 of 206 instances); for
Scoped, share_c genuinely moves.

lambda-hat is computed DIRECTLY from the plans as SUM(entities)/SUM(days);
the pool never enters that arithmetic.  LC and gating act in estimation
only as an inclusion filter (skip a record whose constrained pool is
empty), which drops 0 of 289 (record, entity) pairs -- inert.

lambda-hat is estimated PER MODEL and as a RATE PER DAY -- sum(entities) /
sum(days) over the stratum, not the mean of per-record rates, which would
let short trips dominate.  Per model rather than pooled so that model
style cancels in COMPRESSION.

--------------------------------------------------------------------------
n_max, AND WHY 4/day FOR ATTRACTIONS IS NOT AN ARBITRARY ANCHOR
--------------------------------------------------------------------------
n_max = min( SUM_c share_c * slots_c , SUM_c min(qual_c, slots_c) ),
identical to window_compression.py.

  Restaurant  3/day is STRUCTURAL -- the plan format has exactly three
              meal keys, so a fourth slot does not exist.
  Attraction  4/day is corroborated INDEPENDENTLY by two sources: it is
              the maximum over TravelPlanner's 45 human-annotated plans,
              AND 99.8% of the ~32,000 model plan-days schedule at most 4
              (99.1% at most 3).  The evaluator itself sets no ceiling.
  no-repeat   is_valid_restaurants / is_valid_attractions forbid reuse
              trip-wide, so the qualifying pool is a hard cap.

The slots term binds on 197 of 206 numeric instances; the pool cap binds
on 9.  So slots_per_day carries the ceiling, which is why its provenance
matters.

USAGE
    python3 evaluation/wci/entity_utilization.py --split test_large \
        --model qwen3.8-27b --plan-file <structured_plans.jsonl> \
        --out runs/entity_util_qwen3.8-27b_test_large.txt \
        --json-out runs/entity_util_qwen3.8-27b_test_large.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pool_common import (build_pools, qualify as _qualify,        # noqa: E402
                         loads as _loads, gating_all_leaves)
from window_compression import (                                   # noqa: E402
    TARGET_ENTITIES, SLOTS_PER_DAY, FLOOR_PER_REQUIRED_DAY,
    _REQUIRED_DAY_KEY, incentive_class, plan_geography,
    target_instances, _plan_entities, _COMPLEX_PARADIGMS,
    _entity_types_touched)

# local_constraint key -> the pool predicate it implies
_LC_PREDICATE = {
    "cuisine":   ("Restaurant", "cuisine", "in"),
    "house rule": ("Accommodation", "house_rules", "not_in"),
    "room type": ("Accommodation", "room_type", "=="),
}


def lc_leaf(local_constraint: Any, entity_type: str) -> dict | None:
    """The pool predicate implied by this record's local hard constraint,
    or None.  Applied to BOTH strata so the pools are comparable."""
    lc = _loads(local_constraint) or {}
    if not isinstance(lc, dict):
        return None
    for key, (et, attr, op) in _LC_PREDICATE.items():
        if et != entity_type:
            continue
        val = lc.get(key)
        if val in (None, "", [], {}):
            continue
        if op == "in" and not isinstance(val, (list, tuple)):
            val = [val]
        return {"entity_type": et, "attribute": attr, "op": op,
                "value": val, "scope": "any"}
    return None


def constrained_pool(pools: dict, city: str, et: str,
                     gating: list[dict], lc: dict | None) -> list[dict]:
    """City pool after partner gating and the local hard constraint.
    Identical treatment in the baseline and the main analysis."""
    pool = (pools.get(city) or {}).get(et) or []
    for leaf in gating:
        pool = _qualify(pool, leaf)
    if lc is not None:
        pool = _qualify(pool, lc)
    return pool


# =========================================================================== #
# STAGE 1 — estimate lambda-hat from the unpressured stratum                   #
# =========================================================================== #

def estimate_lambda(dataset: str, split: str, model: str,
                    plan_file: Path) -> dict[str, Any]:
    """Per-day entity rate for each target entity, from records where no
    complex preference touches it.  Returns rates plus the diagnostics
    needed to judge them."""
    from datasets import load_dataset
    import evaluate_preferences as EP

    plans: dict[int, Any] = {}
    with plan_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("id") is not None:
                plans[int(r["id"])] = r.get("plan")

    ds = load_dataset(dataset, split=split, verification_mode="no_checks")
    db = EP._DB()
    ent_sum: dict[str, int] = defaultdict(int)
    day_sum: dict[str, float] = defaultdict(float)
    rec_n: dict[str, int] = defaultdict(int)

    for rec in ds:
        raw = plans.get(int(rec["id"]))
        if not raw:
            continue
        prefs = _loads(rec.get("preferences_json")) or []
        pressured: set[str] = set()
        for entry in prefs:
            if entry.get("paradigm") in _COMPLEX_PARADIGMS:
                pressured |= _entity_types_touched(entry)
        free = [et for et in TARGET_ENTITIES if et not in pressured]
        if not free:
            continue
        pools = build_pools(rec.get("reference_information"),
                            entity_types=TARGET_ENTITIES)
        pooled = {c for c, per in pools.items()
                  if any(per.get(e) for e in TARGET_ENTITIES)}
        geo = plan_geography(raw, pooled_cities=pooled)
        if not geo["days_c"] or geo["multi_hop_days"]:
            continue
        try:
            plan_days = EP.transform_plan(
                raw, db=db, date_seq=rec.get("date"),
                people_number=int(rec.get("people_number") or 1))
        except Exception:
            continue
        for et in free:
            # NO partner gating and NO local constraint here.  lambda-hat is
            # a BEHAVIOURAL rate -- entities scheduled per day -- and the
            # numerator it will be compared against (n_used) is likewise
            # unfiltered.  Filtering one side only would make the ratio
            # compare two different populations.  (Measured separately:
            # applying them changed lambda-hat by +0.7% for LC and +2.0%
            # for a simple preference, and their inclusion filter dropped
            # 0 of 289 records -- so nothing is lost by removing them.)
            # The only requirement is that the entity type exists in this
            # trip's choice set at all; otherwise the plan cannot contain
            # one and the record would drag the rate down spuriously.
            if not any((pools.get(c) or {}).get(et) for c in geo["days_c"]):
                continue
            used = _plan_entities(plan_days, et)
            ent_sum[et] += sum(len(v) for v in used.values())
            # Rate per DAY, weighted by days -- not the mean of per-record
            # rates, which would let short trips dominate.
            day_sum[et] += sum(geo["days_c"].values())
            rec_n[et] += 1

    return {
        "model": model, "split": split,
        "lambda_per_day": {et: (ent_sum[et] / day_sum[et]) if day_sum[et] else None
                           for et in TARGET_ENTITIES},
        "n_records": dict(rec_n),
        "n_days": {et: round(day_sum[et], 2) for et in TARGET_ENTITIES},
        "n_entities": dict(ent_sum),

    }


# =========================================================================== #
# STAGE 2 — score each in-scope instance against lambda-hat and n_max          #
# =========================================================================== #

def compute_entity_utilization(inst: dict, record: dict, geo: dict, plan_days: list,
               pools: dict, lam: dict[str, float]) -> dict | None:
    """One row carrying COMPRESSION, HEADROOM and ABSOLUTE."""
    et = inst["entity_type"]
    rate = lam.get(et)
    if not rate:
        return None
    slots_per_day = SLOTS_PER_DAY[et]
    filt = inst["filter"]

    used = _plan_entities(plan_days, et)
    if filt is not None:
        used = {c: _qualify(v, filt) for c, v in used.items()}
    n_used = sum(len(v) for v in used.values())

    gating = gating_all_leaves(_loads(record.get("preferences_json")) or [],
                               et, exclude=(inst["paradigm"], inst["bank_id"]))
    lc = lc_leaf(record.get("local_constraint"), et)

    per_city: dict[str, dict] = {}
    for city, days_c in geo["days_c"].items():
        pool = constrained_pool(pools, city, et, gating, lc)
        if not pool:
            continue
        qual = _qualify(pool, filt)
        share = (len(qual) / len(pool)) if pool else 0.0
        per_city[city] = {"days_c": days_c,
                          "slots_c": slots_per_day * days_c,
                          "pool_total": len(pool), "pool_qual": len(qual),
                          "share_c": share,
                          # lambda-hat applied to THIS record's structure:
                          # the indirect-standardisation step.
                          "expected_c": rate * days_c * share}
    if not per_city:
        return None

    n_expected = sum(c["expected_c"] for c in per_city.values())
    expectation = sum(c["share_c"] * c["slots_c"] for c in per_city.values())
    no_repeat_cap = sum(min(c["pool_qual"], c["slots_c"])
                        for c in per_city.values())
    n_max = min(expectation, float(no_repeat_cap))
    n_min = (FLOOR_PER_REQUIRED_DAY[et] * geo[_REQUIRED_DAY_KEY[et]]
             if inst["construct"] == "numeric" else 0)

    # Neutral names: the ratio is two-sided and ~55% of instances land
    # ABOVE 1.0, so "compression" would assert a direction the data does
    # not support.  relative_entity_utilization < 1 = less of the window used
    # norm, > 1 = larger; no claim either way is built into the name.
    compression = (n_used / n_expected) if n_expected > 0 else None
    headroom = (n_expected / n_max) if n_max > 0 else None
    absolute = (n_used / n_max) if n_max > 0 else None
    return {
        "id": record.get("id"), "split": record.get("_split"),
        "model": record.get("_model"),
        **{k: inst.get(k) for k in ("construct", "paradigm", "bank_id",
                                    "entity_type", "attribute", "aggregation",
                                    "direction", "filter")},
        "incentive": incentive_class(inst.get("aggregation"),
                                     inst.get("direction")),
        "days": geo["days"], "days_c": geo["days_c"], "per_city": per_city,
        "lambda_per_day": rate,
        "n_used": n_used, "n_expected": n_expected, "n_max": n_max,
        "n_min_commonsense": n_min,
        "relative_entity_utilization": compression, "expected_entity_utilization": headroom,
        "realized_entity_utilization": absolute,
        # subfloor is a FLAG, never a normalisation term -- folding it into
        # the denominator produced negative spans for gemma.
        "subfloor": bool(n_used < n_min),
        "subfloor_deviation": max(n_min - n_used, 0),
        "n_gating_partners": len(gating), "has_lc": lc is not None,
    }


# =========================================================================== #
# Driver + report                                                              #
# =========================================================================== #

def analyse(dataset: str, split: str, model: str,
            plan_file: Path) -> tuple[list[dict], dict]:
    from datasets import load_dataset
    import evaluate_preferences as EP

    lam_info = estimate_lambda(dataset, split, model, plan_file)
    lam = lam_info["lambda_per_day"]

    plans: dict[int, Any] = {}
    with plan_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("id") is not None:
                plans[int(r["id"])] = r.get("plan")

    ds = load_dataset(dataset, split=split, verification_mode="no_checks")
    db = EP._DB()
    rows: list[dict] = []
    for rec in ds:
        raw = plans.get(int(rec["id"]))
        if not raw:
            continue
        insts = target_instances(_loads(rec.get("preferences_json")) or [])
        if not insts:
            continue
        pools = build_pools(rec.get("reference_information"),
                            entity_types=TARGET_ENTITIES)
        pooled = {c for c, per in pools.items()
                  if any(per.get(e) for e in TARGET_ENTITIES)}
        geo = plan_geography(raw, pooled_cities=pooled)
        if not geo["days_c"] or geo["multi_hop_days"]:
            continue
        try:
            plan_days = EP.transform_plan(
                raw, db=db, date_seq=rec.get("date"),
                people_number=int(rec.get("people_number") or 1))
        except Exception:
            continue
        ctx = dict(rec)
        ctx["_split"], ctx["_model"] = split, model
        for inst in insts:
            row = compute_entity_utilization(inst, ctx, geo, plan_days, pools, lam)
            if row:
                rows.append(row)
    return rows, lam_info


def _pool(rows: list[dict], key_num: str, key_den: str) -> float | None:
    """Ratio of SUMS, not the mean of ratios -- the aggregate of a ratio
    metric, and it keeps the three columns' identity exact."""
    num = sum(r[key_num] for r in rows if r.get(key_num) is not None)
    den = sum(r[key_den] for r in rows if r.get(key_den) is not None)
    return (num / den) if den > 0 else None


def report(model: str, split: str, rows: list[dict], lam: dict) -> str:
    def f(v):
        return "--" if v is None else f"{v:.3f}"
    L = [f"ENTITY UTILIZATION — {model} / {split}", "=" * 78,
         "    n_used/n_max   =  (n_used/n_expected)  x  (n_expected/n_max)",
         "     REALIZED      =      RELATIVE         x      EXPECTED",
         "    <------------------- ENTITY UTILIZATION ------------------->",
         "",
         "  RELATIVE   window size vs what THIS model does unpressured.",
         "             < 1 below its own norm, > 1 above.  Model style",
         "             cancels.  Two-sided by construction: an O/E against",
         "             a MEAN, so ~half exceed 1.",
         "  EXPECTED   that unpressured norm as a fraction of the",
         "             structural ceiling.  No preference involved.",
         "  REALIZED   the plan as a fraction of that ceiling.", ""]
    L.append("  EXPECTED RATE (lambda-hat) = SUM(entities) / SUM(days), over records")
    L.append("  where NO complex preference touches that entity.  Unfiltered by")
    L.append("  pool, gating or local constraints -- n_used is unfiltered too,")
    L.append("  so filtering one side would compare different populations:")
    for et in TARGET_ENTITIES:
        r = lam["lambda_per_day"].get(et)
        L.append(f"    {et:<12s} {f(r)} /day   "
                 f"(records={lam['n_records'].get(et,0)}, "
                 f"days={lam['n_days'].get(et,0)}, "
                 f"entities={lam['n_entities'].get(et,0)})")
    L.append("")
    L.append(f"  {'construct':<9s} {'entity':<11s} {'n':>4s} "
             f"{'RELATIVE':>10s} {'EXPECTED':>10s} {'REALIZED':>10s} {'sub':>4s}")
    L.append("  " + "-" * 62)
    for con in ("numeric", "scoped"):
        for et in TARGET_ENTITIES:
            sel = [r for r in rows if r["construct"] == con
                   and r["entity_type"] == et and not r["subfloor"]]
            if not sel:
                continue
            n_sub = sum(1 for r in rows if r["construct"] == con
                        and r["entity_type"] == et and r["subfloor"])
            L.append(f"  {con:<9s} {et:<11s} {len(sel):>4d} "
                     f"{f(_pool(sel,'n_used','n_expected')):>11s} "
                     f"{f(_pool(sel,'n_expected','n_max')):>10s} "
                     f"{f(_pool(sel,'n_used','n_max')):>9s} "
                     f"{n_sub:>4d}")
    L.append("")
    L.append("  RELATIVE ENTITY UTILIZATION by incentive class")
    L.append("  (numeric only; prediction shrink < neutral < expand):")
    for et in TARGET_ENTITIES:
        cells = []
        for cls in ("shrink", "neutral", "expand"):
            sel = [r for r in rows if r["construct"] == "numeric"
                   and r["entity_type"] == et and r["incentive"] == cls
                   and not r["subfloor"]]
            v = _pool(sel, "n_used", "n_expected") if sel else None
            cells.append(f"{cls}=" + (f"{v:.3f}(n={len(sel)})" if v else "--"))
        L.append(f"    {et:<12s} " + "  ".join(cells))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="UKPLab/PreferTripPlan")
    ap.add_argument("--split", default="test_large")
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows, lam = analyse(args.dataset, args.split, args.model, args.plan_file)
    txt = report(args.model, args.split, rows, lam)
    print(txt)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"lambda": lam, "rows": rows}, indent=2, default=str))
        print(f"[out] {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
