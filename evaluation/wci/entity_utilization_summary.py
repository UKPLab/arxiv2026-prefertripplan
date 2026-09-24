#!/usr/bin/env python3
"""entity_utilization_summary.py — cross-model tables for the
entity-utilization analysis.

Reads the per-model JSON from ``entity_utilization.py`` and writes one
self-contained file: a single headline number per model, then the method,
the lambda-hat baselines, and -- in the appendix -- the full per-construct
detail and the STRONG / WEAK / NONE incentive contrast.  Regenerate after
any re-run so the prose and the numbers cannot drift apart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ["gpt-5.6-terra", "nemotron", "deepseek", "qwen3.8-27b",
          "gemma-4-26b-a4b"]
SPLITS = ["test", "test_large"]
ENTITIES = ["Restaurant", "Attraction"]

METHOD = r"""
================================================================================
ENTITY UTILIZATION AS AN OBSERVED / EXPECTED RATIO
================================================================================

THE IDENTITY
------------
        n_used            n_used          n_expected
      ---------   =   ------------   x   ------------
        n_max          n_expected           n_max

       REALIZED         RELATIVE            EXPECTED
      <------------ ENTITY UTILIZATION ------------->

All three are ENTITY UTILIZATION of the same structural window, which is
what makes the identity legible: the two absolute terms share the
denominator n_max, so their ratio IS the relative term.

  RELATIVE   realized / expected -- entity utilization relative to what
             THIS model does when unpressured.  Model style cancels, so
             this isolates the preference's effect: the headline column.
             < 1 = less of the window used than the model's own norm,
             > 1 = more.  DELIBERATELY TWO-SIDED: an O/E against a MEAN,
             so roughly half of instances exceed 1 by construction and
             the name asserts no direction.  (Measured: 55.4% above 1.0.)
  EXPECTED   the unpressured norm as a fraction of the structural
             ceiling -- what this model would use anyway.  No preference
             involved; a model x record property.
  REALIZED   what the plan actually used of that same ceiling.

The decomposition exists because a low REALIZED ENTITY UTILIZATION has
two very different explanations.  A model scheduling 1.13 restaurants/day
against a 3/day ceiling utilizes little because its EXPECTED ENTITY
UTILIZATION is low, not because any preference changed its window.  A
single number conflates the two; the product recovers it exactly.

WHY O/E, AND WHY THE FLOOR IS NOT SUBTRACTED
--------------------------------------------
This is INDIRECT STANDARDISATION: reference rates applied to the study
population's own structure (days per city) to form an expected count, then
observed / expected.  The same construction as a standardised mortality
ratio in epidemiology or expected species richness in ecology -- an
established estimator, not an invention.

The floor-subtracted form (n_used - n_min)/(n_expected - n_min) was tested
and REJECTED.  A model whose baseline rate falls below the commonsense
floor rate produces a NEGATIVE denominator -- measured: 7 days / 4
required gives n_min = 12 against n_expected = 7.91, i.e. -4.09 -- and the
ratio flips sign.  Plain O/E is always positive and symmetric about 1.
Sub-floor plans are reported as a separate FLAG, which is what they should
always have been.

BASELINE ESTIMATION (lambda-hat)
--------------------------------
A record contributes to lambda-hat for entity ``et`` iff NO COMPLEX
preference touches ``et``.  Complex = anything other than Atomic or
Composite: Numeric, Conditional, Lexicographic, Compensatory, Scoped,
Temporal.  Atomic and Composite are excluded from "complex" because
neither creates a window incentive -- an Atomic [all] predicate must hold
on every entity however many there are.  Complex preferences on OTHER
entity types are left alone; they do not touch this window.

Because the rule removes every complex preference on ``et`` regardless of
which preference slot it occupies, the paired-preference channel is absent
by construction.

NO POOL FILTERING IN lambda-hat
-------------------------------
Partner gating and local constraints are deliberately NOT applied when
estimating lambda-hat.  The quantity it is compared against, ``n_used``,
is itself unfiltered -- for bare Numeric it counts every plan entity of
the type regardless of compliance -- so filtering only the denominator
would make the ratio compare two different populations.

Measured before removal: applying them shifted lambda-hat by +0.7% (local
constraint: 2.597 vs 2.615 restaurants/day) and +2.0% (simple preference:
2.600 vs 2.653), and their inclusion filter dropped 0 of 289 records.
Nothing is lost by removing them.

They ARE still applied in SCORING, where the pool feeds share_c and the
no-repeat cap.  For bare Numeric that is nearly inert -- filt is None, so
share_c == 1 regardless, and the cap binds on 9 of 206 instances.  For
Scoped, share_c genuinely moves.

HOW lambda-hat IS COMPUTED.  Directly from the plans:

      lambda-hat = SUM(entities scheduled) / SUM(days)

over the stratum.  A RATE PER DAY, formed as the RATIO OF SUMS rather
than the mean of per-record rates -- day-weighting is what makes
lambda-hat calibrated (summed over the stratum, expected reproduces
observed exactly) and is also the inverse-variance weighting, since a
7-day trip is seven day-observations of the rate and a 3-day trip only
three.  Estimated PER MODEL, so model style cancels inside RELATIVE
ENTITY UTILIZATION.  The only record requirement is that the entity type
appears in the trip's choice set at all; otherwise the plan could not
contain one and the record would drag the rate down spuriously.

n_max AND THE PROVENANCE OF ITS CEILING
---------------------------------------
n_max = min( SUM_c share_c * slots_c , SUM_c min(qual_c, slots_c) ).

  Restaurant  3/day is STRUCTURAL -- the plan format has exactly three
              meal keys, so a fourth slot does not exist.
  Attraction  4/day is corroborated INDEPENDENTLY by two sources: it is
              the maximum over TravelPlanner's 45 human-annotated plans,
              AND 99.8% of ~32,000 model plan-days schedule at most 4
              (99.1% at most 3).  The evaluator sets no ceiling of its own.
  no-repeat   is_valid_restaurants / is_valid_attractions forbid reuse
              trip-wide, so the qualifying pool is a hard cap.

The slots term binds on 197 of 206 numeric instances and the pool cap on
9, so slots_per_day carries the ceiling -- which is why its provenance is
spelled out rather than assumed.

SCOPE
-----
Numeric x {Restaurant, Attraction} and Scoped -> Numeric, for the reason
given in window_compression.py: F2 is measurable only where the score
AGGREGATES WITHOUT SELECTING.  A selecting inner makes the denominator
unanswerable in both directions.  Accommodation and Transportation are
excluded (one slot of leeway and none).  Agentic runs are excluded: a
tool-using planner queries the DB directly, so reference_information is
not its choice set.

READING THE TABLES
------------------
All pooled figures are RATIOS OF SUMS, never means of ratios -- the
natural aggregate of a ratio metric, and the only form that keeps the
three columns' identity exact.
""".lstrip("\n")


def _pool(rows, num_key, den_key):
    num = sum(r[num_key] for r in rows if r.get(num_key) is not None)
    den = sum(r[den_key] for r in rows if r.get(den_key) is not None)
    return (num / den) if den > 0 else None


def _f(v):
    return "--" if v is None else f"{v:.3f}"


def headline(runs: Path) -> str:
    """ONE number per model: Relative Utilization, pooled over every
    eligible instance across Numeric and Scoped, Restaurant and
    Attraction.

    POOLED FROM THE UNDERLYING COUNTS, not averaged over the per-construct
    cells:

        Relative Utilization = SUM(n_used) / SUM(n_expected)

    Averaging the cells would weight an 8-instance cell like a 69-instance
    one and would make the headline depend on how the constructs happen to
    be balanced.  Pooling the counts is the natural aggregate of a ratio
    metric and keeps the identity exact at every level.

    Expected and Realized are pooled the same way, so the identity
    REALIZED = RELATIVE x EXPECTED holds on the headline row too.
    """
    L = ["=" * 104,
         "HEADLINE — entity utilization per model",
         "=" * 104,
         "One number per model, pooled over ALL eligible instances (Numeric",
         "and Scoped, Restaurant and Attraction).  Ratios of SUMS, never",
         "means of ratios -- so the identity holds at this level too:",
         "",
         "      ENTITY UTILIZATION:  REALIZED = RELATIVE x EXPECTED",
         "",
         "  RELATIVE   the headline.  Window used vs what THIS model does",
         "             unpressured; model style cancels.  < 1 = less than",
         "             its own norm, > 1 = more.  Two-sided by construction",
         "             (an O/E against a mean), so the name asserts no",
         "             direction.",
         "  EXPECTED   what that unpressured norm uses of the structural",
         "             ceiling.  No preference involved.",
         "  REALIZED   what the plan actually used of that ceiling.", ""]
    for split in SPLITS:
        L.append(f"  [{split}]")
        L.append(f"  {'model':<19s} {'RELATIVE':>10s} {'EXPECTED':>10s} "
                 f"{'REALIZED':>10s} {'inst':>6s} {'sub':>5s}")
        L.append("  " + "-" * 66)
        for mo in MODELS:
            fp = runs / f"entity_util_{mo}_{split}.json"
            if not fp.exists():
                L.append(f"  {mo:<19s} (no run)")
                continue
            rows = json.loads(fp.read_text())["rows"]
            sel = [r for r in rows if not r["subfloor"]
                   and r.get("relative_entity_utilization") is not None]
            nsub = sum(1 for r in rows if r["subfloor"])
            L.append(f"  {mo:<19s} "
                     f"{_f(_pool(sel,'n_used','n_expected')):>10s} "
                     f"{_f(_pool(sel,'n_expected','n_max')):>10s} "
                     f"{_f(_pool(sel,'n_used','n_max')):>10s} "
                     f"{len(sel):>6d} {nsub:>5d}")
        L.append("")
    return "\n".join(L)


def build(runs: Path) -> str:
    L = ["=" * 104,
         "BASELINE RATES (lambda-hat) — entities per day, unpressured stratum",
         "=" * 104,
         f"  {'model':<17s} {'split':<11s} "
         f"{'Restaurant':>22s} {'Attraction':>22s}",
         "  " + "-" * 76]
    lam_cache = {}
    for split in SPLITS:
        for mo in MODELS:
            fp = runs / f"entity_util_{mo}_{split}.json"
            if not fp.exists():
                continue
            data = json.loads(fp.read_text())
            lam_cache[(mo, split)] = data
            lam = data["lambda"]
            cells = []
            for et in ENTITIES:
                r = lam["lambda_per_day"].get(et)
                cells.append(f"{_f(r)}/day (n={lam['n_records'].get(et,0)})")
            L.append(f"  {mo:<17s} {split:<11s} " +
                     "".join(f"{c:>22s}" for c in cells))
    L += ["", "=" * 104,
          "APPENDIX A — DECOMPOSITION per construct x entity",
          "=" * 104,
          "RELATIVE < 1 = less of the window used than this model's own",
          "unpressured norm, > 1 = more; ~half exceed 1 by construction.",
          "EXPECTED = what that norm uses of the structural ceiling.",
          "REALIZED = their product, i.e. the outcome.", ""]
    for split in SPLITS:
        L.append(f"  [{split}]")
        L.append(f"  {'model':<17s} {'construct':<9s} {'entity':<11s} {'n':>4s} "
                 f"{'RELATIVE':>10s} {'EXPECTED':>10s} {'REALIZED':>10s} {'sub':>4s}")
        L.append("  " + "-" * 80)
        for mo in MODELS:
            d = lam_cache.get((mo, split))
            if not d:
                continue
            rows = d["rows"]
            for con in ("numeric", "scoped"):
                for et in ENTITIES:
                    sel = [r for r in rows if r["construct"] == con
                           and r["entity_type"] == et and not r["subfloor"]]
                    if not sel:
                        continue
                    nsub = sum(1 for r in rows if r["construct"] == con
                               and r["entity_type"] == et and r["subfloor"])
                    L.append(f"  {mo:<17s} {con:<9s} {et:<11s} {len(sel):>4d} "
                             f"{_f(_pool(sel,'n_used','n_expected')):>11s} "
                             f"{_f(_pool(sel,'n_expected','n_max')):>10s} "
                             f"{_f(_pool(sel,'n_used','n_max')):>9s} "
                             f"{nsub:>4d}")
        L.append("")
    L += ["=" * 104,
          "APPENDIX B — RELATIVE ENTITY UTILIZATION by incentive class (numeric only)",
          "=" * 104,
          "STRONG = aggregation and direction oppose, so shrinking raises the",
          "score.  WEAK = avg, roughly half the incentive (order statistics).",
          "NONE = they align, no incentive -- a POSITIVE CONTROL: if",
          "RELATIVE ENTITY UTILIZATION does not rise there, the metric is not tracking",
          "incentive at all.  Prediction: STRONG < WEAK < NONE.", ""]
    for split in SPLITS:
        L.append(f"  [{split}]")
        L.append(f"  {'model':<17s} {'entity':<11s} {'STRONG':>15s} "
                 f"{'WEAK':>15s} {'NONE':>15s}")
        L.append("  " + "-" * 76)
        for mo in MODELS:
            d = lam_cache.get((mo, split))
            if not d:
                continue
            for et in ENTITIES:
                cells = []
                for cls in ("shrink", "neutral", "expand"):
                    sel = [r for r in d["rows"] if r["construct"] == "numeric"
                           and r["entity_type"] == et
                           and r["incentive"] == cls and not r["subfloor"]]
                    v = _pool(sel, "n_used", "n_expected") if sel else None
                    cells.append(f"{_f(v)}(n={len(sel)})" if sel else "--")
                L.append(f"  {mo:<17s} {et:<11s} " +
                         "".join(f"{c:>15s}" for c in cells))
        L.append("")
    return "\n".join(L)


def worked(runs: Path) -> str:
    """Full arithmetic for one instance per (construct x entity).

    Generated from the per-instance JSON, so these derivations regenerate
    with the results and cannot drift from them.  Selection is
    deterministic: prefer a multi-city record so the per-city summation is
    visible, then the lowest id."""
    pool_rows = []
    for split in SPLITS:
        for mo in MODELS:
            fp = runs / f"entity_util_{mo}_{split}.json"
            if fp.exists():
                pool_rows.extend(json.loads(fp.read_text())["rows"])
    L = ["", "=" * 104,
         "WORKED EXAMPLES — the identity, computed in full",
         "=" * 104,
         "Each block shows lambda-hat applied to that record's own",
         "structure (the indirect-standardisation step), then the three",
         "ratios.  REALIZED should equal RELATIVE x EXPECTED.", ""]
    for con in ("numeric", "scoped"):
        for et in ENTITIES:
            sel = [r for r in pool_rows if r["construct"] == con
                   and r["entity_type"] == et and not r["subfloor"]
                   and r.get("relative_entity_utilization") is not None]
            if not sel:
                L += [f"-- {con.upper()} + {et}: no eligible instance --", ""]
                continue
            sel.sort(key=lambda r: (-len(r["per_city"]), r["id"]))
            r = sel[0]
            L.append("-" * 104)
            L.append(f"CASE: {con.upper()} + {et.upper()}")
            L.append("-" * 104)
            L.append(f"  id={r['id']}  bank={r['bank_id']}  model={r['model']}"
                     f"  split={r['split']}")
            if con == "numeric":
                L.append(f"  preference : Numeric {et}.{r['attribute']}  "
                         f"aggregation={r['aggregation']} "
                         f"direction={r['direction']}"
                         f"  -> incentive={r['incentive'].upper()}")
                L.append("  window     : every plan entity of this type "
                         "(no filter, so share_c = 1)")
            else:
                f_ = r["filter"] or {}
                L.append(f"  preference : Scoped  FILTER {f_.get('entity_type')}."
                         f"{f_.get('attribute')} {f_.get('op')} "
                         f"{f_.get('value')} [{f_.get('scope')}]")
                L.append("  window     : plan entities passing that filter")
            L.append(f"  lambda-hat : {r['lambda_per_day']:.3f} {et}/day "
                     f"(this model's unpressured baseline)")
            L.append(f"  gating partners={r['n_gating_partners']}  "
                     f"local constraint={'yes' if r['has_lc'] else 'no'}")
            L.append("")
            exp_t = cap_t = 0.0
            for city, v in r["per_city"].items():
                exp_t += v["share_c"] * v["slots_c"]
                cap_t += min(v["pool_qual"], v["slots_c"])
                L.append(f"    {city:<18s} days_c={v['days_c']}  "
                         f"slots_c={v['slots_c']}")
                L.append(f"    {'':18s} pool={v['pool_total']} "
                         f"qual={v['pool_qual']}  share_c={v['share_c']:.3f}")
                L.append(f"    {'':18s} expected_c = {r['lambda_per_day']:.3f}"
                         f" x {v['days_c']} x {v['share_c']:.3f} = "
                         f"{v['expected_c']:.2f}")
            L.append("")
            L.append(f"    n_expected = SUM expected_c        = "
                     f"{r['n_expected']:.2f}")
            L.append(f"    n_max      = min({exp_t:.2f}, {cap_t:.0f})"
                     f"           = {r['n_max']:.2f}")
            L.append(f"    n_used     = {r['n_used']}")
            L.append("")
            L.append(f"    RELATIVE  = {r['n_used']} / {r['n_expected']:.2f}"
                     f"  = {r['relative_entity_utilization']:.3f}")
            L.append(f"    EXPECTED  = {r['n_expected']:.2f} / "
                     f"{r['n_max']:.2f} = {r['expected_entity_utilization']:.3f}")
            L.append(f"    REALIZED  = {r['n_used']} / {r['n_max']:.2f}"
                     f"  = {r['realized_entity_utilization']:.3f}")
            L.append(f"    check     : {r['relative_entity_utilization']:.3f} x "
                     f"{r['expected_entity_utilization']:.3f} = "
                     f"{r['relative_entity_utilization']*r['expected_entity_utilization']:.3f}")
            L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path,
                    default=Path(__file__).parent / "runs")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "ENTITY_UTILIZATION_approach_and_results.txt")
    args = ap.parse_args()
    txt = (headline(args.runs_dir) + "\n" + METHOD + "\n"
           + build(args.runs_dir) + worked(args.runs_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(txt)
    print(f"[out] {args.out}  ({len(txt.splitlines())} lines)")


if __name__ == "__main__":
    main()
