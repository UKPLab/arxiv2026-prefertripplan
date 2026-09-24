#!/usr/bin/env python3
r"""pfi.py — PFI, the Profile-Following Index: one drift-impact score.

PFI asks a single question of every drifted preference, whatever its
paradigm, operator or direction:

    did the planner enact the DRIFTED PROFILE instead of the STATED QUERY?

    PFI > 0   followed the drifted profile, under-served the query
    PFI = 0   drift had no behavioural effect
    PFI < 0   doubled down on the query

--------------------------------------------------------------------------
THE FORM
--------------------------------------------------------------------------
        PFI(p)  =  mean over drifted leaves of  D(l)

A plain leaf-count mean, following Keeney & Raiffa (1976) additive utility
under preferential independence: leaves inside one preference are
conjunctive obligations of comparable standing, so equal weighting is the
right default and needs no multiplicative-utility assumption.

NO APPLICABILITY FACTOR.  An earlier version was a two-part (hurdle) model
after Cragg (1971), ``PFI = A(p) x mean D``, with A gating out vacuous
preferences on the grounds that intensity cannot be observed where there
was no obligation.  It is dropped, because VACUITY IS ITSELF A
DRIFT-INDUCED OUTCOME: when drift lands on a Conditional's condition and
the gate then fails to fire, the empty plan IS the behavioural response,
and R = 0 records what the plan actually contains.  Gating it out
discarded exactly that.

The effect was large, not cosmetic.  Measured on gpt-5.6-terra /
test_large: 26 of 594 preferences had A = 0 (23 Conditional, 3 Temporal)
and they carried 63% of the total D mass, mean D = +0.357 against +0.009
for the rest.  A and D are strongly ANTI-correlated -- a vacuous
preference always yields R = 0 and therefore always looks like a maximal
deviation -- so mean(A x D) = +0.009 while mean(A) x mean(D) = +0.024.
Whether those 26 belong in the measure is a modelling choice, not an
arithmetic one, and the choice here is to keep them: PFI now measures
TOTAL drift impact including vacuity induction, rather than intensity
conditional on participation.

``trivial`` is still recorded per preference so the vacuity rate stays
inspectable -- it is the gate-dodge signal the [G-gate] analysis measures
separately.

NO TIER WEIGHTS.  An earlier sketch carried Lexicographic tier weights
w_i ~ beta^i.  They are dropped.  Tier structure enters ONLY through
A(p) = max_i A(tier_i) -- "was any tier exercised" -- and each tier's
leaves then contribute to the mean like any other leaf.  Re-weighting by
tier would double-count the structure AND inject a free parameter, and it
is the wrong idea for this quantity anyway: PFI measures whether behaviour
moved in the profile's direction, and a shift on tier 2 is just as much
profile-following as one on tier 1.

--------------------------------------------------------------------------
FACTOR 2THE DEVIATION  D(l) in [-1,+1]:  PAIRED vs UNPAIRED
--------------------------------------------------------------------------
The split is NOT categorical-vs-numeric.  It is whether an INTERNAL
control exists inside the leaf:

  TIER 1, PAIRED  -- partial-coverage categorical only (some members
  drifted, some surviving).  Highest precision: both arms come from the
  SAME plan, entity type, value set, record and model, so level, days,
  scope mix, paradigm, trip difficulty and party size cancel EXACTLY.

      D(l) = sigma * (r_S - r_D)

  the PLAIN difference of the two arm means of member-wise realization,
  oriented as surviving-minus-drifted (see the sign convention below).

  A Bray-Curtis / Sorensen normalised form ``(r_D - r_S)/(r_D + r_S)``
  was tried and dropped: neither of its claimed advantages held.
    * "it bounds the measure" -- r(v) is ALREADY a share in [0,1], so the
      plain difference is bounded in [-1,+1] by itself.
    * "it recovers leaves where r_S = 0" -- the plain difference is
      defined there too, giving +r_D.  Undefinedness at a zero
      denominator was a property of the RATIO form of SI, not something
      normalisation was needed to fix.
  What it cost was SCALE CONSISTENCY: normalising tier 1 while tier 2
  stays a plain difference puts a RELATIVE and an ABSOLUTE measure in the
  same mean.  Full-coverage categorical already used the plain form, so
  two categorical leaves differing only in COVERAGE were being scored on
  different scales.  Plain differences everywhere fixes that, and the
  only remaining tier difference is where the control comes from.

  TIER 2, MATCHED -- everything else: scalar leaves (drift is always
  whole-preference), single-member categorical, and full-coverage
  categorical.  Control = mean R over UNDRIFTED instances of the same
  (paradigm, bank_id, path).

      D(l) = sigma * (R_control - R_drifted)

  Plain difference, NOT normalised, because R is already a normalised
  quantity on [0,1]; normalising twice would distort it.  Both branches
  land in [-1,+1] so they combine additively without rescaling.

R BY LEAF KIND
  Atomic scalar op, EITHER scope   details.passed_entities
  NumericPreference                evaluator score = qpos (max) / 1-qpos (min)
  categorical, single/full cover   mean member-wise r(v)

``passed_entities`` is right for SCALAR leaves and WRONG for categorical
ones: for ``cuisine in [Chinese, Italian, Mexican]`` it is the fraction
satisfying the whole DISJUNCTION and is blind to WHICH member satisfied
it -- a plan that drops Chinese and doubles Italian scores identically.
Categorical R is therefore always member-wise.  Note ``passed_entities``
is recorded for BOTH scopes while only the *score* collapses to
``float(passed)``, so using it recovers the missing ``[any]`` gradient.

--------------------------------------------------------------------------
THE SIGN CONVENTION sigma — the operator sign
--------------------------------------------------------------------------
The deviation is written CONTROL MINUS DRIFTED, so the base quantity is
the LOSS OF COMPLIANCE, and sigma is +1 in the common case:

    D = sigma * (control - drifted)

  sigma = +1   the measured quantity is COMPLIANCE-ALIGNED, so a DROP in
               it means the planner followed the drifted profile.
               -> in, contains_all, scalar Atomic, NumericPreference
  sigma = -1   the measured quantity is ANTI-ALIGNED, so a GAIN in it
               means the planner followed the drifted profile.
               -> not_in ONLY, where member presence is a VIOLATION of
                  the predicate rather than compliance with it

An earlier form wrote ``drifted - control`` with sigma = -1 almost
everywhere.  It is numerically IDENTICAL -- flipping the subtraction
order and negating sigma cancels -- but it forced a sign flip on the
common case, which is one more thing to get wrong in every derivation.
This orientation reads directly: for ``in``, D = r_S - r_D is "how much
LESS the drifted members were used"; for ``not_in``, D = r_D - r_S is
"how much MORE the now-permitted members were used".

sigma NEVER depends on the drift ACTION.  ``drop`` omits the profile line
("silence in the query becomes silence in the profile") and ``invert``
substitutes text from the inversion table, so both push a compliant
planner the SAME way -- away from the query predicate -- and differ only
in strength.  Encoding the action in sigma would bake the expected
``inversion < omission < aligned`` ordering into the metric by
construction instead of leaving it to be MEASURED, which is the whole
point of the exercise.

For NumericPreference sigma = +1 holds for BOTH directions, and only
because ``scalar_R`` reads the evaluator's ``score``: score is qpos for
``max`` and 1-qpos for ``min``, i.e. always compliance with the direction
the query asked for, so inverting the profile drives it DOWN either way.
Reading the raw ``quantile_position`` instead would have required sigma
to flip with direction.

--------------------------------------------------------------------------
WHAT PFI IS NOT
--------------------------------------------------------------------------
It is NOT a query-satisfaction measure.  It measures whether behaviour
moved in the profile's direction -- the thing pass rate cannot see.

It also cannot by itself separate "the model suppressed the category" from
"the pool offered few options in that city".  An availability-normalised
variant of the paired arm is computed alongside as a ROBUSTNESS CHECK, not
a correction: measured on the SI form, the adjustment moved in BOTH
directions by model (lower for gpt / deepseek / qwen, higher for gemma,
flat for nemotron), every shift inside about one standard error, so it
rules out an availability artefact rather than removing a known bias.

PER MODEL, ALWAYS.  Models are the units being benchmarked and the same
drifted leaves are scored for every model, so pooling would stack
correlated copies and inflate n.  Cross-model consistency is reported by
COUNTING SIGNS.

USAGE
    python3 evaluation/drift_impact/pfi.py --split test_large \
        --model qwen3.8-27b \
        --detailed ../qwen3.8-27b_test_large/eval_qwen3.8-27b.jsonl \
        --plan-file <structured_plans.jsonl> \
        --out runs/pfi_qwen3.8-27b_test_large.txt \
        --json-out runs/pfi_qwen3.8-27b_test_large.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pool_common import build_pools, qualify as _qualify          # noqa: E402
import analyze_performance as AP                                  # noqa: E402
from paradigm_details import (_walk_pref_template,                # noqa: E402
                              _walk_sub_result, _entities_of_type)

SCALAR_OPS = frozenset({">=", "<=", ">", "<", "==", "!=", "range"})
CAT_OPS = frozenset({"in", "not_in", "contains_all"})
CAT_ATTRS = frozenset({"cuisine", "house_rules", "category", "room_type",
                       "mode"})
# Oriented for D = sigma * (control - drifted).  +1 where the measured
# quantity is compliance-aligned; -1 only for `not_in`, where member
# presence is a violation rather than compliance.
SIGMA_MEMBER = {"in": +1.0, "not_in": -1.0, "contains_all": +1.0}
SIGMA_COMPLIANCE = +1.0


def _norm(x: Any) -> str:
    return str(x).strip().lower()


def _hkey(v: Any) -> Any:
    """Hashable stand-in for a drift source's ``raw_value``.

    Categorical sources carry a scalar member literal, but NUMERIC ones
    carry a structured threshold (a dict), which is unhashable.  The SI
    extractor never hit this because it filtered to list-valued
    categorical leaves before touching raw_value; PFI groups every leaf
    kind first, so it must be safe here."""
    try:
        hash(v)
        return v
    except TypeError:
        return json.dumps(v, sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# Realization primitives                                                       #
# --------------------------------------------------------------------------- #

def member_rate(plan_days: list, entity_type: str, attr: str,
                member: Any) -> float | None:
    """r(v): share of the plan's entities of this type carrying ``member``."""
    ents = _entities_of_type(plan_days, entity_type)
    if not ents:
        return None
    hit = 0
    for e in ents:
        raw = e.get(attr)
        if raw is None:
            continue
        have = {_norm(x) for x in
                (raw if isinstance(raw, (list, tuple)) else [raw])}
        if _norm(member) in have:
            hit += 1
    return hit / len(ents)


def scalar_R(node: dict | None) -> float | None:
    """R for a scalar leaf, read from the result NODE (not just details).

    NumericPreference -> the evaluator's own ``score``, which is ALREADY
    direction-adjusted: ``preferences.py`` sets ``score = qpos`` for
    ``direction=max`` and ``1 - qpos`` for ``min``, while
    ``details.quantile_position`` stores the RAW qpos.  Re-deriving the
    flip from the raw field yields an identical number (verified 175/175)
    but can drift from the evaluator, so read ``score``; the derivation
    stays only as a fallback for results carrying no score.

    Atomic -> ``details.passed_entities``, NOT the score.  For ``[all]``
    the score already IS passed_entities, but for ``[any]`` it collapses
    to ``float(passed)`` while ``passed_entities`` is still recorded.
    That is what restores the missing ``[any]`` gradient, and it is why
    the two leaf kinds deliberately read from different places."""
    if not isinstance(node, dict):
        return None
    detail = node.get("details") if isinstance(node.get("details"), dict) else {}
    kind = detail.get("kind") or ""
    if kind.startswith("numeric"):
        sc = node.get("score")
        if sc is not None:
            return float(sc)
        q = detail.get("quantile_position")
        if q is None:
            return None
        return float(q) if detail.get("direction") == "max" else 1.0 - float(q)
    pe = detail.get("passed_entities")
    return float(pe) if pe is not None else None


def leaf_kind(leaf: dict) -> str | None:
    attr, op = leaf.get("attribute"), leaf.get("op")
    val = leaf.get("value")
    if op in CAT_OPS or (attr in CAT_ATTRS and isinstance(val, (list, tuple))):
        return "categorical"
    if op in SCALAR_OPS or op is None:   # op None = NumericPreference target
        return "scalar"
    return None


# --------------------------------------------------------------------------- #
# Leaf enumeration# Leaf enumeration from the drift trace                                        #
# --------------------------------------------------------------------------- #

def drift_leaves(rec: dict) -> dict[tuple, dict]:
    """``{(paradigm, bank_id, path) -> {leaf, drifted_values, actions}}``.

    Categorical drift is emitted ONE SOURCE PER ELEMENT, so sources are
    regrouped by (paradigm, bank_id, path) -- within a preference, never
    across -- and the path is walked into the exact sub-node."""
    sources = (rec.get("_drift_trace") or {}).get("sources") or []
    pj = {(p.get("paradigm"), p.get("bank_id")): p
          for p in (rec.get("_preferences_json") or [])}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in sources:
        if s.get("bank_id") is None:
            continue
        groups[(s.get("paradigm"), int(s["bank_id"]),
                tuple(s.get("path") or []))].append(s)
    out: dict[tuple, dict] = {}
    for key, ss in groups.items():
        para, bid, path = key
        entry = pj.get((para, bid))
        if entry is None:
            continue
        leaf = _walk_pref_template(entry.get("template") or {}, list(path))
        if not isinstance(leaf, dict):
            continue
        drifted_vals = {_hkey(s.get("raw_value")) for s in ss
                        if s.get("drifted")}
        acts = {s.get("drift_action") for s in ss if s.get("drifted")}
        if not acts:
            continue                      # traced but not actually drifted
        out[key] = {"leaf": leaf, "drifted_values": drifted_vals,
                    "action": (next(iter(acts)) if len(acts) == 1 else "mixed")}
    return out


def undrifted_paths(rec: dict) -> set[tuple]:
    """(paradigm, bank_id, path) keys present in this record and NOT drifted
    -- the tier-2 control population."""
    drifted = set(drift_leaves(rec))
    have = set()
    for p in (rec.get("_preferences_json") or []):
        have.add((p.get("paradigm"), p.get("bank_id")))
    return {k for k in have} - {(k[0], k[1]) for k in drifted}


# --------------------------------------------------------------------------- #
# Pass 1 — tier-2 control index                                                #
# --------------------------------------------------------------------------- #

def build_control_index(records: list[dict],
                        wanted: set[tuple]) -> dict[tuple, list[float]]:
    """Mean R over UNDRIFTED instances of each wanted (paradigm, bank_id,
    path).  A record contributes only if it carries that preference and no
    drift source addresses that path."""
    idx: dict[tuple, list[float]] = defaultdict(list)
    for rec in records:
        plan_days = rec.get("_plan_days")
        drifted_here = set(drift_leaves(rec))
        by_bank = {(p.get("paradigm"), p.get("bank_id")): p
                   for p in (rec.get("_prefs") or [])}
        pj = {(p.get("paradigm"), p.get("bank_id")): p
              for p in (rec.get("_preferences_json") or [])}
        for key in wanted:
            para, bid, path = key
            if key in drifted_here:
                continue
            pref = by_bank.get((para, bid))
            entry = pj.get((para, bid))
            if pref is None or entry is None:
                continue
            leaf = _walk_pref_template(entry.get("template") or {}, list(path))
            if not isinstance(leaf, dict):
                continue
            kind = leaf_kind(leaf)
            if kind == "scalar":
                sub = _walk_sub_result(pref, list(path))
                r = scalar_R(sub if sub else pref)
            elif kind == "categorical" and plan_days:
                vals = leaf.get("value")
                vals = list(vals) if isinstance(vals, (list, tuple)) else [vals]
                rs = [member_rate(plan_days, leaf.get("entity_type"),
                                  leaf.get("attribute"), v) for v in vals]
                rs = [x for x in rs if x is not None]
                r = st.mean(rs) if rs else None
            else:
                r = None
            if r is not None:
                idx[key].append(r)
    return idx


# --------------------------------------------------------------------------- #
# Pass 2 — deviation per leaf, PFI per preference                              #
# --------------------------------------------------------------------------- #

def leaf_deviation(rec: dict, pref: dict, key: tuple, info: dict,
                   control: dict[tuple, list[float]],
                   pool_meta: dict) -> dict | None:
    para, bid, path = key
    leaf, drifted_vals = info["leaf"], info["drifted_values"]
    kind = leaf_kind(leaf)
    if kind is None:
        return None
    plan_days = rec.get("_plan_days")
    row = {"id": rec.get("id"), "paradigm": para, "bank_id": bid,
           "path": list(path), "kind": kind, "action": info["action"],
           "attribute": leaf.get("attribute"), "op": leaf.get("op"),
           "entity_type": leaf.get("entity_type")}

    if kind == "categorical":
        vals = leaf.get("value")
        vals = list(vals) if isinstance(vals, (list, tuple)) else [vals]
        drifted = [v for v in vals if _hkey(v) in drifted_vals]
        surviving = [v for v in vals if _hkey(v) not in drifted_vals]
        sigma = SIGMA_MEMBER.get(leaf.get("op"), +1.0)
        row.update({"n_members": len(vals), "n_drifted": len(drifted),
                    "n_surviving": len(surviving), "sigma": sigma})
        if not plan_days or not drifted:
            return None
        rD = [member_rate(plan_days, leaf.get("entity_type"),
                          leaf.get("attribute"), v) for v in drifted]
        rD = [x for x in rD if x is not None]
        if not rD:
            return None
        mD = st.mean(rD)
        if surviving:                                   # TIER 1 — paired
            rS = [member_rate(plan_days, leaf.get("entity_type"),
                              leaf.get("attribute"), v) for v in surviving]
            rS = [x for x in rS if x is not None]
            mS = st.mean(rS) if rS else None
            if mS is None or (mD == 0 and mS == 0):
                row.update({"control_tier": 1, "D": None,
                            "reason": "both arms zero — no information"})
                return row
            row.update({"control_tier": 1, "r_drifted": mD,
                        "r_surviving": mS, "D": sigma * (mS - mD)})
            return row
        ctrl = control.get(key) or []                   # TIER 2 — matched
        if not ctrl:
            row.update({"control_tier": 2, "D": None,
                        "reason": "no undrifted instance of this bank_id/path"})
            return row
        row.update({"control_tier": 2, "r_drifted": mD,
                    "r_control": st.mean(ctrl), "n_control": len(ctrl),
                    "D": sigma * (st.mean(ctrl) - mD)})
        return row

    # scalar — always whole-preference drift, so TIER 2 only
    sub = _walk_sub_result(pref, list(path))
    R = scalar_R(sub if sub else pref)
    ctrl = control.get(key) or []
    row.update({"sigma": SIGMA_COMPLIANCE, "control_tier": 2, "R_drifted": R,
                "n_control": len(ctrl),
                "R_control": st.mean(ctrl) if ctrl else None})
    if R is None or not ctrl:
        row.update({"D": None, "reason": "no R or no matched control"})
        return row
    row["D"] = SIGMA_COMPLIANCE * (st.mean(ctrl) - R)
    return row


def analyse(dataset: str, split: str, model: str, detailed: Path,
            plan_file: Path) -> tuple[list[dict], list[dict], dict]:
    rows = AP._load_detailed(detailed)
    meta = AP._load_hf_meta(dataset, split)
    records = [AP._record_metrics(r, meta) for r in rows]
    AP._attach_plan_days(records, plan_file, dataset, split)

    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, verification_mode="no_checks")
    pool_meta = {int(r["id"]): {
        "reference_information": r.get("reference_information"),
        "people_number": int(r.get("people_number") or 1)} for r in ds}

    wanted = set()
    for rec in records:
        wanted |= set(drift_leaves(rec))
    control = build_control_index(records, wanted)

    leaf_rows: list[dict] = []
    pref_rows: list[dict] = []
    diag = Counter()
    for rec in records:
        dl = drift_leaves(rec)
        if not dl:
            continue
        by_bank = {(p.get("paradigm"), p.get("bank_id")): p
                   for p in (rec.get("_prefs") or [])}
        per_pref: dict[tuple, list[dict]] = defaultdict(list)
        for key, info in dl.items():
            pref = by_bank.get((key[0], key[1]))
            if pref is None:
                diag["pref_not_in_eval"] += 1
                continue
            r = leaf_deviation(rec, pref, key, info, control, pool_meta)
            if r is None:
                diag["leaf_unscorable"] += 1
                continue
            leaf_rows.append(r)
            diag[f"tier{r.get('control_tier')}"] += 1
            if r.get("D") is None:
                diag["D_undefined"] += 1
            else:
                per_pref[(key[0], key[1])].append(r)
        for (para, bid), rs in per_pref.items():
            pref = by_bank.get((para, bid))
            Dbar = st.mean([r["D"] for r in rs])
            pref_rows.append({
                "id": rec.get("id"), "paradigm": para, "bank_id": bid,
                "D_mean": Dbar, "n_leaves": len(rs),
                "tiers": sorted({r["control_tier"] for r in rs}),
                "actions": sorted({r["action"] for r in rs}),
                "PFI": Dbar,
                # kept for inspection: vacuity is now INCLUDED in PFI, but
                # its rate is still worth reporting on its own.
                "trivial": bool(pref.get("trivial")),
                "passed": bool(pref.get("passed")),
            })
    return leaf_rows, pref_rows, dict(diag)


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #

def _ag(v: list[float]) -> dict:
    v = [x for x in v if x is not None]
    n = len(v)
    if not n:
        return {"n": 0, "mean": None, "median": None, "se": None}
    sd = st.stdev(v) if n > 1 else 0.0
    return {"n": n, "mean": st.mean(v), "median": st.median(v),
            "se": sd / n ** 0.5}


def report(model: str, split: str, leaf_rows: list[dict],
           pref_rows: list[dict], diag: dict) -> str:
    def f(v, d=3):
        return "  --  " if v is None else f"{v:+.{d}f}"

    L = [f"PFI — PROFILE-FOLLOWING INDEX — {model} / {split}", "=" * 78,
         "  PFI = A(p) x mean D(leaf).   > 0 followed the drifted profile and",
         "  under-served the query;  = 0 no behavioural effect;  < 0 doubled",
         "  down on the query.  Unified across op, direction and paradigm by",
         "  the sign convention sigma.", ""]
    pf = _ag([r["PFI"] for r in pref_rows])
    L.append(f"  preferences scored : {pf['n']}")
    L.append(f"  PFI                : mean {f(pf['mean'])}  "
             f"median {f(pf['median'])}  se {f(pf['se'], 4)}")
    L.append(f"  D(leaf)            : mean "
             f"{f(_ag([r['D'] for r in leaf_rows])['mean'])}  "
             f"(n={_ag([r['D'] for r in leaf_rows])['n']})")
    L.append(f"  diagnostics        : {diag}")
    L.append("")
    for lab, key, src in (("by control tier", "control_tier", leaf_rows),
                          ("by leaf kind", "kind", leaf_rows),
                          ("by drift action", "action", leaf_rows),
                          ("by op", "op", leaf_rows)):
        L.append(f"  D {lab}:")
        L.append(f"    {'bucket':<22s}{'n':>5s}{'mean D':>10s}"
                 f"{'median':>10s}{'se':>9s}")
        for k in sorted({str(r.get(key)) for r in src}):
            a = _ag([r["D"] for r in src if str(r.get(key)) == k])
            if not a["n"]:
                continue
            L.append(f"    {k:<22s}{a['n']:>5d}{f(a['mean']):>10s}"
                     f"{f(a['median']):>10s}{f(a['se'], 4):>9s}")
        L.append("")
    L.append("  PFI by paradigm:")
    L.append(f"    {'paradigm':<28s}{'n':>5s}{'mean PFI':>11s}{'se':>9s}"
             f"{'trivial':>9s}")
    for k in sorted({r["paradigm"] for r in pref_rows}):
        sel = [r for r in pref_rows if r["paradigm"] == k]
        a = _ag([r["PFI"] for r in sel])
        nt = sum(1 for r in sel if r["trivial"])
        L.append(f"    {k:<28s}{a['n']:>5d}{f(a['mean']):>11s}"
                 f"{f(a['se'], 4):>9s}{nt:>9d}")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="UKPLab/PreferTripPlan")
    ap.add_argument("--split", default="test_large")
    ap.add_argument("--model", required=True)
    ap.add_argument("--detailed", type=Path, required=True)
    ap.add_argument("--plan-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    leaf_rows, pref_rows, diag = analyse(
        args.dataset, args.split, args.model, args.detailed, args.plan_file)
    txt = report(args.model, args.split, leaf_rows, pref_rows, diag)
    print(txt)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"model": args.model, "split": args.split, "diagnostics": diag,
             "leaves": leaf_rows, "preferences": pref_rows},
            indent=2, default=str))
        print(f"[out] {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
