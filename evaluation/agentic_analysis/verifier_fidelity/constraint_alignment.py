#!/usr/bin/env python3
"""Constraint-level correspondence between the agent's self-coded preference
checks and the gold preference evaluator, on one agentic run.

`verifier_fidelity.py` answers "did any authored check fire on a plan the
evaluator rejects".  That is a firing test: a check may fire for a reason
unrelated to the rule that was actually violated, so every number it produces is
an upper bound on real recall.  This script removes that caveat by pairing each
GOLD preference with the self-coded check(s) the agent wrote for it, and then
comparing the two verdicts on the same plan.

The pairing is structural, not semantic.  A gold preference in
`preferences_json` is a recursive composition of atoms, each carrying
`entity_type`, `attribute`, `op` and `value`.  Those literals reappear almost
verbatim in the check the agent authors, so a gold preference is matched to the
self-coded check whose text, pseudocode and source share its values and
attributes.  Alignment quality is reported rather than assumed, and the verdict
comparison is restricted to pairs above a confidence floor.

Three quantities come out of it:

  COVERAGE     did the agent write any check at all aimed at this preference?
               This needs no verdict and is robust to alignment error.
  AGREEMENT    on aligned pairs, does the authored check reach the same verdict
               as the evaluator, plan by plan?  This is fidelity proper.
  FRAGMENTATION how many checks the agent spends on one gold preference.  A
               structurally composed preference is often split into independent
               conjuncts, which loses the composition.

Inputs are the trajectory (for the authored specs), the per-plan judgements
`verifier_fidelity.py` writes (for self-check verdicts), and the per-plan
iteration records (for `pref_state`, the gold per-preference verdicts).
"""
from __future__ import annotations

import argparse
import json
import ast
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parents[1]
AGENTIC = EVAL.parent / "plan-generation" / "agentic"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(AGENTIC))

STRUCTURAL = {"Conditional", "Compensatory", "Lexicographic", "Temporal", "Scoped"}
_NUM = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z_]+")


def _norm_word(w: str) -> str:
    w = w.lower()
    return w[:-1] if len(w) > 4 and w.endswith("s") else w


def _norm_num(x) -> str:
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x).lower()


def gold_signature(node, attrs, vals, ents, ops=None, quants=None) -> None:
    """Collect the five slots of the benchmark's leaf predicate.

    The predicate is (entity, attribute, operator, value, quantifier) and all
    five are used. Dropping the operator would let a check testing
    `rating <= 3.0` pair with a gold preference demanding `rating >= 3.0`;
    dropping the quantifier would not separate "every restaurant" from "at
    least one restaurant".
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "attribute" and isinstance(v, str):
                attrs.add(_norm_word(v))
            elif k == "entity_type" and isinstance(v, str):
                ents.add(_norm_word(v))
            elif k == "op" and isinstance(v, str) and ops is not None:
                ops.add(v.strip().lower())
            elif k == "scope" and isinstance(v, str) and quants is not None:
                quants.add(v.strip().lower())
            elif k == "value":
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, bool):
                        continue
                    if isinstance(x, (int, float)):
                        vals.add(_norm_num(x))
                    elif isinstance(x, str):
                        parts = [w for w in _WORD.findall(x.lower()) if len(w) > 2]
                        vals.update(_norm_word(w) for w in parts) if len(parts) > 1 \
                            else vals.add(_norm_word(x))
            else:
                gold_signature(v, attrs, vals, ents, ops, quants)
    elif isinstance(node, list):
        for x in node:
            gold_signature(x, attrs, vals, ents, ops, quants)


_CMP = {ast.GtE: ">=", ast.LtE: "<=", ast.Gt: ">", ast.Lt: "<",
        ast.Eq: "==", ast.NotEq: "!=", ast.In: "in", ast.NotIn: "not_in"}
_POP = [(r">=|at least|or higher|no less than|minimum|not below", ">="),
        (r"<=|at most|or less|or lower|no more than|maximum|not exceed", "<="),
        (r"not in\b|must not|may not|never|exclude|avoid", "not_in"),
        (r"\bin\b|belongs? to|among|one of|includes?", "in"),
        (r"==|exactly|equals|must be\b", "==")]
_PQ = [(r"for all|every|each\b|\ball\b", "all"),
       (r"exists|at least one|\bany\b|\bsome\b", "any"),
       (r"per.?day|each day|on any day", "per_day"),
       (r"across the trip|whole trip|entire trip|globally", "global")]


def self_ops(c):
    """Operators and quantifiers the authored check actually uses."""
    ops, quants = set(), set()
    try:
        for n in ast.walk(ast.parse(str(c.get("python") or ""))):
            if isinstance(n, ast.Compare):
                for o in n.ops:
                    if type(o) in _CMP:
                        ops.add(_CMP[type(o)])
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                if n.func.id in ("all", "any"):
                    quants.add(n.func.id)
    except (SyntaxError, ValueError):
        pass
    blob = (str(c.get("text") or "") + " " + str(c.get("pseudocode") or "")).lower()
    for pat, sym in _POP:
        if re.search(pat, blob):
            ops.add(sym)
    for pat, sym in _PQ:
        if re.search(pat, blob):
            quants.add(sym)
    return ops, quants


def self_tokens(c: dict) -> tuple[set, set]:
    """Words and numbers appearing anywhere in an authored check."""
    blob = " ".join(str(c.get(k) or "") for k in ("text", "pseudocode", "python"))
    low = blob.lower()
    return ({_norm_word(w) for w in _WORD.findall(low)},
            {_norm_num(n) for n in _NUM.findall(low)})


def score(gold: dict, check: dict) -> float:
    """Weighted share of the gold predicate's five slots the check reproduces."""
    attrs, vals, ents, ops, quants = set(), set(), set(), set(), set()
    gold_signature(gold.get("template"), attrs, vals, ents, ops, quants)
    # AND/OR and the temporal operators are composition, not leaf comparisons
    cmp_ops = ops & {">=", "<=", ">", "<", "==", "!=", "in", "not_in"}
    words, nums = self_tokens(check)
    seen = words | nums
    c_ops, c_quants = self_ops(check)

    parts, wts = [], []
    for got, w, have in ((vals, 0.40, seen), (attrs, 0.20, seen), (ents, 0.15, seen),
                         (cmp_ops, 0.15, c_ops), (quants, 0.10, c_quants)):
        if got:
            parts.append(sum(1 for x in got if x in have) / len(got) * w)
            wts.append(w)
    return sum(parts) / sum(wts) if wts else 0.0


# --------------------------------------------------------------------------- #
# gold signatures for the FIXED rules (commonsense / hard)                      #
# --------------------------------------------------------------------------- #
# A preference's gold form is DATA: `preferences_json` states entity_type,
# attribute, op and value, so its signature is read straight off the record.
# A commonsense or hard rule's gold form is CODE, so its signature is read off
# the evaluator's own source instead: the string literals it can return as a
# failure message, and the plan keys it inspects. That is a weaker signal than
# an explicit value, and the panels say so.


def rule_signatures(py_path: Path) -> dict:
    """{function name: (words, numbers)} harvested from each rule body."""
    tree = ast.parse(py_path.read_text())
    out = {}
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        words, nums = set(), set()
        for w in _WORD.findall(fn.name.lower()):
            words.add(_norm_word(w))
        for node in ast.walk(fn):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, str):
                    for w in _WORD.findall(node.value.lower()):
                        if len(w) > 2:
                            words.add(_norm_word(w))
                elif isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                    nums.add(_norm_num(node.value))
        words -= {"the", "and", "for", "not", "are", "day", "you", "with", "that",
                  "this", "from", "have", "must", "false", "true", "none", "info",
                  "question", "tested_data", "unit", "return", "len", "str", "int"}
        out[fn.name] = (words, nums)
    return out


def rule_idf(sigs: dict) -> dict:
    """Inverse document frequency of each token across the rule signatures.

    A plain overlap fraction rewards small rule bodies and punishes large ones,
    and it treats 'days' -- which every rule mentions -- as evidence. Weighting
    by rarity across the rule set makes the distinctive token ('absent',
    'repeated', 'budget') carry the match, which is what a reader would use.
    """
    df = Counter()
    for words, nums in sigs.values():
        for t in (words | nums):
            df[t] += 1
    n = max(len(sigs), 1)
    return {t: math.log(n / c) + 1.0 for t, c in df.items()}


def rule_score(sig, check: dict, idf: dict | None = None) -> float:
    words, nums = sig
    w, n = self_tokens(check)
    seen = w | n
    both = words | nums
    if not both:
        return 0.0
    if idf is None:
        return sum(1 for x in both if x in seen) / len(both)
    num = sum(idf.get(x, 1.0) for x in both if x in seen)
    den = sum(idf.get(x, 1.0) for x in both)
    return num / den if den else 0.0

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--fidelity", required=True, help="per-plan jsonl from verifier_fidelity.py")
    ap.add_argument("--iter", required=True, help="per-plan jsonl from iterative_repair.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--floor", type=float, default=0.5, help="alignment floor, preferences")
    ap.add_argument("--floor-rule", dest="floor_rule", type=float, default=0.30,
                    help="alignment floor for commonsense/hard rules")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--with-rules", dest="with_rules", action="store_true",
                    help="additionally align commonsense/hard checks to the "
                         "evaluator's fixed rule set (weaker procedure, off by default)")
    ap.add_argument("--audit", type=int, default=12, help="aligned pairs to print for manual audit")
    a = ap.parse_args()

    import _paths                                                # noqa: PLC0415
    gold_rows = {int(r["id"]): r for r in _paths.load_split(a.split)}

    specs = {}
    for line in Path(a.traj).open():
        if not line.strip():
            continue
        t = json.loads(line)
        sp = t["spec_json"] if isinstance(t["spec_json"], dict) else json.loads(t["spec_json"] or "{}")
        if sp.get("constraints"):
            specs[t["id"]] = sp

    fid = {}
    for line in Path(a.fidelity).open():
        if line.strip():
            r = json.loads(line)
            fid[(r["id"], r["plan_index"])] = r
    itr = {}
    for line in Path(a.iter).open():
        if line.strip():
            r = json.loads(line)
            itr[(r["id"], r["plan_index"])] = r

    # ---- align, once per record -------------------------------------------- #
    pairs = []          # one per (record, gold preference)
    for rid, sp in specs.items():
        if rid not in gold_rows:
            continue
        gold = json.loads(gold_rows[rid]["preferences_json"])
        checks = [c for c in sp["constraints"] if c.get("kind") == "preference"]
        for gi, g in enumerate(gold):
            ranked = sorted(((score(g, c), c) for c in checks), key=lambda x: -x[0])
            best = [(s, c) for s, c in ranked if s >= a.floor]
            pairs.append({"id": rid, "gold_index": gi,
                          "paradigm": g["paradigm"].replace("Preference", ""),
                          "n_checks_available": len(checks),
                          "aligned": [c["id"] for _, c in best],
                          "top_score": ranked[0][0] if ranked else 0.0,
                          "gold_template": g.get("template")})

    L = []

    def H(t):
        L.append("")
        L.append("=" * 96)
        L.append(t)
        L.append("=" * 96)

    H("1. COVERAGE  (did the agent author a check aimed at this gold preference?)")
    cov = defaultdict(lambda: [0, 0])
    for p in pairs:
        cls = "structural" if p["paradigm"] in STRUCTURAL else "flat"
        cov[p["paradigm"]][1] += 1
        cov["ALL " + cls][1] += 1
        if p["aligned"]:
            cov[p["paradigm"]][0] += 1
            cov["ALL " + cls][0] += 1
    L.append(f"  {'paradigm':<18}{'class':<12}{'gold prefs':>11}{'covered':>9}{'rate':>8}")
    for k, (hit, tot) in sorted(cov.items(), key=lambda x: (x[0].startswith("ALL"), -x[1][1])):
        if k.startswith("ALL"):
            continue
        L.append(f"  {k:<18}{('structural' if k in STRUCTURAL else 'flat'):<12}"
                 f"{tot:>11}{hit:>9}{hit/tot:>8.3f}")
    for cls in ("flat", "structural"):
        hit, tot = cov["ALL " + cls]
        if tot:
            L.append(f"  -> all {cls:<13}{'':<12}{tot:>11}{hit:>9}{hit/tot:>8.3f}")
    L.append(f"  Alignment floor {a.floor}. Coverage needs no verdict, so it is the")
    L.append("  quantity least exposed to alignment error.")

    H("2. FRAGMENTATION  (checks the agent spends on one gold preference)")
    frag = defaultdict(Counter)
    for p in pairs:
        if p["aligned"]:
            cls = "structural" if p["paradigm"] in STRUCTURAL else "flat"
            frag[cls][min(len(p["aligned"]), 3)] += 1
    L.append(f"  {'class':<14}{'1 check':>9}{'2 checks':>10}{'3+':>6}{'mean':>8}")
    for cls in ("flat", "structural"):
        c = frag[cls]
        n = sum(c.values())
        if not n:
            continue
        mean = sum(k * v for k, v in c.items()) / n
        L.append(f"  {cls:<14}{c[1]:>9}{c[2]:>10}{c[3]:>6}{mean:>8.2f}")
    L.append("  A composed preference split into independent conjuncts loses the")
    L.append("  composition, whatever each conjunct tests correctly.")

    H("3. VERDICT AGREEMENT  (fidelity proper, split on whether aggregation is needed)")
    L.append("  When one gold preference draws ONE check the comparison is direct. When it")
    L.append("  draws several, combining them needs the gold composition semantics: a")
    L.append("  Lexicographic preference holds if its top tier holds, so treating the tier")
    L.append("  checks as a conjunction ('any fires -> fail') is wrong, and the same applies")
    L.append("  to Composite-OR and to Compensatory fallbacks. Tier-level correspondence is")
    L.append("  not recoverable from the authored checks, so the 1:1 subset is reported as")
    L.append("  the measurement and the many:1 subset separately, under conjunction, which")
    L.append("  makes the verifier report failure more readily than the gold semantics would")
    L.append("  and therefore OVERSTATES its recall.")

    def tally(only):
        agg = defaultdict(Counter)
        for p in pairs:
            k = len(p["aligned"])
            if not k or not only(k):
                continue
            cls = "structural" if p["paradigm"] in STRUCTURAL else "flat"
            for (rid, pi), f in fid.items():
                if rid != p["id"]:
                    continue
                t = itr.get((rid, pi))
                if not t or not t.get("pref_state"):
                    continue
                st = t["pref_state"]
                if p["gold_index"] >= len(st):
                    continue
                ran = set(f.get("self_ran") or [])
                if not any(c in ran for c in p["aligned"]):
                    continue
                gold_fail = not st[p["gold_index"]]["p"]
                self_fail = any(c in set(f["self_fired"]) for c in p["aligned"])
                for key in (p["paradigm"], "ALL " + cls):
                    agg[key]["n"] += 1
                    agg[key]["tp"] += gold_fail and self_fail
                    agg[key]["fp"] += (not gold_fail) and self_fail
                    agg[key]["fn"] += gold_fail and (not self_fail)
                    agg[key]["tn"] += (not gold_fail) and (not self_fail)
        return agg

    for title, only in (("3a. ONE check per gold preference (no aggregation rule)",
                         lambda k: k == 1),
                        ("3b. SEVERAL checks per gold preference (conjunction imposed)",
                         lambda k: k > 1)):
        agg = tally(only)
        L.append("")
        L.append(f"  {title}")
        L.append(f"  {'paradigm':<18}{'pairs':>7}{'viol.':>7}{'caught':>8}"
                 f"{'recall':>8}{'prec':>8}")
        for k in sorted(agg, key=lambda x: (x.startswith("ALL"), -agg[x]["n"])):
            c = agg[k]
            viol = c["tp"] + c["fn"]
            pr = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else float("nan")
            rc = c["tp"] / viol if viol else float("nan")
            L.append(f"  {k:<18}{c['n']:>7}{viol:>7}{c['tp']:>8}{rc:>8.3f}{pr:>8.3f}")

    # Commonsense and hard rules are gold as CODE, not as data, so their
    # signatures must be reverse-engineered from the evaluator source and a
    # check must be matched many-to-many against a fixed rule set. That is a
    # materially weaker and more assumption-laden procedure than the preference
    # pairing above, so it is off by default and reported separately when asked.
    if not a.with_rules:
        rep = "\n".join(L)
        print(rep)
        if a.summary:
            Path(a.summary).write_text(rep + "\n")
            print(f"\n[out] summary -> {a.summary}")
        return

    H("4. THE SAME PIPELINE ON COMMONSENSE AND HARD RULES")
    # Reported rule name -> the evaluator function that implements it. The two
    # differences are a spelling slip in the upstream source and the cost rule,
    # which is named for the quantity it computes rather than the test it makes.
    ALIAS = {"is_valid_accommodation": "is_valid_accommodaton",
             "valid_cost": "get_total_cost"}
    sig = {}
    sig.update(rule_signatures(EVAL / "commonsense_constraint.py"))
    sig.update(rule_signatures(EVAL / "hard_constraint.py"))
    idf = rule_idf({k: v for k, v in sig.items() if k.startswith(("is_", "get_"))})

    def sig_for(rule):
        fn = ALIAS.get(rule) or (rule if rule in sig else "is_" + rule)
        return sig.get(fn)

    fam_rules = {"commonsense": set(), "hard": set()}
    for t in itr.values():
        fam_rules["commonsense"].update(t.get("commonsense_failed") or [])
        fam_rules["hard"].update(t.get("hard_failed") or [])

    # align every authored commonsense/hard check to its best-matching gold rule
    # A check is assigned to EVERY rule it clears the floor on, not just its best
    # match. One authored check often bears on several rules at once -- a check
    # requiring one breakfast, one lunch, one dinner and an attraction each day
    # speaks to restaurant validity and to completeness -- and the question being
    # asked is whether any authored check bore on the violated rule, so absorbing
    # it into a single winner would understate coverage.
    assign = defaultdict(list)       # (record, check id) -> [rules]
    for rid, sp in specs.items():
        for c in sp["constraints"]:
            if c.get("kind") not in ("commonsense", "hard"):
                continue
            for fam in ("commonsense", "hard"):
                for rule in fam_rules[fam]:
                    sg = sig_for(rule)
                    if sg and rule_score(sg, c, idf) >= a.floor_rule:
                        assign[(rid, c["id"])].append(rule)

    L.append(f"  Alignment floor {a.floor_rule} (rules are matched on identifiers and failure")
    L.append("  messages read out of the evaluator source, a weaker signal than the explicit")
    L.append("  values a gold preference carries, so the floor is set lower).")
    L.append("")
    L.append(f"  {'evaluator rule':<40}{'family':<13}{'records w/ a check':>19}"
             f"{'viol.':>7}{'caught':>8}{'recall':>8}")
    fam_tot = defaultdict(Counter)
    for fam in ("commonsense", "hard"):
        for rule in sorted(fam_rules[fam]):
            recs_with = sum(1 for rid in specs
                            if any(rule in v for (r2, _), v in assign.items() if r2 == rid))
            tp = fn_ = 0
            for (rid, pi), f in fid.items():
                t = itr.get((rid, pi))
                if not t:
                    continue
                failed = set(t.get("commonsense_failed") or []) | set(t.get("hard_failed") or [])
                if rule not in failed:
                    continue
                ran, fired = set(f.get("self_ran") or []), set(f["self_fired"])
                mine = [cid for (r2, cid), v in assign.items() if r2 == rid and rule in v]
                if not any(c in ran for c in mine):
                    continue
                tp += any(c in fired for c in mine)
                fn_ += not any(c in fired for c in mine)
            viol = tp + fn_
            fam_tot[fam]["tp"] += tp
            fam_tot[fam]["viol"] += viol
            fam_tot[fam]["recs"] += recs_with
            L.append(f"  {rule:<40}{fam:<13}{recs_with:>19}{viol:>7}{tp:>8}"
                     f"{(tp / viol if viol else float('nan')):>8.3f}")
    for fam in ("commonsense", "hard"):
        c = fam_tot[fam]
        L.append(f"  -> all {fam:<34}{'':<13}{'':>19}{c['viol']:>7}{c['tp']:>8}"
                 f"{(c['tp'] / c['viol'] if c['viol'] else float('nan')):>8.3f}")
    L.append("  'viol.' counts only plans on which an aligned check actually ran, so the")
    L.append("  denominator is violations the authored verifier had a chance to catch.")

    H("5. ALIGNMENT AUDIT  (sample for manual inspection)")
    seen = 0
    for p in pairs:
        if not p["aligned"] or seen >= a.audit:
            continue
        sp = specs[p["id"]]
        by = {c["id"]: c for c in sp["constraints"]}
        L.append(f"  record {p['id']}  [{p['paradigm']}]  score={p['top_score']:.2f}")
        L.append(f"    GOLD {json.dumps(p['gold_template'])[:150]}")
        for cid in p["aligned"]:
            L.append(f"    SELF {cid}: {by[cid].get('text', '')[:120]}")
        seen += 1
    L.append(f"  Unaligned gold preferences: {sum(1 for p in pairs if not p['aligned'])}"
             f" of {len(pairs)}. Sample of those:")
    seen = 0
    for p in pairs:
        if p["aligned"] or seen >= 5:
            continue
        L.append(f"    record {p['id']} [{p['paradigm']}] best score {p['top_score']:.2f}"
                 f" over {p['n_checks_available']} authored preference check(s)")
        seen += 1

    rep = "\n".join(L)
    print(rep)
    if a.summary:
        Path(a.summary).write_text(rep + "\n")
        print(f"\n[out] summary -> {a.summary}")


if __name__ == "__main__":
    main()
