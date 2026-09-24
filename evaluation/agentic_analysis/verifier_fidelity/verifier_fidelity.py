#!/usr/bin/env python3
"""How close does a self-authored verifier come to the ground-truth one?

ChinaTravel's strongest baseline pairs an LLM planner with a ground-truth
symbolic verifier. This harness withholds that: the environment supplies only
entity existence and cost, and every other check is written by the agent from
the query alone. This script measures what that costs, by treating each verifier
as a binary classifier of plan validity and scoring it against the benchmark
evaluator over every plan the run produced, accepted or rejected.

Three verifiers are compared on the same plans:

  ENV      the environment oracle: entity resolution plus the budget comparison.
           Supplied and correct by construction, but blind to preferences and to
           anything spanning days.
  SELF     the agent's compiled checks, re-executed here so that every plan is
           judged by one check-set rather than by whichever revision was live at
           the time.
  BOTH     their conjunction, which is what the loop actually gated on.

Ground truth is the evaluator's own verdict, taken from the per-plan rows that
`iterative_repair.py` writes.

Scoring is reported for detecting an INVALID plan, since that is the decision a
verifier exists to make, and with Matthews correlation alongside accuracy
because the classes are unbalanced. Rule-level recall asks a stricter question:
of the individual evaluator rules a plan violates, how many does some authored
check fire on.

    python3 verifier_fidelity.py \
        --traj ../plan-generation/agentic-runs/traj_test_openrouter_qwen3.8-27b.jsonl \
        --iter ../plan-generation/agentic-runs/iter_qwen3.8-27b_test.jsonl \
        --out  ../plan-generation/agentic-runs/fidelity_qwen3.8-27b_test.jsonl \
        --summary ../plan-generation/agentic-runs/fidelity_qwen3.8-27b_test.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parents[1]          # evaluation/, which holds eval.py etc.
AGENTIC = EVAL.parent / "plan-generation" / "agentic"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(AGENTIC))


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- #
# plan recovery, identical to eval_agentic_iterations so the two rows line up  #
# --------------------------------------------------------------------------- #
def plan_series(traj: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for line in traj.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("n_tool_calls"):
            continue
        series = []
        for st in (r.get("steps") or []):
            for tc in (st.get("tool_calls") or []):
                if not isinstance(tc, dict) or tc.get("name") != "submit_plan":
                    continue
                args = tc.get("args")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        continue
                text = (args or {}).get("travel_plan") or ""
                if text.strip() and (not series or _sha(text) != series[-1]["sha"]):
                    series.append({"index": len(series) + 1, "text": text,
                                   "sha": _sha(text)})
        if series and len(series) > len(out.get(int(r["id"]), [])):
            out[int(r["id"])] = series
    return out


def specs(traj: Path) -> dict[int, dict]:
    out = {}
    for line in traj.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("n_tool_calls"):
            continue
        sp = r["spec_json"] if isinstance(r.get("spec_json"), dict) else \
            json.loads(r.get("spec_json") or "{}")
        if sp.get("constraints"):
            out[int(r["id"])] = sp
    return out


# --------------------------------------------------------------------------- #
# worker: judge one batch of plans with SELF and ENV                          #
# --------------------------------------------------------------------------- #
def _worker(inp: Path, outp: Path) -> None:
    import verifier as V
    import db_tools as T
    import runtime as RT
    sys.path.insert(0, str(EVAL))
    from convert_plans import convert_plan_text

    rows = []
    for job in json.loads(inp.read_text()):
        sp_j, text = job["spec"], job["text"]
        days = convert_plan_text(text, day_base=1)
        spec = V.Spec()
        spec.facts = sp_j.get("facts") or {}
        for c in sp_j["constraints"]:
            spec.constraints.append(V.Constraint(
                id=c["id"], kind=c["kind"], source=c["source"], text=c.get("text", ""),
                pseudocode=c.get("pseudocode", ""), python=c.get("python")))
        # usability is decided on the harness's own pre-flight fixture, not on
        # the plan under test, so a check cannot be admitted by the plan it judges
        try:
            V.preflight(spec, {**spec.ctx(), "items": T.resolve_plan(
                RT._synthetic_days(int(spec.facts.get("days") or 3)))})
        except Exception:
            pass
        runnable = {c.id: c.python for c in spec.constraints if c.usable}
        kind = {c.id: c.kind for c in spec.constraints}
        ctx = {**spec.ctx(), "items": T.resolve_plan(days)}
        fired, errored = [], []
        if runnable and days:
            for v in V.run(runnable, days, ctx, timeout=10):
                if v.ok is False:
                    fired.append(v.id)
                elif v.error:
                    errored.append(v.id)
        env_ok, env_cost = None, None
        if days:
            try:
                ov = RT.evaluate_with_oracle(days, int(spec.facts.get("people") or 1),
                                             budget=job.get("budget"))
                env_ok, env_cost = ov.ok, ov.cost
            except Exception:
                pass
        rows.append({"id": job["id"], "plan_index": job["plan_index"],
                     "n_runnable": len(runnable),
                     # ids that actually executed, so a later analysis can tell a
                     # check that ran and passed from one that never ran at all
                     "self_ran": sorted(runnable),
                     "self_pref_ids": sorted(c.id for c in spec.constraints
                                             if c.kind == "preference"),
                     "self_fired": fired, "self_errored": errored,
                     "self_fired_kinds": sorted({kind.get(x, "?") for x in fired}),
                     "env_ok": env_ok, "env_cost": env_cost})
    outp.write_text("\n".join(json.dumps(r) for r in rows))


# --------------------------------------------------------------------------- #
# classification statistics                                                    #
# --------------------------------------------------------------------------- #
def stats(pred_invalid: list[bool], truth_invalid: list[bool]) -> dict:
    tp = sum(1 for p, t in zip(pred_invalid, truth_invalid) if p and t)
    fp = sum(1 for p, t in zip(pred_invalid, truth_invalid) if p and not t)
    fn = sum(1 for p, t in zip(pred_invalid, truth_invalid) if not p and t)
    tn = sum(1 for p, t in zip(pred_invalid, truth_invalid) if not p and not t)
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else 0.0
    spec_ = tn / (tn + fp) if tn + fp else 0.0
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / n if n else 0.0,
            "precision": prec, "recall": rec, "f1": f1,
            "balanced_acc": (rec + spec_) / 2, "mcc": mcc}


def row(label: str, s: dict) -> str:
    return (f"  {label:<26}{s['n']:>5}{s['tp']:>5}{s['fp']:>5}{s['fn']:>5}{s['tn']:>5}"
            f"{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}"
            f"{s['balanced_acc']:>8.3f}{s['mcc']:>8.3f}")


HEAD = (f"  {'verifier':<26}{'n':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
        f"{'prec':>8}{'recall':>8}{'F1':>8}{'bal.acc':>8}{'MCC':>8}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        _worker(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    ap.add_argument("--traj", required=True)
    ap.add_argument("--iter", required=True)
    ap.add_argument("--dataset", default="UKPLab/PreferTripPlan")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--reuse", action="store_true",
                    help="re-render the summary from an existing --out file "
                         "instead of re-judging every plan")
    a = ap.parse_args()

    if a.reuse:
        recs = [json.loads(x) for x in Path(a.out).read_text().splitlines() if x.strip()]
        print(f"[in] reusing {len(recs)} judged plans from {a.out}")
        summarize(recs, a)
        return

    traj = Path(a.traj)
    series, sp = plan_series(traj), specs(traj)
    truth = {}
    for line in Path(a.iter).open():
        r = json.loads(line)
        truth[(r["id"], r["plan_index"])] = r
    import _paths                                                # noqa: PLC0415
    budget = {int(r["id"]): r.get("budget") for r in _paths.load_split(a.split)}

    jobs = [{"id": rid, "plan_index": p["index"], "text": p["text"],
             "spec": sp[rid], "budget": budget.get(rid)}
            for rid, plans in series.items() if rid in sp for p in plans
            if (rid, p["index"]) in truth]
    print(f"[in] {len(series)} records, {len(jobs)} plans to judge", flush=True)

    judged = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        batches = [jobs[i:i + a.chunk] for i in range(0, len(jobs), a.chunk)]
        for b, batch in enumerate(batches, 1):
            fi, fo = tmp / f"in{b}.json", tmp / f"out{b}.jsonl"
            fi.write_text(json.dumps(batch))
            pr = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 "--_worker", str(fi), str(fo)],
                                capture_output=True, text=True)
            if pr.returncode != 0 or not fo.exists():
                why = (f"signal {-pr.returncode}" if pr.returncode < 0
                       else ((pr.stderr or "").strip().splitlines() or ["?"])[-1][:80])
                print(f"[warn] batch {b}/{len(batches)} failed: {why}", flush=True)
                continue
            judged += [json.loads(x) for x in fo.read_text().splitlines() if x.strip()]
            print(f"  [batch] {b}/{len(batches)}: {len(batch)} plans", flush=True)

    recs = []
    for j in judged:
        t = truth[(j["id"], j["plan_index"])]
        recs.append({**j,
                     "truth_invalid": not t["final_pass"],
                     "truth_invalid_prefs": not t["final_pass_prefs"],
                     "truth_cs_failed": t["commonsense_failed"],
                     "truth_hard_failed": t["hard_failed"],
                     "truth_pref_failed": t["pref_failed"],
                     "self_invalid": bool(j["self_fired"]),
                     "env_invalid": (j["env_ok"] is False)})
    Path(a.out).write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    print(f"[out] {len(recs)} rows -> {a.out}")
    summarize(recs, a)


def summarize(recs, a) -> None:

    L = []
    H = lambda t: (L.append(""), L.append("=" * 96), L.append(t), L.append("=" * 96))
    T_ = [r["truth_invalid"] for r in recs]
    H("1. DETECTING AN INVALID PLAN  (Plan-PR as ground truth, all judged plans)")
    L.append(f"  base rate: {sum(T_)}/{len(T_)} = {sum(T_)/len(T_):.3f} of plans are invalid")
    L.append("")
    L.append(HEAD)
    L.append(row("ENV (supplied)", stats([r["env_invalid"] for r in recs], T_)))
    L.append(row("SELF (authored)", stats([r["self_invalid"] for r in recs], T_)))
    L.append(row("BOTH (the loop's gate)",
                 stats([r["env_invalid"] or r["self_invalid"] for r in recs], T_)))

    Tp = [r["truth_invalid_prefs"] for r in recs]
    H("2. THE SAME, WITH PREFERENCES IN THE GROUND TRUTH (PreferPlan-PR)")
    L.append(f"  base rate: {sum(Tp)}/{len(Tp)} = {sum(Tp)/len(Tp):.3f}")
    L.append("")
    L.append(HEAD)
    L.append(row("ENV (supplied)", stats([r["env_invalid"] for r in recs], Tp)))
    L.append(row("SELF (authored)", stats([r["self_invalid"] for r in recs], Tp)))
    L.append(row("BOTH (the loop's gate)",
                 stats([r["env_invalid"] or r["self_invalid"] for r in recs], Tp)))

    H("3. WHAT SELF ADDS OVER THE SUPPLIED ORACLE")
    only_self = [r for r in recs if r["self_invalid"] and not r["env_invalid"]]
    only_env = [r for r in recs if r["env_invalid"] and not r["self_invalid"]]
    L.append(f"  plans flagged by SELF alone : {len(only_self):>4}, "
             f"of which genuinely invalid {sum(1 for r in only_self if r['truth_invalid']):>4} "
             f"({(sum(1 for r in only_self if r['truth_invalid'])/len(only_self) if only_self else 0):.3f})")
    L.append(f"  plans flagged by ENV alone  : {len(only_env):>4}, "
             f"of which genuinely invalid {sum(1 for r in only_env if r['truth_invalid']):>4} "
             f"({(sum(1 for r in only_env if r['truth_invalid'])/len(only_env) if only_env else 0):.3f})")
    missed = [r for r in recs if r["truth_invalid"] and not r["self_invalid"]
              and not r["env_invalid"]]
    L.append(f"  invalid plans neither caught: {len(missed):>4} "
             f"({len(missed)/max(sum(T_),1):.3f} of all invalid plans)")

    H("4. RULE-LEVEL RECALL  (did any authored check fire on a violated rule?)")
    by_rule = defaultdict(lambda: [0, 0])
    for r in recs:
        for fam, key in (("commonsense", "truth_cs_failed"), ("hard", "truth_hard_failed")):
            for rule in r[key]:
                by_rule[rule][1] += 1
                if r["self_fired"]:
                    by_rule[rule][0] += 1
    L.append(f"  {'evaluator rule violated':<44}{'plans':>7}{'some check fired':>18}{'rate':>8}")
    for rule, (hit, tot) in sorted(by_rule.items(), key=lambda x: -x[1][1]):
        L.append(f"  {rule:<44}{tot:>7}{hit:>18}{hit/tot:>8.3f}")
    L.append("  Firing is not the same as identifying the rule; this is an upper bound on")
    L.append("  rule-level recall, since the check that fired may target something else.")

    H("5. AGREEMENT BY CONSTRAINT FAMILY  (two taxonomies, read with care)")
    L.append("  The family of an authored check is DECLARED BY THE AGENT: the spec stage")
    L.append("  makes every check name itself commonsense/hard/preference and the harness")
    L.append("  validates only that the string is one of the three (verifier.py, parse_spec).")
    L.append("  It is not the evaluator's taxonomy. On this run 41.5% of the 532 checks the")
    L.append("  agent filed as 'hard' test trip-envelope facts (cities, travellers, days)")
    L.append("  that the evaluator scores elsewhere or not at all, and 2.8% mention budget.")
    L.append("  The 'hard' row below therefore measures taxonomy mismatch as much as it")
    L.append("  measures verification failure. Panel 6's 'any check' column is free of this.")
    L.append("")
    L.append(f"  {'family':<22}{'n':>6}{'truth fails':>13}{'self fires':>12}"
             f"{'precision':>11}{'recall':>9}{'MCC':>8}")
    for fam, tkey in (("commonsense", "truth_cs_failed"), ("hard", "truth_hard_failed"),
                      ("preference", "truth_pref_failed")):
        tv = [bool(r[tkey]) for r in recs]
        pv = [fam in r["self_fired_kinds"] for r in recs]
        st = stats(pv, tv)
        L.append(f"  {fam:<22}{st['n']:>6}{sum(tv):>13}{sum(pv):>12}"
                 f"{st['precision']:>11.3f}{st['recall']:>9.3f}{st['mcc']:>8.3f}")

    H("6. PARADIGM-LEVEL DETECTION  (preferences the agent had to verify unaided)")
    STRUCT = {"Conditional", "Compensatory", "Lexicographic", "Temporal", "Scoped"}
    tot, hit_p, hit_a = Counter(), Counter(), Counter()
    for r in recs:
        for par in {x.split(":")[0] for x in r["truth_pref_failed"]}:
            tot[par] += 1
            if "preference" in r["self_fired_kinds"]:
                hit_p[par] += 1
            if r["self_fired"]:
                hit_a[par] += 1
    L.append(f"  {'paradigm':<16}{'class':<12}{'violated':>9}"
             f"{'pref check fired':>18}{'any check':>11}")
    for par, n in sorted(tot.items(), key=lambda x: -x[1]):
        cls = "structural" if par in STRUCT else "flat"
        L.append(f"  {par:<16}{cls:<12}{n:>9}{hit_p[par]/n:>18.3f}{hit_a[par]/n:>11.3f}")
    for lab, keep in (("flat", lambda x: x not in STRUCT),
                      ("structural", lambda x: x in STRUCT)):
        n = sum(v for k, v in tot.items() if keep(k))
        if not n:
            continue
        a_ = sum(v for k, v in hit_p.items() if keep(k))
        b_ = sum(v for k, v in hit_a.items() if keep(k))
        L.append(f"  -> {lab:<13}{'':<12}{n:>9}{a_/n:>18.3f}{b_/n:>11.3f}")
    L.append("  Flat paradigms state a condition over entities; structural ones quantify,")
    L.append("  scope, order or trade off conditions, and carry no counterpart in the")
    L.append("  environment oracle, so the authored check is the only verifier of them.")

    H("7. WHICH VERIFIER PRECEDED EACH REPAIR")
    by_id = defaultdict(dict)
    for r in recs:
        by_id[r["id"]][r["plan_index"]] = r
    tag = Counter()
    fam_fired = Counter()
    for _, d in by_id.items():
        for k in sorted(d)[:-1]:            # every plan but the last was superseded
            e, sf = d[k]["env_invalid"], d[k]["self_invalid"]
            tag["both" if (e and sf) else "env" if e else "self" if sf else "neither"] += 1
            if sf:
                fam_fired.update(d[k]["self_fired_kinds"])
    n_sup = sum(tag.values())
    L.append(f"  superseded plans (each one was repaired into its successor): {n_sup}")
    for k, lab in (("both", "flagged by BOTH (attribution ambiguous)"),
                   ("env", "flagged by ENV only (env-driven)"),
                   ("self", "flagged by SELF only (self-coded-driven)"),
                   ("neither", "flagged by NEITHER (revised for another reason)")):
        L.append(f"    {lab:<46}{tag[k]:>5}{(tag[k]/n_sup if n_sup else 0):>9.3f}")
    L.append(f"  Self-coded checks were the sole trigger on {tag['self']}/{n_sup} repairs and")
    L.append(f"  were firing on {tag['self'] + tag['both']}/{n_sup}. Families they fired on: "
             f"{dict(fam_fired)}.")
    L.append("  A trigger is not the same as a cause: this says which verdict preceded the")
    L.append("  repair, not which evaluator rule the repair went on to fix.")

    rep = "\n".join(L)
    print(rep)
    if a.summary:
        Path(a.summary).write_text(rep + "\n")
        print(f"\n[out] summary -> {a.summary}")


if __name__ == "__main__":
    main()
