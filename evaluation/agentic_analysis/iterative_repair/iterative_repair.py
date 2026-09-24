#!/usr/bin/env python3
"""Score EVERY plan an agentic run produced, not just the one it delivered.

`eval.py` answers "how good is the final plan". A trajectory also contains the
plans the loop rejected along the way, and the question this answers is whether
the CONSTRUCT -> VERIFY -> REPAIR cycle actually improves a plan: does P2 score
better than P1, does P3 beat P2, and on which metric.

Plans are recovered from the trajectory at
``steps[].tool_calls[].args.travel_plan`` for every call named ``submit_plan``.
Consecutive identical submissions are collapsed (an agent re-submitting an
unchanged plan is one plan, not two); pass --keep-duplicates to score them
separately.

Scoring reuses ``eval.eval_score`` unchanged, once per plan index, so a number
here is directly comparable with the published per-record numbers. Records that
have no k-th plan are simply absent from index k rather than counted as failures.

    python3 iterative_repair.py \
        --traj ../plan-generation/agentic-runs/traj_test_openrouter_qwen3.8-27b.jsonl \
        --out iter_qwen3.8-27b_test.jsonl --summary iter_qwen3.8-27b_test.txt

`--traj` accepts several paths or globs, so sharded runs can be pooled:

    --traj '../plan-generation/agentic-runs/shard*/traj_*.jsonl'
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parents[1]          # evaluation/, which holds eval.py and convert_plans.py
sys.path.insert(0, str(EVAL))

from convert_plans import convert_plan_text                       # noqa: E402

# `eval` pulls in the whole TravelPlanner data layer (flights, accommodations,
# restaurants, attractions, the distance matrix) at import time. The parent
# process only orchestrates, so importing it here would hold a second copy of
# those tables alongside every worker's -- which is what put this over the
# machine's limit. Import it where it is actually used instead.


def hard_denominators(dataset: str, split: str) -> dict[int, int]:
    """{id -> number of hard-constraint slots the QUERY states}.

    eval.py sizes the hard micro rate by what the query asks for -- budget on
    every record plus each non-null local_constraint slot, 526 across `test` --
    not by what the evaluator managed to score. The difference matters: when a
    plan trips the is_not_absent / sandbox gate its hard block comes back None,
    and counting only scored constraints silently drops those records instead
    of failing them. That read 0.882 here against eval.py's 0.753 on the same
    plans. Use the query's slots so the two agree."""
    from eval import load_queries, parse_local_constraint       # noqa: PLC0415
    out: dict[int, int] = {}
    for q in load_queries(dataset, split):
        lc = parse_local_constraint(q.get("local_constraint"))
        out[int(q["id"])] = 1 + sum(1 for v in lc.values() if v is not None)
    return out


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- #
# 1. recover the plan series from the trajectories                            #
# --------------------------------------------------------------------------- #
def plan_series(traj_paths: list[Path], keep_duplicates: bool) -> dict[int, list[dict]]:
    """{record id -> [{index, phase, step, text, sha}, ...]} in emission order.

    A record appearing in more than one trajectory file (a resumed or re-run
    shard) keeps the LONGEST series, on the assumption that the longer one is
    the completed attempt.
    """
    out: dict[int, list[dict]] = {}
    for p in traj_paths:
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                rid = int(r["id"])
                series: list[dict] = []
                for i, step in enumerate(r.get("steps") or []):
                    for tc in (step.get("tool_calls") or []):
                        if not isinstance(tc, dict) or tc.get("name") != "submit_plan":
                            continue
                        args = tc.get("args")
                        if isinstance(args, str):          # some models serialise args as text
                            try:
                                args = json.loads(args)
                            except Exception:
                                continue
                        text = (args or {}).get("travel_plan") or ""
                        if not text.strip():
                            continue
                        if (not keep_duplicates and series
                                and _sha(text) == series[-1]["sha"]):
                            continue                        # re-submitted unchanged
                        series.append({"index": len(series) + 1,
                                       "phase": step.get("phase"),
                                       "step": i, "text": text, "sha": _sha(text)})
                if series and len(series) > len(out.get(rid, [])):
                    out[rid] = series
    return out


def delivered_shas(cache: Path | None) -> dict[int, str]:
    """{id -> sha of the plan actually returned}. Empty when no cache is given."""
    out: dict[int, str] = {}
    if not cache or not cache.exists():
        return out
    for line in cache.open():
        if not line.strip():
            continue
        c = json.loads(line)
        try:
            out[int(c["id"])] = _sha(json.loads(c["content"])["travel_plan"])
        except Exception:
            continue
    return out


def traj_meta(traj_paths: list[Path]) -> dict[int, dict]:
    """Per-record loop telemetry, so outcomes can be read against how the loop ran."""
    out: dict[int, dict] = {}
    for p in traj_paths:
        for line in p.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("n_tool_calls"):
                continue
            seq = [(st.get("phase"), t["name"],
                    bool((t.get("args") if isinstance(t.get("args"), dict) else {}) or {}))
                   for st in (r.get("steps") or [])
                   for t in (st.get("tool_calls") or [])
                   if isinstance(t, dict) and t.get("name")]
            feats = set()
            for st in (r.get("steps") or []):
                for t in (st.get("tool_calls") or []):
                    if not isinstance(t, dict):
                        continue
                    args = t.get("args")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    args = args or {}
                    if t.get("name") == "filter_items" and args.get("sort_by"):
                        feats.add("sort_by")
                    if t.get("name") == "aggregate_items":
                        feats.add("aggregate")
                        if args.get("limit"):
                            feats.add("aggregate_limit")
            out[int(r["id"])] = {
                "seq": [(ph, nm) for ph, nm, _ in seq],
                "tool_feats": sorted(feats),
                "stop_reason": r.get("stop_reason", ""),
                "n_tool_calls": r.get("n_tool_calls", 0),
                "n_revisions": len(r.get("spec_revisions") or []),
                "repair_cycle": bool(r.get("repair_cycle")),
                "reverted_to_best": bool(r.get("reverted_to_best")),
                "verifier_coverage": r.get("verifier_coverage"),
                "verify_repair_rounds": r.get("verify_repair_rounds", 0),
                "wall_s": round(r.get("wall_s", 0.0), 1),
            }
    return out


def direct_baseline(path: Path | None, hdenom: dict[int, int]) -> dict[int, dict]:
    """{id -> the same metric block} for the direct run, or {} if not supplied."""
    if not path or not path.exists():
        return {}
    return {int(r["id"]): metrics(r, hdenom.get(int(r["id"])))
            for r in map(json.loads, path.open())}


# --------------------------------------------------------------------------- #
# 2. score one plan index through the real evaluator                          #
# --------------------------------------------------------------------------- #
def _worker(plan_file: Path, detail: Path, dataset: str, split: str) -> None:
    """Score one chunk and exit. Run as a subprocess so the evaluator's tables
    are freed between chunks -- the whole 225 in one process peaks past what a
    loaded 15 GB machine can give, and the OOM killer takes it with no
    traceback."""
    from eval import eval_score                                  # noqa: PLC0415
    with contextlib.redirect_stdout(io.StringIO()):
        eval_score(dataset, split, plan_file, detailed_out=detail, with_preferences=True)


def score_index(series: dict[int, list[dict]], k: int, dataset: str, split: str,
                tmp: Path, quiet: bool, chunk: int) -> dict[int, dict]:
    """Run eval_score over every record that has a k-th plan. {id -> eval row}."""
    rows = []
    for rid, plans in series.items():
        if len(plans) < k:
            continue
        rows.append({"id": rid, "source_id": None,
                     "plan": convert_plan_text(plans[k - 1]["text"], day_base=1)})
    if not rows:
        return {}
    out: dict[int, dict] = {}
    batches = [rows[i:i + chunk] for i in range(0, len(rows), chunk)] if chunk > 0 else [rows]
    for b, batch in enumerate(batches, 1):
        # Only this batch's ids. eval_score scores every query in the split, so
        # a detail file also contains all the records whose plan was NOT in this
        # batch, marked undelivered. Filtering on the whole index's id set let a
        # later batch overwrite an earlier batch's real row with an all-fail one.
        want = {r["id"] for r in batch}
        pf = tmp / f"plans_idx{k}_b{b}.jsonl"
        pf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in batch))
        detail = tmp / f"eval_idx{k}_b{b}.jsonl"
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_worker",
             str(pf), str(detail), dataset, split],
            capture_output=True, text=True)
        if proc.returncode != 0 or not detail.exists():
            why = (f"killed by signal {-proc.returncode} (out of memory?)"
                   if proc.returncode < 0 else
                   f"exit {proc.returncode}: " +
                   ((proc.stderr or "").strip().splitlines() or ["no output"])[-1][:80])
            print(f"[warn] index {k} batch {b}/{len(batches)} failed -- {why}", flush=True)
            continue
        for r in map(json.loads, detail.open()):
            if int(r["id"]) in want:
                out[int(r["id"])] = r
        if not quiet:
            print(f"  [batch] index {k} {b}/{len(batches)}: {len(batch)} plans", flush=True)
    return out


# --------------------------------------------------------------------------- #
# 3. per-plan metric block                                                    #
# --------------------------------------------------------------------------- #
def metrics(ev: dict, hard_denom: int | None = None) -> dict:
    """Micro counts plus the pass/fail flags, in the same shape used elsewhere."""
    cs, hd = ev.get("commonsense") or {}, ev.get("hard") or {}
    prefs = ev.get("preferences") or []
    cs_p = sum(1 for v in cs.values() if v[0] is True)
    cs_n = sum(1 for v in cs.values() if v[0] is not None)
    hd_p = sum(1 for v in hd.values() if v[0] is True)
    # denominator from the query, not from what got scored -- see hard_denominators
    hd_n = (hard_denom if hard_denom is not None
            else sum(1 for v in hd.values() if v[0] is not None))
    pr_p = sum(1 for p in prefs if p.get("passed") is True)
    pr_n = len(prefs)
    # A "trivial" pass is one the evaluator could not actually test -- a
    # conditional whose condition never fired, a scoped preference with no
    # matching day, an aggregate over an empty set. It counts as passed, but the
    # plan did not earn it, and a repair loop can manufacture one by removing
    # whatever triggered the preference rather than by satisfying it.
    tri = [p for p in prefs if p.get("trivial")]
    tri_p = sum(1 for p in tri if p.get("passed") is True)
    nt = [p for p in prefs if not p.get("trivial")]
    nt_p = sum(1 for p in nt if p.get("passed") is True)
    # `hard` is None when the upstream gate (is_not_absent / sandbox) refused to
    # score it, which is a failure of the plan, not an absence of constraints.
    cs_all = bool(cs) and all(v[0] is not False for v in cs.values())
    hd_all = bool(hd) and all(v[0] is not False for v in hd.values())
    pr_all = all(p.get("passed") is not False for p in prefs)
    final = cs_all and hd_all
    return {
        "delivered": bool(ev.get("delivered")),
        "commonsense_pass": cs_p, "commonsense_total": cs_n,
        "hard_pass": hd_p, "hard_total": hd_n, "hard_scored": bool(hd),
        "pref_pass": pr_p, "pref_total": pr_n,
        "pref_trivial": len(tri), "pref_trivial_pass": tri_p,
        "pref_nontrivial": len(nt), "pref_nontrivial_pass": nt_p,
        # index-aligned with the query's preference list, so the same preference
        # can be followed across a record's plans
        "pref_state": [{"n": p.get("name", "")[:64],
                        "para": p.get("paradigm", ""),
                        "t": bool(p.get("trivial")),
                        "p": p.get("passed")} for p in prefs],
        "commonsense_all": cs_all, "hard_all": hd_all, "pref_all": pr_all,
        "final_pass": final, "final_pass_prefs": final and pr_all,
        "commonsense_failed": sorted(k for k, v in cs.items() if v[0] is False),
        "hard_failed": sorted(k for k, v in hd.items() if v[0] is False),
        "pref_failed": [p.get("name", "")[:70] for p in prefs if p.get("passed") is False],
    }


# --------------------------------------------------------------------------- #
# 4. trends                                                                   #
# --------------------------------------------------------------------------- #
def _rate(rs, num, den):
    a, b = sum(r[num] for r in rs), sum(r[den] for r in rs)
    return a / b if b else 0.0


def _macro(rs, key):
    """eval.py's macro rate: the fraction of RECORDS where every constraint of
    that type passed. Verified to reproduce the published 0.2978 / 0.6267 /
    0.8089 on the direct run."""
    return sum(r[key] for r in rs) / len(rs) if rs else 0.0


def _blk(rs, label, width=34):
    f = lambda n, d: _rate(rs, n, d)
    m = lambda k: _macro(rs, k)
    return (f"  {label:<{width}} n={len(rs):>3} | "
            f"cs {f('commonsense_pass','commonsense_total'):.3f}/{m('commonsense_all'):.3f} "
            f"hard {f('hard_pass','hard_total'):.3f}/{m('hard_all'):.3f} "
            f"pref {f('pref_pass','pref_total'):.3f}/{m('pref_all'):.3f} | "
            f"Final {m('final_pass'):.3f} +pref {m('final_pass_prefs'):.3f}")


_BLKHEAD = ("  {:<34} {:>5} | {:<13} {:<13} {:<13} | {}".format(
    "population", "n", "cs mic/mac", "hard mic/mac", "pref mic/mac", "Final  +pref"))


def summarise(recs, series, meta, direct) -> str:
    L: list[str] = []
    H = lambda t: (L.append(""), L.append("=" * 78), L.append(t), L.append("=" * 78))
    by_id: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        by_id[r["id"]].append(r)
    for v in by_id.values():
        v.sort(key=lambda x: x["plan_index"])
    revised = {i for i, v in by_id.items() if len(v) > 1}
    first = {i: v[0] for i, v in by_id.items()}
    # the DELIVERED plan, not the last submitted -- the loop returns its best
    # candidate, and on this run it reverted to an earlier one 13 times.
    deliv = {i: next((x for x in v if x.get("is_delivered")), v[-1])
             for i, v in by_id.items()}
    last = {i: v[-1] for i, v in by_id.items()}

    H("1. HOW MUCH ITERATION HAPPENED")
    c = Counter(len(v) for v in by_id.values())
    for k in sorted(c):
        L.append(f"  {k} plan(s): {c[k]:>4} records")
    L.append(f"  total plans scored: {sum(len(v) for v in by_id.values())}")
    nd = sum(1 for i in by_id if deliv[i] is not last[i])
    L.append(f"  delivered plan is NOT the last submitted: {nd} record(s) "
             f"-- the loop reverted to an earlier, better-scoring candidate")

    H("2. DID ITERATION HELP?  first plan vs DELIVERED plan, same records")
    ids = sorted(revised)
    if ids:
        L.append(f"  population: the {len(ids)} records that produced more than one plan")
        L.append("")
        L.append(f"  {'metric':<24}{'first':>9}{'delivered':>11}{'delta':>9}")
        for name, num, den in (("commonsense micro", "commonsense_pass", "commonsense_total"),
                               ("hard micro", "hard_pass", "hard_total"),
                               ("preferences micro", "pref_pass", "pref_total")):
            a = _rate([first[i] for i in ids], num, den)
            b = _rate([deliv[i] for i in ids], num, den)
            L.append(f"  {name:<24}{a:>9.3f}{b:>11.3f}{b-a:>+9.3f}")
        for name, key in (("commonsense macro", "commonsense_all"),
                          ("hard macro", "hard_all"),
                          ("preferences macro", "pref_all")):
            a = _macro([first[i] for i in ids], key)
            b = _macro([deliv[i] for i in ids], key)
            L.append(f"  {name:<24}{a:>9.3f}{b:>11.3f}{b-a:>+9.3f}")
        for name, key in (("Final Pass", "final_pass"), ("Final Pass + prefs", "final_pass_prefs")):
            a = sum(first[i][key] for i in ids) / len(ids)
            b = sum(deliv[i][key] for i in ids) / len(ids)
            L.append(f"  {name:<24}{a:>9.3f}{b:>11.3f}{b-a:>+9.3f}")
        L.append("")
        L.append("  crossing the pass threshold (a record either fully passes or does not):")
        for name, key in (("Final Pass", "final_pass"), ("Final Pass + prefs", "final_pass_prefs"),
                          ("all commonsense", "commonsense_all"), ("all hard", "hard_all"),
                          ("all preferences", "pref_all")):
            g = sum(1 for i in ids if not first[i][key] and deliv[i][key])
            w = sum(1 for i in ids if first[i][key] and not deliv[i][key])
            L.append(f"    {name:<22} improved {g:>3}   worsened {w:>3}   unchanged {len(ids)-g-w:>3}")
        L.append("")
        L.append("  individual checks gained or lost (records, not ratios):")
        for name, num in (("commonsense", "commonsense_pass"), ("hard", "hard_pass"),
                          ("preferences", "pref_pass")):
            g = sum(1 for i in ids if deliv[i][num] > first[i][num])
            w = sum(1 for i in ids if deliv[i][num] < first[i][num])
            L.append(f"    {name:<22} more passed {g:>3}   fewer {w:>3}   same {len(ids)-g-w:>3}")

    H("3. WHICH CHECKS THE REPAIR LOOP ACTUALLY FIXED")
    fixed, broke = Counter(), Counter()
    for i in ids:
        for k in ("commonsense_failed", "hard_failed"):
            a, b = set(first[i][k]), set(deliv[i][k])
            for x in a - b:
                fixed[x] += 1
            for x in b - a:
                broke[x] += 1
        if first[i]["hard_scored"] and not deliv[i]["hard_scored"]:
            broke["<hard became unscorable>"] += 1
        if not first[i]["hard_scored"] and deliv[i]["hard_scored"]:
            fixed["<hard became scorable>"] += 1
        a, b = set(first[i]["pref_failed"]), set(deliv[i]["pref_failed"])
        for x in a - b:
            fixed["pref: " + x.split(":")[0]] += 1
        for x in b - a:
            broke["pref: " + x.split(":")[0]] += 1
    L.append("  fixed between the first plan and the delivered one:")
    for k, v in fixed.most_common(12):
        L.append(f"    {v:>4}  {k}")
    L.append("  newly broken:")
    for k, v in (broke.most_common(8) or [(None, 0)]):
        L.append(f"    {v:>4}  {k}" if k else "     none")

    H("4. WHERE THE GAIN COMES FROM  (which step in the series)")
    steps = Counter()
    for i in ids:
        v = by_id[i]
        for a, b in zip(v, v[1:]):
            if not a["final_pass"] and b["final_pass"]:
                steps[f"P{a['plan_index']} -> P{b['plan_index']}"] += 1
    L.append("  transitions that turned a failing plan into a Final Pass:")
    for k, v in sorted(steps.items()):
        L.append(f"    {v:>4}  {k}")
    L.append(f"    (records reaching Final Pass at the delivered plan: "
             f"{sum(deliv[i]['final_pass'] for i in ids)}/{len(ids)})")

    H("5. SELECTION EFFECT  who needs iteration, and do they catch up?")
    L.append(_BLKHEAD)
    never = [first[i] for i in by_id if i not in revised]
    L.append(_blk(never, "never revised (single plan)"))
    if ids:
        L.append(_blk([first[i] for i in ids], "revised: their FIRST plan"))
        L.append(_blk([deliv[i] for i in ids], "revised: their DELIVERED plan"))
    L.append(_blk([deliv[i] for i in by_id], "ALL delivered (what eval.py reports)"))

    if direct:
        H("6. AGENTIC vs DIRECT, per record")
        common = [i for i in by_id if i in direct]
        L.append(_BLKHEAD)
        L.append(_blk([direct[i] for i in common], "direct"))
        L.append(_blk([deliv[i] for i in common], "agentic (delivered)"))
        L.append("")
        for name, key in (("Final Pass", "final_pass"), ("Final Pass + prefs", "final_pass_prefs")):
            g = sum(1 for i in common if not direct[i][key] and deliv[i][key])
            w = sum(1 for i in common if direct[i][key] and not deliv[i][key])
            L.append(f"  {name:<20} agentic gains {g:>3}   loses {w:>3}   net {g-w:>+4}")
        lost = [i for i in common if direct[i]["final_pass"] and not deliv[i]["final_pass"]]
        if lost:
            never_passed = sum(1 for i in lost if not first[i]["final_pass"])
            L.append(f"  of the {len(lost)} agentic loses, {never_passed} never passed at ANY "
                     f"iteration (construction, not repair)")
            cc = Counter(x for i in lost for x in deliv[i]["commonsense_failed"] + deliv[i]["hard_failed"])
            for k, v in cc.most_common(6):
                L.append(f"     {v:>3}  {k}")

    if meta:
        H("7. HOW THE LOOP RAN  (trajectory telemetry vs outcome)")
        rows = [(i, meta[i], deliv[i]) for i in by_id if i in meta]
        L.append(f"  {'stop_reason':<34}{'n':>5}{'Final':>8}{'+pref':>8}{'plans':>7}{'tools':>7}")
        for sr in sorted({m["stop_reason"][:34] for _, m, _ in rows}):
            g = [(m, d) for _, m, d in rows if m["stop_reason"][:34] == sr]
            L.append(f"  {sr:<34}{len(g):>5}"
                     f"{sum(d['final_pass'] for _, d in g)/len(g):>8.3f}"
                     f"{sum(d['final_pass_prefs'] for _, d in g)/len(g):>8.3f}"
                     f"{sum(d['n_plans'] for _, d in g)/len(g):>7.2f}"
                     f"{sum(m['n_tool_calls'] for m, _ in g)/len(g):>7.1f}")
        L.append("")
        L.append(f"  {'revisions used':<34}{'n':>5}{'Final':>8}{'+pref':>8}{'plans':>7}")
        for nrev in sorted({m["n_revisions"] for _, m, _ in rows}):
            g = [(m, d) for _, m, d in rows if m["n_revisions"] == nrev]
            L.append(f"  {nrev:<34}{len(g):>5}"
                     f"{sum(d['final_pass'] for _, d in g)/len(g):>8.3f}"
                     f"{sum(d['final_pass_prefs'] for _, d in g)/len(g):>8.3f}"
                     f"{sum(d['n_plans'] for _, d in g)/len(g):>7.2f}")
        L.append("")
        cov = [(m["verifier_coverage"], d) for _, m, d in rows if m["verifier_coverage"] is not None]
        full = [d for c, d in cov if c == 1.0]
        part = [d for c, d in cov if c < 1.0]
        L.append(f"  verifier coverage == 1.00 : n={len(full):>3}  "
                 f"Final {sum(d['final_pass'] for d in full)/max(len(full),1):.3f}")
        L.append(f"  verifier coverage <  1.00 : n={len(part):>3}  "
                 f"Final {sum(d['final_pass'] for d in part)/max(len(part),1):.3f}")

    H("8. FIXED POPULATION: all records at each round (carry-forward)")
    L.append("  Every record is present in every row. A record that stopped at plan k")
    L.append("  keeps plan k from then on, which is what the loop would have returned")
    L.append("  had it been asked to stop there. So the trend IS comparable.")
    L.append("")
    maxk = max((len(v) for v in by_id.values()), default=0)
    L.append(f"  {'round':>5}{'n':>5}{'upd':>5} |{'cs micro':>10}{'cs macro':>10}"
             f"{'hd micro':>10}{'hd macro':>10}{'pf micro':>10}{'pf macro':>10} |"
             f"{'Final':>8}{'+pref':>8}")
    prev_state = None
    for k in range(1, maxk + 1):
        state = {i: v[min(k, len(v)) - 1] for i, v in by_id.items()}
        rs = list(state.values())
        upd = 0 if prev_state is None else sum(
            1 for i in state if state[i]["sha"] != prev_state[i]["sha"])
        L.append(f"  {k:>5}{len(rs):>5}{upd:>5} |"
                 f"{_rate(rs,'commonsense_pass','commonsense_total'):>10.3f}"
                 f"{_macro(rs,'commonsense_all'):>10.3f}"
                 f"{_rate(rs,'hard_pass','hard_total'):>10.3f}{_macro(rs,'hard_all'):>10.3f}"
                 f"{_rate(rs,'pref_pass','pref_total'):>10.3f}{_macro(rs,'pref_all'):>10.3f} |"
                 f"{_macro(rs,'final_pass'):>8.3f}{_macro(rs,'final_pass_prefs'):>8.3f}")
        prev_state = state
    rs = [deliv[i] for i in by_id]
    nrev = sum(1 for i in by_id if deliv[i]["sha"] != prev_state[i]["sha"])
    L.append(f"  {'deliv':>5}{len(rs):>5}{nrev:>5} |"
             f"{_rate(rs,'commonsense_pass','commonsense_total'):>10.3f}"
             f"{_macro(rs,'commonsense_all'):>10.3f}"
             f"{_rate(rs,'hard_pass','hard_total'):>10.3f}{_macro(rs,'hard_all'):>10.3f}"
             f"{_rate(rs,'pref_pass','pref_total'):>10.3f}{_macro(rs,'pref_all'):>10.3f} |"
             f"{_macro(rs,'final_pass'):>8.3f}{_macro(rs,'final_pass_prefs'):>8.3f}")
    L.append("  ('deliv' = what the loop actually returned, which reverts to an earlier")
    L.append("   candidate where a later one scored worse)")

    H("9. TRIVIAL SATISFACTION  (passes the plan did not earn)")
    L.append("  A preference is `trivial` when the evaluator had nothing to test: a")
    L.append("  conditional whose condition never fired, a scoped preference with no")
    L.append("  matching day, an aggregate over an empty set. It scores as passed. The")
    L.append("  risk in a repair loop is that a plan stops failing a preference by")
    L.append("  removing what triggered it rather than by satisfying it.")
    L.append("")
    L.append(f"  {'population':<34}{'prefs':>7}{'trivial':>9}{'triv%':>8}"
             f"{'trivPass':>10}{'nonTrivPass%':>14}")
    def trow(rs, lbl):
        n = sum(r["pref_total"] for r in rs)
        t = sum(r["pref_trivial"] for r in rs)
        tp = sum(r["pref_trivial_pass"] for r in rs)
        ntn = sum(r["pref_nontrivial"] for r in rs)
        ntp = sum(r["pref_nontrivial_pass"] for r in rs)
        L.append(f"  {lbl:<34}{n:>7}{t:>9}{(t/n if n else 0):>8.1%}{tp:>10}"
                 f"{(ntp/ntn if ntn else 0):>14.1%}")
    if direct:
        trow([direct[i] for i in by_id if i in direct], "direct (baseline)")
    trow([first[i] for i in by_id], "agentic, every FIRST plan")
    trow([deliv[i] for i in by_id], "agentic, every DELIVERED plan")
    if ids:
        trow([first[i] for i in ids], "  of the revised: FIRST")
        trow([deliv[i] for i in ids], "  of the revised: DELIVERED")

    L.append("")
    L.append("  carry-forward by round, all records (does triviality creep up?):")
    maxk2 = max((len(v) for v in by_id.values()), default=0)
    L.append(f"  {'round':>6}{'prefs':>7}{'trivial':>9}{'triv%':>8}{'nonTrivPass%':>14}")
    for k in range(1, maxk2 + 1):
        rs = [v[min(k, len(v)) - 1] for v in by_id.values()]
        n = sum(r["pref_total"] for r in rs); t = sum(r["pref_trivial"] for r in rs)
        ntn = sum(r["pref_nontrivial"] for r in rs); ntp = sum(r["pref_nontrivial_pass"] for r in rs)
        L.append(f"  {k:>6}{n:>7}{t:>9}{(t/n if n else 0):>8.1%}{(ntp/ntn if ntn else 0):>14.1%}")

    L.append("")
    L.append("  what happened to each preference between the first plan and the delivered")
    L.append("  one, on the records that revised (transitions, index-aligned):")
    trans = Counter(); gaming = []
    for i in ids:
        for a, b in zip(first[i]["pref_state"], deliv[i]["pref_state"]):
            sa = ("trivial" if a["t"] else "real") + ("-pass" if a["p"] else "-fail")
            sb = ("trivial" if b["t"] else "real") + ("-pass" if b["p"] else "-fail")
            if sa != sb:
                trans[f"{sa:<13} -> {sb}"] += 1
                if not a["t"] and a["p"] is False and b["t"] and b["p"] is True:
                    gaming.append((i, b["para"], b["n"]))
    if trans:
        for k, v in trans.most_common():
            L.append(f"    {v:>4}  {k}")
    else:
        L.append("    (no preference changed state)")
    L.append("")
    if gaming:
        L.append(f"  {len(gaming)} preference(s) went from a REAL FAILURE to a TRIVIAL PASS --")
        L.append("  the plan stopped being testable on them rather than satisfying them:")
        for i, para, nm in gaming:
            L.append(f"    id {i:<5} {para:<24} {nm[:44]}")
    else:
        L.append("  no preference went from a real failure to a trivial pass: the loop did")
        L.append("  not buy any preference by making it untestable.")

    if meta and any(m.get("seq") for m in meta.values()):
        H("10. TOOL USE: what was called, in what order, and did it pay")
        seqs = {i: m["seq"] for i, m in meta.items() if m.get("seq")}
        allc = Counter(n for v in seqs.values() for _, n in v)
        users = Counter(n for v in seqs.values() for n in {x[1] for x in v})
        L.append(f"  {len(seqs)} sequences, {sum(allc.values())} calls, "
                 f"median length {sorted(len(v) for v in seqs.values())[len(seqs)//2]}")
        L.append("")
        L.append(f"  {'tool':<24}{'calls':>7}{'records':>9}{'per rec':>9}{'med pos':>9}")
        pos = defaultdict(list)
        for v in seqs.values():
            den = max(len(v) - 1, 1)
            for j, (_, n) in enumerate(v):
                pos[n].append(j / den)
        med = lambda xs: sorted(xs)[len(xs) // 2]
        for n, c in sorted(allc.items(), key=lambda x: med(pos[x[0]])):
            L.append(f"  {n:<24}{c:>7}{users[n]:>9}{c/max(users[n],1):>9.1f}"
                     f"{med(pos[n]):>9.2f}")
        L.append("  ('med pos' is the median position in the episode, 0 = first call,")
        L.append("   1 = last -- it reads as the pipeline the agent actually follows)")

        L.append("")
        L.append("  ORDER the prompt suggests, against what happens:")
        first = Counter(v[0][1] for v in seqs.values())
        top, topn = first.most_common(1)[0]
        L.append(f"    first call is {top} on {topn}/{len(seqs)} records "
                 f"({topn/len(seqs):.0%}) -- prompt step 1 is get_trip_dates")
        # step 3 (per-city entities) vs step 4 (per-leg transport)
        ENT = {"search_attractions", "search_restaurants", "search_accommodations"}
        LEG = {"search_flights", "get_ground_transport"}
        inv = same = 0
        for v in seqs.values():
            e = [j for j, (_, n) in enumerate(v) if n in ENT]
            g = [j for j, (_, n) in enumerate(v) if n in LEG]
            if not e or not g:
                continue
            if med(g) < med(e):
                inv += 1
            else:
                same += 1
        L.append(f"    prompt orders per-city entities (step 3) BEFORE per-leg transport")
        L.append(f"    (step 4); observed transport-first on {inv}/{inv+same} records.")
        L.append(f"    Pricing legs first is how you choose cities when the destination is")
        L.append(f"    a state, so this is a sensible inversion rather than drift.")

        L.append("")
        L.append("  most common consecutive pairs:")
        big = Counter()
        for v in seqs.values():
            for a, b in zip(v, v[1:]):
                big[(a[1], b[1])] += 1
        for (a, b), n in big.most_common(8):
            tag = "   (a run of the same tool)" if a == b else ""
            L.append(f"    {n:>5}  {a} -> {b}{tag}")

        L.append("")
        L.append("  RANKING tools -- adopted, and did they pay?")
        for k, lbl in (("sort_by", "filter_items with sort_by"),
                       ("aggregate", "aggregate_items"),
                       ("aggregate_limit", "aggregate_items with limit (the k-best form)")):
            n = sum(1 for m in meta.values() if k in (m.get("tool_feats") or []))
            L.append(f"    {lbl:<44}{n:>4} / {len(meta)} records")
        if direct:
            RANK = {"NumericPreference", "CompensatoryPreference", "LexicographicPreference"}
            THRESH = {"AtomicPreference", "CompositePreference", "ScopedPreference"}
            STRUCT = {"TemporalPreference", "ConditionalPreference"}
            L.append("")
            L.append("  preference outcome by paradigm shape, split on whether the record")
            L.append("  sorted its pools (difference-in-differences against direct):")
            L.append(f"    {'shape':<20}{'sorted?':<10}{'n':>5}{'direct':>9}{'agentic':>9}{'delta':>9}")
            for lbl, P in (("ranking-shaped", RANK), ("threshold-shaped", THRESH),
                           ("structural", STRUCT)):
                for tag, want in (("yes", True), ("no", False)):
                    d = a = n = 0
                    for i in by_id:
                        if i not in direct or i not in meta:
                            continue
                        if (("sort_by" in (meta[i].get("tool_feats") or [])) != want):
                            continue
                        for x, y in zip(direct[i].get("pref_state") or [],
                                        deliv[i].get("pref_state") or []):
                            if y["para"] in P:
                                n += 1
                                d += x["p"] is True
                                a += y["p"] is True
                    if n:
                        L.append(f"    {lbl:<20}{tag:<10}{n:>5}{d/n:>9.3f}{a/n:>9.3f}"
                                 f"{(a-d)/n:>+9.3f}")
            L.append("")
            L.append("  ranking-shaped = Numeric / Compensatory / Lexicographic, the paradigms")
            L.append("  preferences.py defines by comparison against alternatives; those are")
            L.append("  what sorting a pool is supposed to serve.")

        L.append("")
        L.append("  does the loop gather NEW information when repairing?")
        rep = Counter(n for v in seqs.values() for ph, n in v
                      if ph == "repair" and n not in ("submit_plan", "revise_checks"))
        nrec = sum(1 for v in seqs.values()
                   if any(ph == "repair" and n not in ("submit_plan", "revise_checks")
                          for ph, n in v))
        L.append(f"    {nrec}/{len(seqs)} records make any database call during repair")
        if rep:
            L.append(f"    {sum(rep.values())} such calls in total: "
                     + ", ".join(f"{n}x {k}" for k, n in rep.most_common(4)))
        L.append("    Repair is dominated by revise_checks and re-submission, so the loop")
        L.append("    mostly re-reasons over what it already retrieved.")

    if meta:
        H("11. WHY A RECORD STOPPED AT ONE PLAN")
        singles = [i for i, v in by_id.items() if len(v) == 1 and i in meta]
        L.append(f"  {len(singles)} of {len(by_id)} records produced only one distinct plan.")
        L.append("")
        buckets = Counter()
        for i in singles:
            m = meta[i]
            sr, rnd, nrev = m["stop_reason"][:24], m["verify_repair_rounds"], m["n_revisions"]
            if sr == "verified" and rnd <= 1:
                buckets["verified first try -- nothing needed fixing"] += 1
            elif sr == "verified":
                buckets["verified after revising its CHECKS, plan untouched"] += 1
            elif sr == "repair_cycle":
                buckets["repair_cycle -- had nothing better to submit"] += 1
            else:
                buckets[sr] += 1
        for k, v in buckets.most_common():
            L.append(f"    {v:>4}  {k}")
        L.append("")
        L.append("  the second bucket is the revise mechanism doing the work: the loop")
        L.append("  reported a failure, the agent judged its own check wrong rather than")
        L.append("  the plan, and the unchanged plan then passed.")

    return "\n".join(L)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        _worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5])
        return
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", nargs="+", required=True,
                    help="trajectory jsonl path(s) or glob(s)")
    ap.add_argument("--dataset", default="UKPLab/PreferTripPlan")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, help="per (id, plan_index) JSONL")
    ap.add_argument("--summary", default=None, help="trend report destination")
    ap.add_argument("--cache", default=None,
                    help="plans_*_cache_*.jsonl -- identifies which plan was DELIVERED. "
                         "The loop returns its best-scoring candidate, which is not "
                         "always the last one submitted.")
    ap.add_argument("--direct", default=None,
                    help="eval_*.jsonl from the direct run, to attach a per-record baseline")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="score a re-submitted identical plan as its own index")
    ap.add_argument("--chunk", type=int, default=60,
                    help="records per subprocess; smaller uses less peak memory "
                         "(0 = one process for everything)")
    ap.add_argument("--verbose", action="store_true", help="let eval.py print")
    a = ap.parse_args()

    paths = sorted({Path(p) for pat in a.traj for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"[error] no trajectory files matched {a.traj}")
    print(f"[in] {len(paths)} trajectory file(s)")

    series = plan_series(paths, a.keep_duplicates)
    dshas  = delivered_shas(Path(a.cache) if a.cache else None)
    meta   = traj_meta(paths)
    hdenom = hard_denominators(a.dataset, a.split)   # imports eval, then releases
    direct = direct_baseline(Path(a.direct) if a.direct else None, hdenom)
    if a.cache and not dshas:
        print(f"[warn] no delivered plans read from {a.cache}; falling back to last-submitted")
    total = sum(len(v) for v in series.values())
    maxk = max((len(v) for v in series.values()), default=0)
    print(f"[in] {len(series)} records with >=1 plan, {total} plans, longest series {maxk}")

    recs: list[dict] = []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".partial.jsonl")
    part.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for k in range(1, maxk + 1):
            evs = score_index(series, k, a.dataset, a.split, tmp,
                              quiet=not a.verbose, chunk=a.chunk)
            print(f"[eval] plan index {k}: {len(evs)} records scored", flush=True)
            n_before = len(recs)
            for rid, ev in evs.items():
                p = series[rid][k - 1]
                dsha = dshas.get(rid)
                recs.append({"id": rid, "plan_index": k, "n_plans": len(series[rid]),
                             "is_final": k == len(series[rid]),
                             # the loop returns its best candidate, which is not
                             # always the last one written -- this is the one graded
                             "is_delivered": (p["sha"] == dsha) if dsha
                                             else (k == len(series[rid])),
                             "phase": p["phase"], "step": p["step"], "sha": p["sha"],
                             "level": ev.get("level"), "days": ev.get("days"),
                             **metrics(ev, hdenom.get(rid)), **{f"traj_{kk}": vv
                                               for kk, vv in (meta.get(rid) or {}).items()}})

            # Bank each index as it completes. Scoring 225 records alongside the
            # evaluator's own tables peaks around 1.3 GB, and on a loaded machine
            # the OOM killer takes the process with no traceback -- losing every
            # index already computed. Partial results survive that.
            with part.open("a") as pf:
                for r in recs[n_before:]:
                    pf.write(json.dumps(r, ensure_ascii=False) + "\n")

    recs.sort(key=lambda r: (r["id"], r["plan_index"]))
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    part.unlink(missing_ok=True)
    print(f"[out] {len(recs)} rows -> {out}")

    report = summarise(recs, series, meta, direct)
    print("\n" + report)
    if a.summary:
        Path(a.summary).write_text(report + "\n")
        print(f"\n[out] summary -> {a.summary}")


if __name__ == "__main__":
    main()
