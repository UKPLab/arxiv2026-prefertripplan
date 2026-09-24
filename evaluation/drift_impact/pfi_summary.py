#!/usr/bin/env python3
"""pfi_summary.py — cross-model tables for PFI.

PER MODEL, ALWAYS.  Models are the units being benchmarked, and the same
drifted leaves are scored for every model, so pooling would stack
correlated copies and inflate n.  Cross-model consistency is reported by
COUNTING SIGNS, never by averaging observations into one arm.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

MODELS = ["gpt-5.6-terra", "nemotron", "deepseek", "qwen3.8-27b",
          "gemma-4-26b-a4b"]
SPLITS = ["test", "test_large"]
W = 104


def _load(runs: Path, m: str, s: str) -> dict | None:
    fp = runs / f"pfi_{m}_{s}.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def _ag(v):
    v = [x for x in v if x is not None]
    n = len(v)
    if not n:
        return {"n": 0, "mean": None, "se": None, "median": None}
    sd = st.stdev(v) if n > 1 else 0.0
    return {"n": n, "mean": st.mean(v), "median": st.median(v),
            "se": sd / n ** 0.5}


def _f(v, d=3):
    return "  --  " if v is None else f"{v:+.{d}f}"


def _action_of(r: dict) -> str:
    """omission / inversion / mixed for one preference.  Drift mode is
    assigned per RECORD, so every leaf of a preference shares an action
    and `mixed` is expected to be empty -- it is kept so a violation of
    that assumption shows up rather than being silently bucketed."""
    a = set(r.get("actions") or [])
    if a == {"drop"}:
        return "omission"
    if a == {"invert"}:
        return "inversion"
    return "mixed"


def _block(runs: Path, split: str, keep) -> list[str]:
    L = [f"  {'model':<18s}{'prefs':>6s}{'PFI':>9s}{'se':>9s}"
         f"{'median':>9s}{'  |':>3s}"
         f"{'D tier1':>9s}{'n':>5s}{'D tier2':>9s}{'n':>5s}{'  trivial':>10s}"]
    L.append("  " + "-" * 92)
    signs = []
    for mo in MODELS:
        d = _load(runs, mo, split)
        if not d:
            continue
        pr = [r for r in d["preferences"] if keep(r)]
        if not pr:
            continue
        ids = {(r["id"], r["paradigm"], r["bank_id"]) for r in pr}
        lv = [r for r in d["leaves"]
              if (r["id"], r["paradigm"], r["bank_id"]) in ids]
        p = _ag([r["PFI"] for r in pr])
        t1 = _ag([r["D"] for r in lv if r.get("control_tier") == 1])
        t2 = _ag([r["D"] for r in lv if r.get("control_tier") == 2])
        if p["mean"] is not None:
            signs.append((mo, p["mean"]))
        L.append(f"  {mo:<18s}{p['n']:>6d}{_f(p['mean']):>9s}"
                 f"{_f(p['se'], 4):>9s}{_f(p['median']):>9s}{'  |':>3s}"
                 f"{_f(t1['mean']):>9s}{t1['n']:>5d}"
                 f"{_f(t2['mean']):>9s}{t2['n']:>5d}"
                 f"{sum(1 for r in pr if r['trivial']):>10d}")
    if signs:
        pos = sum(1 for _, x in signs if x > 0)
        L.append("")
        L.append(f"  consistency (sign count, NOT a pooled test): "
                 f"{pos}/{len(signs)} models with PFI > 0")
    return L


def headline(runs: Path) -> str:
    L = ["=" * W, "HEADLINE — PFI per model", "=" * W,
         "PFI(p) = mean D(leaf) over the preference's drifted leaves.",
         "  > 0  followed the drifted profile, under-served the query",
         "  = 0  drift had no behavioural effect",
         "  < 0  doubled down on the query",
         "One scale across op, direction and paradigm, via the sign",
         "convention sigma.  D is split by CONTROL TIER because the two",
         "have very different precision:",
         "  tier 1  paired within the leaf (surviving members) -- all",
         "          confounds cancel exactly",
         "  tier 2  matched undrifted instances of the same bank_id/path",
         "          -- covers scalar, single-member and full-coverage",
         "          categorical leaves; noisier",
         "`trivial` counts vacuous preferences.  They are INCLUDED in PFI:",
         "vacuity is itself a drift-induced outcome, so R = 0 records what",
         "the plan contains.  The count is reported so its weight is visible.",
         ""]
    for split in SPLITS:
        L.append(f"  [{split}]   ALL DRIFT")
        L += _block(runs, split, lambda r: True)
        L.append("")
    L += ["=" * W, "HEADLINE — PFI split by drift action", "=" * W,
          "omission  = the profile LINE IS OMITTED (silence in the query",
          "            becomes silence in the profile)",
          "inversion = the profile asserts the OPPOSITE, from the inversion",
          "            table",
          "Both push a compliant planner the same way, so sigma is identical",
          "for the two; any difference here is MAGNITUDE, which is what the",
          "expected `inversion > omission` ordering predicts.", ""]
    for split in SPLITS:
        for act in ("omission", "inversion", "mixed"):
            rows = _block(runs, split, lambda r, a=act: _action_of(r) == a)
            if len(rows) <= 2:
                continue
            L.append(f"  [{split}]   {act.upper()}")
            L += rows
            L.append("")
    return "\n".join(L)


def detail(runs: Path) -> str:
    L = ["=" * W, "DETAIL — per model", "=" * W,
         "Every block is ONE model.  Nothing is averaged across models.", ""]
    for split in SPLITS:
        for mo in MODELS:
            d = _load(runs, mo, split)
            if not d:
                continue
            lv, pr = d["leaves"], d["preferences"]
            L.append(f"  [{split}]  {mo}")
            for lab, key, src in (("D by leaf kind", "kind", lv),
                                  ("D by drift action", "action", lv),
                                  ("D by op", "op", lv),
                                  ("D by control tier", "control_tier", lv)):
                L.append(f"    {lab}")
                L.append(f"      {'bucket':<22s}{'n':>5s}{'mean':>10s}"
                         f"{'median':>10s}{'se':>9s}")
                for k in sorted({str(r.get(key)) for r in src}):
                    a = _ag([r["D"] for r in src if str(r.get(key)) == k])
                    if not a["n"]:
                        continue
                    L.append(f"      {k:<22s}{a['n']:>5d}{_f(a['mean']):>10s}"
                             f"{_f(a['median']):>10s}{_f(a['se'], 4):>9s}")
            L.append("    PFI by paradigm x action")
            L.append(f"      {'paradigm':<26s}{'action':<10s}{'n':>5s}"
                     f"{'mean PFI':>11s}{'se':>9s}{'trivial':>9s}")
            for k in sorted({r["paradigm"] for r in pr}):
                for act in ("omission", "inversion", "mixed"):
                    sel = [r for r in pr if r["paradigm"] == k
                           and _action_of(r) == act]
                    if not sel:
                        continue
                    a = _ag([r["PFI"] for r in sel])
                    L.append(f"      {k:<26s}{act:<10s}{a['n']:>5d}"
                             f"{_f(a['mean']):>11s}{_f(a['se'], 4):>9s}"
                             f"{sum(1 for r in sel if r['trivial']):>9d}")
            L.append(f"    diagnostics: {d['diagnostics']}")
            L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path,
                    default=Path(__file__).parent / "runs")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent
                    / "PFI_approach_and_results.txt")
    args = ap.parse_args()
    txt = headline(args.runs_dir) + "\n" + detail(args.runs_dir)
    args.out.write_text(txt)
    print(f"[out] {args.out}  ({len(txt.splitlines())} lines)")


if __name__ == "__main__":
    main()
