#!/usr/bin/env python3
"""tier_completion_summary.py — cross-model tables for the F4
tier-completion (branch-shortcutting) analysis.

Reads the per-model JSON from ``tier_completion.py`` and writes one
self-contained file: the two contrasts per model and split, the method,
then appendices.  Levels are printed with every difference, so a 0.000 is
always readable as a ceiling or as equal means.  Regenerate after any
re-run so prose and numbers cannot drift apart.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MODELS = ["gpt-5.6-terra", "nemotron", "deepseek", "qwen3.8-27b",
          "gemma-4-26b-a4b"]
SPLITS = ["test", "test_large"]
PAIRING_ORDER = ("any->all", "any->any", "all->any", "all->all")
VARIANT_B = ("any->all", "all->any")
W = 104


def _load(runs: Path, model: str, split: str) -> list[dict] | None:
    fp = runs / f"tier_{model}_{split}.json"
    return json.loads(fp.read_text())["rows"] if fp.exists() else None


def _m(v):
    return sum(v) / len(v) if v else None


def _f3(v):
    return " --  " if v is None else f"{v:.3f}"


def _fd(v):
    return "  --  " if v is None else f"{v:+.3f}"


def headline(runs: Path) -> str:
    """One ROW PER MODEL for each level, with the pairing / scope as COLUMNS.

    Collapsing a level to a single averaged number was tried and dropped:
    averaging the two Level-2 scope differences turned gpt's +0.055 [any]
    and +0.026 [all] into one +0.040, which reads as a third, different
    result rather than a summary of the two.  Keeping pairing and scope as
    columns gives one row per model AND leaves every number traceable to
    the detail sections.

    Level 2 uses VARIANT A (all pairings), so any->any contributes to both
    the [any]-as-t1 and [any]-as-t2 groups.

    Base is `scorable` -- both bases appear in the detail sections below.
    """
    L = ["=" * W, "HEADLINE — one row per model", "=" * W,
         "Base: SCORABLE instances (both tiers jointly attainable from the",
         "shown pool).  The `all instances` base is in the sections below.",
         "g = graded (details.tiers[i].score);  b = binary (.passed);",
         "identical for [any], whose score is float(passed).", ""]
    for split in SPLITS:
        L.append("-" * W)
        L.append(f"LEVEL 1 — within pairing, t1 - t2      [{split}]")
        L.append("-" * W)
        counts = _pairing_counts(runs, split)
        L.append("  n per pairing: " + "   ".join(
            f"{pr}={counts.get(pr, 0)}" for pr in PAIRING_ORDER))
        for kind, k1, k2 in (("graded", "graded_1", "graded_2"),
                             ("binary", "binary_1", "binary_2")):
            L.append(f"  {kind}")
            L.append(f"  {'model':<18s}" + "".join(
                f"{pr:^19s}" for pr in PAIRING_ORDER))
            L.append(f"  {'':<18s}" + "".join(
                f"{'t1':>6s}{'t2':>6s}{'diff':>7s}" for _ in PAIRING_ORDER))
            L.append("  " + "-" * (18 + 19 * len(PAIRING_ORDER)))
            acc: list = []
            for mo in MODELS:
                rows = _load(runs, mo, split)
                if rows is None:
                    continue
                sel = [r for r in rows if r["scorable"]]
                acc.extend(sel)
                L.append("  " + _l1_row(mo, sel, k1, k2))
            if acc:
                L.append("  " + "-" * (18 + 19 * len(PAIRING_ORDER)))
                L.append("  " + _l1_row("POOLED", acc, k1, k2))
        L.append("")
        L.append("-" * W)
        L.append(f"LEVEL 2 — within scope, position varied, variant A      [{split}]")
        L.append("-" * W)
        L.append(f"  {'model':<16s}"
                 f"{'[any] t1':>9s}{'t2':>7s}{'diff g':>8s}{'diff b':>8s}"
                 f"{'  |':>3s}"
                 f"{'[all] t1':>9s}{'t2':>7s}{'diff g':>8s}{'diff b':>8s}"
                 f"{'n any':>10s}{'n all':>9s}")
        L.append("  " + "-" * 102)
        acc = []
        for mo in MODELS:
            rows = _load(runs, mo, split)
            if rows is None:
                continue
            sel = [r for r in rows if r["scorable"]]
            acc.extend(sel)
            L.append("  " + _l2_row(mo, sel))
        if acc:
            L.append("  " + "-" * 102)
            L.append("  " + _l2_row("POOLED", acc))
        L.append("")
    return "\n".join(L)


def _pairing_counts(runs: Path, split: str) -> dict:
    """Scorable n per pairing.  Model-independent, so the first model that
    has a run file answers for all of them."""
    for mo in MODELS:
        rows = _load(runs, mo, split)
        if rows:
            return Counter(r["pairing"] for r in rows if r["scorable"])
    return {}


def _l1_row(label: str, sel: list[dict], k1: str, k2: str) -> str:
    cells = ""
    for pr in PAIRING_ORDER:
        S = [r for r in sel if r["pairing"] == pr]
        if not S:
            cells += f"{'--':>6s}{'--':>6s}{'--':>7s}"
            continue
        a, b = _m([r[k1] for r in S]), _m([r[k2] for r in S])
        cells += f"{_f3(a):>6s}{_f3(b):>6s}{_fd(a-b):>7s}"
    return f"{label:<18s}" + cells


def _l2_row(label: str, sel: list[dict]) -> str:
    out = f"{label:<16s}"
    sizes = []
    for scope in ("any", "all"):
        g1 = [r["graded_1"] for r in sel if r["scope_1"] == scope]
        g2 = [r["graded_2"] for r in sel if r["scope_2"] == scope]
        b1 = [r["binary_1"] for r in sel if r["scope_1"] == scope]
        b2 = [r["binary_2"] for r in sel if r["scope_2"] == scope]
        dg = (_m(g1) - _m(g2)) if g1 and g2 else None
        db = (_m(b1) - _m(b2)) if b1 and b2 else None
        out += (f"{_f3(_m(g1)):>9s}{_f3(_m(g2)):>7s}"
                f"{_fd(dg):>8s}{_fd(db):>8s}")
        if scope == "any":
            out += f"{'  |':>3s}"
        sizes.append(f"{len(g1)}/{len(g2)}")
    return out + f"{sizes[0]:>10s}{sizes[1]:>9s}"


def level1(runs: Path) -> str:
    L = ["=" * W, "LEVEL 1 — WITHIN PAIRING:  t1 - t2", "=" * W,
         "Positive = tier 1 scores higher than tier 2.  g = graded",
         "(details.tiers[i].score), b = binary (details.tiers[i].passed).",
         "",
         "This contrast cannot separate POSITION from SCOPE on its own: the",
         "any->all and all->any pairings put the hard scope in opposite tiers.",
         "Level 2 does that separation.", ""]
    for split in SPLITS:
        for base, keep in (("scorable", lambda r: r["scorable"]),
                           ("all instances", lambda r: True)):
            L.append(f"  [{split}]  base: {base}")
            L.append(f"  {'model':<18s}{'pairing':<10s}{'n':>4s}"
                     f"{'t1 g':>7s}{'t2 g':>7s}{'diff g':>8s}"
                     f"{'   ':>3s}{'t1 b':>7s}{'t2 b':>7s}{'diff b':>8s}")
            L.append("  " + "-" * 82)
            pooled: dict = {}
            for mo in MODELS:
                rows = _load(runs, mo, split)
                if rows is None:
                    continue
                sel = [r for r in rows if keep(r)]
                for pr in PAIRING_ORDER:
                    S = [r for r in sel if r["pairing"] == pr]
                    if not S:
                        L.append(f"  {mo:<18s}{pr:<10s}{0:>4d}"
                                 + f"{' --  ':>7s}" * 2 + f"{'  --  ':>8s}"
                                 + f"{'   ':>3s}" + f"{' --  ':>7s}" * 2
                                 + f"{'  --  ':>8s}")
                        continue
                    g1, g2 = (_m([r["graded_1"] for r in S]),
                              _m([r["graded_2"] for r in S]))
                    b1, b2 = (_m([r["binary_1"] for r in S]),
                              _m([r["binary_2"] for r in S]))
                    pooled.setdefault(pr, []).extend(S)
                    L.append(f"  {mo:<18s}{pr:<10s}{len(S):>4d}"
                             f"{_f3(g1):>7s}{_f3(g2):>7s}{_fd(g1-g2):>8s}"
                             f"{'   ':>3s}{_f3(b1):>7s}{_f3(b2):>7s}"
                             f"{_fd(b1-b2):>8s}")
            L.append("  " + "-" * 82)
            for pr in PAIRING_ORDER:
                S = pooled.get(pr)
                if not S:
                    continue
                g1, g2 = (_m([r["graded_1"] for r in S]),
                          _m([r["graded_2"] for r in S]))
                b1, b2 = (_m([r["binary_1"] for r in S]),
                          _m([r["binary_2"] for r in S]))
                L.append(f"  {'POOLED':<18s}{pr:<10s}{len(S):>4d}"
                         f"{_f3(g1):>7s}{_f3(g2):>7s}{_fd(g1-g2):>8s}"
                         f"{'   ':>3s}{_f3(b1):>7s}{_f3(b2):>7s}"
                         f"{_fd(b1-b2):>8s}")
            L.append("")
    return "\n".join(L)


def level2(runs: Path) -> str:
    L = ["=" * W,
         "LEVEL 2 — WITHIN SCOPE, POSITION VARIED:  (as tier 1) - (as tier 2)",
         "=" * W,
         "Holds scope fixed, so this is the POSITION effect net of scope",
         "difficulty.  Positive = the same scope scores higher when it sits in",
         "tier 1.  n1/n2 are the two group sizes.",
         "",
         "Reported under two pairing sets, because the choice changes one scope",
         "and not the other:",
         "  VARIANT A  all pairings",
         "  VARIANT B  only any->all and all->any",
         "",
         "[all] is IDENTICAL under both -- structurally, since any->any has no",
         "[all] leaf and all->all has zero scorable instances, so B removes",
         "nothing from it.  [any] SWINGS, because any->any's tier-2 scores are",
         "the low ones.  So the robust result is the [all] half; [any] sits at",
         "ceiling in both positions and should not be claimed.", ""]
    for split in SPLITS:
        for base, keep in (("scorable", lambda r: r["scorable"]),
                           ("all instances", lambda r: True)):
            for vlab, vkeep in (("A: all pairings", PAIRING_ORDER),
                                ("B: any->all + all->any", VARIANT_B)):
                L.append(f"  [{split}]  base: {base}   variant {vlab}")
                L.append(f"  {'model':<18s}{'scope':<7s}"
                         f"{'as t1':>7s}{'as t2':>7s}{'diff g':>8s}"
                         f"{'  n1/n2':>10s}{'   ':>3s}"
                         f"{'as t1':>7s}{'as t2':>7s}{'diff b':>8s}")
                L.append("  " + "-" * 86)
                pool: dict = {}
                for mo in MODELS:
                    rows = _load(runs, mo, split)
                    if rows is None:
                        continue
                    sel = [r for r in rows
                           if keep(r) and r["pairing"] in vkeep]
                    for scope in ("any", "all"):
                        a1 = [r["graded_1"] for r in sel
                              if r["scope_1"] == scope]
                        a2 = [r["graded_2"] for r in sel
                              if r["scope_2"] == scope]
                        b1 = [r["binary_1"] for r in sel
                              if r["scope_1"] == scope]
                        b2 = [r["binary_2"] for r in sel
                              if r["scope_2"] == scope]
                        d = (_m(a1) - _m(a2)) if a1 and a2 else None
                        db = (_m(b1) - _m(b2)) if b1 and b2 else None
                        p = pool.setdefault(scope, [[], [], [], []])
                        for dst, src in zip(p, (a1, a2, b1, b2)):
                            dst.extend(src)
                        L.append(f"  {mo:<18s}[{scope}]{'':<2s}"
                                 f"{_f3(_m(a1)):>7s}{_f3(_m(a2)):>7s}"
                                 f"{_fd(d):>8s}"
                                 f"{f'{len(a1)}/{len(a2)}':>10s}{'   ':>3s}"
                                 f"{_f3(_m(b1)):>7s}{_f3(_m(b2)):>7s}"
                                 f"{_fd(db):>8s}")
                L.append("  " + "-" * 86)
                for scope in ("any", "all"):
                    p = pool.get(scope)
                    if not p or not p[0] or not p[1]:
                        continue
                    a1, a2, b1, b2 = p
                    L.append(f"  {'POOLED':<18s}[{scope}]{'':<2s}"
                             f"{_f3(_m(a1)):>7s}{_f3(_m(a2)):>7s}"
                             f"{_fd(_m(a1)-_m(a2)):>8s}"
                             f"{f'{len(a1)}/{len(a2)}':>10s}{'   ':>3s}"
                             f"{_f3(_m(b1)):>7s}{_f3(_m(b2)):>7s}"
                             f"{_fd(_m(b1)-_m(b2)):>8s}")
                L.append("")
    return "\n".join(L)


METHOD = r"""
====================================================================================================
METHOD
====================================================================================================

THE MECHANISM
-------------
``LexicographicPreference.evaluate`` takes the verdict from tier 1 alone:

      passed = sub_results[0].passed                  # preferences.py:815

Lower tiers move the scalar score but can never flip ``passed``.  A planner that honours
its top priority and abandons the rest passes identically to one that honours the whole
ordering, while lexicographic semantics intend every tier to matter.

WHY TWO LEVELS AND NOT ONE NUMBER
---------------------------------
Every Lexicographic entry in the bank has exactly two tiers (test 34/34, test_large
149/149), and the tier SCOPES are not uniform.  The four possible pairings are:

      any->all      tier 1 cheap, tier 2 costly       banks 2, 3, 5, 6, 15
      any->any      both cheap                        banks 7, 14
      all->any      tier 1 costly, tier 2 cheap       banks 8, 9, 10, 12, 13
      all->all      bank 11 -- mutually exclusive, 0 scorable

LEVEL 1 compares t1 against t2 inside a pairing.  On its own it cannot say whether a gap
comes from POSITION (tier 2 does not gate the verdict, so it is skippable) or from SCOPE
(``[all]`` is simply harder than ``[any]``), because any->all and all->any put the hard
scope in opposite tiers.

LEVEL 2 holds the scope fixed and varies only the position, which isolates the position
effect.  It is reported under two pairing sets because the choice changes one scope and
not the other -- see the note above the Level 2 tables.

GRADED vs BINARY
----------------
GRADED is ``details.tiers[i].score``; BINARY is ``details.tiers[i].passed``.  For an
Atomic ``[all]`` leaf the score is ``passed_count / n_entities``, a real gradation.  For
``[any]`` it is ``float(passed)``, so graded and binary are IDENTICAL there by
construction -- not a duplication bug.

Binary runs roughly 3x graded on any->all because it counts a tier-2 score of 0.94 as a
total failure.  Of 44 tier-2 failures only 3 scored below 0.1 while 16 scored above 0.8.
Both are reported; graded is the honest magnitude.

THE BIG-M SCALAR IS DELIBERATELY UNUSED.  ``details.scalar`` is exact and invertible
(reconstruction error 0.00e+00 over 745 instances), but ``M = max(plan entity count) + 1``
varies per instance -- measured 2 to 40, median 8 -- so identical behaviour scores
differently, and tier 2 occupies only ``1/(M+1)`` of the range (0.024 at M=40).  M is also
planner-controllable in the shortcutting direction: scheduling more entities raises M and
shrinks the cost of abandoning tier 2.  The per-tier scores are on a common [0,1] scale
and need none of that correction.

ATTAINABILITY — NOT SATISFYING THE IMPOSSIBLE IS NOT SHORTCUTTING
-----------------------------------------------------------------
An instance is SCORABLE only if BOTH tiers were JOINTLY attainable from the shown choice
set.  ``[all]`` binds PER CITY (every scheduled entity must comply, and entities come from
the city they sit in); ``[any]`` binds TRIP-WIDE.  With A = {e : t1(e)}, B = {e : t2(e)}
over the gated city pool and R_c the floor requirement:

  all/all   forall c: |A n B|_c >= R_c
  all/any   forall c: |A|_c >= R_c       and  sum_c |A n B|_c >= 1
  any/all   forall c: |B|_c >= R_c       and  sum_c |A n B|_c >= 1
  any/any   forall c: |pool|_c >= R_c    and  ( sum_c |A n B|_c >= 1  or  capacity >= 2
                                                with a witness for each )

The all/all row subsumes the structural mutual-exclusivity check: bank 11
(``mode == flight [all]`` over ``mode in [self-driving, taxi] [all]``) has |A n B| = 0 and
drops out unaided.  A shape-based rule was considered and REJECTED -- it would wrongly
exclude a NESTED pair such as ``rating >= 3 [all]`` over ``rating >= 4 [all]``, which is
perfectly compatible.

FLOOR FOR [all], CEILING FOR [any] -- THE ASYMMETRY
---------------------------------------------------
Attainability asks whether a VALID plan satisfying both tiers exists, so each tier gets
the most permissive valid plan, and permissive points in OPPOSITE directions:

  [all]   a SMALLER plan is easier (fewer entities must comply)          ->  FLOOR
  [any]   a LARGER plan is easier (more slots host distinct witnesses)   ->  CEILING

  FLOOR    per city   Restaurant 3, Attraction 1, Accommodation 1, Transportation 1
  CEILING  per trip   Restaurant 3/day, Attraction 4/day, Accommodation and
                      Transportation 1 per city

Using the floor for both was a bug caught in implementation: it capped attraction capacity
at 1 per city, forced a SINGLE entity to satisfy both [any] tiers of bank 14, and wrongly
excluded instances a 4-attraction day hosts easily.  Accommodation is where floor and
ceiling coincide -- its ceiling really is one stay per city -- which is why bank 7's
exclusions survive and are correct.

``is_not_absent`` exempts meals on transit days and attractions on transit-or-travel days,
so a minimum-viable valid plan needs entities only on the REQUIRED days.  Measured:
exactly ONE required day per city (99.7% / 90.6% / 90.6% of plans at 1 / 2 / 3 cities).
That makes R an INTEGER and removes the per-city day split from the filter entirely.  An
earlier draft used ``floor_per_day * days_c`` -- about 7 restaurants per city on a 7-day
trip -- which was over-strict and over-excluded (any->all kept 54 instead of 59).

``plan_geography``'s fractional ``days_c`` is correct for an EXPECTATION (it multiplies a
rate, and 2.5 days x 3 meals = 7.5 slots is meaningful) but wrong for a FEASIBILITY
THRESHOLD: "at least 7.5 qualifying candidates" has no meaning as a count, and the
rounding direction would change the verdict.

MODEL-INDEPENDENT BY CONSTRUCTION
---------------------------------
Requirement and geography come from the RECORD, never the plan.  Using ``plan_geography``
would (a) let a model shrink its own denominator and thereby move its own attainability --
the F2 denominator-control problem imported into F4's gate -- and (b) give every model a
different scorable set, destroying the cross-model comparison.  Verified: scorable counts
are identical across all five models.

Pool construction, predicate testing and partner gating are imported from ``pool_common``,
so F4 and F2 cannot disagree about what the choice set is or whether a candidate
qualifies.

TIER STATES — REPORTED BECAUSE 0.5 IS AMBIGUOUS
-----------------------------------------------
``n_passed_tiers / n_tiers`` maps two OPPOSITE behaviours onto the same 0.5:

      tier1 PASS, tier2 FAIL   ->  verdict PASS  ->  the shortcut
      tier1 FAIL, tier2 PASS   ->  verdict FAIL  ->  priority INVERSION

So prefix depth is kept instead: k = the largest k with tiers 1..k ALL passed, m = total
passed, and ``m > k`` means a satisfied tier sits below a failed one.  Measured, not
hypothesised: gemma-4-26b-a4b posts 7 inversions in 31 scorable all->any instances.

WHAT F4 DOES NOT COVER, AND WHY
-------------------------------
  Compensatory   ``tier_pass_counts`` would support a Compensation Reliance measure, but
                 compensation is a DESIGNED affordance -- using it is not prima facie
                 lazy.  Not reported.
  Composite OR   ``details`` keeps ``passed_children`` as a COUNT, not which child, so
                 "took the cheapest branch" is untestable without a preferences.py change.
  Conditional    its branch choice is the gate dodge, already measured in the drift/gate
                 analysis (triviality 16% -> 64% when drift hits the condition).  Counting
                 it here would double-count one mechanism.

LIMITS OF THE BANK
------------------
Only two ``(entity, attribute, scope)`` keys appear in BOTH tier positions, and both are
``[any]``.  A fully matched position test is therefore not possible with the current data;
Level 2 is a between-groups contrast.  If F4 is to carry more weight, the fix is at
construction: mirror the same predicate pair in both tier orders.
"""


def appendix(runs: Path) -> str:
    L = ["=" * W, "APPENDIX A — tier states and attainability accounting", "=" * W,
         "complete     both tiers passed",
         "prefix_stop  tier 1 passed, tier 2 did not -- THE SHORTCUT (verdict secured)",
         "inverted     tier 1 failed, tier 2 passed -- the preference FAILS",
         "none         neither passed",
         "",
         "prefix_stop and inverted are the two states a n_passed/n_tiers scalar",
         "would collapse onto 0.5.", ""]
    for split in SPLITS:
        L.append(f"  [{split}]  scorable instances")
        L.append(f"  {'model':<18s}{'pairing':<10s}{'n':>4s}{'complete':>10s}"
                 f"{'prefix_stop':>13s}{'inverted':>10s}{'none':>6s}")
        L.append("  " + "-" * 72)
        for mo in MODELS:
            rows = _load(runs, mo, split)
            if rows is None:
                continue
            for pr in PAIRING_ORDER:
                S = [r for r in rows if r["scorable"] and r["pairing"] == pr]
                if not S:
                    continue
                c = Counter(r["state"] for r in S)
                L.append(f"  {mo:<18s}{pr:<10s}{len(S):>4d}{c['complete']:>10d}"
                         f"{c['prefix_stop']:>13d}{c['inverted']:>10d}"
                         f"{c['none']:>6d}")
        L.append("")
    L += ["=" * W, "APPENDIX B — attainability filter", "=" * W,
          "Model-independent, so these counts are identical for every model;",
          "any divergence would be a bug and is worth checking here first.", ""]
    for split in SPLITS:
        rows = next((_load(runs, mo, split) for mo in MODELS
                     if _load(runs, mo, split)), None)
        if not rows:
            continue
        L.append(f"  [{split}]")
        L.append(f"  {'pairing':<10s}{'total':>7s}{'scorable':>10s}"
                 f"{'excluded':>10s}{'keep':>8s}")
        L.append("  " + "-" * 46)
        tot = keep = 0
        for pr in PAIRING_ORDER:
            S = [r for r in rows if r["pairing"] == pr]
            if not S:
                continue
            k = sum(1 for r in S if r["scorable"])
            tot += len(S)
            keep += k
            L.append(f"  {pr:<10s}{len(S):>7d}{k:>10d}{len(S)-k:>10d}"
                     f"{100*k/len(S):>7.1f}%")
        L.append("  " + "-" * 46)
        L.append(f"  {'TOTAL':<10s}{tot:>7d}{keep:>10d}{tot-keep:>10d}"
                 f"{100*keep/tot:>7.1f}%")
        L.append("")
        L.append("  exclusion reasons:")
        for (pr, why), n in sorted(
                Counter((r["pairing"], r["exclusion_reason"]) for r in rows
                        if not r["scorable"]).items(), key=lambda kv: -kv[1]):
            L.append(f"    {pr:<10s}{why:<38s}{n:>4d}")
        L.append("")
        L.append("  per bank (total -> scorable):")
        banks: dict = {}
        for r in rows:
            b = banks.setdefault(r["bank_id"], {
                "n": 0, "k": 0, "pr": r["pairing"],
                "ent": f"{r['entity_1']}/{r['entity_2']}"})
            b["n"] += 1
            b["k"] += int(r["scorable"])
        for bid in sorted(banks, key=str):
            b = banks[bid]
            L.append(f"    bank {str(bid):>3s}  {b['n']:>4d} -> {b['k']:>4d}   "
                     f"{b['pr']:<10s} {b['ent']}")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path,
                    default=Path(__file__).parent / "runs")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent
                    / "TIER_COMPLETION_approach_and_results.txt")
    args = ap.parse_args()
    txt = (headline(args.runs_dir) + "\n"
           + level1(args.runs_dir) + "\n" + level2(args.runs_dir) + "\n"
           + METHOD + "\n" + appendix(args.runs_dir))
    args.out.write_text(txt)
    print(f"[out] {args.out}  ({len(txt.splitlines())} lines)")


if __name__ == "__main__":
    main()
