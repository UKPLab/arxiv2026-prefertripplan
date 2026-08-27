#!/usr/bin/env python3
"""Distributional analysis of the PUBLISHED PreferTripPlan dataset.

Companion to ``analyze_distributions.py``, which profiles the raw
generation-side artefact (``prefertripplan.jsonl``) and has access to
construction-time fields -- feasibility gates, budget multipliers,
per-city floor shortfalls -- that never reach the Hub.  This script
instead reads the PUBLISHED splits straight from the Hub
(``UKPLab/PreferTripPlan``), so what it reports is exactly what a user
of the benchmark sees.  It therefore covers the subset of sections that
survive publication, and adds the leaf-scope analysis, which is a
property of the released ``preferences_json`` column.

Sections
--------
  [1]  Records per level / days / party size, per split.
  [2]  Preference paradigm + sub-paradigm distribution.
  [3]  Preferences per record.
  [4]  Pairing type and subtype (single / independent / overlapping;
       competing / non_competing), overall and per level.
  [5]  Profile drift mode (aligned / omission / inversion) per level.
  [6]  Local-constraint (hard constraint) coverage.
  [7]  LEAF-SCOPE COMPOSITION per paradigm            <- Table 1
  [8]  Scope-tuple distribution per paradigm          <- Table 2
  [9]  Preferences by universal share of their leaves <- Table 3

Quantifier accounting
---------------------
A preference decomposes into LEAF predicates, each carrying a quantifier
scope: ``all`` (universal -- must hold for every matching entity) or
``any`` (existential -- one witness suffices).  Numeric-optimization
leaves carry a DIRECTION (min / max) instead of a quantifier.

**Optimization leaves are counted as UNIVERSAL.**  Choosing the
cheapest / highest-rated entity is only decidable after examining the
whole candidate set; satisfying it on a subset, or on one arbitrary
witness, does not satisfy it at all.  Functionally that is the universal
obligation, not the existential one, so folding ``opt`` into ``all`` is
the faithful accounting.  Every table below uses the folded view, and an
``of which opt`` column keeps the fold auditable rather than hidden.
``--split-opt`` reports the three-way breakdown instead.

Usage
-----
  python3 data-generation/analyze_hf_distributions.py
  python3 data-generation/analyze_hf_distributions.py --splits test test_large
  python3 data-generation/analyze_hf_distributions.py --no-figures
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DATASET = "UKPLab/PreferTripPlan"
SPLITS = ("test", "test_large")

LEVEL_ORDER = ["easy", "medium", "hard"]
DRIFT_ORDER = ["aligned", "omission", "inversion"]
PAIR_TYPE_ORDER = ["single", "independent", "overlapping"]
PAIR_SUBTYPE_ORDER = ["competing", "non_competing", "--"]

PARADIGM_ORDER = [
    "AtomicPreference",
    "CompositePreference",
    "ConditionalPreference",
    "LexicographicPreference",
    "CompensatoryPreference",
    "ScopedPreference",
    "NumericPreference",
    "TemporalPreference",
]

# Short labels for the figure's x axis, matching the evaluation figures.
PARADIGM_SHORT = {
    "AtomicPreference":        "atom.",
    "CompositePreference":     "compos.",
    "ConditionalPreference":   "cond.",
    "LexicographicPreference": "lexico.",
    "CompensatoryPreference":  "compen.",
    "ScopedPreference":        "scoped",
    "NumericPreference":       "numeric",
    "TemporalPreference":      "tempo.",
}


# --------------------------------------------------------------------------- #
# Leaf-scope decomposition                                                     #
# --------------------------------------------------------------------------- #
# Mirrors ``evaluation/analyze_performance.py::_leaf_scopes``.  Kept as a
# local copy so ``data-generation`` carries no dependency on the
# evaluation package; the two must stay in sync if the preference schema
# gains a paradigm.  Verified to classify every leaf in both published
# splits with no unhandled node shape.

_LEAF_CHILD_KEYS: dict[str, tuple[str, ...]] = {
    "CompositePreference":     ("children",),
    "ConditionalPreference":   ("condition", "then_pref", "else_pref"),
    "LexicographicPreference": ("preferences",),
    "CompensatoryPreference":  ("primary_ap", "margin_ap", "secondary_ap"),
    "ScopedPreference":        ("scope_filters", "inner"),
    "TemporalPreference":      ("subject_ap", "reference_ap"),
}


def _leaf_scopes(paradigm: str | None, template: Any,
                 out: list[str], depth: int = 0) -> None:
    """Append every leaf's raw scope label: ``all`` / ``any`` / ``opt``."""
    if depth > 12 or not isinstance(template, dict):
        return
    if paradigm in _LEAF_CHILD_KEYS:
        for key in _LEAF_CHILD_KEYS[paradigm]:
            value = template.get(key)
            if value is None:
                continue
            for child in (value if isinstance(value, list) else [value]):
                if not isinstance(child, dict):
                    continue
                _leaf_scopes(child.get("class") or child.get("paradigm"),
                             child.get("template", child), out, depth + 1)
        return
    if paradigm == "NumericPreference":
        out.append("opt")
        return
    scope = template.get("scope")
    if scope in ("all", "any"):
        out.append(scope)
    elif template.get("direction") in ("min", "max"):
        out.append("opt")


def leaf_scope_tuple(entry: dict) -> tuple[str, ...]:
    out: list[str] = []
    _leaf_scopes(entry.get("paradigm"), entry.get("template") or {}, out)
    return tuple(out)


def fold_opt(t: tuple[str, ...]) -> tuple[str, ...]:
    """Optimization leaves counted as universal -- see the module
    docstring for why ``opt`` is functionally an ``all`` obligation."""
    return tuple("all" if s == "opt" else s for s in t)


def universal_share(t: tuple[str, ...]) -> float:
    """Fraction of a preference's leaves that are universal, AFTER the
    opt fold.  This is the axis Table 3 buckets on."""
    if not t:
        return 0.0
    f = fold_opt(t)
    return f.count("all") / len(f)


# --------------------------------------------------------------------------- #
# Collection                                                                   #
# --------------------------------------------------------------------------- #

def _loads(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (TypeError, ValueError):
            return None
    return None


class SplitStats:
    """Everything the report needs from one published split."""

    def __init__(self, name: str):
        self.name = name
        self.n_records = 0
        self.level = Counter()
        self.days = Counter()
        self.people = Counter()
        self.drift = Counter()
        self.drift_by_level: dict[str, Counter] = defaultdict(Counter)
        self.pair_type = Counter()
        self.pair_subtype = Counter()
        self.pair_subtype_by_level: dict[str, Counter] = defaultdict(Counter)
        self.paradigm = Counter()
        self.subparadigm = Counter()
        self.n_prefs: list[int] = []
        self.lc_fields = Counter()
        self.lc_present = 0
        # Leaf-scope: raw three-way counts, plus tuple census.
        self.leaves: dict[str, Counter] = defaultdict(Counter)   # para -> raw scope
        self.tuples: dict[str, Counter] = defaultdict(Counter)   # para -> raw tuple
        self.prefs_per_para = Counter()

    def update(self, row: dict) -> None:
        self.n_records += 1
        self.level[row.get("level")] += 1
        try:
            self.days[int(row.get("days") or 0)] += 1
        except (TypeError, ValueError):
            pass
        try:
            self.people[int(row.get("people_number") or 0)] += 1
        except (TypeError, ValueError):
            pass
        drift = row.get("profile_drift")
        self.drift[drift] += 1
        self.drift_by_level[row.get("level")][drift] += 1

        pair = _loads(row.get("preference_pair")) or {}
        ptype = pair.get("type") or "single"
        psub = pair.get("subtype") or "--"
        self.pair_type[ptype] += 1
        self.pair_subtype[psub] += 1
        self.pair_subtype_by_level[row.get("level")][psub] += 1

        lc = _loads(row.get("local_constraint")) or {}
        if isinstance(lc, dict):
            hit = False
            for k, v in lc.items():
                if v not in (None, "", [], {}):
                    self.lc_fields[k] += 1
                    hit = True
            self.lc_present += int(hit)

        prefs = _loads(row.get("preferences_json")) or []
        self.n_prefs.append(len(prefs))
        for p in prefs:
            para = p.get("paradigm") or "?"
            self.paradigm[para] += 1
            self.prefs_per_para[para] += 1
            sub = p.get("sub_paradigm") or _sub_of(p)
            self.subparadigm[f"{para}[{sub}]" if sub else para] += 1
            t = leaf_scope_tuple(p)
            self.tuples[para][t] += 1
            for s in t:
                self.leaves[para][s] += 1


def _sub_of(entry: dict) -> str | None:
    """Best-effort sub-paradigm from the template (op / direction)."""
    t = entry.get("template") or {}
    if not isinstance(t, dict):
        return None
    for key in ("op", "direction"):
        v = t.get(key)
        if isinstance(v, str):
            return v
    return None


# --------------------------------------------------------------------------- #
# Formatting                                                                   #
# --------------------------------------------------------------------------- #

def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def fmt_counter(c: Counter, total: int, *, order: list | None = None,
                key_hdr: str = "key", width: int = 26) -> list[str]:
    keys = ([k for k in order if k in c] + sorted(
        (k for k in c if not order or k not in order), key=str)) if order \
        else sorted(c, key=lambda k: (-c[k], str(k)))
    lines = [f"  {key_hdr:<{width}s} {'n':>7s} {'share':>8s}",
             "  " + "-" * (width + 17)]
    for k in keys:
        lines.append(f"  {str(k):<{width}s} {c[k]:>7d} "
                     f"{_pct(c[k] / total) if total else '-':>8s}")
    return lines


def table1_rows(st: SplitStats, split_opt: bool) -> list[tuple]:
    """(paradigm, prefs, leaves, %any, %all, n_opt) with opt folded into
    all unless ``split_opt``."""
    rows = []
    for para in PARADIGM_ORDER:
        c = st.leaves.get(para)
        if not c:
            continue
        n = sum(c.values())
        n_any, n_all, n_opt = c["any"], c["all"], c["opt"]
        if split_opt:
            rows.append((para, st.prefs_per_para[para], n,
                         n_any / n, n_all / n, n_opt / n, n_opt))
        else:
            rows.append((para, st.prefs_per_para[para], n,
                         n_any / n, (n_all + n_opt) / n, None, n_opt))
    return rows


def write_report(stats: dict[str, SplitStats], out: Path,
                 split_opt: bool) -> str:
    P: list[str] = []
    A = P.append
    A("=" * 78)
    A("PreferTripPlan — published-dataset distribution report")
    A(f"source: {DATASET} (HuggingFace Hub)")
    A("=" * 78)
    A("")
    A("Optimization ('opt') leaves are counted as UNIVERSAL ('all'): an")
    A("optimization is only decidable over the whole candidate set, so it")
    A("carries a universal obligation rather than an existential one."
      if not split_opt else
      "Reporting the RAW three-way split (--split-opt); 'opt' is NOT folded.")
    A("")

    for name, st in stats.items():
        A("#" * 78)
        A(f"SPLIT: {name}   ({st.n_records} records, "
          f"{sum(st.paradigm.values())} preferences)")
        A("#" * 78)
        A("")

        A("[1] Records per level")
        P.extend(fmt_counter(st.level, st.n_records, order=LEVEL_ORDER,
                             key_hdr="level")); A("")
        A("[1b] Records per trip length (days)")
        P.extend(fmt_counter(st.days, st.n_records, key_hdr="days")); A("")
        A("[1c] Records per party size (people_number)")
        P.extend(fmt_counter(st.people, st.n_records, key_hdr="people")); A("")

        A("[2] Preference paradigm")
        P.extend(fmt_counter(st.paradigm, sum(st.paradigm.values()),
                             order=PARADIGM_ORDER, key_hdr="paradigm",
                             width=26)); A("")
        A("[2b] Preference sub-paradigm")
        P.extend(fmt_counter(st.subparadigm, sum(st.subparadigm.values()),
                             key_hdr="paradigm[sub]", width=34)); A("")

        A("[3] Preferences per record")
        if st.n_prefs:
            A(f"  n={len(st.n_prefs)}  mean={statistics.mean(st.n_prefs):.2f}  "
              f"median={statistics.median(st.n_prefs):.1f}  "
              f"min={min(st.n_prefs)}  max={max(st.n_prefs)}")
            P.extend(fmt_counter(Counter(st.n_prefs), len(st.n_prefs),
                                 key_hdr="prefs/record"))
        A("")

        A("[4] Pairing type")
        P.extend(fmt_counter(st.pair_type, st.n_records,
                             order=PAIR_TYPE_ORDER, key_hdr="type")); A("")
        A("[4b] Pairing subtype")
        P.extend(fmt_counter(st.pair_subtype, st.n_records,
                             order=PAIR_SUBTYPE_ORDER, key_hdr="subtype")); A("")
        A("[4c] Pairing subtype per level")
        A(f"  {'level':<10s}" + "".join(f"{s:>16s}" for s in PAIR_SUBTYPE_ORDER))
        A("  " + "-" * (10 + 16 * len(PAIR_SUBTYPE_ORDER)))
        for lv in LEVEL_ORDER:
            c = st.pair_subtype_by_level.get(lv, Counter()); tot = sum(c.values())
            A(f"  {lv:<10s}" + "".join(
                f"{f'{c[s]} ({100*c[s]/tot:.0f}%)' if tot else '-':>16s}"
                for s in PAIR_SUBTYPE_ORDER))
        A("")

        A("[5] Profile drift mode")
        P.extend(fmt_counter(st.drift, st.n_records, order=DRIFT_ORDER,
                             key_hdr="drift")); A("")
        A("[5b] Drift mode per level")
        A(f"  {'level':<10s}" + "".join(f"{d:>16s}" for d in DRIFT_ORDER))
        A("  " + "-" * (10 + 16 * len(DRIFT_ORDER)))
        for lv in LEVEL_ORDER:
            c = st.drift_by_level.get(lv, Counter()); tot = sum(c.values())
            A(f"  {lv:<10s}" + "".join(
                f"{f'{c[d]} ({100*c[d]/tot:.0f}%)' if tot else '-':>16s}"
                for d in DRIFT_ORDER))
        A("")

        A("[6] Local-constraint coverage")
        A(f"  records with >=1 local constraint: {st.lc_present}"
          f" / {st.n_records} ({100*st.lc_present/st.n_records:.1f}%)")
        P.extend(fmt_counter(st.lc_fields, st.n_records, key_hdr="lc field"))
        A("")

        # ---- Table 1 -------------------------------------------------
        A("[7] LEAF-SCOPE COMPOSITION per paradigm   (Table 1)")
        if split_opt:
            A(f"  {'paradigm':<26s} {'prefs':>6s} {'leaves':>7s} "
              f"{'%any':>7s} {'%all':>7s} {'%opt':>7s}")
            A("  " + "-" * 63)
            for para, npref, nleaf, f_any, f_all, f_opt, _n_opt in \
                    table1_rows(st, split_opt):
                A(f"  {para:<26s} {npref:>6d} {nleaf:>7d} "
                  f"{_pct(f_any):>7s} {_pct(f_all):>7s} {_pct(f_opt):>7s}")
        else:
            A(f"  {'paradigm':<26s} {'prefs':>6s} {'leaves':>7s} "
              f"{'%any':>7s} {'%all':>7s} {'of which opt':>13s}")
            A("  " + "-" * 69)
            for para, npref, nleaf, f_any, f_all, _f_opt, n_opt in \
                    table1_rows(st, split_opt):
                A(f"  {para:<26s} {npref:>6d} {nleaf:>7d} "
                  f"{_pct(f_any):>7s} {_pct(f_all):>7s} {n_opt:>13d}")
        tot = Counter()
        for c in st.leaves.values():
            tot.update(c)
        n = sum(tot.values())
        A("  " + "-" * (63 if split_opt else 69))
        if split_opt:
            A(f"  {'ALL':<26s} {sum(st.prefs_per_para.values()):>6d} {n:>7d} "
              f"{_pct(tot['any']/n):>7s} {_pct(tot['all']/n):>7s} "
              f"{_pct(tot['opt']/n):>7s}")
        else:
            A(f"  {'ALL':<26s} {sum(st.prefs_per_para.values()):>6d} {n:>7d} "
              f"{_pct(tot['any']/n):>7s} "
              f"{_pct((tot['all']+tot['opt'])/n):>7s} {tot['opt']:>13d}")
        A("")

        # ---- Table 2 -------------------------------------------------
        A("[8] Scope-tuple distribution per paradigm   (Table 2)")
        A(f"  {'paradigm':<26s} {'scope tuple (opt→all)':<22s} "
          f"{'%all':>7s} {'%any':>7s} {'n':>7s} {'share':>8s}")
        A("  " + "-" * 80)
        for para in PARADIGM_ORDER:
            tups = st.tuples.get(para)
            if not tups:
                continue
            total = sum(tups.values())
            for i, (t, cnt) in enumerate(sorted(
                    tups.items(), key=lambda kv: (universal_share(kv[0]),
                                                  len(kv[0]), kv[0]))):
                f = fold_opt(t)
                A(f"  {para if i == 0 else '':<26s} "
                  f"{'(' + ','.join(f) + ')':<22s} "
                  f"{_pct(f.count('all')/len(f)):>7s} "
                  f"{_pct(f.count('any')/len(f)):>7s} "
                  f"{cnt:>7d} {_pct(cnt/total):>8s}")
            A("")

        # ---- Table 3 -------------------------------------------------
        A("[9] Preferences by universal share of their leaves   (Table 3)")
        buckets = Counter()
        for para, tups in st.tuples.items():
            for t, cnt in tups.items():
                buckets[round(100 * universal_share(t), 1)] += cnt
        total = sum(buckets.values())
        A(f"  {'universal share':<26s} {'n':>7s} {'share':>8s}")
        A("  " + "-" * 43)
        for k in sorted(buckets):
            A(f"  {str(k) + '%':<26s} {buckets[k]:>7d} "
              f"{_pct(buckets[k]/total):>8s}")
        A("  " + "-" * 43)
        A(f"  {'TOTAL':<26s} {total:>7d} {_pct(1.0):>8s}")
        A("")

    report = "\n".join(P)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return report


# --------------------------------------------------------------------------- #
# Figure                                                                       #
# --------------------------------------------------------------------------- #

def plot_table1(st: SplitStats, out_path: Path, fmt: str) -> bool:
    """Stacked bar of leaf-scope composition per paradigm, one split.

    Three segments, not two: existential, universal-by-quantifier, and
    universal-by-optimization.  The opt segment shares the universal
    hue and is hatched, so the bar reads as a two-way any/all split at a
    glance while still showing HOW MUCH of the universal share came from
    the fold.  Collapsing it to two segments would make the fold
    unfalsifiable from the figure alone."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:                                # pragma: no cover
        print(f"  [fig] matplotlib unavailable ({e}); skipping figure")
        return False

    paras = [p for p in PARADIGM_ORDER if st.leaves.get(p)]
    if not paras:
        return False
    labels = [PARADIGM_SHORT.get(p, p) for p in paras]
    share_any, share_all, share_opt, totals = [], [], [], []
    for p in paras:
        c = st.leaves[p]
        n = sum(c.values())
        totals.append(n)
        share_any.append(100 * c["any"] / n)
        share_all.append(100 * c["all"] / n)
        share_opt.append(100 * c["opt"] / n)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = range(len(paras))
    C_ANY, C_ALL = "#4c78a8", "#e45756"
    b1 = ax.bar(x, share_any, width=0.62, color=C_ANY,
                label="existential  (any)")
    b2 = ax.bar(x, share_all, width=0.62, bottom=share_any, color=C_ALL,
                label="universal  (all)")
    bot = [a + b for a, b in zip(share_any, share_all)]
    b3 = ax.bar(x, share_opt, width=0.62, bottom=bot, color=C_ALL,
                hatch="///", edgecolor="white", linewidth=0.0,
                label="universal via optimization  (opt → all)")

    for xi, (a, al, o, n) in enumerate(zip(share_any, share_all,
                                           share_opt, totals)):
        for val, base in ((a, 0.0), (al, a), (o, a + al)):
            if val >= 6.0:
                ax.text(xi, base + val / 2, f"{val:.0f}", ha="center",
                        va="center", fontsize=8, color="white",
                        fontweight="bold")
        ax.text(xi, 101.5, f"n={n}", ha="center", va="bottom", fontsize=7,
                color="0.35")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Share of leaf predicates (%)")
    ax.set_xlabel("Preference paradigm")
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, fontsize=8.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = out_path.with_suffix(f".{fmt}")
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {target}")
    return True


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET,
                    help=f"HuggingFace dataset repo (default: {DATASET}).")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS),
                    help="Splits to analyse (default: test test_large).")
    ap.add_argument("--out", type=Path,
                    default=here / "analysis" / "hf_distribution_report.txt")
    ap.add_argument("--fig-dir", type=Path, default=here / "analysis" / "figures")
    ap.add_argument("--fmt", choices=["png", "svg", "pdf"], default="pdf")
    ap.add_argument("--split-opt", action="store_true",
                    help="Report the raw three-way any/all/opt split instead "
                         "of folding optimization leaves into 'all'.")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset

    stats: dict[str, SplitStats] = {}
    for split in args.splits:
        ds = load_dataset(args.dataset, split=split, verification_mode="no_checks")
        st = SplitStats(split)
        for row in ds:
            st.update(row)
        stats[split] = st
        print(f"[in] {args.dataset}:{split}  {st.n_records} records")

    report = write_report(stats, args.out, args.split_opt)
    print(f"[out] report -> {args.out}")
    if args.stdout:
        print(); print(report)

    if not args.no_figures:
        for split, st in stats.items():
            plot_table1(st, args.fig_dir / f"leaf_scope_composition_{split}",
                        args.fmt)


if __name__ == "__main__":
    main()
