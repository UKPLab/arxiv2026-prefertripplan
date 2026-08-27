#!/usr/bin/env python3
"""plot_scope_scatter.py — the leaf-scope effect on preference paradigms,
all models in one panel.

WHAT THIS FIGURE ARGUES
-----------------------
Headline per-paradigm pass rates are not comparable across paradigms,
because a paradigm's difficulty is dominated by the quantifier scope of
its LEAVES rather than by its combinator.  Paradigms differ sharply in
leaf-scope composition (Conditional is ~80% existential, Atomic ~62%
universal), so the apparent superiority of structurally COMPLEX
paradigms over Atomic is an artefact of the mix each one is averaged
over.

This plots, for every paradigm and every model, the pass rate at that
paradigm's most-existential scope tuple and at its most-universal one.
The vertical gap between the two markers is the scope effect, measured
WITHIN a paradigm -- so neither the combinator nor the paradigm mix can
confound it.  Cross-paradigm reading is possible but secondary and does
additionally reflect combinator semantics; say so in the caption.

    python3 evaluation/plot_figures/plot_scope_scatter.py \\
        --report evaluation/gpt-5.6-terra_test_large/analyze_gpt-5.6-terra.txt \\
        ... \\
        --out-dir evaluation/figures_crossmodel/ --fmt pdf

ENCODING
--------
colour  -> model (shared with every other figure via ``_common``)
shape   -> max-%any tuple (circle) vs max-%all tuple (triangle-down)
hollow  -> diamond, for a paradigm with only ONE scope tuple: its
           most-existential and most-universal tuple are the SAME tuple,
           so there is no contrast to draw.  Plotting two filled markers
           at one point would imply two measurements where the dataset
           provides one, so these get a single distinct marker instead.
           They are NOT excluded -- the level is real and is shown.

SPLITS
------
Both splits are drawn.  ``test_large`` is the primary series (filled
markers, solid connector); ``test`` is the secondary one (hollow
markers, dotted connector), nudged a hair to the right inside the same
model slot.  Fill and line style carry the split together, because
either cue alone is easy to lose at this marker size -- the same
reasoning as ``_common.SPLIT_STYLE``.

WHICH TUPLE REPRESENTS A PARADIGM is resolved ONCE, from ``test_large``,
and both splits are then plotted at those same tuples.  Resolving it per
split would silently compare the two splits on DIFFERENT tuples wherever
the threshold bites differently -- at ``--min-n 10`` that happened for
Temporal (its ``(all)`` tuple is n=32 on test_large but n=8 on test) and
for Lexicographic -- which would destroy exactly the comparison the
overlay exists to make.  The default of 5 keeps both splits present for
those two; the selection rule stays regardless, since a higher --min-n
would re-introduce the divergence.

MIN-N
-----
``--min-n`` (default 5, exclusive) is a reliability guard on a rate
ESTIMATE: a 5-preference cell moves 20 points if one plan changes.  It
does two things:

  * a tuple must clear it on ``test_large`` to be eligible as a
    paradigm's representative at all;
  * an individual marker is omitted when ITS OWN split's n fails it, so
    a thin ``test`` cell drops without taking the ``test_large`` marker
    with it.

Omissions are always reported on stderr, never silent.  This guard
applies to rates only and has no place in dataset-composition figures,
where the quantity is a count with no sampling noise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for _p in (_HERE, _PARENT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import (save, display_model, model_colours,        # noqa: E402
                     order_models, parse_report_spec)

_SCOPE_SECTION = "Preferences: by (paradigm x leaf-scope tuple)"

# Canonical paradigm order + short x labels, matching the other figures.
_PARADIGMS = [
    ("AtomicPreference",        "atom."),
    ("CompositePreference",     "compos."),
    ("ConditionalPreference",   "cond."),
    ("LexicographicPreference", "lexico."),
    ("CompensatoryPreference",  "compen."),
    ("ScopedPreference",        "scoped"),
    ("NumericPreference",       "numeric"),
    ("TemporalPreference",      "tempo."),
]

_MARKER_ANY    = "o"
_MARKER_ALL    = "v"
_MARKER_SINGLE = "D"


def _scope_rows(report) -> dict[tuple[str, tuple[str, ...]], tuple[int, int, float]]:
    """``{(paradigm, scope_tuple) -> (n, pass, pass_rate)}``.

    ``pass`` is carried as a raw count, not recovered from the rate, so
    tuples can be POOLED exactly when several share a composition."""
    out: dict[tuple[str, tuple[str, ...]], tuple[int, int, float]] = {}
    for row in report.section(_SCOPE_SECTION):
        bucket = (row.get("bucket") or "").strip()
        if "(" not in bucket or not bucket.endswith(")"):
            continue
        para, _, tail = bucket.partition("(")
        tup = tuple(t.strip() for t in tail[:-1].split(",") if t.strip())
        try:
            n = int(row.get("n"))
            npass = int(row.get("pass"))
            pr = float((row.get("pass%") or "").rstrip("%"))
        except (TypeError, ValueError):
            continue
        out[(para.strip(), tup)] = (n, npass, pr)
    return out


def _frac(t: tuple[str, ...], quant: str) -> float:
    """Share of a tuple's leaves carrying ``quant``.  ``opt`` leaves sit
    in the denominator but satisfy neither quantifier, so (any, opt) is
    50% existential and 0% universal."""
    return t.count(quant) / len(t) if t else 0.0


def _pick_buckets(tuples: list[tuple[str, ...]]
                  ) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Group tuples by QUANTIFIER COMPOSITION and return the two extreme
    groups: every tuple at the highest existential share, and every
    tuple at the highest universal share.

    Composition is a fraction, so tuples of different arity are
    comparable: ``(any, all)`` is 50% universal, ``(any, all, all)`` is
    66.7%, ``(all, all)`` and ``(all)`` are both 100%.  Selecting a
    SINGLE representative tuple per side instead would silently discard
    same-composition siblings -- e.g. Temporal's ``(any)`` and
    ``(any, any)`` are both 100% existential and belong in one group."""
    if not tuples:
        return [], []
    f_any_max = max(_frac(t, "any") for t in tuples)
    f_all_max = max(_frac(t, "all") for t in tuples)
    return ([t for t in tuples if _frac(t, "any") == f_any_max],
            [t for t in tuples if _frac(t, "all") == f_all_max])


_PRIMARY_SPLIT = "test_large"

# Split -> (marker fill, connector linestyle, x nudge within the model
# slot, z-order).  Filled+solid reads as the primary series; hollow+
# dotted as the secondary one.
_SPLIT_ENC = {
    "test_large": {"filled": True,  "ls": "-",  "nudge": -0.018, "z": 4},
    "test":       {"filled": False, "ls": ":",  "nudge": +0.018, "z": 3},
}


def plot_scope_scatter(report_specs: list[tuple[Path, str, str]],
                       out_path: Path, fmt: str, *,
                       min_n: int = 5,
                       figsize: tuple[float, float] = (11.0, 5.6),
                       ymin: float = 25.0, ymax: float = 103.0,
                       connect: bool = True,
                       band: float = 0.62,
                       annotate_n: bool = True) -> bool:
    try:
        from latex import _parse_report
    except ImportError as e:                            # pragma: no cover
        print(f"  [scatter] cannot import latex._parse_report ({e}); skipped")
        return False

    parsed: dict[tuple[str, str], dict] = {}
    for path, model, split in report_specs:
        if not path.exists():
            print(f"  [scatter] missing {path}; skipped")
            continue
        parsed[(model, split)] = _scope_rows(_parse_report(path, f"{model}/{split}"))
    if not parsed:
        return False
    models = order_models(sorted({m for m, _ in parsed}))
    splits = [s for s in (_PRIMARY_SPLIT, "test") if any(s == sp for _, sp in parsed)]
    splits += sorted({sp for _, sp in parsed} - set(splits))
    colour = model_colours(models)

    def _n(para, tup, split) -> int:
        """Largest n any model reports for this cell -- the count is a
        dataset property, so models agreeing is expected and taking the
        max is robust to a model whose report omits the row."""
        return max((parsed[(m, split)].get((para, tup), (0, 0, 0.0))[0]
                    for m in models if (m, split) in parsed), default=0)

    def _pool(para, bucket, model, split):
        """POOLED rate over a composition bucket: sum(pass)/sum(n), not a
        naive mean of rates.  A bucket can mix a 190-preference tuple with
        a 5-preference one, and an unweighted mean would let the small one
        move the point as much as the large one."""
        rows = parsed.get((model, split))
        if rows is None:
            return None
        num = den = 0
        for t in bucket:
            if _n(para, t, split) <= min_n:
                continue
            cell = rows.get((para, t))
            if cell is None:
                continue
            den += cell[0]; num += cell[1]
        return (100.0 * num / den, den) if den else None

    # Composition buckets, resolved ONCE on the primary split so both
    # splits are plotted on the same grouping.  See the SPLITS note above.
    sel_split = _PRIMARY_SPLIT if _PRIMARY_SPLIT in splits else splits[0]
    chosen: dict[str, tuple | None] = {}
    for para, _lbl in _PARADIGMS:
        tuples = sorted({t for (m, sp) in parsed if sp == sel_split
                         for (p, t) in parsed[(m, sp)] if p == para})
        eligible = [t for t in tuples if _n(para, t, sel_split) > min_n]
        chosen[para] = _pick_buckets(eligible) if eligible else None

    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(_PARADIGMS), dtype=float)
    offs = (np.arange(len(models)) - (len(models) - 1) / 2.0) * (band / max(len(models), 1))

    dropped: list[str] = []
    thin: list[str] = []
    for xi, (para, _lbl) in enumerate(_PARADIGMS):
        pick = chosen.get(para)
        if pick is None:
            dropped.append(f"{para} (no tuple with n>{min_n} on {sel_split})")
            continue
        b_any, b_all = pick
        single = (set(b_any) == set(b_all))
        for split in splits:
            enc = _SPLIT_ENC.get(split, _SPLIT_ENC["test"])
            for bucket, tag in ((b_any, "max-%any"), (b_all, "max-%all")):
                if single and tag == "max-%all":
                    continue
                for t in bucket:
                    if _n(para, t, split) <= min_n:
                        thin.append(f"{para} ({','.join(t)}) in {tag} on "
                                    f"{split}: n={_n(para, t, split)}")
            for mi, m in enumerate(models):
                if (m, split) not in parsed:
                    continue
                x = xs[xi] + offs[mi] + enc["nudge"]
                a = _pool(para, b_any, m, split)
                b = _pool(para, b_all, m, split)
                if single:
                    if a is None:
                        continue
                    # One composition only: no contrast exists in the data.
                    ax.scatter([x], [a[0]], marker=_MARKER_SINGLE, s=46,
                               facecolors=colour[m] if enc["filled"] else "none",
                               edgecolors=colour[m], linewidths=1.4, zorder=enc["z"])
                    continue
                if a is not None and b is not None and connect:
                    ax.plot([x, x], [a[0], b[0]], color=colour[m], linestyle=enc["ls"],
                            linewidth=1.0, alpha=0.6, zorder=enc["z"] - 1)
                for val, mk, sz in ((a, _MARKER_ANY, 30), (b, _MARKER_ALL, 34)):
                    if val is None:
                        continue
                    ax.scatter([x], [val[0]], marker=mk, s=sz,
                               facecolors=colour[m] if enc["filled"] else "none",
                               edgecolors=colour[m], linewidths=1.2, zorder=enc["z"])

    # Paradigm name only; the composition detail belongs in the caption.
    labels = [lbl for _para, lbl in _PARADIGMS]

    # What each column actually pools, to stderr -- the figure stays
    # clean but the mapping is still on the record.
    print("  [scatter] composition buckets pooled per paradigm "
          f"(resolved on {sel_split}, n>{min_n}):")
    for para, lbl in _PARADIGMS:
        pick = chosen.get(para)
        if pick is None:
            print(f"             {lbl:<9s} -- none eligible"); continue
        b_any, b_all = pick
        fa = _frac(b_any[0], "any") * 100 if b_any else 0.0
        fl = _frac(b_all[0], "all") * 100 if b_all else 0.0
        na = sum(_n(para, t, sel_split) for t in b_any)
        nl = sum(_n(para, t, sel_split) for t in b_all)
        if set(b_any) == set(b_all):
            print(f"             {lbl:<9s} single composition "
                  f"[{'; '.join('(' + ','.join(t) + ')' for t in b_any)}] n={na}")
        else:
            print(f"             {lbl:<9s} {fa:5.1f}%any "
                  f"[{'; '.join('(' + ','.join(t) + ')' for t in b_any)}] n={na}"
                  f"   ->  {fl:5.1f}%all "
                  f"[{'; '.join('(' + ','.join(t) + ')' for t in b_all)}] n={nl}")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_xlim(-0.6, len(_PARADIGMS) - 0.4)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Preference micro pass rate (%)")
    ax.set_xlabel("Preference paradigm")
    # No title: the caption carries it.
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    for xi in range(len(_PARADIGMS) - 1):
        ax.axvline(xi + 0.5, color="0.92", linewidth=0.8, zorder=0)

    from matplotlib.lines import Line2D
    model_handles = [Line2D([], [], marker="s", linestyle="none",
                            color=colour[m], markersize=6,
                            label=display_model(m)) for m in models]
    shape_handles = [
        Line2D([], [], marker=_MARKER_ANY, linestyle="none", color="0.25",
               markersize=6, label="max %any"),
        Line2D([], [], marker=_MARKER_ALL, linestyle="none", color="0.25",
               markersize=6, label="max %all"),
        Line2D([], [], marker=_MARKER_SINGLE, linestyle="none",
               markerfacecolor="none", markeredgecolor="0.25",
               markersize=6, label="single composition"),
    ]
    split_handles = [
        Line2D([], [], marker="o", linestyle=_SPLIT_ENC[s]["ls"], color="0.25",
               markerfacecolor="0.25" if _SPLIT_ENC[s]["filled"] else "none",
               markeredgecolor="0.25", markersize=6, label=s)
        for s in splits if s in _SPLIT_ENC
    ]
    leg1 = fig.legend(handles=model_handles, title="Model", loc="upper left",
                      bbox_to_anchor=(0.005, 1.0), ncol=len(models),
                      frameon=False, fontsize=8.5, alignment="left")
    leg1.get_title().set_fontsize(9)
    fig.add_artist(leg1)
    leg2 = fig.legend(handles=shape_handles, title="Leaf-scope composition", loc="upper right",
                      bbox_to_anchor=(0.885, 1.0), frameon=False,
                      fontsize=8.5, alignment="left")
    leg2.get_title().set_fontsize(9)
    fig.add_artist(leg2)
    if split_handles:
        leg3 = fig.legend(handles=split_handles, title="Split", loc="upper right",
                          bbox_to_anchor=(0.998, 1.0), frameon=False,
                          fontsize=8.5, alignment="left")
        leg3.get_title().set_fontsize(9)

    # Every omission is announced.  A marker missing from the figure must
    # be traceable to a stated threshold, never to silent filtering.
    if dropped:
        print(f"  [scatter] paradigm with no eligible tuple: {'; '.join(dropped)}")
    if thin:
        print(f"  [scatter] {len(thin)} marker(s) omitted for n<={min_n}:")
        for t in sorted(set(thin)):
            print(f"             {t}")
    fig.subplots_adjust(top=0.845)
    save(fig, out_path, fmt)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="append", required=True,
                    metavar="PATH[:MODEL:SPLIT]",
                    help="analyze_*.txt report; repeatable.")
    ap.add_argument("--min-n", type=int, default=5,
                    help="Exclusive threshold on a cell's n (default: 5).  A "
                         "tuple must clear it on test_large to represent its "
                         "paradigm, and an individual marker is omitted when "
                         "its own split's n fails it.  Raising it to 10 costs "
                         "the test-split markers for Temporal and "
                         "Lexicographic.  Reliability guard on RATES only -- "
                         "never applied to dataset-composition figures, where "
                         "the quantity is a count with no sampling noise.")
    ap.add_argument("--no-connect", action="store_true",
                    help="Omit the vertical line joining a model's two markers.")
    ap.add_argument("--no-n", action="store_true",
                    help="Omit per-tuple n from the x tick labels.")
    ap.add_argument("--ymin", type=float, default=25.0)
    ap.add_argument("--ymax", type=float, default=103.0)
    ap.add_argument("--width", type=float, default=11.0)
    ap.add_argument("--height", type=float, default=5.6)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out-stem", default="scatter_scope")
    ap.add_argument("--fmt", choices=["png", "svg", "pdf"], default="pdf")
    args = ap.parse_args()

    specs = [parse_report_spec(s) for s in args.report]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok = plot_scope_scatter(specs, args.out_dir / args.out_stem, args.fmt,
                            min_n=args.min_n,
                            figsize=(args.width, args.height),
                            ymin=args.ymin, ymax=args.ymax,
                            connect=not args.no_connect,
                            annotate_n=not args.no_n)
    if not ok:
        raise SystemExit("no figure produced")


if __name__ == "__main__":
    main()
