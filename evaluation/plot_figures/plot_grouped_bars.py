#!/usr/bin/env python3
"""plot_grouped_bars.py — cross-model grouped bar charts of preference
pass rate over a categorical axis.

Reads the plain-text ``analyze_*.txt`` reports produced by
``evaluation/analyze_performance.py`` for several model x split runs and
draws one bar per (model, split) within each category group.

    python3 evaluation/plot_figures/plot_grouped_bars.py --chart paradigm \\
        --report evaluation/nemotron_test/analyze_nemotron.txt \\
        --report evaluation/nemotron_test_large/analyze_nemotron.txt \\
        ... \\
        --out-dir evaluation/figures_crossmodel/

``--chart``      pairing | paradigm | temporal
``--bar-width``  per-bar width in x-data units.  Default: derived from the
                 series count so the group footprint stays constant as
                 models are added; pass a value to override.
``--drop-split`` exclude a split entirely, e.g. ``--drop-split test``

Bars are ordered SPLIT-MAJOR within each group: every model's ``test``
bar first, then every model's ``test_large`` bar.  That puts the two
split blocks side by side so a split-level shift reads as a block-level
shift rather than having to be traced across alternating bars.

Model identity is colour; split is fill weight AND hatch together, so
the distinction survives greyscale printing.  Both come from
``_common.py``, shared with every other figure in this directory.

Note on the truncated baseline: ``--ymin`` defaults to 40 because the
pass rates occupy roughly the 40-100 band and a zero-based axis spends
half its height empty.  Bar length then encodes value ABOVE the
baseline, not value itself, so the bars' *ratios* are not meaningful --
only their ordering and their differences.  State the baseline in the
caption.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for _p in (_HERE, _PARENT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import (save, CHARTS, SPLIT_STYLE, SPLIT_ORDER,      # noqa: E402
                     display_model, model_colours, order_models,
                     parse_report_spec, metric_from_report)


# Fraction of the 1.0 category pitch occupied by a group's bars; the
# remainder is the gap between adjacent groups.  0.72 reproduces the
# previous four-model look (8 series x 0.09).
_GROUP_SPAN = 0.72


def plot_grouped_bars(report_specs: list[tuple[Path, str, str]],
                      out_path: Path, fmt: str,
                      axis_spec: list[tuple[str, str, str]],
                      *,
                      column: str = "pass%",
                      metric_label: str = "Preference pass rate (%)",
                      title: str = "",
                      xlabel: str = "",
                      bar_width: float | None = None,
                      figsize: tuple[float, float] = (12.0, 5.4),
                      ymin: float = 40.0,
                      ymax: float = 100.0,
                      show_values: bool = True,
                      drop_splits: tuple[str, ...] = ()) -> bool:
    """Grouped bars: one group per category, one bar per (model, split),
    ordered split-major within the group."""
    try:
        from latex import _parse_report
    except ImportError as e:                       # pragma: no cover
        print(f"  [bars] cannot import latex._parse_report ({e}); skipped")
        return False

    parsed: dict[tuple[str, str], object] = {}
    for path, model, split in report_specs:
        if split in drop_splits:
            continue
        if not path.exists():
            print(f"  [bars] missing {path}; skipped")
            continue
        parsed[(model, split)] = _parse_report(path, f"{model}/{split}")
    if not parsed:
        return False

    models, splits = [], []
    for (model, split) in parsed:
        if model not in models:
            models.append(model)
        if split not in splits:
            splits.append(split)
    models = order_models(models)
    splits.sort(key=lambda s: SPLIT_ORDER.index(s) if s in SPLIT_ORDER else 99)
    colour = model_colours(models)

    # SPLIT-MAJOR: all models' test, then all models' test_large.
    series = [(m, s) for s in splits for m in models if (m, s) in parsed]
    n = len(series)

    labels = [lbl for lbl, _s, _b in axis_spec]
    xs = np.arange(len(axis_spec), dtype=float)
    # Bar width defaults to a FRACTION OF THE GROUP PITCH rather than a
    # fixed constant.  Categories sit 1.0 apart, so a hardcoded width
    # silently eats the inter-group gap as models are added: at 0.09 the
    # group spans 0.72 with four models but 0.90 with five, leaving only
    # 0.10 of daylight.  Deriving it keeps _GROUP_SPAN of the pitch used
    # and the remainder as the gap, whatever the model count.  An
    # explicit --bar-width still overrides.
    if bar_width is None:
        bar_width = _GROUP_SPAN / max(n, 1)
    offsets = (np.arange(n) - (n - 1) / 2.0) * bar_width

    fig, ax = plt.subplots(figsize=figsize)
    below: list[tuple[str, str, float]] = []

    for (model, split), off in zip(series, offsets):
        report = parsed[(model, split)]
        vals = [metric_from_report(report, sec, bkt, column)
                for _lbl, sec, bkt in axis_spec]
        xpos = [x + off for x, v in zip(xs, vals) if v is not None]
        yval = [v for v in vals if v is not None]
        if not yval:
            continue
        st = SPLIT_STYLE.get(split, SPLIT_STYLE["test"])
        # A truncated baseline silently swallows anything below it -- the
        # bar height goes negative and matplotlib draws nothing at all.
        # Clamp those to the baseline and flag them with a caret so an
        # under-baseline value is visibly present rather than missing.
        heights = [max(v - ymin, 0.0) for v in yval]
        # Alpha is baked into the facecolor so the border and hatch stay
        # at full strength while only the fill goes translucent.
        ax.bar(xpos, heights, bottom=ymin,
               width=bar_width * 0.88,
               facecolor=to_rgba(colour[model], st["face_alpha"]),
               hatch=st["hatch"], edgecolor=colour[model],
               linewidth=st.get("lw", 0.0), zorder=2)
        for x, v in zip(xpos, yval):
            if v < ymin:
                below.append((model, split, v))
                ax.plot([x], [ymin], marker="v", markersize=4.5,
                        color=colour[model], clip_on=False, zorder=5)
        if show_values:
            for x, v in zip(xpos, yval):
                # Under-baseline labels sit just above the axis, since
                # their bar has no height to sit on top of.
                y_at = max(v, ymin)
                ax.annotate(f"{v:.1f}", xy=(x, y_at), xytext=(0, 4.5),
                            textcoords="offset points", rotation=90,
                            ha="center", va="bottom", fontsize=5.5,
                            color=colour[model], zorder=4)

    if below:
        print(f"  [bars] {len(below)} value(s) fall below the "
              f"ymin={ymin:g} baseline and are shown as carets:")
        for m, sp, v in below:
            print(f"           {m:<18s} {sp:<11s} {v:.1f}")
        print(f"  [bars] consider a lower --ymin so these are drawn as bars.")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_xlim(-0.5, len(axis_spec) - 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_label)
    ax.set_title(title, pad=34)
    ax.set_ylim(ymin, ymax * 1.12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6)

    # Colour = model, fill+hatch = split.  Both legends sit above the
    # axes so the plot area stays free for the value labels.
    from matplotlib.patches import Patch
    model_handles = [Patch(facecolor=colour[m], edgecolor=colour[m],
                           label=display_model(m)) for m in models]
    split_handles = [
        Patch(facecolor=to_rgba("0.35", SPLIT_STYLE[s]["face_alpha"]),
              edgecolor="0.35", linewidth=SPLIT_STYLE[s].get("lw", 0.0),
              hatch=SPLIT_STYLE[s]["hatch"], label=s)
        for s in splits]
    leg_model = ax.legend(handles=model_handles, title="Model",
                          loc="lower left", bbox_to_anchor=(0.0, 1.02),
                          ncol=len(model_handles), frameon=False,
                          fontsize=8.5, alignment="left",
                          handlelength=1.6, columnspacing=1.3,
                          handletextpad=0.5, borderpad=0.2)
    leg_model.get_title().set_fontsize(9)
    ax.add_artist(leg_model)
    if len(split_handles) > 1:
        leg_split = ax.legend(handles=split_handles, title="Split",
                              loc="lower right", bbox_to_anchor=(1.0, 1.02),
                              ncol=len(split_handles), frameon=False,
                              fontsize=8.5, alignment="left",
                              handlelength=1.6, columnspacing=1.3,
                              handletextpad=0.5, borderpad=0.2)
        leg_split.get_title().set_fontsize(9)

    save(fig, out_path, fmt)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="append", metavar="PATH[:MODEL:SPLIT]",
                    required=True,
                    help="Plain-text analyze_*.txt report; repeatable.  "
                         "MODEL/SPLIT default to the parent directory name.")
    ap.add_argument("--chart", choices=tuple(CHARTS), default="paradigm",
                    help="Which categorical axis to draw (default: paradigm).")
    ap.add_argument("--bar-width", type=float, default=None,
                    help="Per-bar width in x-data units.  Default: derived "
                         f"as {_GROUP_SPAN}/n_series so the group keeps the "
                         "same footprint regardless of how many models are "
                         "plotted (0.09 for the original 8-series case).")
    ap.add_argument("--drop-split", action="append", default=[],
                    metavar="SPLIT",
                    help="Exclude a split entirely; repeatable.  Used for the "
                         "temporal chart, where the test split carries only "
                         "3-4 preferences per operator and is too small to "
                         "read reliably.")
    ap.add_argument("--metric", default=None,
                    help="Report column to plot (default depends on --chart).")
    ap.add_argument("--metric-label", default=None,
                    help="Y-axis label (default follows --metric).")
    ap.add_argument("--ymin", type=float, default=40.0,
                    help="Baseline of the y-axis (default: 40).  Bar length "
                         "then encodes value above this baseline, so state it "
                         "in the caption.")
    ap.add_argument("--ymax", type=float, default=100.0,
                    help="Top of the data range (default: 100).")
    ap.add_argument("--no-values", action="store_true",
                    help="Omit the per-bar value labels.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory where the figure is written.")
    ap.add_argument("--out-stem", default=None,
                    help="Filename stem (default: bar_<chart>).")
    ap.add_argument("--fmt", choices=("png", "svg", "pdf"), default="pdf",
                    help="Figure format (default: pdf).")
    ap.add_argument("--width", type=float, default=12.0,
                    help="Figure width in inches (default: 12.0).")
    ap.add_argument("--height", type=float, default=5.4,
                    help="Figure height in inches (default: 5.4).")
    args = ap.parse_args()

    specs = [parse_report_spec(s) for s in args.report]
    axis_spec, default_metric, noun = CHARTS[args.chart]
    metric = args.metric or default_metric
    stem   = args.out_stem or f"bar_{args.chart}"
    label  = args.metric_label or (
        "Preference micro pass rate (%)" if metric == "pass%"
        else "Preference macro pass rate (%)" if metric == "pf_M"
        else metric)

    print(f"[in]  {len(specs)} analyze report(s):")
    for path, model, split in specs:
        skip = "  (dropped)" if split in args.drop_split else ""
        print(f"        {model:<20s} {split:<12s} {path}{skip}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / stem
    ok = plot_grouped_bars(specs, out_path, args.fmt, axis_spec,
                           column=metric, metric_label=label,
                           title=f"Preference pass rate across {noun}",
                           xlabel=noun.capitalize(),
                           bar_width=args.bar_width,
                           figsize=(args.width, args.height),
                           ymin=args.ymin, ymax=args.ymax,
                           show_values=not args.no_values,
                           drop_splits=tuple(args.drop_split))
    if ok:
        print(f"[out] {out_path.with_suffix('.' + args.fmt)}")
    else:
        print("[out] skipped (no usable report data)")


if __name__ == "__main__":
    main()
