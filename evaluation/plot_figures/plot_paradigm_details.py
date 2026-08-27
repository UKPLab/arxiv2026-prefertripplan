#!/usr/bin/env python3
"""plot_paradigm_details.py — per-model matplotlib figures from the raw
JSON paradigm-detail dump emitted by ``analyze_performance.py --raw-out``.

Produces one figure per FIG-tagged analysis under ``--out-dir``:

    A1_atomic_partial_credit            # A1 histogram
    L1_lex_per_tier_pass                # L1 per-tier bar
    N1_numeric_quantile_position        # N1 overlaid histograms by direction
    CO1_compensatory_tier_fractions     # CO1 stacked bar
    CO3_compensatory_by_cross_entity    # CO3 grouped stacked bars
    CO4_compensatory_by_drift           # CO4 grouped stacked bars
    T4_temporal_within_first_pos        # T4 histogram
    T6_temporal_timed_day_curve         # T6 line
    T11_temporal_always_within          # T11 histogram

Every plotter is defensive: if the source extract is empty (e.g. the
detailed JSONL was produced under the legacy string-details schema),
the figure is skipped rather than crashing.

The cross-model pairing slope chart lives in its own script,
``plot_pairing_slope.py``, so it can be run independently of these.

Usage:
    python3 evaluation/plot_figures/plot_paradigm_details.py \\
        --raw-in  evaluation/nemotron_test/analyze_nemotron.raw.json \\
        --out-dir evaluation/figures_nemotron/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import save, mean     # noqa: E402


# --------------------------------------------------------------------------- #
# Palettes                                                                    #
# --------------------------------------------------------------------------- #

# Palettes.
_TIER_COLORS = {
    "tier1":        "#2ca02c",   # ideal        (green)
    "tier2_comp":   "#98df8a",   # compensated  (light green)
    "tier2_uncomp": "#ff7f0e",   # uncompensated (amber)
    "tier3":        "#d62728",   # below margin (red)
}
_TIERS = ("tier1", "tier2_comp", "tier2_uncomp", "tier3")

_DRIFT_COLORS = {
    "aligned":    "#2ca02c",
    "omission":   "#ff7f0e",
    "inversion":  "#d62728",
}



# --------------------------------------------------------------------------- #
# Plotters                                                                    #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Plotters                                                                    #
# --------------------------------------------------------------------------- #

def plot_A1(data: dict, out_path: Path, fmt: str) -> bool:
    """A1 — Atomic [all]-scope partial-credit histogram."""
    values = data.get("values") or []
    if not values:
        return False
    fig, ax = plt.subplots()
    ax.hist(values, bins=20, range=(0, 1),
             edgecolor="black", alpha=0.75, color="#4c78a8")
    m = mean(values)
    ax.axvline(m, color="red", linestyle="--", linewidth=1.2,
               label=f"mean = {m:.3f}")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("passed_entities  (partial-credit ratio)")
    ax.set_ylabel("# Atomic-[all] preferences")
    ax.set_title(f"A1 · Atomic [all]-scope partial-credit distribution "
                 f"(n = {len(values)})")
    save(fig, out_path, fmt)
    return True


def plot_L1(data: dict, out_path: Path, fmt: str) -> bool:
    """L1 — Lex per-tier pass-rate bar."""
    if not data.get("n"):
        return False
    tp = data.get("tier_pass") or []
    tt = data.get("tier_total") or []
    if not tp or not tt:
        return False
    rates = [(p / t) if t else 0.0 for p, t in zip(tp, tt)]
    xs = [f"P{i + 1}" for i in range(len(rates))]
    fig, ax = plt.subplots()
    bars = ax.bar(xs, rates, color="#4c78a8", edgecolor="black")
    for i, (bar, r, p, t) in enumerate(zip(bars, rates, tp, tt)):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02,
                f"{r * 100:.1f}%\n({p}/{t})",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("pass rate")
    ax.set_xlabel("tier (P1 = highest priority)")
    ax.set_title(f"L1 · Lex per-tier pass rate  "
                 f"(n_records = {data.get('n', 0)})")
    save(fig, out_path, fmt)
    return True


def plot_N1(data: dict, out_path: Path, fmt: str) -> bool:
    """N1 — Numeric quantile-position density, overlaid by direction."""
    max_v = data.get("max") or []
    min_v = data.get("min") or []
    if not max_v and not min_v:
        return False
    fig, ax = plt.subplots()
    if max_v:
        ax.hist(max_v, bins=20, range=(0, 1), alpha=0.55,
                 color="#4c78a8", edgecolor="black",
                 label=f"direction = max  (n={len(max_v)}, "
                       f"mean={mean(max_v):.3f})")
    if min_v:
        ax.hist(min_v, bins=20, range=(0, 1), alpha=0.55,
                 color="#e45756", edgecolor="black",
                 label=f"direction = min  (n={len(min_v)}, "
                       f"mean={mean(min_v):.3f})")
    ax.axvline(1.0, color="#4c78a8", linestyle=":", alpha=0.8,
               label="max target (extreme)")
    ax.axvline(0.0, color="#e45756", linestyle=":", alpha=0.8,
               label="min target (extreme)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("quantile_position   (0 = worst-end anchor, "
                  "1 = best-end anchor)")
    ax.set_ylabel("# Numeric preferences")
    ax.set_title("N1 · Numeric quantile-position density by direction")
    ax.legend(loc="upper center", fontsize=8, frameon=False)
    save(fig, out_path, fmt)
    return True


def _stacked_tier_bar(ax, y_positions, key_frac_map: dict[str, dict],
                       keys: list[str]) -> None:
    """Draw one horizontal stacked bar per key, using tier colors.
    y_positions is a list of y-locs aligned with ``keys``."""
    for y, key in zip(y_positions, keys):
        left = 0.0
        frac = key_frac_map.get(key, {})
        for t in _TIERS:
            v = float(frac.get(t, 0.0))
            ax.barh(y, v, left=left, color=_TIER_COLORS[t],
                    edgecolor="black", linewidth=0.5)
            if v >= 0.03:
                ax.text(left + v / 2, y, f"{v * 100:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color=("white" if v > 0.4 else "black"))
            left += v


def _tier_legend(ax) -> None:
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_TIER_COLORS[t], edgecolor="black",
                     label=t) for t in _TIERS]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.30), ncol=4, frameon=False)


def plot_CO1(data: dict, out_path: Path, fmt: str) -> bool:
    """CO1 — Compensatory aggregate tier fractions (single stacked bar)."""
    frac = data.get("aggregate_frac") or {}
    if not any(frac.get(t, 0) for t in _TIERS):
        return False
    fig, ax = plt.subplots(figsize=(8.0, 2.2))
    _stacked_tier_bar(ax, [0], {"all": frac}, ["all"])
    ax.set_yticks([0]); ax.set_yticklabels(["all Comp entities"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of primary-entity occurrences")
    ax.set_title(f"CO1 · Compensatory aggregate tier fractions  "
                 f"(n_entities = {data.get('n_entities_total', 0)})")
    ax.grid(False)
    _tier_legend(ax)
    save(fig, out_path, fmt)
    return True


def plot_CO3(data: dict, out_path: Path, fmt: str) -> bool:
    """CO3 — Compensatory tier fractions × cross_entity."""
    by = data.get("by_cross_entity_frac") or {}
    keys = [k for k in ("same", "cross") if k in by]
    if not keys:
        return False
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    _stacked_tier_bar(ax, list(range(len(keys))), by, keys)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{k} entity" for k in keys])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of primary-entity occurrences")
    ax.set_title("CO3 · Compensatory tier fractions × cross_entity")
    ax.grid(False)
    _tier_legend(ax)
    save(fig, out_path, fmt)
    return True


def plot_CO4(data: dict, out_path: Path, fmt: str) -> bool:
    """CO4 — Compensatory tier fractions × profile_drift."""
    by = data.get("by_drift_frac") or {}
    keys = [k for k in ("aligned", "omission", "inversion") if k in by]
    if not keys:
        return False
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    _stacked_tier_bar(ax, list(range(len(keys))), by, keys)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of primary-entity occurrences")
    ax.set_title("CO4 · Compensatory tier fractions × profile_drift  "
                 "(domination-dilution: expect T1 to shrink under "
                 "inversion)")
    ax.grid(False)
    _tier_legend(ax)
    save(fig, out_path, fmt)
    return True


def plot_T4(data: dict, out_path: Path, fmt: str) -> bool:
    """T4 — Temporal.within first-subject-position density."""
    values = data.get("normalised") or []
    if not values:
        return False
    fig, ax = plt.subplots()
    ax.hist(values, bins=20, range=(0, 1),
             edgecolor="black", alpha=0.75, color="#4c78a8")
    m = mean(values)
    ax.axvline(m, color="red", linestyle="--", linewidth=1.2,
               label=f"mean = {m:.3f}")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("first-subject position  (0 = group start, 1 = group end)")
    ax.set_ylabel("# Temporal.within groups")
    ax.set_title(f"T4 · Temporal.within first-subject-position density  "
                 f"(n = {len(values)}; "
                 f"no-subject: {data.get('n_no_subject_groups', 0)})")
    save(fig, out_path, fmt)
    return True


def plot_T6(data: dict, out_path: Path, fmt: str) -> bool:
    """T6 — Temporal.timed per-window-day pass curve."""
    curve = data.get("window_pos_curve") or []
    counts = data.get("window_pos_counts") or []
    if not curve:
        return False
    fig, ax = plt.subplots()
    xs = list(range(len(curve)))
    ax.plot(xs, curve, marker="o", color="#4c78a8", linewidth=1.8)
    ax.fill_between(xs, 0, curve, alpha=0.18, color="#4c78a8")
    for x, v, c in zip(xs, curve, counts):
        ax.text(x, v + 0.03, f"n={c}", ha="center",
                 fontsize=8, color="gray")
    ax.set_xlabel("window position  (0 = window start)")
    ax.set_ylabel("mean day_score  (0-1)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"T6 · Temporal.timed per-window-day pass curve  "
                 f"(n_groups = {data.get('n_groups', 0)})")
    save(fig, out_path, fmt)
    return True


def plot_T11(data: dict, out_path: Path, fmt: str) -> bool:
    """T11 — Temporal.pair always_within passed_subjects density."""
    values = data.get("values") or []
    if not values:
        return False
    fig, ax = plt.subplots()
    ax.hist(values, bins=20, range=(0, 1),
             edgecolor="black", alpha=0.75, color="#4c78a8")
    m = mean(values)
    ax.axvline(m, color="red", linestyle="--", linewidth=1.2,
               label=f"mean = {m:.3f}")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("passed_subjects  (fraction of subject triggers "
                  "with in-window reference)")
    ax.set_ylabel("# always_within groups")
    ax.set_title(f"T11 · Temporal.pair always_within passed_subjects  "
                 f"(n_groups = {data.get('n_groups', 0)})")
    save(fig, out_path, fmt)
    return True


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# ----------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

_PLOTTERS = [
    ("A1",  "A1_atomic_partial_credit",           "A1_atomic_partial_credit",       plot_A1),
    ("L1",  "L1_L2_lex_tiers",                    "L1_lex_per_tier_pass",           plot_L1),
    ("N1",  "N1_numeric_quantile_position",       "N1_numeric_quantile_position",   plot_N1),
    ("CO1", "CO_compensatory_tiers",              "CO1_compensatory_tier_fractions", plot_CO1),
    ("CO3", "CO_compensatory_tiers",              "CO3_compensatory_by_cross_entity", plot_CO3),
    ("CO4", "CO_compensatory_tiers",              "CO4_compensatory_by_drift",      plot_CO4),
    ("T4",  "T4_temporal_within_first_pos",       "T4_temporal_within_first_pos",   plot_T4),
    ("T6",  "T6_temporal_timed_day_curve",        "T6_temporal_timed_day_curve",    plot_T6),
    ("T11", "T11_temporal_always_within",         "T11_temporal_always_within",     plot_T11),
]



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-in", type=Path, required=True,
                    help="JSON file emitted by `analyze_performance.py --raw-out`.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory where the figures are written.")
    ap.add_argument("--fmt", choices=("png", "svg", "pdf"), default="pdf",
                    help="Figure format (default: pdf).")
    args = ap.parse_args()

    with args.raw_in.open() as f:
        data = json.load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[in]  {args.raw_in}")
    print(f"[out] {args.out_dir}/  (fmt={args.fmt})")

    made = skipped = 0
    for tag, data_key, out_stem, fn in _PLOTTERS:
        section = data.get(data_key) or {}
        out_path = args.out_dir / out_stem
        ok = fn(section, out_path, args.fmt)
        if ok:
            print(f"  [{tag:>3s}] wrote {out_path.with_suffix('.' + args.fmt).name}")
            made += 1
        else:
            print(f"  [{tag:>3s}] skipped (empty source data)")
            skipped += 1

    print(f"\n[done] {made} figure(s) written, {skipped} skipped.")


if __name__ == "__main__":
    main()
