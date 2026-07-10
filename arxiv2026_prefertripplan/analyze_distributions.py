#!/usr/bin/env python3
"""Distribution analysis for the preference-augmented TravelPlanner JSONL.

Reads the augmented file (default: ``prefertripplan.jsonl``)
and emits:

  analysis/distribution_analysis.txt   : human-readable tables.
  analysis/plots/*.png                 : bar / grouped-bar charts.

Counts surfaced:

  * queries per difficulty level
  * pairing_type per level (single / independent / overlapping)
  * preference paradigm (atomic / composite / conditional / lexicographic /
    temporal.atmost_once / temporal.sometime_before / temporal.always_within)
    overall and per level
  * entity (Accommodation / Attraction / Day / Restaurant / Transportation)
    overall and per level
  * entity.attribute combined labels (e.g. Restaurant.cost)
  * bank-entry trace usage per paradigm

Run:

  python analyze_distributions.py --in prefertripplan.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEVEL_ORDER = ["easy", "medium", "hard"]
PARADIGM_ORDER = [
    "AtomicPreference",
    "CompositePreference",
    "NumericPreference",
    "ConditionalPreference",
    "LexicographicPreference",
    "CompensatoryPreference",
    "TemporalPreference.always",
    "TemporalPreference.sometime",
    "TemporalPreference.within",
    "TemporalPreference.atmost_once",
    "TemporalPreference.sometime_before",
    "TemporalPreference.sometime_after",
    "TemporalPreference.always_within",
    "TemporalPreference.hold_during",
    "TemporalPreference.hold_after",
    "ScopedPreference",
]
PARADIGM_TOP_ORDER = [
    "AtomicPreference", "CompositePreference", "NumericPreference",
    "ConditionalPreference", "LexicographicPreference",
    "CompensatoryPreference", "TemporalPreference", "ScopedPreference",
]
ENTITY_ORDER = ["Accommodation", "Attraction", "Day", "Restaurant", "Transportation"]
PAIRING_ORDER = ["single", "independent", "overlapping"]
PAIRING_SUB_ORDER = ["competing", "non_competing"]


# --------------------------------------------------------------------------- #
# Template traversal                                                          #
# --------------------------------------------------------------------------- #

def _atomic_ea(node: dict[str, Any]) -> tuple[str | None, str | None]:
    """Read entity_type/attribute from either direct or {class, template:{...}} nesting."""
    if "template" in node and isinstance(node["template"], dict) and "entity_type" in node["template"]:
        t = node["template"]
        return t.get("entity_type"), t.get("attribute")
    return node.get("entity_type"), node.get("attribute")


def iter_entity_attrs(paradigm: str, template: dict[str, Any]) -> Iterable[tuple[str, str]]:
    def emit(node: dict[str, Any]) -> Iterable[tuple[str, str]]:
        e, a = _atomic_ea(node)
        if e and a:
            yield (e, a)

    if paradigm == "AtomicPreference":
        e = template.get("entity_type"); a = template.get("attribute")
        if e and a:
            yield (e, a)
    elif paradigm == "CompositePreference":
        for c in template.get("children", []):
            yield from emit(c)
    elif paradigm == "NumericPreference":
        e = template.get("entity_type"); a = template.get("attribute")
        if e and a:
            yield (e, a)
    elif paradigm == "ConditionalPreference":
        for k in ("condition", "then_pref"):
            n = template.get(k)
            if n:
                yield from emit(n)
    elif paradigm == "LexicographicPreference":
        for p in template.get("preferences", []):
            yield from emit(p)
    elif paradigm == "CompensatoryPreference":
        for k in ("primary_ap", "margin_ap", "secondary_ap"):
            n = template.get(k)
            if n:
                yield from emit(n)
    elif paradigm == "TemporalPreference":
        for k in ("subject_ap", "reference_ap"):
            n = template.get(k)
            if n:
                yield from emit(n)
    elif paradigm == "ScopedPreference":
        inner = template.get("inner") or {}
        inner_cls = inner.get("class")
        if inner_cls and "template" in inner:
            yield from iter_entity_attrs(inner_cls, inner["template"])
        for sf in template.get("scope_filters", []) or []:
            yield from emit(sf)


def paradigm_key(pref: dict[str, Any]) -> str:
    base = pref.get("paradigm", "?")
    sub = pref.get("subtype")
    return f"{base}.{sub}" if sub else base


def top_paradigm(pref: dict[str, Any]) -> str:
    return pref.get("paradigm", "?")


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #

class Stats:
    def __init__(self) -> None:
        self.n_records = 0
        self.level_counts: Counter[str] = Counter()
        self.pairing_counts: Counter[str] = Counter()
        self.pairing_by_level: dict[str, Counter[str]] = defaultdict(Counter)
        self.pairing_sub_counts: Counter[str] = Counter()
        self.pairing_sub_by_level: dict[str, Counter[str]] = defaultdict(Counter)

        self.paradigm_counts: Counter[str] = Counter()           # with temporal subtype
        self.paradigm_by_level: dict[str, Counter[str]] = defaultdict(Counter)
        self.top_paradigm_counts: Counter[str] = Counter()       # temporal collapsed to one
        self.top_paradigm_by_level: dict[str, Counter[str]] = defaultdict(Counter)

        self.entity_counts: Counter[str] = Counter()
        self.entity_by_level: dict[str, Counter[str]] = defaultdict(Counter)

        self.entity_attr_counts: Counter[str] = Counter()
        self.entity_attr_by_level: dict[str, Counter[str]] = defaultdict(Counter)

        self.trace_counts: Counter[str] = Counter()
        self.trace_by_paradigm: dict[str, Counter[str]] = defaultdict(Counter)

        # paradigm-pair distribution for two-preference records
        self.paradigm_pair_counts: Counter[tuple[str, str]] = Counter()
        self.paradigm_pair_by_level: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)

    def update(self, record: dict[str, Any]) -> None:
        self.n_records += 1
        level = record.get("level", "?")
        prefs: list[dict[str, Any]] = record.get("preferences", []) or []
        if not prefs:
            pairing = "skipped"
        else:
            pairing = record.get("pairing_type") or "single"

        self.level_counts[level] += 1
        self.pairing_counts[pairing] += 1
        self.pairing_by_level[level][pairing] += 1
        sub = record.get("pairing_subtype")
        if sub:
            self.pairing_sub_counts[sub] += 1
            self.pairing_sub_by_level[level][sub] += 1

        pks = [paradigm_key(p) for p in prefs]
        tops = [top_paradigm(p) for p in prefs]
        for p, pk, tp in zip(prefs, pks, tops):
            self.paradigm_counts[pk] += 1
            self.paradigm_by_level[level][pk] += 1
            self.top_paradigm_counts[tp] += 1
            self.top_paradigm_by_level[level][tp] += 1
            seen_entities: set[str] = set()
            seen_ea: set[str] = set()
            for entity, attr in iter_entity_attrs(tp, p["template"]):
                seen_entities.add(entity)
                seen_ea.add(f"{entity}.{attr}")
            for e in seen_entities:
                self.entity_counts[e] += 1
                self.entity_by_level[level][e] += 1
            for ea in seen_ea:
                self.entity_attr_counts[ea] += 1
                self.entity_attr_by_level[level][ea] += 1
            tr = p.get("trace") or "?"
            self.trace_counts[tr] += 1
            self.trace_by_paradigm[pk][tr] += 1
        if len(pks) == 2:
            pair = tuple(sorted(pks))
            self.paradigm_pair_counts[pair] += 1
            self.paradigm_pair_by_level[level][pair] += 1


# --------------------------------------------------------------------------- #
# Plot helpers                                                                #
# --------------------------------------------------------------------------- #

def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def bar_plot(counts: Counter[str] | dict[str, int], title: str, ylabel: str,
             out: Path, order: list[str] | None = None) -> None:
    items: list[tuple[str, int]]
    if order is not None:
        items = [(k, counts.get(k, 0)) for k in order]
    else:
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    width = max(6.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(width, 4.5))
    x = list(range(len(labels)))
    bars = ax.bar(x, vals, color="#4C72B0")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out)


def grouped_bar(by_level: dict[str, Counter[str]], categories: list[str],
                title: str, ylabel: str, out: Path,
                levels: list[str] | None = None) -> None:
    levels = levels or [lv for lv in LEVEL_ORDER if lv in by_level]
    n = len(levels)
    if n == 0:
        return
    x = list(range(len(categories)))
    width = 0.8 / n
    width_in = max(6.5, 0.85 * len(categories))
    fig, ax = plt.subplots(figsize=(width_in, 4.8))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for i, lvl in enumerate(levels):
        vals = [by_level[lvl].get(c, 0) for c in categories]
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        ax.bar(offsets, vals, width=width, label=lvl, color=palette[i % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(title="level")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out)


def trace_heatbars(by_paradigm: dict[str, Counter[str]], out: Path) -> None:
    """One horizontal bar plot per paradigm showing bank-entry usage."""
    paradigms = [p for p in PARADIGM_ORDER if p in by_paradigm]
    if not paradigms:
        return
    rows = len(paradigms)
    fig, axes = plt.subplots(rows, 1, figsize=(9, 1.5 + 1.4 * rows))
    if rows == 1:
        axes = [axes]
    for ax, pk in zip(axes, paradigms):
        c = by_paradigm[pk]
        items = sorted(c.items(), key=lambda kv: int(kv[0].split(":")[-1]))
        labels = [kv[0].split(":")[-1] for kv in items]
        vals = [kv[1] for kv in items]
        ax.bar(labels, vals, color="#4C72B0")
        ax.set_title(f"{pk}  (entries used: {len(labels)})", fontsize=10)
        ax.set_ylabel("count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    axes[-1].set_xlabel("bank id")
    _save(fig, out)


# --------------------------------------------------------------------------- #
# Text report                                                                 #
# --------------------------------------------------------------------------- #

def fmt_table(rows: list[tuple[str, int]], total: int, header: tuple[str, str] = ("key", "count")) -> str:
    if not rows:
        return "  (empty)\n"
    w = max(len(r[0]) for r in rows)
    w = max(w, len(header[0]))
    lines = [f"  {header[0]:<{w}}  {header[1]:>7}  {'%':>6}"]
    lines.append(f"  {'-'*w}  {'-'*7}  {'-'*6}")
    for k, v in rows:
        pct = (100.0 * v / total) if total else 0.0
        lines.append(f"  {k:<{w}}  {v:>7d}  {pct:>5.1f}%")
    return "\n".join(lines) + "\n"


def write_report(stats: Stats, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("Preference Augmentation — Distribution Report\n")
    lines.append("=" * 56 + "\n\n")
    lines.append(f"Total records analyzed: {stats.n_records}\n\n")

    lines.append("[1] Records per difficulty level\n")
    rows = [(lv, stats.level_counts.get(lv, 0)) for lv in LEVEL_ORDER]
    lines.append(fmt_table(rows, stats.n_records, ("level", "count")))
    lines.append("\n")

    lines.append("[2] Pairing type (overall)\n")
    extra_pairing = ["skipped"]
    rows = [(p, stats.pairing_counts.get(p, 0)) for p in PAIRING_ORDER + extra_pairing
            if stats.pairing_counts.get(p, 0) > 0]
    lines.append(fmt_table(rows, stats.n_records, ("pairing", "count")))
    lines.append("\n")

    lines.append("[2b] Overlap sub-tag distribution\n")
    rows = [(s, stats.pairing_sub_counts.get(s, 0)) for s in PAIRING_SUB_ORDER
            if stats.pairing_sub_counts.get(s, 0) > 0]
    tot_sub = sum(stats.pairing_sub_counts.values())
    lines.append(fmt_table(rows, tot_sub, ("subtype", "count")))
    lines.append("\n")

    lines.append("[3] Pairing type per level\n")
    for lv in LEVEL_ORDER:
        denom = stats.level_counts.get(lv, 0)
        if not denom:
            continue
        lines.append(f"  -- {lv} --\n")
        sub = stats.pairing_by_level.get(lv, Counter())
        rows = [(p, sub.get(p, 0)) for p in PAIRING_ORDER + ["skipped"] if sub.get(p, 0) > 0]
        lines.append(fmt_table(rows, denom, ("pairing", "count")))
        sub2 = stats.pairing_sub_by_level.get(lv, Counter())
        if sum(sub2.values()):
            rows = [(s, sub2.get(s, 0)) for s in PAIRING_SUB_ORDER if sub2.get(s, 0) > 0]
            lines.append(fmt_table(rows, sum(sub2.values()), ("subtype", "count")))
    lines.append("\n")

    n_top = sum(stats.top_paradigm_counts.values())
    lines.append(f"[3b] Top paradigm distribution (Temporal counted as 1, n={n_top})\n")
    rows = [(p, stats.top_paradigm_counts.get(p, 0)) for p in PARADIGM_TOP_ORDER
            if stats.top_paradigm_counts.get(p, 0) > 0]
    lines.append(fmt_table(rows, n_top, ("paradigm", "count")))
    lines.append("\n")

    lines.append("[3c] Top paradigm per level\n")
    for lv in LEVEL_ORDER:
        sub = stats.top_paradigm_by_level.get(lv, Counter())
        total = sum(sub.values())
        if not total:
            continue
        lines.append(f"  -- {lv} (n={total}) --\n")
        rows = [(p, sub.get(p, 0)) for p in PARADIGM_TOP_ORDER if sub.get(p, 0) > 0]
        lines.append(fmt_table(rows, total, ("paradigm", "count")))
    lines.append("\n")

    n_prefs = sum(stats.paradigm_counts.values())
    lines.append(f"[4] Preference paradigm (overall over {n_prefs} preferences)\n")
    rows = [(p, stats.paradigm_counts.get(p, 0)) for p in PARADIGM_ORDER
            if stats.paradigm_counts.get(p, 0) > 0]
    extra = [(p, c) for p, c in stats.paradigm_counts.items()
             if p not in PARADIGM_ORDER]
    rows.extend(sorted(extra))
    lines.append(fmt_table(rows, n_prefs, ("paradigm", "count")))
    lines.append("\n")

    lines.append("[5] Preference paradigm per level\n")
    for lv in LEVEL_ORDER:
        sub = stats.paradigm_by_level.get(lv, Counter())
        total = sum(sub.values())
        if not total:
            continue
        lines.append(f"  -- {lv} (n={total}) --\n")
        rows = [(p, sub.get(p, 0)) for p in PARADIGM_ORDER if sub.get(p, 0) > 0]
        lines.append(fmt_table(rows, total, ("paradigm", "count")))
    lines.append("\n")

    n_ea = sum(stats.entity_counts.values())
    lines.append(f"[6] Entity coverage (occurrences across {n_ea} pref-touches)\n")
    rows = [(e, stats.entity_counts.get(e, 0)) for e in ENTITY_ORDER
            if stats.entity_counts.get(e, 0) > 0]
    lines.append(fmt_table(rows, n_ea, ("entity", "count")))
    lines.append("\n")

    lines.append("[7] Entity per level\n")
    for lv in LEVEL_ORDER:
        sub = stats.entity_by_level.get(lv, Counter())
        total = sum(sub.values())
        if not total:
            continue
        lines.append(f"  -- {lv} (n={total}) --\n")
        rows = [(e, sub.get(e, 0)) for e in ENTITY_ORDER if sub.get(e, 0) > 0]
        lines.append(fmt_table(rows, total, ("entity", "count")))
    lines.append("\n")

    n_ea_full = sum(stats.entity_attr_counts.values())
    lines.append(f"[8] Entity.attribute coverage (n={n_ea_full})\n")
    rows = sorted(stats.entity_attr_counts.items(),
                  key=lambda kv: (-kv[1], kv[0]))
    lines.append(fmt_table(rows, n_ea_full, ("entity.attribute", "count")))
    lines.append("\n")

    lines.append("[9] Entity.attribute per level\n")
    for lv in LEVEL_ORDER:
        sub = stats.entity_attr_by_level.get(lv, Counter())
        total = sum(sub.values())
        if not total:
            continue
        lines.append(f"  -- {lv} (n={total}) --\n")
        rows = sorted(sub.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append(fmt_table(rows, total, ("entity.attribute", "count")))
    lines.append("\n")

    lines.append("[10] Bank-entry usage per paradigm (trace → count)\n")
    for pk in PARADIGM_ORDER:
        sub = stats.trace_by_paradigm.get(pk, Counter())
        if not sub:
            continue
        used = len(sub)
        lines.append(f"  -- {pk}  (distinct entries used: {used}) --\n")
        rows = sorted(sub.items(),
                      key=lambda kv: int(kv[0].split(":")[-1]))
        lines.append(fmt_table(rows, sum(sub.values()),
                               ("trace", "count")))
    lines.append("\n")

    lines.append("[11] Paradigm-pair frequency (two-pref records)\n")
    rows_pair = sorted(stats.paradigm_pair_counts.items(),
                       key=lambda kv: (-kv[1], kv[0]))
    rows = [(f"{a}  +  {b}", c) for (a, b), c in rows_pair]
    total_pairs = sum(stats.paradigm_pair_counts.values())
    lines.append(fmt_table(rows, total_pairs, ("pair", "count")))
    lines.append("\n")

    out.write_text("".join(lines))


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def main() -> None:
    project = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=project / "prefertripplan.jsonl")
    ap.add_argument("--out-dir", type=Path, default=project / "analysis")
    args = ap.parse_args()

    stats = Stats()
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats.update(json.loads(line))

    out_dir = args.out_dir
    plots = out_dir / "plots"
    write_report(stats, out_dir / "distribution_analysis.txt")

    # Plots
    bar_plot(stats.level_counts, "Queries per difficulty level",
             "queries", plots / "level_counts.png", order=LEVEL_ORDER)
    bar_plot(stats.pairing_counts, "Pairing type (overall)",
             "queries", plots / "pairing_overall.png", order=PAIRING_ORDER)

    pairing_categories = [p for p in PAIRING_ORDER
                          if any(stats.pairing_by_level[lv].get(p, 0)
                                 for lv in LEVEL_ORDER)]
    grouped_bar(stats.pairing_by_level, pairing_categories,
                "Pairing type by level", "queries",
                plots / "pairing_by_level.png")

    if sum(stats.pairing_sub_counts.values()):
        bar_plot(stats.pairing_sub_counts,
                 "Overlap sub-tag (competing vs non_competing)",
                 "overlap pairs", plots / "pairing_subtype_overall.png",
                 order=PAIRING_SUB_ORDER)
        grouped_bar(stats.pairing_sub_by_level, PAIRING_SUB_ORDER,
                    "Overlap sub-tag by level", "overlap pairs",
                    plots / "pairing_subtype_by_level.png")

    top_paradigm_categories = [p for p in PARADIGM_TOP_ORDER
                               if stats.top_paradigm_counts.get(p, 0) > 0]
    bar_plot(stats.top_paradigm_counts,
             "Top paradigm (Temporal counted as one)",
             "preferences", plots / "top_paradigm_overall.png",
             order=top_paradigm_categories)
    grouped_bar(stats.top_paradigm_by_level, top_paradigm_categories,
                "Top paradigm by level", "preferences",
                plots / "top_paradigm_by_level.png")

    paradigm_categories = [p for p in PARADIGM_ORDER
                           if stats.paradigm_counts.get(p, 0) > 0]
    bar_plot(stats.paradigm_counts, "Preference paradigm (overall)",
             "preferences", plots / "paradigm_overall.png",
             order=paradigm_categories)
    grouped_bar(stats.paradigm_by_level, paradigm_categories,
                "Preference paradigm by level", "preferences",
                plots / "paradigm_by_level.png")

    entity_categories = [e for e in ENTITY_ORDER
                         if stats.entity_counts.get(e, 0) > 0]
    bar_plot(stats.entity_counts, "Entity coverage (overall)",
             "preference touches", plots / "entity_overall.png",
             order=entity_categories)
    grouped_bar(stats.entity_by_level, entity_categories,
                "Entity coverage by level", "preference touches",
                plots / "entity_by_level.png")

    ea_categories = [k for k, _ in sorted(stats.entity_attr_counts.items(),
                                          key=lambda kv: (-kv[1], kv[0]))]
    bar_plot(stats.entity_attr_counts, "Entity.attribute coverage",
             "preference touches", plots / "entity_attribute.png",
             order=ea_categories)
    grouped_bar(stats.entity_attr_by_level, ea_categories,
                "Entity.attribute coverage by level",
                "preference touches",
                plots / "entity_attribute_by_level.png")

    trace_heatbars(stats.trace_by_paradigm, plots / "bank_trace_usage.png")

    print(f"Wrote {out_dir / 'distribution_analysis.txt'}")
    print(f"Wrote plots to {plots}/")


if __name__ == "__main__":
    main()
