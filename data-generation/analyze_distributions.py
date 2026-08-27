#!/usr/bin/env python3
"""Distributional analysis of the augmented + profile-rendered dataset.

Reads ``data-generation/prefertripplan.jsonl`` and prints a text report
covering:

  [1]  Records per level, plus pass-through/skipped counts.
  [2]  Preference-paradigm distribution — top-level and (for Temporal)
       per subtype (op).
  [3]  Preferences per record and per level.
  [4]  Bank-id usage per paradigm.  For Temporal, bank ids are keyed by
       the op (bank ids are op-scoped in the source bank).
  [5]  Pairing type + pairing subtype (single/independent/overlapping;
       competing/non_competing) — overall and per level.
  [6]  Level-wise drift-mode distribution (aligned/omission/inversion),
       plus drift proportion averaged over sources.
  [7]  Local-constraint (hard constraint) coverage.
  [8]  Feasibility gate results — flight_only_feasible, budget_multiplier
       distribution, per-city floor shortfalls, transport-cap headroom.
  [9]  Per-record templated_nl_profile / templated_nl_query length stats.
  [10] Paradigm distribution WITHIN COMPETING pairs, split per level.
  [11] Same, for non_competing and independent (per-level).
  [12] Compensatory bank_id landing across (competing/nc/indep/single).
       Plus Compensatory competing pair combos (Comp × partner × level).
  [13] Lexicographic bank_id landing across buckets, plus Lex competing
       pair combos.
  [14] Cross-paradigm competing pair combos (bidirectional).

Also emits ``analysis/summary.txt`` — a concise headline snapshot with
level splits, competing-share per level, paradigm competing counts,
Compensatory/Lex competing detail, and quality checks.

Text-only; no matplotlib dependency.  Run::

  python3 data-generation/analyze_distributions.py
  python3 data-generation/analyze_distributions.py --in <path> --out <path>
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


LEVEL_ORDER = ["easy", "medium", "hard"]

PARADIGM_TOP_ORDER = [
    "AtomicPreference",
    "CompositePreference",
    "NumericPreference",
    "ConditionalPreference",
    "LexicographicPreference",
    "CompensatoryPreference",
    "TemporalPreference",
    "ScopedPreference",
]

TEMPORAL_OP_ORDER = [
    "always", "sometime", "within", "atmost_once",
    "sometime_before", "sometime_after", "always_within",
    "hold_during", "hold_after",
]

PAIRING_ORDER    = ["single", "independent", "overlapping"]
PAIRING_SUB_ORDER = ["competing", "non_competing"]
DRIFT_MODES      = ["aligned", "omission", "inversion"]


# --------------------------------------------------------------------------- #
# Preference introspection                                                    #
# --------------------------------------------------------------------------- #

def paradigm_key(pref: dict[str, Any]) -> str:
    """Full paradigm key, with the Temporal op appended when present."""
    base = pref.get("paradigm", "?")
    if base == "TemporalPreference":
        op = (pref.get("template") or {}).get("op") or pref.get("subtype")
        return f"TemporalPreference.{op}" if op else base
    return base


def bank_key(pref: dict[str, Any]) -> tuple[str, Any]:
    """(paradigm-with-subtype, bank_id) used to name a bank entry uniquely."""
    return (paradigm_key(pref), pref.get("bank_id"))


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #

class Stats:
    def __init__(self) -> None:
        self.n_records = 0
        self.n_pass_through = 0                # 0 preferences
        self.n_with_prefs = 0

        # [1] level counts
        self.level_counts: Counter[str] = Counter()
        self.level_pass_through: Counter[str] = Counter()

        # [2] paradigm counts
        self.paradigm_top: Counter[str] = Counter()
        self.paradigm_top_by_level: dict[str, Counter[str]] = defaultdict(Counter)
        self.paradigm_full: Counter[str] = Counter()          # incl. Temporal.op
        self.paradigm_full_by_level: dict[str, Counter[str]] = defaultdict(Counter)
        self.temporal_op: Counter[str] = Counter()
        self.temporal_op_by_level: dict[str, Counter[str]] = defaultdict(Counter)

        # [3] prefs-per-record
        self.n_prefs_per_record: Counter[int] = Counter()
        self.n_prefs_per_record_by_level: dict[str, Counter[int]] = defaultdict(Counter)

        # [4] bank-id usage
        self.bank_usage: dict[str, Counter[Any]] = defaultdict(Counter)

        # [5] pairing
        self.pairing: Counter[str] = Counter()
        self.pairing_by_level: dict[str, Counter[str]] = defaultdict(Counter)
        self.pairing_sub: Counter[str] = Counter()
        self.pairing_sub_by_level: dict[str, Counter[str]] = defaultdict(Counter)

        # [6] drift
        self.drift: Counter[str] = Counter()
        self.drift_by_level: dict[str, Counter[str]] = defaultdict(Counter)
        self.source_kind_counts: Counter[str] = Counter()
        self.source_drift_fraction: list[float] = []          # per-record

        # [7] local_constraint (hard) coverage
        self.lc_key_counts: Counter[str] = Counter()
        self.lc_records: int = 0

        # [8] feasibility gates
        self.flight_only_feasible: Counter[str] = Counter()   # True/False/None
        self.budget_multipliers: list[float] = []
        self.transport_headroom_fraction: list[float] = []    # transport_cost / budget
        self.solution_pool_shortfalls: int = 0
        self.pool_shortfall_examples: list[dict] = []

        # [9] NL text stats
        self.nl_profile_chars: list[int] = []
        self.nl_query_chars: list[int] = []
        self.nl_profile_empty: int = 0
        self.nl_query_empty: int = 0

        # [10] Paradigm × level × bucket slot counts.  Bucket is one of
        # competing / non_competing / independent / single.  Used to
        # break the competing pool by paradigm-per-level (the piece the
        # standard [5] pairing section reports only in aggregate).
        # Key: (paradigm, level, bucket) -> count.
        self.para_level_bucket: Counter[tuple[str, str, str]] = Counter()

        # [12/13] Per-bank-id landing across buckets, for Comp and Lex.
        # Key: bank_id -> {competing, non_competing, independent, single}.
        self.comp_bid_bucket: dict[Any, Counter[str]] = defaultdict(Counter)
        self.lex_bid_bucket:  dict[Any, Counter[str]] = defaultdict(Counter)

        # Competing pair details: (bank_id, partner_paradigm, partner_bid,
        # level) -> count.  Restricted to pairs of length 2 whose
        # pairing_subtype == "competing".
        self.comp_competing_pairs: Counter[tuple[Any, str, Any, str]] = Counter()
        self.lex_competing_pairs:  Counter[tuple[Any, str, Any, str]] = Counter()

        # [14] Cross-paradigm competing pair combos (bidirectional).
        # Key: paradigm -> Counter of partner_paradigm counts.
        self.para_competing_partners: dict[str, Counter[str]] = defaultdict(Counter)

    # -----------------------------------------------------------------

    def update(self, rec: dict[str, Any]) -> None:
        self.n_records += 1
        level = rec.get("level", "?")
        self.level_counts[level] += 1

        prefs = rec.get("preferences") or []
        n_p = len(prefs)
        self.n_prefs_per_record[n_p] += 1
        self.n_prefs_per_record_by_level[level][n_p] += 1

        if n_p == 0:
            self.n_pass_through += 1
            self.level_pass_through[level] += 1
        else:
            self.n_with_prefs += 1

        # [2] + [4] paradigm & bank-id
        for pref in prefs:
            top = pref.get("paradigm", "?")
            full = paradigm_key(pref)
            self.paradigm_top[top] += 1
            self.paradigm_top_by_level[level][top] += 1
            self.paradigm_full[full] += 1
            self.paradigm_full_by_level[level][full] += 1
            if top == "TemporalPreference":
                op = (pref.get("template") or {}).get("op") or pref.get("subtype")
                if op:
                    self.temporal_op[op] += 1
                    self.temporal_op_by_level[level][op] += 1
            bid = pref.get("bank_id")
            if bid is not None:
                self.bank_usage[full][bid] += 1

        # [5] pairing
        if n_p == 0:
            pairing = "pass_through"
        else:
            pairing = rec.get("pairing_type") or "single"
        self.pairing[pairing] += 1
        self.pairing_by_level[level][pairing] += 1
        sub = rec.get("pairing_subtype")
        if sub:
            self.pairing_sub[sub] += 1
            self.pairing_sub_by_level[level][sub] += 1

        # [6] drift
        drift_mode = rec.get("profile_drift_mode")
        if drift_mode:
            self.drift[drift_mode] += 1
            self.drift_by_level[level][drift_mode] += 1
        trace = (rec.get("profile_trace") or {}).get("sources") or []
        n_src = len(trace)
        n_drift = 0
        for s in trace:
            self.source_kind_counts[s.get("kind") or "?"] += 1
            if s.get("drifted"):
                n_drift += 1
        if n_src > 0:
            self.source_drift_fraction.append(n_drift / n_src)

        # [7] hard constraints
        lc = rec.get("local_constraint") or {}
        active = [k for k, v in lc.items() if v not in (None, "", [])]
        if active:
            self.lc_records += 1
            for k in active:
                self.lc_key_counts[k] += 1

        # [8] feasibility gates
        fm = rec.get("feasibility_metadata") or {}
        fof = fm.get("flight_only_feasible")
        self.flight_only_feasible[repr(fof)] += 1
        bm = rec.get("budget_multiplier")
        if bm is not None:
            self.budget_multipliers.append(float(bm))
        tcost = fm.get("prehoc_transport_cost")
        bused = fm.get("budget_used")
        if tcost is not None and bused:
            try:
                self.transport_headroom_fraction.append(float(tcost) / float(bused))
            except (TypeError, ValueError):
                pass
        shortfalls = (fm.get("per_city_check_breakdown") or {}).get("shortfalls") \
                     if isinstance(fm.get("per_city_check_breakdown"), dict) \
                     else fm.get("solution_pool_shortfalls") or []
        # Some records also carry it top-level.
        if not shortfalls:
            shortfalls = rec.get("solution_pool_shortfalls") or []
        if shortfalls:
            self.solution_pool_shortfalls += 1
            if len(self.pool_shortfall_examples) < 5:
                self.pool_shortfall_examples.append({
                    "id": rec.get("query_id") or rec.get("id"),
                    "shortfalls": shortfalls[:3],
                })

        # [9] NL text
        p_text = rec.get("templated_nl_profile") or ""
        q_text = rec.get("templated_nl_query") or ""
        self.nl_profile_chars.append(len(p_text))
        self.nl_query_chars.append(len(q_text))
        if not p_text.strip(): self.nl_profile_empty += 1
        if not q_text.strip(): self.nl_query_empty += 1

        # [10]-[14] Paradigm/bank landing per (level, bucket) + competing
        # pair details.  Bucket derived from pairing_type / pairing_subtype:
        #   pairing_type == "single"      -> "single"
        #   pairing_type == "independent" -> "independent"
        #   pairing_subtype == "competing"     -> "competing"
        #   pairing_subtype == "non_competing" -> "non_competing"
        # Pass-through records (n_p == 0) are skipped from these buckets.
        pt_ = rec.get("pairing_type")
        ps_ = rec.get("pairing_subtype")
        if n_p > 0:
            if pt_ == "single":
                bucket = "single"
            elif pt_ == "independent":
                bucket = "independent"
            elif ps_ == "competing":
                bucket = "competing"
            elif ps_ == "non_competing":
                bucket = "non_competing"
            else:
                bucket = None
            if bucket is not None:
                for pref in prefs:
                    pa = pref.get("paradigm")
                    if pa is None: continue
                    self.para_level_bucket[(pa, level, bucket)] += 1
                    bid = pref.get("bank_id")
                    if pa == "CompensatoryPreference":
                        self.comp_bid_bucket[bid][bucket] += 1
                    elif pa == "LexicographicPreference":
                        self.lex_bid_bucket[bid][bucket] += 1
                if bucket == "competing" and len(prefs) == 2:
                    p0, p1 = prefs[0], prefs[1]
                    a, b = p0.get("paradigm"), p1.get("paradigm")
                    if a and b:
                        self.para_competing_partners[a][b] += 1
                        self.para_competing_partners[b][a] += 1
                    if "CompensatoryPreference" in (a, b):
                        comp = p0 if a == "CompensatoryPreference" else p1
                        other = p1 if a == "CompensatoryPreference" else p0
                        self.comp_competing_pairs[(
                            comp.get("bank_id"), other.get("paradigm"),
                            other.get("bank_id"), level)] += 1
                    if "LexicographicPreference" in (a, b):
                        lex_ = p0 if a == "LexicographicPreference" else p1
                        other = p1 if a == "LexicographicPreference" else p0
                        self.lex_competing_pairs[(
                            lex_.get("bank_id"), other.get("paradigm"),
                            other.get("bank_id"), level)] += 1


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #

def fmt_kv_table(counts: dict, total: int, *, key_hdr: str = "key",
                 count_hdr: str = "count", order: list | None = None,
                 top_k: int | None = None) -> str:
    if not counts:
        return "  (empty)\n"
    items: list[tuple[Any, int]]
    if order is not None:
        items = [(k, counts.get(k, 0)) for k in order if counts.get(k, 0) > 0]
        # anything ordered but absent from `order` gets appended
        for k, v in counts.items():
            if k not in order:
                items.append((k, v))
    else:
        items = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    if top_k:
        items = items[:top_k]
    w = max(len(str(k)) for k, _ in items)
    w = max(w, len(key_hdr))
    lines = [f"  {key_hdr:<{w}}  {count_hdr:>7}  {'%':>6}",
             f"  {'-'*w}  {'-'*7}  {'-'*6}"]
    for k, v in items:
        pct = 100.0 * v / total if total else 0.0
        lines.append(f"  {str(k):<{w}}  {v:>7d}  {pct:>5.1f}%")
    return "\n".join(lines) + "\n"


def fmt_grouped_by_level(by_level: dict, categories: list,
                         *, key_hdr: str = "key") -> str:
    levels = [lv for lv in LEVEL_ORDER if lv in by_level]
    if not levels:
        return "  (empty)\n"
    if categories:
        used = [c for c in categories
                if any(by_level[lv].get(c, 0) > 0 for lv in levels)]
        # tail — anything not in the fixed order
        seen = set(categories)
        for lv in levels:
            for k in by_level[lv].keys():
                if k not in seen:
                    used.append(k); seen.add(k)
    else:
        used = sorted({k for lv in levels for k in by_level[lv].keys()})
    if not used:
        return "  (empty)\n"
    w = max(len(str(k)) for k in used)
    w = max(w, len(key_hdr))
    header = f"  {key_hdr:<{w}}" + "".join(f"  {lv:>7}" for lv in levels) + f"  {'total':>7}"
    sep    = f"  {'-'*w}"      + "".join(f"  {'-'*7}"    for _ in levels) + f"  {'-'*7}"
    lines = [header, sep]
    for k in used:
        row_vals = [by_level[lv].get(k, 0) for lv in levels]
        total = sum(row_vals)
        lines.append(f"  {str(k):<{w}}"
                     + "".join(f"  {v:>7d}" for v in row_vals)
                     + f"  {total:>7d}")
    return "\n".join(lines) + "\n"


def fmt_stats(values: list[float], unit: str = "") -> str:
    if not values:
        return "  (no samples)\n"
    q = statistics.quantiles(values, n=100) if len(values) > 3 else None
    def _pct(p): return q[p-1] if q else values[0]
    return (f"  n={len(values)}  min={min(values):.3f}{unit}  "
            f"p25={_pct(25):.3f}{unit}  median={statistics.median(values):.3f}{unit}  "
            f"p75={_pct(75):.3f}{unit}  p95={_pct(95):.3f}{unit}  "
            f"max={max(values):.3f}{unit}  mean={statistics.fmean(values):.3f}{unit}\n")


def write_report(stats: Stats, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    push = lines.append

    push("PreferTripPlan (data-generation) — Distributional Report\n")
    push("=" * 60 + "\n\n")
    push(f"Records analyzed          : {stats.n_records}\n")
    push(f"Records with preferences  : {stats.n_with_prefs}\n")
    push(f"Records without prefs     : {stats.n_pass_through}  (pass-through)\n\n")

    # [1] level counts
    push("[1] Records per difficulty level\n")
    combined = {lv: stats.level_counts.get(lv, 0) for lv in LEVEL_ORDER}
    combined.update({k: v for k, v in stats.level_counts.items() if k not in combined})
    push(fmt_kv_table(combined, stats.n_records, key_hdr="level"))
    if any(stats.level_pass_through.values()):
        push("  Pass-through records by level:\n")
        for lv in LEVEL_ORDER:
            v = stats.level_pass_through.get(lv, 0)
            if v: push(f"    {lv:<6} {v}\n")
    push("\n")

    # [2] paradigm distribution
    push("[2] Preference-paradigm distribution\n")
    push("  Top-level paradigm (overall):\n")
    push(fmt_kv_table(stats.paradigm_top, sum(stats.paradigm_top.values()),
                      key_hdr="paradigm", order=PARADIGM_TOP_ORDER))
    push("  Top-level paradigm by level:\n")
    push(fmt_grouped_by_level(stats.paradigm_top_by_level,
                              PARADIGM_TOP_ORDER, key_hdr="paradigm"))
    if stats.temporal_op:
        push("  Temporal subtype (op) distribution:\n")
        push(fmt_kv_table(stats.temporal_op, sum(stats.temporal_op.values()),
                          key_hdr="op", order=TEMPORAL_OP_ORDER))
        push("  Temporal subtype by level:\n")
        push(fmt_grouped_by_level(stats.temporal_op_by_level,
                                  TEMPORAL_OP_ORDER, key_hdr="op"))
    push("\n")

    # [3] preferences per record
    push("[3] Preferences per record\n")
    push(fmt_kv_table({str(k): v for k, v in stats.n_prefs_per_record.items()},
                      stats.n_records, key_hdr="n_prefs",
                      order=[str(k) for k in sorted(stats.n_prefs_per_record.keys())]))
    push("\n")

    # [4] bank ids per paradigm (op-scoped for Temporal)
    push("[4] Bank-id usage per paradigm\n")
    push("  (Temporal ids are op-scoped in the bank — reported per op.)\n\n")
    for para in list(PARADIGM_TOP_ORDER) + \
                sorted(k for k in stats.bank_usage.keys()
                       if k.startswith("TemporalPreference.")):
        if para == "TemporalPreference":
            continue                                          # detailed below
        usage = stats.bank_usage.get(para) or {}
        if not usage:
            continue
        distinct = len(usage)
        total    = sum(usage.values())
        push(f"  ─ {para}  (distinct ids used: {distinct};  total draws: {total})\n")
        top = sorted(usage.items(), key=lambda kv: (-kv[1], kv[0]))
        # print all ids sorted by id (compact one-line if wide)
        ids_sorted = sorted(usage.items(), key=lambda kv: kv[0])
        one_line = "    " + ", ".join(f"{k}:{v}" for k, v in ids_sorted)
        if len(one_line) < 100:
            push(one_line + "\n")
        else:
            push(fmt_kv_table({str(k): v for k, v in ids_sorted}, total,
                              key_hdr="bank_id", top_k=None))
    push("\n")

    # [5] pairing
    push("[5] Pairing type + subtype\n")
    push(fmt_kv_table(stats.pairing, stats.n_records, key_hdr="pairing_type",
                      order=PAIRING_ORDER + ["pass_through"]))
    push("  Pairing type by level:\n")
    push(fmt_grouped_by_level(stats.pairing_by_level,
                              PAIRING_ORDER + ["pass_through"], key_hdr="pairing"))
    if stats.pairing_sub:
        push("  Pairing subtype (overlapping only):\n")
        push(fmt_kv_table(stats.pairing_sub, sum(stats.pairing_sub.values()),
                          key_hdr="subtype", order=PAIRING_SUB_ORDER))
        push("  Pairing subtype by level:\n")
        push(fmt_grouped_by_level(stats.pairing_sub_by_level,
                                  PAIRING_SUB_ORDER, key_hdr="subtype"))
    push("\n")

    # [6] drift
    push("[6] Profile drift mode (aligned / omission / inversion)\n")
    push(fmt_kv_table(stats.drift, sum(stats.drift.values()),
                      key_hdr="drift_mode", order=DRIFT_MODES))
    push("  By level:\n")
    push(fmt_grouped_by_level(stats.drift_by_level, DRIFT_MODES,
                              key_hdr="drift_mode"))
    if stats.source_drift_fraction:
        push("  Fraction of drifted sources per record:\n")
        push(fmt_stats(stats.source_drift_fraction))
    if stats.source_kind_counts:
        push("  Profile-source kind distribution:\n")
        push(fmt_kv_table(stats.source_kind_counts,
                          sum(stats.source_kind_counts.values()),
                          key_hdr="source_kind"))
    push("\n")

    # [7] local_constraint coverage
    push("[7] Local-constraint (hard) coverage\n")
    push(f"  Records with ≥1 hard constraint: {stats.lc_records} "
         f"({100*stats.lc_records/stats.n_records:.1f}%)\n")
    if stats.lc_key_counts:
        push(fmt_kv_table(stats.lc_key_counts, stats.n_records,
                          key_hdr="lc_key"))
    push("\n")

    # [8] feasibility gates
    push("[8] Feasibility gates\n")
    push("  flight_only_feasible:\n")
    push(fmt_kv_table(stats.flight_only_feasible,
                      sum(stats.flight_only_feasible.values()),
                      key_hdr="value"))
    if stats.budget_multipliers:
        push("  Budget multiplier (post-escalation):\n")
        push(fmt_stats(stats.budget_multipliers, unit="×"))
    if stats.transport_headroom_fraction:
        push("  Prehoc transport-cost / budget ratio:\n")
        push(fmt_stats(stats.transport_headroom_fraction))
        over = sum(1 for x in stats.transport_headroom_fraction if x > 0.5 + 1e-6)
        push(f"  Records exceeding 0.5×budget transport cap: {over}\n")
    push(f"  Records with per-city pool shortfalls: {stats.solution_pool_shortfalls}\n")
    if stats.pool_shortfall_examples:
        push("  Sample shortfalls:\n")
        for ex in stats.pool_shortfall_examples:
            push(f"    id={ex['id']}: {ex['shortfalls']}\n")
    push("\n")

    # [9] NL text stats
    push("[9] Templated NL text (character counts)\n")
    push("  templated_nl_profile length:\n")
    push(fmt_stats([float(x) for x in stats.nl_profile_chars], unit=" ch"))
    push(f"  Empty templated_nl_profile records: {stats.nl_profile_empty}\n")
    push("  templated_nl_query length:\n")
    push(fmt_stats([float(x) for x in stats.nl_query_chars], unit=" ch"))
    push(f"  Empty templated_nl_query records: {stats.nl_query_empty}\n\n")

    # [10]-[14] Paradigm × bucket breakdowns (competing/nc/ind), per level,
    # plus Compensatory + Lex bank_id + partner detail, plus cross-paradigm
    # competing pair combos.
    def _slot(pa: str, lvl: str, bkt: str) -> int:
        return stats.para_level_bucket.get((pa, lvl, bkt), 0)

    def _competing_table(bucket: str) -> str:
        lvls = [lv for lv in LEVEL_ORDER if lv in stats.level_counts]
        paras = sorted(
            {p for (p, _l, b) in stats.para_level_bucket if b == bucket},
            key=lambda p: -sum(_slot(p, lv, bucket) for lv in lvls),
        )
        if not paras:
            return "  (none)\n"
        totals_by_lvl = {lv: sum(_slot(p, lv, bucket) for p in paras) for lv in lvls}
        grand_total = sum(totals_by_lvl.values())
        w = max(len(p) for p in paras + ["paradigm"])
        header = f"  {'paradigm':<{w}}" + "".join(f"  {lv:>7}" for lv in lvls) \
                 + f"  {'total':>7}  {'share':>7}"
        sep    = f"  {'-'*w}"        + "".join(f"  {'-'*7}"   for _ in lvls) \
                 + f"  {'-'*7}  {'-'*7}"
        out_lines = [header, sep]
        for pa in paras:
            row = [_slot(pa, lv, bucket) for lv in lvls]
            tot = sum(row)
            share = 100 * tot / grand_total if grand_total else 0.0
            out_lines.append(
                f"  {pa:<{w}}" + "".join(f"  {v:>7d}" for v in row)
                + f"  {tot:>7d}  {share:>6.1f}%"
            )
        out_lines.append(
            f"  {'TOTAL':<{w}}"
            + "".join(f"  {totals_by_lvl.get(lv,0):>7d}" for lv in lvls)
            + f"  {grand_total:>7d}"
        )
        return "\n".join(out_lines) + "\n"

    push("[10] Paradigm competing distribution per level\n")
    push(_competing_table("competing"))
    push("\n")

    push("[11] Paradigm non_competing / independent distribution per level\n")
    push("  non_competing:\n")
    push(_competing_table("non_competing"))
    push("  independent:\n")
    push(_competing_table("independent"))
    push("\n")

    def _bid_bucket_table(bid_bucket: dict) -> str:
        if not bid_bucket:
            return "  (none)\n"
        rows = []
        rows.append(f"  {'bank_id':>7}  {'comp':>6}  {'nc':>6}  {'indep':>6}  {'single':>6}  {'total':>6}")
        rows.append(f"  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
        for bid in sorted(bid_bucket):
            b = bid_bucket[bid]
            tot = sum(b.values())
            rows.append(
                f"  {str(bid):>7}  {b['competing']:>6d}  {b['non_competing']:>6d}  "
                f"{b['independent']:>6d}  {b['single']:>6d}  {tot:>6d}"
            )
        return "\n".join(rows) + "\n"

    def _pair_combo_lines(pairs: dict, label_prefix: str) -> list[str]:
        if not pairs:
            return ["  (none)"]
        out_lines = []
        for (bid, partner_p, partner_bid, lvl), c in sorted(pairs.items()):
            out_lines.append(
                f"  {label_prefix} {str(bid):>3s} × {partner_p:22s} "
                f"id={str(partner_bid):>3s}  [{lvl:6s}]  ×{c}"
            )
        return out_lines

    push("[12] Compensatory bank_id landing per bucket\n")
    push(_bid_bucket_table(stats.comp_bid_bucket))
    push(f"  Competing pair combos ({sum(stats.comp_competing_pairs.values())} total):\n")
    for ln in _pair_combo_lines(stats.comp_competing_pairs, "Comp"):
        push(ln + "\n")
    push("\n")

    push("[13] Lexicographic bank_id landing per bucket\n")
    push(_bid_bucket_table(stats.lex_bid_bucket))
    push(f"  Competing pair combos ({sum(stats.lex_competing_pairs.values())} total):\n")
    for ln in _pair_combo_lines(stats.lex_competing_pairs, "Lex "):
        push(ln + "\n")
    push("\n")

    push("[14] Cross-paradigm competing pair combos (bidirectional)\n")
    for pa in sorted(stats.para_competing_partners,
                     key=lambda p: -sum(stats.para_competing_partners[p].values())):
        total = sum(stats.para_competing_partners[pa].values())
        push(f"  {pa}  ({total} competing slots):\n")
        for partner, c in sorted(stats.para_competing_partners[pa].items(),
                                  key=lambda kv: -kv[1]):
            push(f"    ↔ {partner:25s}  ×{c}\n")
    push("\n")

    out.write_text("".join(lines))


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    project = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in",  dest="in_path",  type=Path,
                    default=project / "prefertripplan.jsonl")
    ap.add_argument("--out", dest="out_path", type=Path,
                    default=project / "analysis" / "distribution_report.txt")
    ap.add_argument("--stdout", action="store_true",
                    help="Also echo the report to stdout.")
    args = ap.parse_args()

    stats = Stats()
    with args.in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats.update(json.loads(line))

    write_report(stats, args.out_path)
    print(f"Report written to {args.out_path}  ({stats.n_records} records)")
    if args.stdout:
        print()
        print(args.out_path.read_text())


if __name__ == "__main__":
    main()
