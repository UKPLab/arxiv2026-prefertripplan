"""Build a balanced ~225-record subset of prefertripplan.jsonl.

Balance axes (primary → secondary → tertiary):

  primary   : (level, pairing_type, pairing_subtype)   -- 10 cells
  secondary : profile_drift_mode                       -- aligned / omission / inversion
  tertiary  : preference paradigm diversity            -- 8 paradigms + Temporal ops

Targets (30% independent vs 70% overlapping within pairing-eligible
levels -- matching the corpus-level bias; overlap is split 50/50
competing vs non_competing in the subset to give downstream
evaluation an equal-power test of both subtypes, even though the
corpus itself sits at ~40/60):

  easy      : 75 single           (no pairing)                       = 75
  medium    : 23 independent
              + 26 overlapping-competing
              + 26 overlapping-non_competing                          = 75
              (indep/overlap = 23/52 = 30.7/69.3,
               comp/non_comp within overlap = 26/52 = 50/50)
  hard      : same as medium                                          = 75
                                                             total    = 225

Within each cell we further stratify by profile_drift_mode (aim ~30 / 35 /
35 split), and within each (cell, drift) subgroup we pick records that
maximise paradigm diversity via round-robin over the paradigm-pair
signature.  For Temporal-containing records we additionally spread the
sub-op (always / sometime / within / atmost_once / sometime_before /
sometime_after / always_within / hold_during / hold_after) as evenly
as the cell size permits.

The picked subset is deterministic given a seed (default 20260712).

Emits ``data-generation/prefertripplan.subset.jsonl`` (JSONL, same schema
as source) plus a ``.summary.txt`` next to it reporting the achieved
balance across every axis.

Usage:
    python3 data-generation/build_balanced_subset.py                    # default target 225
    python3 data-generation/build_balanced_subset.py --target 250       # tune size
    python3 data-generation/build_balanced_subset.py --seed 42          # different picks
    python3 data-generation/build_balanced_subset.py --in <path> --out <path>
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "prefertripplan.jsonl"
OUT  = ROOT / "prefertripplan.subset.jsonl"

# --- primary stratum quotas (proportion of the target size) --- #
# Structure: level -> [(pairing, subtype_or_None, share_within_level), ...].
# Shares are pre-normalised: they sum to 1.0 per level.  Level shares
# themselves are equal (1/3 each).
STRATUM_LAYOUT: dict[str, list[tuple[str, str | None, float]]] = {
    "easy": [
        ("single",       None,             1.000),      # 75/75
    ],
    "medium": [
        # 30 / 70 indep-vs-overlap within pairing-eligible records
        # (matches the corpus-level bias enforced by the augmenter);
        # overlap is split 50 / 50 competing vs non_competing in the
        # SUBSET so downstream evaluation gets equal-power coverage
        # of both subtypes.  The corpus's own 40 / 60 comp/non_comp
        # is intentionally NOT mirrored here -- the subset is
        # curated for balanced testing, not proportional sampling.
        # comp share = 0.70 * 0.50 = 0.35; non_comp = 0.70 * 0.50 = 0.35.
        ("independent",  None,             0.300),      # ~22/75
        ("overlapping",  "competing",      0.350),      # ~26/75
        ("overlapping",  "non_competing",  0.350),      # ~26/75
    ],
    "hard": [
        ("independent",  None,             0.300),      # ~22/75
        ("overlapping",  "competing",      0.350),      # ~26/75
        ("overlapping",  "non_competing",  0.350),      # ~26/75
    ],
}
DRIFT_TARGET_SHARE = {"aligned": 0.30, "omission": 0.35, "inversion": 0.35}


# Complex paradigms competing for non-anchor slots in medium/hard pairs.
# The Anchor paradigms (Composite/Atomic) are pinned by pairing structure:
# medium pairs anchor to Atomic, hard pairs anchor to Composite.  Easy
# singles can be any of the 8 paradigms.
_COMPLEX_PARADIGMS: list[str] = [
    "NumericPreference",
    "ConditionalPreference",
    "LexicographicPreference",
    "CompensatoryPreference",
    "ScopedPreference",
    "TemporalPreference",
]
_ANCHOR_PARADIGMS: list[str] = ["AtomicPreference", "CompositePreference"]
_ALL_PARADIGMS: list[str] = _ANCHOR_PARADIGMS + _COMPLEX_PARADIGMS

# TemporalPreference sub-ops (9 flavours).  Sub-op quota is enforced
# within the Temporal-family paradigm quota.
_TEMPORAL_OPS: list[str] = [
    "always", "sometime", "within", "atmost_once",
    "sometime_before", "sometime_after",
    "always_within", "hold_during", "hold_after",
]

# Soft-hard cap penalty weights.  A record that pushes a paradigm over
# its target quota gets `_OVER_QUOTA_PENALTY_PARADIGM` added to its
# score; the sampler then prefers any in-quota alternative.  Only when
# EVERY remaining record would exceed quotas is the penalty overridden
# (that's the "soft" part -- keeps sampling from stalling).  Set high
# enough that a single over-quota record loses to any in-quota record
# regardless of the diversity term.
_OVER_QUOTA_PENALTY_PARADIGM: float = 10000.0
_OVER_QUOTA_PENALTY_TEMPORAL: float =  5000.0


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def paradigm_full(pref: dict[str, Any]) -> str:
    par = pref.get("paradigm")
    if par == "TemporalPreference":
        op = (pref.get("template") or {}).get("op") or "?"
        return f"Temporal.{op}"
    return par or "?"


def temporal_op_of(pref: dict[str, Any]) -> str | None:
    if pref.get("paradigm") != "TemporalPreference":
        return None
    return (pref.get("template") or {}).get("op")


def stratum_key(rec: dict) -> tuple[str, str, str | None]:
    """(level, pairing_type, pairing_subtype) -- ``pass_through`` for
    records with no preferences."""
    prefs = rec.get("preferences") or []
    lv = rec.get("level")
    if not prefs:
        return (lv, "pass_through", None)
    pt = rec.get("pairing_type") or "single"
    ps = rec.get("pairing_subtype")
    return (lv, pt, ps)


def paradigm_signature(rec: dict) -> tuple[str, ...]:
    """Sorted tuple of paradigm-full labels; used for diversity round-robin."""
    return tuple(sorted(paradigm_full(p) for p in (rec.get("preferences") or [])))


def temporal_ops_in(rec: dict) -> list[str]:
    return [op for op in (temporal_op_of(p) for p in (rec.get("preferences") or []))
            if op]


# --------------------------------------------------------------------------- #
# Sampler                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class Cell:
    level: str
    pairing: str
    subtype: str | None
    target: int
    records: list[dict]            # eligible source records

    @property
    def key(self) -> tuple[str, str, str | None]:
        return (self.level, self.pairing, self.subtype)


def build_cells(recs: list[dict], target: int) -> list[Cell]:
    """Group source records by (level, pairing, subtype) and assign a
    per-cell target size using ``STRATUM_LAYOUT``.  Any cell whose
    source pool is smaller than its target absorbs the deficit into
    other cells of the same level."""
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        by_key[stratum_key(r)].append(r)

    level_target = target // 3   # equal split; small remainder redistributed below
    remainder    = target - level_target * 3

    cells: list[Cell] = []
    for level_idx, level in enumerate(("easy", "medium", "hard")):
        lvl_target = level_target + (1 if level_idx < remainder else 0)
        layout = STRATUM_LAYOUT[level]
        raw = [(pt, ps, int(round(share * lvl_target))) for pt, ps, share in layout]
        # correct rounding drift to hit lvl_target exactly
        diff = lvl_target - sum(t for _, _, t in raw)
        if diff:
            raw[0] = (raw[0][0], raw[0][1], raw[0][2] + diff)
        for pt, ps, t in raw:
            pool = list(by_key.get((level, pt, ps), []))
            # cap target at available pool
            eff = min(t, len(pool))
            cells.append(Cell(level=level, pairing=pt, subtype=ps,
                              target=eff, records=pool))
    return cells


def compute_paradigm_targets(cells: list["Cell"]) -> dict[str, int]:
    """Return per-paradigm target counts across the whole subset.

    Distribution logic:
      * Complex slots (non-anchor preference in medium/hard pairs) are
        divided EQUALLY across the 6 complex paradigms.
      * Anchor slots are pinned to Composite (hard) / Atomic (medium)
        by pairing structure -- not distributed.
      * Easy singles are distributed EQUALLY across all 8 paradigms.

    The returned dict is the target `taken_paradigm_counts[paradigm]`
    the sampler should aim to hit; the sampler penalizes going over it.
    """
    complex_slots = sum(c.target for c in cells
                        if c.level != "easy" and c.pairing != "single")
    per_complex = complex_slots // len(_COMPLEX_PARADIGMS)
    extras = complex_slots - per_complex * len(_COMPLEX_PARADIGMS)
    complex_t = {p: per_complex for p in _COMPLEX_PARADIGMS}
    for p in _COMPLEX_PARADIGMS[:extras]:
        complex_t[p] += 1

    easy_slots = sum(c.target for c in cells if c.level == "easy")
    per_easy = easy_slots // len(_ALL_PARADIGMS)
    easy_extras = easy_slots - per_easy * len(_ALL_PARADIGMS)
    easy_t = {p: per_easy for p in _ALL_PARADIGMS}
    for p in _ALL_PARADIGMS[:easy_extras]:
        easy_t[p] += 1

    hard_anchor_slots   = sum(c.target for c in cells
                              if c.level == "hard"   and c.pairing != "single")
    medium_anchor_slots = sum(c.target for c in cells
                              if c.level == "medium" and c.pairing != "single")

    targets: dict[str, int] = {}
    for p in _COMPLEX_PARADIGMS:
        targets[p] = complex_t[p] + easy_t[p]
    targets["CompositePreference"] = hard_anchor_slots + easy_t["CompositePreference"]
    targets["AtomicPreference"]    = medium_anchor_slots + easy_t["AtomicPreference"]
    return targets


def compute_temporal_op_targets(temporal_family_target: int) -> dict[str, int]:
    """Distribute the Temporal-family target across the 9 sub-ops
    evenly.  Called after compute_paradigm_targets has fixed the
    Temporal quota."""
    per_op = temporal_family_target // len(_TEMPORAL_OPS)
    extras = temporal_family_target - per_op * len(_TEMPORAL_OPS)
    targets = {op: per_op for op in _TEMPORAL_OPS}
    for op in _TEMPORAL_OPS[:extras]:
        targets[op] += 1
    return targets


def _quota_over_penalty(rec: dict,
                        taken_para: Counter,
                        para_targets: dict[str, int],
                        taken_op: Counter,
                        op_targets: dict[str, int]) -> float:
    """Compute the total over-quota penalty this record would incur
    if picked next.  Iterates over the record's preferences and adds
    the paradigm-level penalty for any paradigm currently at-or-over
    its target, plus the Temporal sub-op penalty for any at-or-over
    sub-op.  Zero if picking the record stays within all quotas."""
    penalty = 0.0
    for p in (rec.get("preferences") or []):
        para = p.get("paradigm")
        if para and para in para_targets \
                and taken_para.get(para, 0) >= para_targets[para]:
            penalty += _OVER_QUOTA_PENALTY_PARADIGM
        op = temporal_op_of(p)
        if op and op in op_targets \
                and taken_op.get(op, 0) >= op_targets[op]:
            penalty += _OVER_QUOTA_PENALTY_TEMPORAL
    return penalty


def score_paradigm(rec: dict, taken_paradigm_counts: Counter) -> float:
    """Lower is better -- prefer records whose paradigms are currently
    under-represented in the picked subset.  Uses BARE paradigm labels
    (TemporalPreference counts as one bucket, not nine) so the 8 top-
    level paradigms compete on equal footing.  Temporal sub-op spread
    is handled separately by ``score_temporal_ops``."""
    paras = [p.get("paradigm") for p in (rec.get("preferences") or [])
             if p.get("paradigm")]
    if not paras:
        return 0.0
    return sum(taken_paradigm_counts.get(pa, 0) for pa in paras) / len(paras)


def score_temporal_ops(rec: dict, taken_op_counts: Counter) -> float:
    """Lower is better -- prefer under-represented Temporal ops."""
    ops = temporal_ops_in(rec)
    if not ops:
        return 0.0
    return sum(taken_op_counts.get(o, 0) for o in ops) / len(ops)


def pick_from_cell(cell: Cell, rng: random.Random,
                   taken_paradigm_counts: Counter,
                   taken_op_counts: Counter,
                   taken_drift_counts: Counter,
                   paradigm_targets: dict[str, int],
                   temporal_op_targets: dict[str, int]) -> list[dict]:
    """Pick ``cell.target`` records from ``cell.records`` such that:
       - drift-mode distribution matches DRIFT_TARGET_SHARE within the
         cell (as before);
       - subset-wide paradigm counts stay within ``paradigm_targets``
         (soft-hard: an over-quota record only wins when no in-quota
         alternative exists);
       - subset-wide Temporal sub-op counts stay within
         ``temporal_op_targets`` (same soft-hard mechanism);
       - within remaining slack, paradigm and Temporal sub-op diversity
         are maximised via the existing round-robin diversity scores.

    ``taken_paradigm_counts`` and ``taken_op_counts`` are GLOBAL (shared
    across cells) so the quotas hold subset-wide; ``taken_drift_counts``
    is per-cell (each cell owns its own drift split).
    """
    if cell.target <= 0 or not cell.records:
        return []

    # Bucket by drift mode
    by_drift: dict[str, list[dict]] = defaultdict(list)
    for r in cell.records:
        by_drift[r.get("profile_drift_mode") or "aligned"].append(r)

    # Per-drift target
    drift_targets: dict[str, int] = {}
    for drift, share in DRIFT_TARGET_SHARE.items():
        drift_targets[drift] = min(int(round(share * cell.target)),
                                    len(by_drift.get(drift, [])))
    # Redistribute rounding drift
    drift_targets["omission"] += cell.target - sum(drift_targets.values())
    drift_targets["omission"] = max(0, min(drift_targets["omission"],
                                            len(by_drift.get("omission", []))))
    def backfill():
        deficit = cell.target - sum(drift_targets.values())
        if deficit <= 0:
            return
        for d in ("aligned", "omission", "inversion"):
            room = len(by_drift.get(d, [])) - drift_targets[d]
            take = min(deficit, room)
            if take > 0:
                drift_targets[d] += take
                deficit -= take
                if deficit <= 0:
                    break
    backfill()

    picks: list[dict] = []
    for drift, dtarget in drift_targets.items():
        if dtarget <= 0:
            continue
        pool = list(by_drift[drift])
        rng.shuffle(pool)
        picked_local: list[dict] = []
        while pool and len(picked_local) < dtarget:
            best = None
            best_score = None
            for r in pool:
                # Over-quota penalty (soft-hard): if picking this record
                # would push any paradigm past its subset-wide target, or
                # any Temporal sub-op past its target, add a heavy
                # penalty.  Only when EVERY remaining record has some
                # penalty does an over-quota pick actually win.
                over = _quota_over_penalty(r,
                                            taken_paradigm_counts,
                                            paradigm_targets,
                                            taken_op_counts,
                                            temporal_op_targets)
                # Diversity signals (bumped Temporal weight from 1 -> 3
                # so sub-op spread competes with paradigm spread).
                base = (score_paradigm(r, taken_paradigm_counts) * 3
                        + score_temporal_ops(r, taken_op_counts) * 3
                        + taken_drift_counts.get(drift, 0) * 0.01)
                s = over + base
                if best_score is None or s < best_score:
                    best_score = s; best = r
            picked_local.append(best); pool.remove(best)
            for p in (best.get("preferences") or []):
                if p.get("paradigm"):
                    taken_paradigm_counts[p["paradigm"]] += 1
                op = temporal_op_of(p)
                if op:
                    taken_op_counts[op] += 1
            taken_drift_counts[drift] += 1
        picks.extend(picked_local)

    return picks


def build_subset(recs: list[dict], target: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cells = build_cells(recs, target)

    # Compute subset-wide targets for hard-cap enforcement.
    paradigm_targets    = compute_paradigm_targets(cells)
    temporal_op_targets = compute_temporal_op_targets(
        paradigm_targets.get("TemporalPreference", 0))

    # Paradigm and Temporal-op counters are GLOBAL so quota enforcement
    # holds subset-wide (a Temporal.always record picked in medium
    # discourages another Temporal.always pick in easy).  Drift counter
    # is per-cell -- each cell owns its own aligned/omission/inversion
    # split independently.
    taken_paradigm: Counter = Counter()
    taken_op:       Counter = Counter()

    # Fill cells in order of *severity* -- rarest cells first so their
    # limited pool isn't blocked by earlier greedy choices from a big
    # cell.  (Overlapping-competing cells are the tightest pool.)
    ordered = sorted(cells, key=lambda c: len(c.records))
    subset: list[dict] = []
    for c in ordered:
        picks = pick_from_cell(c, rng,
                                taken_paradigm,        # shared across cells
                                taken_op,              # shared across cells
                                Counter(),             # fresh drift counter per cell
                                paradigm_targets,
                                temporal_op_targets)
        subset.extend(picks)

    # Sort by query_id so the emitted file is deterministic / readable.
    subset.sort(key=lambda r: r.get("query_id") or 0)
    return subset


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #

def make_summary(source: list[dict], subset: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"Balanced subset: {len(subset)} records "
                 f"(from source of {len(source)})\n")

    def counter_line(title: str, key_fn, order=None):
        cnt = Counter()
        for r in subset: cnt[key_fn(r)] += 1
        lines.append(f"[{title}]")
        keys = order if order is not None else sorted(cnt.keys(), key=str)
        for k in keys:
            v = cnt.get(k, 0)
            pct = 100.0 * v / len(subset) if subset else 0
            lines.append(f"    {str(k):40}  {v:>4}  ({pct:.1f}%)")
        lines.append("")

    counter_line("Level", lambda r: r.get("level"),
                 order=["easy", "medium", "hard"])
    counter_line("Pairing type",  lambda r: r.get("pairing_type") or "pass_through",
                 order=["single", "independent", "overlapping", "pass_through"])
    counter_line("Pairing subtype",
                 lambda r: r.get("pairing_subtype") or "--",
                 order=["competing", "non_competing", "--"])
    counter_line("Profile drift",  lambda r: r.get("profile_drift_mode") or "?",
                 order=["aligned", "omission", "inversion"])

    # Level × pairing × subtype
    lines.append("[Level × Pairing × Subtype]")
    c = Counter((r.get("level"),
                 r.get("pairing_type") or "pass_through",
                 r.get("pairing_subtype") or "--") for r in subset)
    for k, n in sorted(c.items()):
        lines.append(f"    {str(k):50}  {n:>4}")
    lines.append("")

    # Paradigm distribution (multi-count)
    lines.append("[Preference paradigm (multi-count across all preferences)]")
    p_cnt = Counter()
    for r in subset:
        for p in (r.get("preferences") or []): p_cnt[paradigm_full(p)] += 1
    for k, n in sorted(p_cnt.items(), key=lambda x: -x[1]):
        lines.append(f"    {k:40}  {n:>4}")
    lines.append("")

    # Paradigm × level × pairing-bucket (competing / non_competing /
    # independent) -- mirrors sections [10]-[11] of the full
    # distribution report so the subset's pairing balance can be
    # eyeballed against the corpus at the same granularity.
    LEVELS = ["easy", "medium", "hard"]
    def _bucket(rec: dict) -> str | None:
        pt = rec.get("pairing_type")
        ps = rec.get("pairing_subtype")
        if pt == "single":       return "single"
        if pt == "independent":  return "independent"
        if ps == "competing":     return "competing"
        if ps == "non_competing": return "non_competing"
        return None

    para_level_bucket: Counter = Counter()
    for r in subset:
        b = _bucket(r)
        lv = r.get("level")
        if b is None or lv is None:
            continue
        for p in (r.get("preferences") or []):
            pa = p.get("paradigm")
            if pa:
                para_level_bucket[(pa, lv, b)] += 1

    def _para_bucket_table(bucket: str) -> list[str]:
        rows = [(pa, [para_level_bucket.get((pa, lv, bucket), 0) for lv in LEVELS])
                for pa in sorted({p for (p, _l, b) in para_level_bucket if b == bucket},
                                 key=lambda p: -sum(para_level_bucket.get((p, lv, bucket), 0)
                                                    for lv in LEVELS))]
        if not rows:
            return ["    (none)"]
        totals = [sum(v[lv_i] for _, v in rows) for lv_i in range(len(LEVELS))]
        grand = sum(totals)
        w = max(len(pa) for pa, _ in rows + [("paradigm", None)])
        out = [f"    {'paradigm':<{w}}"
               + "".join(f"  {lv:>7}" for lv in LEVELS)
               + f"  {'total':>7}  {'share':>7}"]
        for pa, row in rows:
            tot = sum(row)
            share = 100 * tot / grand if grand else 0.0
            out.append(f"    {pa:<{w}}"
                       + "".join(f"  {v:>7d}" for v in row)
                       + f"  {tot:>7d}  {share:>6.1f}%")
        out.append(f"    {'TOTAL':<{w}}"
                   + "".join(f"  {t:>7d}" for t in totals)
                   + f"  {grand:>7d}")
        return out

    lines.append("[Paradigm competing distribution per level]")
    lines.extend(_para_bucket_table("competing"))
    lines.append("")
    lines.append("[Paradigm non_competing distribution per level]")
    lines.extend(_para_bucket_table("non_competing"))
    lines.append("")
    lines.append("[Paradigm independent distribution per level]")
    lines.extend(_para_bucket_table("independent"))
    lines.append("")

    # Compensatory / Lex bank_id competing pair combos -- mirrors
    # sections [12]/[13] of the report; useful for confirming that the
    # subset preserves reverse-polarity coverage.
    comp_pairs: Counter = Counter()
    lex_pairs:  Counter = Counter()
    for r in subset:
        if _bucket(r) != "competing":
            continue
        prefs = r.get("preferences") or []
        if len(prefs) != 2:
            continue
        p0, p1 = prefs
        a, b = p0.get("paradigm"), p1.get("paradigm")
        lv = r.get("level")
        if "CompensatoryPreference" in (a, b):
            comp = p0 if a == "CompensatoryPreference" else p1
            other = p1 if a == "CompensatoryPreference" else p0
            comp_pairs[(comp.get("bank_id"), other.get("paradigm"),
                        other.get("bank_id"), lv)] += 1
        if "LexicographicPreference" in (a, b):
            lex_ = p0 if a == "LexicographicPreference" else p1
            other = p1 if a == "LexicographicPreference" else p0
            lex_pairs[(lex_.get("bank_id"), other.get("paradigm"),
                       other.get("bank_id"), lv)] += 1

    lines.append(f"[Compensatory competing pair combos]  "
                 f"{sum(comp_pairs.values())} pairs")
    if comp_pairs:
        for (bid, partner_p, partner_bid, lv), c in sorted(comp_pairs.items()):
            lines.append(
                f"    Comp {str(bid):>3s} × {partner_p:22s} "
                f"id={str(partner_bid):>3s}  [{lv:6s}]  ×{c}"
            )
    else:
        lines.append("    (none)")
    lines.append("")

    lines.append(f"[Lexicographic competing pair combos]  "
                 f"{sum(lex_pairs.values())} pairs")
    if lex_pairs:
        for (bid, partner_p, partner_bid, lv), c in sorted(lex_pairs.items()):
            lines.append(
                f"    Lex  {str(bid):>3s} × {partner_p:22s} "
                f"id={str(partner_bid):>3s}  [{lv:6s}]  ×{c}"
            )
    else:
        lines.append("    (none)")
    lines.append("")

    # Temporal op distribution
    lines.append("[Temporal sub-op distribution]")
    t_cnt = Counter()
    for r in subset:
        for p in (r.get("preferences") or []):
            op = temporal_op_of(p)
            if op: t_cnt[op] += 1
    for k, n in sorted(t_cnt.items(), key=lambda x: -x[1]):
        lines.append(f"    {k:20}  {n:>4}")
    lines.append("")

    # Additional dimensions worth eyeballing:
    lines.append("[Days]")
    d_cnt = Counter(r.get("days") for r in subset)
    for k, n in sorted(d_cnt.items()):
        lines.append(f"    days={k}  {n:>4}")
    lines.append("")

    lines.append("[People number]")
    pn_cnt = Counter(r.get("people_number") for r in subset)
    for k, n in sorted(pn_cnt.items()):
        lines.append(f"    people={k}  {n:>4}")
    lines.append("")

    lines.append("[Visiting-city number]")
    vc = Counter(r.get("visiting_city_number") for r in subset)
    for k, n in sorted(vc.items()):
        lines.append(f"    visit_n={k}  {n:>4}")
    lines.append("")

    lines.append("[Distinct destinations covered]")
    dests = sorted({r.get("dest") for r in subset if r.get("dest")})
    lines.append(f"    {len(dests)} distinct destinations")
    lines.append(f"    {', '.join(dests[:20])}{' ...' if len(dests) > 20 else ''}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in",  dest="in_path",  type=Path, default=SRC)
    ap.add_argument("--out", dest="out_path", type=Path, default=OUT)
    ap.add_argument("--target", type=int, default=225,
                    help="Approximate target subset size.  Default 225.")
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.in_path.open()]
    print(f"[in]  {len(recs)} source records from {args.in_path}")

    subset = build_subset(recs, args.target, args.seed)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w") as f:
        for r in subset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[out] wrote {len(subset)} records to {args.out_path}")

    # Companion ids-only file: one query_id per line, ordered as the
    # subset JSONL.  Downstream tools consume this directly:
    #   * render_llm_nl.py --sample-qids <this file>  (renders only the
    #     subset qids)
    #   * build_hf_dataset.py --subset-ids <this file>  (splits into
    #     test vs test_large based on subset membership)
    ids_path = args.out_path.with_suffix(".ids.txt")
    with ids_path.open("w") as f:
        for r in subset:
            f.write(f"{r['query_id']}\n")
    print(f"[out] wrote {len(subset)} query_ids to {ids_path}")

    summary = make_summary(recs, subset)
    summary_path = args.out_path.with_suffix(".summary.txt")
    summary_path.write_text(summary)
    print(f"[out] wrote balance summary to {summary_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
