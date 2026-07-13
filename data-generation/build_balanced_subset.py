"""Build a balanced ~225-record subset of prefertripplan.jsonl.

Balance axes (primary → secondary → tertiary):

  primary   : (level, pairing_type, pairing_subtype)   -- 10 cells
  secondary : profile_drift_mode                       -- aligned / omission / inversion
  tertiary  : preference paradigm diversity            -- 8 paradigms + Temporal ops

Targets:

  easy      : 3 pass-through + 72 single           = 75
  medium    : 5 pass-through + 30 independent
              + 20 overlapping-competing
              + 20 overlapping-non_competing       = 75
  hard      : 3 pass-through + 32 independent
              + 20 overlapping-competing
              + 20 overlapping-non_competing       = 75
                                              total = 225

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
        ("independent",  None,             0.440),      # 33/75
        ("overlapping",  "competing",      0.280),      # 21/75
        ("overlapping",  "non_competing",  0.280),      # 21/75
    ],
    "hard": [
        ("independent",  None,             0.440),      # 33/75
        ("overlapping",  "competing",      0.280),      # 21/75
        ("overlapping",  "non_competing",  0.280),      # 21/75
    ],
}
DRIFT_TARGET_SHARE = {"aligned": 0.30, "omission": 0.35, "inversion": 0.35}


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
                   taken_drift_counts: Counter) -> list[dict]:
    """Pick ``cell.target`` records from ``cell.records`` such that:
       - drift-mode distribution matches DRIFT_TARGET_SHARE as closely
         as possible within the cell,
       - paradigm diversity is maximised via round-robin,
       - Temporal ops (if any records touch Temporal) are spread evenly.
    """
    if cell.target <= 0 or not cell.records:
        return []

    # Bucket by drift mode
    by_drift: dict[str, list[dict]] = defaultdict(list)
    for r in cell.records:
        by_drift[r.get("profile_drift_mode") or "aligned"].append(r)

    # Per-drift target
    remaining = cell.target
    drift_targets: dict[str, int] = {}
    for drift, share in DRIFT_TARGET_SHARE.items():
        drift_targets[drift] = min(int(round(share * cell.target)),
                                    len(by_drift.get(drift, [])))
    # Redistribute rounding drift
    drift_targets["omission"] += cell.target - sum(drift_targets.values())
    drift_targets["omission"] = max(0, min(drift_targets["omission"],
                                            len(by_drift.get("omission", []))))
    # Backfill any deficit from other drift buckets
    def backfill():
        deficit = cell.target - sum(drift_targets.values())
        if deficit <= 0: return
        for d in ("aligned", "omission", "inversion"):
            room = len(by_drift.get(d, [])) - drift_targets[d]
            take = min(deficit, room)
            if take > 0:
                drift_targets[d] += take
                deficit -= take
                if deficit <= 0: break
    backfill()

    picks: list[dict] = []
    for drift, dtarget in drift_targets.items():
        if dtarget <= 0:
            continue
        pool = list(by_drift[drift])
        rng.shuffle(pool)
        # Greedy: score by (paradigm underrepresentation + temporal op
        # underrepresentation).  Keep re-scoring after each pick.
        picked_local: list[dict] = []
        while pool and len(picked_local) < dtarget:
            best = None
            best_score = None
            for r in pool:
                s = (score_paradigm(r, taken_paradigm_counts) * 3
                     + score_temporal_ops(r, taken_op_counts) * 1
                     + taken_drift_counts.get(drift, 0) * 0.01)
                if best_score is None or s < best_score:
                    best_score = s; best = r
            picked_local.append(best); pool.remove(best)
            # update running counters
            for p in (best.get("preferences") or []):
                if p.get("paradigm"):
                    taken_paradigm_counts[p["paradigm"]] += 1
                op = temporal_op_of(p)
                if op: taken_op_counts[op] += 1
            taken_drift_counts[drift] += 1
        picks.extend(picked_local)

    return picks


def build_subset(recs: list[dict], target: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cells = build_cells(recs, target)

    # Temporal-op counter is GLOBAL so the 9 sub-ops spread evenly across
    # the whole subset (a Temporal.always record picked in medium should
    # discourage another Temporal.always pick in easy).  Paradigm and
    # drift counters are PER-CELL: each cell balances its own paradigm
    # mix, so easy/single (which has no forced paradigm) spreads all 8
    # paradigms evenly rather than avoiding Atomic/Composite because
    # medium/hard already saturated the global count.
    taken_op: Counter = Counter()

    # Fill cells in order of *severity* -- rarest cells first so their
    # limited pool isn't blocked by earlier greedy choices from a big
    # cell.  (Overlapping-competing cells are the tightest pool.)
    ordered = sorted(cells, key=lambda c: len(c.records))
    subset: list[dict] = []
    for c in ordered:
        picks = pick_from_cell(c, rng,
                                Counter(),          # fresh paradigm counter per cell
                                taken_op,           # shared across cells
                                Counter())          # fresh drift counter per cell
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
    ap.add_argument("--seed", type=int, default=20260712)
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.in_path.open()]
    print(f"[in]  {len(recs)} source records from {args.in_path}")

    subset = build_subset(recs, args.target, args.seed)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w") as f:
        for r in subset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[out] wrote {len(subset)} records to {args.out_path}")

    summary = make_summary(recs, subset)
    summary_path = args.out_path.with_suffix(".summary.txt")
    summary_path.write_text(summary)
    print(f"[out] wrote balance summary to {summary_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
