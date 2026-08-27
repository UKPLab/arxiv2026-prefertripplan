#!/usr/bin/env python3
"""analyze_performance.py — sliced performance analysis over the per-record
detailed evaluation JSONL emitted by ``evaluation/eval.py --out``.

Inputs:
  * ``--detailed``           per-record JSONL from ``eval.py --out``.
  * ``--dataset`` / ``--split``  HF dataset -- used ONLY to hydrate
                                 ``profile_drift`` + ``preference_pair``
                                 columns that the detailed JSONL doesn't
                                 already carry.
  * ``--out``                optional plain-text destination; report is
                             also printed to stdout.

Emits, in order:

  1. Overall single-bucket constraint metrics (delivery, cs μ/M, hd μ/M,
     pref μ/M, final, final+p, Δpref).
  2. Same metrics sliced across each of five per-record axes:
        days              (3 / 5 / 7)
        level             (easy / medium / hard)
        profile_drift     (aligned / omission / inversion)
        pairing_type      (single / independent / overlapping)
        pairing_subtype   (competing / non_competing / --)
  3. Per-axis × paradigm preference micro pass-rate cross-tabs (same
     five axes × 8 paradigms).
  4. Per-axis × (paradigm, sub_paradigm) preference micro pass-rate
     cross-tabs.
  5. Preference-only breakdowns not tied to any per-record axis:
        by paradigm
        by (paradigm, sub_paradigm)
        by trivial vs non-trivial
        by preference bank_id (top-K only)

Every metric is derived directly from the input JSONL + HF dataset --
nothing is hardcoded per-level or per-split.

Usage:
    python3 evaluation/analyze_performance.py \\
        --detailed evaluation/eval_<model>.jsonl \\
        --dataset  UKPLab/PreferTripPlan --split test \\
        --out      evaluation/analyze_<model>.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


def _tqdm(iterable: Iterable, *, total: int | None = None,
          desc: str = "", disable: bool = False) -> Iterable:
    """Wrap ``iterable`` in a progress bar (tqdm if available, else a
    plain ``\\r``-updated stderr line so runs stay legible without tqdm
    installed).  Same fallback pattern used by ``eval.py`` and
    ``evaluate_preferences.py``."""
    if disable:
        return iterable
    try:
        from tqdm import tqdm as _t
        return _t(iterable, total=total, desc=desc, unit="row",
                  dynamic_ncols=True, mininterval=0.3, leave=True)
    except ImportError:
        pass
    # Fallback: no external dependency.
    import time
    total = total if total is not None else (
        len(iterable) if hasattr(iterable, "__len__") else None)

    def _gen():
        start = time.time()
        last  = start
        n     = 0
        for item in iterable:
            yield item
            n += 1
            now = time.time()
            if now - last > 0.3 or (total and n == total):
                elapsed = now - start
                rate    = n / elapsed if elapsed > 0 else 0.0
                if total:
                    pct = 100.0 * n / total
                    line = (f"[{desc}] {n}/{total} ({pct:5.1f}%)  "
                            f"{rate:.1f} row/s")
                else:
                    line = f"[{desc}] {n}  {rate:.1f} row/s"
                sys.stderr.write("\r" + line)
                sys.stderr.flush()
                last = now
        if n:
            sys.stderr.write("\n")
            sys.stderr.flush()
    return _gen()


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #

def _load_detailed(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_hf_meta(dataset: str, split: str) -> dict[int, dict]:
    """Hydrate the HF-side metadata needed for the analysis passes,
    keyed by 1-based dataset ``id``.  Fields:

      * ``profile_drift``      -- aligned / omission / inversion
      * ``pairing_type`` / ``pairing_subtype``  -- from ``preference_pair``
      * ``people_number``      -- party size (used for drift × party-size)
      * ``preferences_json``   -- structured preferences on this record
                                   (needed to walk drift-source paths into
                                   specific sub-preferences)
      * ``drift_trace``        -- structured trace of every profile source
                                   that fed into this record, including
                                   which ones were drifted, the drift
                                   action (``invert`` / ``drop``), and the
                                   ``path`` locating the affected sub-node
                                   inside the linked preference.

    The two structured columns (``preferences_json`` and ``drift_trace``)
    are JSON-stringified on the Hub; both string and dict forms are
    tolerated defensively."""
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, verification_mode="no_checks")

    def _loads(v: Any) -> Any:
        if isinstance(v, (dict, list)):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (TypeError, ValueError):
                return None
        return None

    meta: dict[int, dict] = {}
    for r in _tqdm(ds, total=len(ds), desc="hf-meta"):
        rid = int(r["id"])
        pair = _loads(r.get("preference_pair")) or {}
        try:
            people_number = int(r.get("people_number") or 0)
        except (TypeError, ValueError):
            people_number = 0
        meta[rid] = {
            "profile_drift":    r.get("profile_drift"),
            "pairing_type":     pair.get("type"),
            "pairing_subtype":  pair.get("subtype"),
            "people_number":    people_number or None,
            "preferences_json": _loads(r.get("preferences_json")) or [],
            "drift_trace":      _loads(r.get("drift_trace")) or {},
        }
    return meta


# --------------------------------------------------------------------------- #
# Per-record derivation                                                       #
# --------------------------------------------------------------------------- #

def _pair(v: Any) -> tuple[Any, Any]:
    """Every commonsense/hard entry in the detailed JSONL is
    ``[bool_or_null, msg_or_null]``.  Return that tuple with the ``None``
    sentinel preserved so ``bool_or_null`` can be distinguished from
    ``False``."""
    if isinstance(v, (list, tuple)) and len(v) >= 1:
        return v[0], (v[1] if len(v) > 1 else None)
    if isinstance(v, dict):
        return v.get("passed"), v.get("message")
    return None, None


def _record_metrics(row: dict, meta: dict) -> dict:
    """Reduce one detailed row into a flat metrics dict."""
    m: dict = meta.get(int(row["id"]), {}) if meta else {}
    cs   = row.get("commonsense")
    hd   = row.get("hard")
    prefs = row.get("preferences") if isinstance(row.get("preferences"), list) else []

    if isinstance(cs, dict) and cs:
        cs_pairs  = [_pair(v) for v in cs.values()]
        cs_pass   = sum(1 for b, _ in cs_pairs if b is True)
        cs_total  = len(cs_pairs)
        cs_macro  = all((b is None) or b for b, _ in cs_pairs)
    else:
        cs_pass = cs_total = 0
        cs_macro = False

    if isinstance(hd, dict) and hd:
        hd_pairs  = [_pair(v) for v in hd.values()]
        hd_pass   = sum(1 for b, _ in hd_pairs if b is True)
        hd_total  = sum(1 for b, _ in hd_pairs if b is not None)
        hd_macro  = all((b is None) or b for b, _ in hd_pairs)
    else:
        hd_pass = hd_total = 0
        hd_macro = False

    n_prefs      = len(prefs)
    n_pref_pass  = sum(1 for p in prefs if p.get("passed"))
    has_prefs    = n_prefs > 0
    pref_macro   = has_prefs and (n_pref_pass == n_prefs)
    pref_vacuous = (not has_prefs) or pref_macro    # vacuously OK on pref-less rows

    delivered   = bool(row.get("delivered"))
    final_pass  = isinstance(cs, dict) and isinstance(hd, dict) and cs_macro and hd_macro
    final_pass_incl = final_pass and pref_vacuous

    return {
        "id":              int(row["id"]),
        "level":           row.get("level"),
        "days":            int(row["days"]) if row.get("days") is not None else None,
        "delivered":       delivered,
        "cs_pass":         cs_pass,   "cs_total":  cs_total,  "cs_macro":  cs_macro,
        "hd_pass":         hd_pass,   "hd_total":  hd_total,  "hd_macro":  hd_macro,
        "n_prefs":         n_prefs,   "n_pref_pass": n_pref_pass,
        "has_prefs":       has_prefs, "pref_macro": pref_macro,
        "final_pass":      final_pass,
        "final_pass_incl": final_pass_incl,
        "profile_drift":    m.get("profile_drift"),
        "pairing_type":     m.get("pairing_type"),
        "pairing_subtype":  m.get("pairing_subtype") or "--",
        # Party size: exact people_number + a 2-way ``solo`` (=1) /
        # ``group`` (>=2) bucket.  The 2-way bucket is what pairs with
        # profile_drift for the joint-bloc bucketing below; the raw
        # people_number is kept for records that want it.
        "people_number":    m.get("people_number"),
        "people_bucket":    ("solo"  if m.get("people_number") == 1 else
                              "group" if (m.get("people_number") or 0) >= 2
                              else None),
        "_prefs":           prefs,   # keep for per-pref buckets
        # HF-side context surfaced verbatim so drift-impact analyses can
        # walk drift-source ``path`` values into the corresponding
        # sub-preference on the SAME record.  Populated only when
        # ``_load_hf_meta`` was called; otherwise these are empty
        # containers.
        "_preferences_json": m.get("preferences_json") or [],
        "_drift_trace":      m.get("drift_trace") or {},
    }


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #

def _summarize_bucket(records: list[dict]) -> dict:
    N = len(records)
    if N == 0:
        return {"N": 0}
    cs_pass  = sum(r["cs_pass"]     for r in records)
    cs_total = sum(r["cs_total"]    for r in records)
    hd_pass  = sum(r["hd_pass"]     for r in records)
    hd_total = sum(r["hd_total"]    for r in records)
    pf_pass  = sum(r["n_pref_pass"] for r in records)
    pf_total = sum(r["n_prefs"]     for r in records)
    pf_records    = sum(1 for r in records if r["has_prefs"])
    pf_macro_pass = sum(1 for r in records if r["pref_macro"])
    fin       = sum(1 for r in records if r["final_pass"])
    fin_incl  = sum(1 for r in records if r["final_pass_incl"])
    return {
        "N":         N,
        "delivered": sum(r["delivered"] for r in records) / N,
        "cs_mu":     (cs_pass / cs_total) if cs_total else 0.0,
        "cs_M":      sum(1 for r in records if r["cs_macro"]) / N,
        "hd_mu":     (hd_pass / hd_total) if hd_total else 0.0,
        "hd_M":      sum(1 for r in records if r["hd_macro"]) / N,
        "pf_mu":     (pf_pass / pf_total) if pf_total else 0.0,
        "pf_M":      (pf_macro_pass / pf_records) if pf_records else 0.0,
        "final":     fin / N,
        "final_p":   fin_incl / N,
        "delta_pref": (fin - fin_incl) / N,
    }


def _bucket_records(records: list[dict], key_fn: Callable[[dict], Any]
                    ) -> dict[Any, dict]:
    """Group per-record metrics by ``key_fn`` and summarise each bucket."""
    buckets: dict[Any, list[dict]] = defaultdict(list)
    for r in records:
        buckets[key_fn(r)].append(r)
    return {k: _summarize_bucket(v) for k, v in buckets.items()}


def _bucket_prefs(records: list[dict], key_fn: Callable[[dict, dict], Any]
                  ) -> dict[Any, dict]:
    """Group per-preference metrics by ``key_fn(pref, record)``.  Returns
    a dict of ``{key -> {n, pass, pass_rate, mean_score, trivial_n,
    trivial_pass_rate, nontrivial_n, nontrivial_pass_rate,
    nontrivial_mean_score}}``.  Preferences with ``passed=None`` are
    counted as failing (defensive).

    The trivial / non-trivial split matters for cross-paradigm reading:
    a paradigm can post a high headline pass rate simply because many of
    its preferences were satisfied VACUOUSLY (a Conditional whose
    condition never fired, a Scoped whose filter matched no day, a
    Compensatory with no primary entity in the plan).  The non-trivial
    columns exclude those, so paradigms can be compared on the
    preferences that actually exercised the planner.

    ``trivial`` here is the flag ``preferences.py`` sets on the
    ``CheckResult`` itself; the two subsets partition the bucket, so
    ``trivial_n + nontrivial_n == n``.

    CAVEAT on grouped Temporal preferences.  ``TemporalPreference``
    under a grouped scope (``per_day`` / ``per_city``) aggregates one
    sub-result per group and sets ``trivial = any(group is trivial)``
    while setting ``passed = all(groups passed)`` -- see the note at
    ``preferences.py::TemporalPreference._aggregate``.  For those rows
    the flag means "CONTAINS vacuous groups", not "was vacuous
    throughout", so ``trivial`` does NOT imply ``passed``: a preference
    that genuinely failed on one day and had no subject on the others
    arrives here as trivial-and-failed.  The partition above then books
    it wholly in ``trivial_n``, which drags ``trivial_pass_rate`` below
    100% and keeps a real failure out of the non-trivial columns.
    Across all five models and both splits this is 15 preferences (10
    always_within, 5 sometime_before) and shifts non-trivial pass rates
    by 0.1-0.3pp -- worth knowing when reading a sub-100%
    ``triv_pass%``, not large enough to change any ordering."""
    buckets: dict[Any, dict] = defaultdict(
        lambda: {"n": 0, "pass": 0, "trivial_n": 0, "trivial_pass": 0,
                  "score_sum": 0.0, "nontrivial_n": 0, "nontrivial_pass": 0,
                  "nontrivial_score_sum": 0.0})
    for r in records:
        for p in r["_prefs"]:
            key = key_fn(p, r)
            if key is None:
                continue
            b = buckets[key]
            b["n"] += 1
            passed = bool(p.get("passed"))
            score  = float(p.get("score") or 0.0)
            b["pass"] += int(passed)
            if bool(p.get("trivial")):
                b["trivial_n"] += 1
                b["trivial_pass"] += int(passed)
            else:
                b["nontrivial_n"] += 1
                b["nontrivial_pass"] += int(passed)
                b["nontrivial_score_sum"] += score
            b["score_sum"] += score
    out: dict[Any, dict] = {}
    for k, b in buckets.items():
        n  = b["n"]
        nt = b["nontrivial_n"]
        out[k] = {
            "n":                 n,
            "pass":              b["pass"],
            "pass_rate":         (b["pass"] / n) if n else 0.0,
            "mean_score":        (b["score_sum"] / n) if n else 0.0,
            "trivial_n":         b["trivial_n"],
            "trivial_pass_rate": (b["trivial_pass"] / b["trivial_n"])
                                  if b["trivial_n"] else 0.0,
            "nontrivial_n":         nt,
            "nontrivial_pass":      b["nontrivial_pass"],
            "nontrivial_pass_rate": (b["nontrivial_pass"] / nt) if nt else 0.0,
            "nontrivial_mean_score": (b["nontrivial_score_sum"] / nt)
                                      if nt else 0.0,
        }
    return out


def _pref_cross_tab(records: list[dict], axis_fn: Callable[[dict], Any],
                    para_fn: Callable[[dict], Any]) -> dict:
    """Two-way per-preference pass_rate table.

    axis_fn : function on record dict producing the outer (row) bucket.
    para_fn : function on pref dict producing the inner (column) bucket.

    Returns ``{axis_key: {para_key: {n, pass, pass_rate, mean_score}, ...},
    'totals': {para_key: {n, pass, pass_rate, mean_score}}}``."""
    outer: dict = defaultdict(lambda: defaultdict(
        lambda: {"n": 0, "pass": 0, "score_sum": 0.0}))
    totals: dict = defaultdict(lambda: {"n": 0, "pass": 0, "score_sum": 0.0})
    for r in records:
        akey = axis_fn(r)
        if akey is None:
            continue
        for p in r["_prefs"]:
            pkey = para_fn(p)
            if pkey is None:
                continue
            for tgt in (outer[akey][pkey], totals[pkey]):
                tgt["n"] += 1
                tgt["pass"] += int(bool(p.get("passed")))
                tgt["score_sum"] += float(p.get("score") or 0.0)

    def _norm(cell):
        n = cell["n"]
        return {"n": n, "pass": cell["pass"],
                "pass_rate": (cell["pass"] / n) if n else 0.0,
                "mean_score": (cell["score_sum"] / n) if n else 0.0}

    result = {"totals": {k: _norm(v) for k, v in totals.items()}}
    for a, inner in outer.items():
        result[a] = {k: _norm(v) for k, v in inner.items()}
    return result


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #

def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _render_constraint_table(title: str, buckets: dict,
                              order: list | None = None) -> str:
    if not buckets:
        return f"=== {title} ===\n  (no records)\n\n"
    keys = order or sorted(buckets.keys(),
                            key=lambda k: str(k) if k is not None else "~")
    lines = [f"=== {title} ==="]
    hdr = (f"  {'bucket':>12s} {'N':>5s} {'deliv':>7s} "
           f"{'cs_μ':>7s} {'cs_M':>7s} {'hd_μ':>7s} {'hd_M':>7s} "
           f"{'pf_μ':>7s} {'pf_M':>7s} "
           f"{'final':>7s} {'final+p':>7s} {'Δpref':>7s}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for k in keys:
        s = buckets.get(k) or {}
        if s.get("N", 0) == 0:
            lines.append(f"  {str(k):>12s} {'':>5s} (no records)")
            continue
        lines.append(
            f"  {str(k):>12s} {s['N']:>5d} {_pct(s['delivered']):>7s} "
            f"{_pct(s['cs_mu']):>7s} {_pct(s['cs_M']):>7s} "
            f"{_pct(s['hd_mu']):>7s} {_pct(s['hd_M']):>7s} "
            f"{_pct(s['pf_mu']):>7s} {_pct(s['pf_M']):>7s} "
            f"{_pct(s['final']):>7s} {_pct(s['final_p']):>7s} "
            f"{_pct(s['delta_pref']):>7s}"
        )
    return "\n".join(lines) + "\n\n"


def _render_pref_table(title: str, buckets: dict,
                       order: list | None = None,
                       label_width: int = 40) -> str:
    if not buckets:
        return f"=== {title} ===\n  (no records)\n\n"
    keys = order or sorted(buckets.keys(),
                            key=lambda k: (-buckets[k]["n"],
                                            str(k) if k is not None else "~"))
    lines = [f"=== {title} ==="]
    # ntriv_* repeats the headline figures with vacuous passes removed,
    # so paradigms can be compared on the preferences that actually
    # exercised the planner.  triv_n + ntriv_n == n by construction.
    hdr = (f"  {'bucket':>{label_width}s} {'n':>5s} {'pass':>5s} "
           f"{'pass%':>7s} {'mean_sc':>8s} "
           f"{'triv_n':>7s} {'triv_pass%':>11s} "
           f"{'ntriv_n':>8s} {'ntriv_pass':>11s} {'ntriv_pass%':>12s} "
           f"{'ntriv_sc':>9s}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for k in keys:
        b = buckets[k]
        lines.append(
            f"  {str(k):>{label_width}s} {b['n']:>5d} {b['pass']:>5d} "
            f"{_pct(b['pass_rate']):>7s} {b['mean_score']:>8.3f} "
            f"{b['trivial_n']:>7d} {_pct(b['trivial_pass_rate']):>11s} "
            f"{b['nontrivial_n']:>8d} {b['nontrivial_pass']:>11d} "
            f"{_pct(b['nontrivial_pass_rate']):>12s} "
            f"{b['nontrivial_mean_score']:>9.3f}"
        )
    return "\n".join(lines) + "\n\n"


def _render_cross_tab(title: str, cross: dict,
                       axis_order: list | None,
                       para_order: list | None,
                       axis_width: int = 12,
                       para_width: int = 24) -> str:
    """Render a two-way pass_rate cross-tab.  ``cross`` is the return
    value of ``_pref_cross_tab``.  Columns are the paradigm-side buckets;
    the ``totals`` key gives an unconditional column-total footer row."""
    totals = cross.get("totals", {})
    all_paras = set(totals)
    for k, inner in cross.items():
        if k == "totals":
            continue
        all_paras.update(inner)
    para_keys = para_order or sorted(all_paras,
                                      key=lambda k: -totals.get(k, {}).get("n", 0))
    axis_keys = axis_order or sorted(k for k in cross if k != "totals")

    lines = [f"=== {title} ==="]
    hdr = f"  {'axis \\\\ paradigm':>{axis_width}s}"
    for pk in para_keys:
        hdr += f"  {str(pk)[:para_width]:>{para_width}s}"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for ak in axis_keys:
        inner = cross.get(ak, {})
        cells = []
        for pk in para_keys:
            cell = inner.get(pk)
            if cell is None or cell["n"] == 0:
                cells.append(f"{'-':>{para_width}s}")
            else:
                cells.append(f"{_pct(cell['pass_rate']) + f' ({cell['n']})':>{para_width}s}")
        lines.append(f"  {str(ak):>{axis_width}s}  " + "  ".join(cells))
    # Total row
    cells = []
    for pk in para_keys:
        cell = totals.get(pk)
        if cell is None or cell["n"] == 0:
            cells.append(f"{'-':>{para_width}s}")
        else:
            cells.append(f"{_pct(cell['pass_rate']) + f' ({cell['n']})':>{para_width}s}")
    lines.append("  " + "-" * (len(hdr) - 2))
    lines.append(f"  {'TOTAL':>{axis_width}s}  " + "  ".join(cells))
    return "\n".join(lines) + "\n\n"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

_STANDARD_ORDERS = {
    "level":            ["easy", "medium", "hard"],
    "profile_drift":    ["aligned", "omission", "inversion"],
    "pairing_type":     ["single", "independent", "overlapping"],
    "pairing_subtype":  ["competing", "non_competing", "--"],
    "paradigm":         ["AtomicPreference", "CompositePreference",
                          "ConditionalPreference", "LexicographicPreference",
                          "NumericPreference", "ScopedPreference",
                          "CompensatoryPreference", "TemporalPreference"],
}


def _order_from_data(records: list[dict], field: str,
                      canonical: list | None = None) -> list:
    """Preferred display order for a bucket: canonical list first
    (filtered to those actually present), then any surprise values
    at the end in sorted order."""
    present = {r[field] for r in records if r[field] is not None}
    if canonical:
        keep = [k for k in canonical if k in present]
        extras = sorted(present - set(canonical), key=str)
        return keep + extras
    return sorted(present, key=lambda k: str(k))


def _pref_paradigm(pref: dict) -> str:
    return pref.get("paradigm") or "?"


def _pref_paradigm_sub(pref: dict) -> str:
    sub = pref.get("sub_paradigm")
    if sub:
        return f"{pref.get('paradigm') or '?'}[{sub}]"
    return pref.get("paradigm") or "?"


# --------------------------------------------------------------------------- #
# Leaf-scope decomposition                                                     #
# --------------------------------------------------------------------------- #
#
# WHY THIS EXISTS.  Headline per-paradigm pass rates are not comparable
# across paradigms, because a paradigm's difficulty is dominated by the
# quantifier scope of its LEAVES rather than by its own combinator.  A
# universally-scoped leaf (``scope="all"``) demands the predicate hold on
# every matching entity in the plan; an existentially-scoped one
# (``scope="any"``) needs a single witness.  The gap between them is far
# larger than the gap between, say, Atomic and Lexicographic -- which is
# how paradigms that are structurally MORE complex than Atomic end up
# with HIGHER pass rates: they are built predominantly from ``any``
# leaves, while plain Atomic preferences skew ``all``.
#
# So we decompose every preference into the ordered tuple of its leaf
# scopes and report performance per (paradigm x leaf-scope tuple).  That
# holds the quantifier load fixed, making the residual differences
# attributable to the combinator itself.  Together with the trivial /
# non-trivial split already carried by ``_bucket_prefs``, this closes the
# two known confounders on cross-paradigm comparison.

# Child-node keys per paradigm, in the order they are traversed.  The
# traversal order fixes the tuple's meaning, so it is part of the
# reported label and must stay stable:
#
#   Composite      children in declaration order
#   Conditional    (condition, then_pref, else_pref)
#   Lexicographic  preferences in priority order (tier 1 first)
#   Compensatory   (primary_ap, margin_ap, secondary_ap)
#   Scoped         (scope_filters ..., inner)  -- filter(s) BEFORE the
#                  inner predicate, matching the reading "among entities
#                  satisfying FILTER, INNER must hold"
#   Temporal       (subject_ap, reference_ap)
#
# Absent optional slots (a Conditional with no else_pref, a Temporal with
# no reference_ap) are skipped, so tuple LENGTH also carries information:
# it is the number of leaf predicates the planner actually has to satisfy.
_LEAF_CHILD_KEYS: dict[str, tuple[str, ...]] = {
    "CompositePreference":     ("children",),
    "ConditionalPreference":   ("condition", "then_pref", "else_pref"),
    "LexicographicPreference": ("preferences",),
    "CompensatoryPreference":  ("primary_ap", "margin_ap", "secondary_ap"),
    "ScopedPreference":        ("scope_filters", "inner"),
    "TemporalPreference":      ("subject_ap", "reference_ap"),
}

# Sort key for scope tuples: fewer universal leaves first, then shorter
# tuples, then lexicographic.  Puts the easy end of the spectrum at the
# top of every table so the cost of each added ``all`` reads downward.
def _scope_tuple_sort_key(t: tuple[str, ...]) -> tuple:
    return (t.count("all"), len(t), t)


def _leaf_scopes(paradigm: str | None, template: Any,
                 out: list[str], depth: int = 0) -> None:
    """Append the scope label of every LEAF predicate under ``template``.

    Labels are ``"all"`` / ``"any"`` for quantified predicates and
    ``"opt"`` for numeric-optimization leaves, which carry a direction
    (min / max) instead of a quantifier and so sit outside the
    all-vs-any axis entirely.

    Nested children arrive as ``{"class": ..., "template": {...}}``
    wrappers while the top-level entry uses ``{"paradigm": ...,
    "template": {...}}``; both spellings are accepted.  Verified to
    classify 3426/3426 leaves across the test and test_large splits with
    no unhandled node shape, so a leaf reaching the final ``return`` is a
    genuine schema surprise rather than an expected gap."""
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
    # Atomic (and any future leaf class): quantified, else optimization.
    scope = template.get("scope")
    if scope in ("all", "any"):
        out.append(scope)
    elif template.get("direction") in ("min", "max"):
        out.append("opt")


def _leaf_scope_tuple(entry: dict) -> tuple[str, ...]:
    """Leaf-scope tuple for one ``preferences_json`` entry."""
    out: list[str] = []
    _leaf_scopes(entry.get("paradigm"), entry.get("template") or {}, out)
    return tuple(out)


def _scope_index(record: dict) -> dict[tuple, tuple[str, ...]]:
    """``{(paradigm, bank_id) -> leaf-scope tuple}`` for one record.

    ``(paradigm, bank_id)`` is the join key between an evaluated
    preference in the detailed JSONL and its structural template in the
    HF ``preferences_json`` column; both carry the pair verbatim.
    Cached on the record so repeated bucketing passes walk each template
    once."""
    cached = record.get("_scope_index")
    if cached is not None:
        return cached
    idx: dict[tuple, tuple[str, ...]] = {}
    for entry in record.get("_preferences_json") or []:
        bank_id = entry.get("bank_id")
        if bank_id is None:
            continue
        idx[(entry.get("paradigm"), int(bank_id))] = _leaf_scope_tuple(entry)
    record["_scope_index"] = idx
    return idx


def _pref_scope_tuple(pref: dict, record: dict) -> tuple[str, ...] | None:
    """Leaf-scope tuple for an evaluated preference, or None when the
    record's structural metadata is absent (``--no-hf-meta``) or the
    join key is missing."""
    bank_id = pref.get("bank_id")
    if bank_id is None:
        return None
    return _scope_index(record).get((pref.get("paradigm"), int(bank_id)))


def _fmt_scope_tuple(t: tuple[str, ...]) -> str:
    """Spaced form, for MATRIX AXIS labels only (rendered by
    ``_render_scope_matrix``, which lays out columns by width rather
    than by whitespace tokens)."""
    return "(" + ", ".join(t) + ")" if t else "(none)"


def _fmt_scope_tuple_compact(t: tuple[str, ...]) -> str:
    """Whitespace-free form, for BUCKET labels in ``_render_pref_table``.

    Report consumers (``latex._parse_report`` and every figure script
    built on it) tokenise a table row with ``line.split()`` and map the
    tokens onto the header positionally.  A bucket label containing a
    space therefore shifts every numeric column right by one and is read
    as silently wrong data rather than failing loudly -- so bucket labels
    must stay a single token."""
    return "(" + ",".join(t) + ")" if t else "(none)"


def _pref_paradigm_scope(pref: dict, record: dict) -> str | None:
    """Bucket label ``"<Paradigm>(any,all)"`` -- single token by design."""
    t = _pref_scope_tuple(pref, record)
    if t is None:
        return None
    return f"{_pref_paradigm(pref)}{_fmt_scope_tuple_compact(t)}"


def _pref_paradigm_sub_scope(pref: dict, record: dict) -> str | None:
    """Bucket label ``"<Paradigm>[<sub>](any,all)"`` -- single token."""
    t = _pref_scope_tuple(pref, record)
    if t is None:
        return None
    return f"{_pref_paradigm_sub(pref)}{_fmt_scope_tuple_compact(t)}"


def _scope_signature(pref: dict, record: dict) -> str | None:
    """Scope tuple collapsed to its QUANTIFIER LOAD, discarding both leaf
    order and paradigm identity: ``"2 leaf | 1 all, 1 any"``.

    This is the axis on which the all-vs-any effect is cleanest -- it
    pools every paradigm that imposes the same quantifier burden, so the
    marginal cost of turning one ``any`` into an ``all`` is read directly
    down the column."""
    t = _pref_scope_tuple(pref, record)
    if t is None:
        return None
    n_all, n_any, n_opt = t.count("all"), t.count("any"), t.count("opt")
    parts = [f"{n} {lbl}" for n, lbl in
             ((n_all, "all"), (n_any, "any"), (n_opt, "opt")) if n]
    # Single token, for the same positional-parsing reason as
    # ``_fmt_scope_tuple_compact``: "2leaf|1all,1any".
    return f"{len(t)}leaf|" + (",".join(parts).replace(" ", "") if parts else "-")


def _render_scope_matrix(title: str, buckets: dict[tuple, dict],
                         *, metric: str = "pass_rate",
                         count_key: str = "n",
                         pass_key: str = "pass",
                         row_width: int = 26,
                         col_width: int = 16) -> str:
    """Pivot ``{(row, col) -> bucket}`` into a matrix of ``metric``.

    EVERY populated cell is rendered, however small its n -- the count is
    printed alongside the rate so the reader judges reliability directly
    rather than having the table decide for them.  ``.`` means the
    combination does not occur in the data at all, not that it was
    suppressed.

    ``pass_key`` must be the numerator matching ``count_key``'s
    denominator (``pass``/``n`` for headline rates,
    ``nontrivial_pass``/``nontrivial_n`` for the vacuous-excluded view);
    mismatching them silently produces nonsense marginals."""
    if not buckets:
        return f"=== {title} ===\n  (no records)\n\n"
    rows = sorted({r for r, _ in buckets})
    cols = sorted({c for _, c in buckets}, key=_scope_tuple_sort_key)

    def _cell(keys: list[tuple]) -> tuple[float, int]:
        num = sum(buckets[k][pass_key] for k in keys if k in buckets)
        den = sum(buckets[k][count_key] for k in keys if k in buckets)
        return ((num / den) if den else 0.0), den

    lines = [f"=== {title} ===",
             f"  metric: {metric} ({pass_key}/{count_key})   "
             f"'.' = combination absent from the data (nothing is suppressed)"]
    hdr = f"  {'paradigm \\ scope':>{row_width}s}"
    for c in cols:
        hdr += f"  {_fmt_scope_tuple(c)[:col_width]:>{col_width}s}"
    hdr += f"  {'ROW':>{col_width}s}"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for r in rows:
        cells = []
        for c in cols:
            b = buckets.get((r, c))
            if b is None or b[count_key] == 0:
                cells.append(f"{'.':>{col_width}s}")
            else:
                cells.append(
                    f"{_pct(b[metric]) + f' ({b[count_key]})':>{col_width}s}")
        rate, n = _cell([(r, c) for c in cols])
        cells.append(f"{_pct(rate) + f' ({n})':>{col_width}s}")
        lines.append(f"  {str(r)[:row_width]:>{row_width}s}  " + "  ".join(cells))
    lines.append("  " + "-" * (len(hdr) - 2))
    cells = []
    for c in cols:
        rate, n = _cell([(r, c) for r in rows])
        cells.append(f"{_pct(rate) + f' ({n})':>{col_width}s}" if n
                     else f"{'.':>{col_width}s}")
    rate, n = _cell(list(buckets))
    cells.append(f"{_pct(rate) + f' ({n})':>{col_width}s}")
    lines.append(f"  {'COLUMN':>{row_width}s}  " + "  ".join(cells))
    return "\n".join(lines) + "\n\n"


# --------------------------------------------------------------------------- #
# Paradigm-detail rendering                                                    #
# --------------------------------------------------------------------------- #

def _histogram(values: list[float], *, bins: int = 10,
                lo: float = 0.0, hi: float = 1.0,
                width: int = 40) -> str:
    """Text histogram over ``[lo, hi]`` in ``bins`` equal-width buckets.
    Empty bins collapse to a single ``|`` marker on the left; otherwise
    a proportional bar of ``█`` characters up to ``width``."""
    if not values:
        return "  (no values)"
    counts = [0] * bins
    step = (hi - lo) / bins
    for v in values:
        if v < lo:
            counts[0] += 1
        elif v >= hi:
            counts[-1] += 1
        else:
            idx = int((v - lo) / step) if step > 0 else 0
            counts[min(idx, bins - 1)] += 1
    mx = max(counts) or 1
    lines: list[str] = []
    for i, c in enumerate(counts):
        lo_i = lo + i * step
        hi_i = lo_i + step
        bar = "█" * int(round(width * c / mx)) if c else "|"
        lines.append(f"  [{lo_i:.2f}, {hi_i:.2f})  {c:>5d}  {bar}")
    return "\n".join(lines)


def _render_paradigm_detail_sections(x: dict) -> list[str]:
    """Text summaries per paradigm-detail extract.  ``x`` is the return
    value of the ``extract_*`` calls keyed by analysis code."""
    parts: list[str] = []

    # A1 -- Atomic [all]-scope partial-credit distribution.
    a1 = x["A1_atomic_partial_credit"]
    parts.append("=== [A1] Atomic [all]-scope partial-credit distribution ===\n"
                 f"  values: {len(a1['values'])}\n"
                 + _histogram(a1["values"]) + "\n\n"
                 "  Split by profile_drift:\n"
                 + "\n".join(f"    {k:>10s}  n={len(v):>3d}  "
                              f"mean={_mean(v):.3f}"
                              for k, v in a1["by_drift"].items())
                 + "\n\n")

    # CD -- Conditional dynamics.
    cd = x["CD_conditional"]
    if cd["n"]:
        cd_rate = lambda a, b: (a / b) if b else 0.0
        parts.append("=== [CD] Conditional dynamics ===\n"
                     f"  total Conditional preferences        : {cd['n']}\n"
                     f"  [CD1] condition-fires rate           : "
                     f"{_pct(cd_rate(cd['n_cond_fired'], cd['n']))}  "
                     f"({cd['n_cond_fired']}/{cd['n']})\n"
                     f"  [CD2] active-pass | condition fired  : "
                     f"{_pct(cd_rate(cd['n_cond_fired_active_pass'], cd['n_cond_fired']))}  "
                     f"({cd['n_cond_fired_active_pass']}/{cd['n_cond_fired']})\n"
                     f"  [CD3] trivial-pass share (true+no-then): "
                     f"{_pct(cd_rate(cd['n_trivial_true_no_then'], cd['n']))}  "
                     f"({cd['n_trivial_true_no_then']}/{cd['n']})\n"
                     f"  [CD3] trivial-pass share (false+no-else): "
                     f"{_pct(cd_rate(cd['n_trivial_false_no_else'], cd['n']))}  "
                     f"({cd['n_trivial_false_no_else']}/{cd['n']})\n\n")

    # L1 / L2 -- Lex per-tier and tier-gap.
    l = x["L1_L2_lex_tiers"]
    if l["n"]:
        parts.append("=== [L1] Lex per-tier pass rate ===\n"
                     "  tier      pass    total   pass%")
        for i, (p_i, t_i) in enumerate(zip(l["tier_pass"], l["tier_total"])):
            rate = (p_i / t_i) if t_i else 0.0
            parts[-1] += f"\n  P{i+1:<2d}      {p_i:>4d}    {t_i:>4d}    {_pct(rate)}"
        parts[-1] += "\n\n"
        parts.append("=== [L2] Lex tier-1-only vs all-tiers-pass gap ===\n"
                     f"  tier-1 pass          : {l['n_tier1_pass']}/{l['n']} "
                     f"({_pct(l['n_tier1_pass']/l['n'])})\n"
                     f"  all-tiers pass       : {l['n_all_tiers_pass']}/{l['n']} "
                     f"({_pct(l['n_all_tiers_pass']/l['n'])})\n"
                     f"  gap (tier-1 − all)   : "
                     f"{_pct((l['n_tier1_pass'] - l['n_all_tiers_pass'])/l['n'])}\n\n")

    # N1 -- Numeric quantile-position density.
    n1 = x["N1_numeric_quantile_position"]
    parts.append("=== [N1] Numeric quantile-position density ===\n"
                 f"  direction=max  (n={len(n1['max'])}, mean={_mean(n1['max']):.3f})\n"
                 + _histogram(n1["max"]) + "\n\n"
                 f"  direction=min  (n={len(n1['min'])}, mean={_mean(n1['min']):.3f})\n"
                 + _histogram(n1["min"]) + "\n\n")

    # CO -- Compensatory tiers.
    co = x["CO_compensatory_tiers"]
    if co["n_evaluations"]:
        parts.append("=== [CO1] Compensatory tier fractions (aggregate) ===")
        for t in ("tier1", "tier2_comp", "tier2_uncomp", "tier3"):
            parts[-1] += (f"\n  {t:>13s}  {co['aggregate'][t]:>5d}  "
                          f"{_pct(co['aggregate_frac'][t])}")
        util = co["utilisation"]
        util_rate = (util["tier2_comp"] / util["tier2_total"]) if util["tier2_total"] else 0.0
        parts[-1] += ("\n\n=== [CO2] Compensation-utilisation ratio ===\n"
                      f"  T2_comp / (T2_comp + T2_uncomp) = "
                      f"{util['tier2_comp']}/{util['tier2_total']} "
                      f"= {_pct(util_rate)}\n\n")
        parts.append("=== [CO3] Compensatory tier fractions × cross_entity ===")
        for key in ("same", "cross"):
            frac = co["by_cross_entity_frac"].get(key)
            if not frac:
                continue
            parts[-1] += f"\n  {key:>6s}:  " + "  ".join(
                f"{t}={_pct(frac[t])}" for t in ("tier1","tier2_comp","tier2_uncomp","tier3"))
        parts[-1] += "\n\n=== [CO4] Compensatory tier fractions × profile_drift ==="
        for drift in ("aligned", "omission", "inversion"):
            frac = co["by_drift_frac"].get(drift)
            if not frac:
                continue
            parts[-1] += f"\n  {drift:>10s}:  " + "  ".join(
                f"{t}={_pct(frac[t])}" for t in ("tier1","tier2_comp","tier2_uncomp","tier3"))
        parts[-1] += "\n\n"

    # T4 -- Temporal.within first-subject-position density.
    t4 = x["T4_temporal_within_first_pos"]
    if t4["normalised"]:
        parts.append("=== [T4] Temporal.within first-subject-position "
                     "(normalised 0=start, 1=end) ===\n"
                     f"  groups: {len(t4['normalised'])}, mean="
                     f"{_mean(t4['normalised']):.3f}, "
                     f"no-subject groups: {t4['n_no_subject_groups']}\n"
                     + _histogram(t4["normalised"]) + "\n\n")

    # T6 -- Temporal.timed per-window-day pass curve.
    t6 = x["T6_temporal_timed_day_curve"]
    if t6["n_groups"]:
        parts.append("=== [T6] Temporal.timed per-window-day pass curve ===\n"
                     f"  groups: {t6['n_groups']}, no-overlap groups: "
                     f"{t6['n_no_overlap']}\n"
                     "  window position → mean day_score (higher = better)")
        for i, (v, c) in enumerate(zip(t6["window_pos_curve"], t6["window_pos_counts"])):
            bar = "█" * int(round(30 * v))
            parts[-1] += f"\n  pos {i:>2d}  n={c:>4d}  {v:5.3f}  {bar}"
        parts[-1] += "\n\n"

    # T9 -- Temporal.pair strict vs non-strict.
    t9 = x["T9_temporal_pair_strict"]
    parts.append("=== [T9] Temporal.pair strict vs non-strict pass rate ===\n"
                 f"      strict:  n={t9['strict']['n']:>4d}  "
                 f"pass={t9['strict']['pass']:>4d}  "
                 f"rate={_pct(t9['strict']['pass_rate'])}\n"
                 f"  non_strict:  n={t9['non_strict']['n']:>4d}  "
                 f"pass={t9['non_strict']['pass']:>4d}  "
                 f"rate={_pct(t9['non_strict']['pass_rate'])}\n\n")

    # T11 -- Temporal.pair always_within passed_subjects.
    t11 = x["T11_temporal_always_within"]
    if t11["values"]:
        parts.append("=== [T11] Temporal.pair always_within passed_subjects ===\n"
                     f"  groups: {t11['n_groups']}, mean="
                     f"{_mean(t11['values']):.3f}, "
                     f"no-subject groups: {t11['n_no_subject_groups']}\n"
                     + _histogram(t11["values"]) + "\n\n")

    # T12 -- Temporal.aggregate by scope.
    t12 = x["T12_temporal_aggregate_by_scope"]
    if t12:
        parts.append("=== [T12] Temporal.aggregate pass rate × scope ===\n"
                     "  scope           n   pass  pass%  mean passed_groups")
        for scope in sorted(t12, key=lambda s: -t12[s]["n"]):
            b = t12[scope]
            parts[-1] += (f"\n  {scope:>13s}  {b['n']:>3d}  {b['pass']:>3d}  "
                          f"{_pct(b['pass_rate']):>5s}  "
                          f"{b['mean_passed_groups']:.3f}")
        parts[-1] += "\n\n"

    # X3 -- Trivial-pass audit.
    x3 = x["X3_trivial_pass_audit"]
    parts.append("=== [X3] Trivial-pass audit (paradigm-side) ===\n"
                 f"  total preferences        : {x3['total']}\n"
                 f"  trivial (any kind)       : {x3['n_trivial_flagged']}  "
                 f"({_pct(x3['n_trivial_flagged']/x3['total']) if x3['total'] else '-'})\n"
                 f"  trivial pass rate        : "
                 f"{_pct(x3['trivial_pass_rate'])}\n"
                 "  by kind:")
    for kind, cnt in sorted(x3["n_trivial_by_kind"].items(), key=lambda kv: -kv[1]):
        parts[-1] += f"\n    {kind:>32s}: {cnt}"
    parts[-1] += "\n  by paradigm  (n_trivial / n_total, trivial-pass-rate):"
    for para in sorted(x3["by_paradigm"], key=lambda p: -x3["by_paradigm"][p]["n_total"]):
        b = x3["by_paradigm"][para]
        parts[-1] += (f"\n    {para:>28s}: {b['n_trivial']:>3d}/{b['n_total']:>3d}, "
                      f"{_pct(b['trivial_pass_rate'])}")
    parts[-1] += "\n\n"

    # D-numeric -- Drift impact on numeric sub-preferences (path-aware).
    dn = x.get("D_numeric_drift_impact") or {}
    if dn and dn.get("counts", {}).get("numeric_hits", 0) > 0:
        c = dn["counts"]
        parts.append(
            "=== [D-num] Path-aware drift impact on numeric sub-preferences ===\n"
            f"  records scanned              : {c['records_scanned']}\n"
            f"  drift sources scanned        : {c['sources_scanned']}\n"
            f"  numeric-leaf sources         : {c['numeric_hits']}\n"
            f"  resolved to a sub_result     : {c['sub_result_hits']}\n"
            f"  fell back to top-level pref  : {c['fallback_top_level']}\n"
            "  (Numeric leaves = Atomic with numeric op {>=,<=,>,<,==,!=,range} "
            "or NumericPreference optimization.  Sub-preference outcome is "
            "read at the exact ``path`` of each drift source; falls back to "
            "the parent pref when the paradigm has no serialised sub_result "
            "at that path.)\n"
            "\n"
            "  Numeric sub-preferences by drift bucket:\n"
            "  (aligned_* buckets isolate non-drifted sources by the\n"
            "   OWNING record's overall drift mode; aligned_all is the\n"
            "   pooled baseline across all record-modes.)\n"
            f"  {'bucket':>32s}  {'n':>5s}  {'pass':>5s}  {'pass%':>7s}  {'mean_score':>10s}"
        )
        _BUCKET_ORDER = (
            "aligned_all",
            "aligned_in_aligned_records",
            "aligned_in_omission_records",
            "aligned_in_inversion_records",
            "omission",
            "inversion",
        )
        for bucket in _BUCKET_ORDER:
            b = dn.get(bucket) or {}
            n = b.get("n", 0)
            parts[-1] += (f"\n  {bucket:>32s}  {n:>5d}  {b.get('n_pass', 0):>5d}  "
                          f"{_pct(b.get('pass_rate', 0.0)):>7s}  "
                          f"{b.get('mean_score', 0.0):>10.3f}")
        parts[-1] += "\n\n"

        # Per-shape breakdown for high-yield inspection.  ``by_shape``
        # keys are ``"entity|attr|op|scope"`` strings (JSON-encodable);
        # render into a legible parenthetical form for the table.
        by_shape = dn.get("by_shape") or {}
        if by_shape:
            parts.append(
                "  Per-shape (entity, attr, op, scope):  "
                "columns are pooled aligned baseline + the two drifted buckets.\n"
                f"  {'shape':<55s}  {'aligned_all':>16s}  {'omission':>16s}  {'inversion':>16s}")
            # Sort by total-n desc across the drifted + pooled-aligned buckets.
            def _row_n(inner):
                return sum(inner.get(d, {}).get("n", 0)
                           for d in ("aligned_all", "omission", "inversion"))
            sorted_shapes = sorted(by_shape.items(), key=lambda kv: -_row_n(kv[1]))
            for shape_key, inner in sorted_shapes[:25]:
                parts_of_shape = shape_key.split("|")
                shape_render = f"({', '.join(parts_of_shape)})"
                cells = []
                for d in ("aligned_all", "omission", "inversion"):
                    b = inner.get(d) or {}
                    n = b.get("n", 0)
                    if n == 0:
                        cells.append(f"{'--':>16s}")
                    else:
                        cells.append(f"{_pct(b['pass_rate'])+f' ({n})':>16s}")
                parts[-1] += f"\n  {shape_render:<55s}  " + "  ".join(cells)
            parts[-1] += "\n\n"

    # D-set -- Drift impact on set-valued categorical sub-preferences.
    ds = x.get("D_set_drift_impact") or {}
    if ds and ds.get("counts", {}).get("set_leaves", 0) > 0:
        c = ds["counts"]
        cov = ds.get("by_coverage") or {}
        act = ds.get("by_action") or {}
        parts.append(
            "=== [D-set] Path-aware drift impact on set-valued categorical "
            "sub-preferences ===\n"
            f"  records scanned              : {c.get('records_scanned', 0)}\n"
            f"  drift-source groups scanned  : {c.get('groups_scanned', 0)}\n"
            f"  set-valued categorical leaves: {c.get('set_leaves', 0)}\n"
            f"  resolved to a sub_result     : {c.get('sub_result_hits', 0)}\n"
            f"  fell back to top-level pref  : {c.get('fallback_top_level', 0)}\n"
            "\n"
            "  Categorical drift is recorded ONE SOURCE PER ELEMENT, so the\n"
            "  sources addressing a leaf are regrouped by (paradigm, bank_id,\n"
            "  path) -- within a preference, never across -- and the leaf is\n"
            "  split into its drifted members and the surviving remainder.\n"
            "  COVERAGE = |drifted| / |value set| grades the perturbation:\n"
            "    none    -- leaf traced but no member drifted (the control:\n"
            "               same attribute / op / scope, same construction\n"
            "               path, simply not perturbed)\n"
            "    partial -- some members drifted, some survived\n"
            "    full    -- every member drifted\n"
            "  Coverage class is a DATASET property, so its n is identical\n"
            "  across models; only the pass rates differ.\n"
            "\n"
            "  By drift coverage:\n"
            f"  {'coverage':>10s}  {'n':>5s}  {'pass':>5s}  {'pass%':>7s}  "
            f"{'mean_score':>10s}  {'mean_cov':>9s}"
        )
        base = (cov.get("none") or {}).get("pass_rate")
        for k in ("none", "partial", "full"):
            b = cov.get(k) or {}
            n = b.get("n", 0)
            delta = ("" if base is None or k == "none" or n == 0
                     else f"   ({100*(b['pass_rate']-base):+.1f}pp vs none)")
            parts[-1] += (f"\n  {k:>10s}  {n:>5d}  {b.get('n_pass', 0):>5d}  "
                          f"{_pct(b.get('pass_rate', 0.0)):>7s}  "
                          f"{b.get('mean_score', 0.0):>10.3f}  "
                          f"{b.get('mean_coverage', 0.0):>9.3f}{delta}")
        parts[-1] += "\n\n  By drift action (same leaves, regrouped):\n"
        parts[-1] += (f"  {'action':>10s}  {'n':>5s}  {'pass':>5s}  "
                      f"{'pass%':>7s}  {'mean_score':>10s}")
        for k in ("none", "drop", "invert", "mixed"):
            b = act.get(k)
            if not b:
                continue
            delta = ("" if base is None or k == "none"
                     else f"   ({100*(b['pass_rate']-base):+.1f}pp vs none)")
            parts[-1] += (f"\n  {k:>10s}  {b['n']:>5d}  {b['n_pass']:>5d}  "
                          f"{_pct(b['pass_rate']):>7s}  "
                          f"{b['mean_score']:>10.3f}{delta}")
        parts[-1] += "\n\n"

        by_attr = ds.get("by_attribute") or {}
        if by_attr:
            parts.append(
                "  Per categorical attribute x coverage:\n"
                f"  {'attribute':<16s}  {'none':>16s}  {'partial':>16s}  {'full':>16s}")
            for attr, inner in sorted(
                    by_attr.items(),
                    key=lambda kv: -sum(v.get("n", 0) for v in kv[1].values())):
                cells = []
                for k in ("none", "partial", "full"):
                    b = inner.get(k) or {}
                    n = b.get("n", 0)
                    cells.append(f"{'--':>16s}" if n == 0
                                 else f"{_pct(b['pass_rate'])+f' ({n})':>16s}")
                parts[-1] += f"\n  {str(attr):<16s}  " + "  ".join(cells)
            parts[-1] += "\n\n"

        ex = ds.get("examples") or []
        if ex:
            parts.append(
                "  Partial-coverage examples (the disintegration case -- one\n"
                "  preference's value set split into drifted and surviving):\n")
            for e in ex[:8]:
                parts[-1] += (
                    f"    id={e['id']:<5} {e['paradigm']}#{e['bank_id']}  "
                    f"{e['attribute']} {e['op']} [{e['scope']}]  "
                    f"cov={e['coverage']}  {e['drift_action']}  "
                    f"passed={e['passed']}\n"
                    f"        drifted={e['drifted']}  surviving={e['surviving']}\n")
            parts[-1] += "\n"

    # G-gate -- does drift on a GATING sub-node manufacture vacuous passes?
    g = x.get("G_gate_drift_triviality") or {}
    if g and g.get("counts", {}).get("gated_prefs", 0) > 0:
        bc = g.get("by_class") or {}
        parts.append(
            "=== [G-gate] Drift on a GATING sub-node vs triviality ===\n"
            "  A GATE is a sub-node whose non-firing makes the WHOLE preference\n"
            "  vacuously satisfied rather than failed:\n"
            "     ConditionalPreference.condition    -- condition false\n"
            "     TemporalPreference.subject_ap      -- no subject to trigger\n"
            "  (ScopedPreference has a gate too -- scope_filters -- but it is\n"
            "   never drifted in either split, so it is excluded rather than\n"
            "   contributing an all-zero column.)\n"
            "  Hypothesis: drift on the gate tells the planner to disregard the\n"
            "  very trigger the preference hinges on.  If it complies the gate\n"
            "  never fires, the preference is satisfied VACUOUSLY, and it is\n"
            "  booked as a PASS -- so drift manufactures easy passes instead of\n"
            "  making the task harder.\n"
            "\n"
            "  UNIT = one evaluated PREFERENCE, restricted to gated paradigms.\n"
            "  The three classes partition them exhaustively, so the column\n"
            "  total is the number of gated preferences in the split.  Class\n"
            "  membership is a DATASET property: n is identical across models,\n"
            "  only trivial%/pass% move.  'nongate_drifted' is the\n"
            "  discriminating control -- drift landed on the preference but not\n"
            "  on its gate, so a jump seen ONLY in gate_drifted cannot be a\n"
            "  generic 'this record was drifted' effect.\n"
            "\n"
            "  READ THE LAST COLUMN.  'pass%' is the OVERALL rate and mixes\n"
            "  vacuous passes with real ones, so a gate-drift effect can be\n"
            "  fully masked: Conditional gate drift leaves overall pass rate\n"
            "  flat while the still-evaluated subset degrades ~10pp.  The\n"
            "  non-trivial column is the one that answers 'did drift hurt?'.\n"
            "  'triv pass%' is 100% for genuine full vacuity (Conditional) but\n"
            "  below 100% for Temporal, whose trivial flag is the\n"
            "  any(group vacuous) partial marker and so contains real failures.\n"
            "\n"
            f"  {'class':>18s}  {'n':>5s}  {'triv':>5s}  {'triv%':>7s}  "
            f"{'OVERALL pass%':>14s}  {'triv pass%':>11s}  "
            f"{'ntriv n':>8s}  {'NON-TRIV pass%':>15s}"
        )
        for k in ("undrifted", "nongate_drifted", "gate_drifted"):
            b = bc.get(k) or {}
            n = b.get("n", 0)
            parts[-1] += (f"\n  {k:>18s}  {n:>5d}  {b.get('n_trivial', 0):>5d}  "
                          f"{_pct(b.get('trivial_rate', 0.0)):>7s}  "
                          f"{_pct(b.get('pass_rate', 0.0)):>14s}  "
                          f"{_pct(b.get('trivial_pass_rate', 0.0)):>11s}  "
                          f"{b.get('nontrivial_n', 0):>8d}  "
                          f"{_pct(b.get('nontrivial_pass_rate', 0.0)):>15s}")
        parts[-1] += "\n\n"
        bp = g.get("by_paradigm") or {}
        if bp:
            parts.append(
                "  Per gated paradigm (trivial% / NON-TRIVIAL pass%):\n"
                f"  {'paradigm':<24s}  {'undrifted':>20s}  {'non-gate':>20s}  {'GATE':>20s}")
            for para in sorted(bp, key=lambda p: -sum(
                    v.get("n", 0) for v in bp[p].values())):
                cells = []
                for k in ("undrifted", "nongate_drifted", "gate_drifted"):
                    b = bp[para].get(k) or {}
                    n = b.get("n", 0)
                    cells.append(f"{'--':>20s}" if n == 0 else
                                 f"{'%.0f/%.0f%% (n=%d)' % (100*b['trivial_rate'], 100*b['nontrivial_pass_rate'], n):>20s}")
                parts[-1] += f"\n  {para:<24s}  " + "  ".join(cells)
            parts[-1] += "\n"
            gp = g.get("gate_positions_seen") or {}
            parts[-1] += (f"\n  gate positions actually drifted: {gp}\n"
                          "  (a gated paradigm with 0 in the GATE column simply never has\n"
                          "   that position drifted in this split -- not a filter)\n\n")

    # M-pairs -- matched-pairs drifted-vs-undrifted contrast.
    mp = x.get("M_drift_matched_pairs") or {}
    if mp and mp.get("counts", {}).get("strata_contributing", 0) > 0:
        c = mp["counts"]
        pl, wi = mp["pooled"], mp["mean_within"]
        parts.append(
            "=== [M-pairs] Matched-pairs drift contrast ===\n"
            "  The ideal design -- two profiles for ONE query, one drifted and\n"
            "  one not -- is unavailable: each record carries a single profile.\n"
            "  This approximates it.  bank_id pins the exact bank entry a\n"
            "  preference instantiates (hence its structure and leaf scopes),\n"
            "  and drift mode is assigned by md5-hash rank independently of\n"
            "  preference content, so within a stratum drifted-vs-undrifted is\n"
            "  close to random assignment.\n"
            f"  stratum key                : {tuple(mp.get('strata_keys') or ())}\n"
            "  ('level' is omitted on purpose -- drift assignment is already\n"
            "   stratified per level, so matching on it only costs power.)\n"
            f"  strata total               : {c.get('strata_total', 0)}\n"
            f"  contributing (both arms)   : {c.get('strata_contributing', 0)}\n"
            f"  discarded (one arm only)   : {c.get('strata_no_contrast', 0)}\n"
            f"  preferences used           : {c.get('prefs_used', 0)}\n"
            "\n"
            f"  pooled   drifted   : {_pct(pl['drifted']['pass_rate'])} "
            f"(n={pl['drifted']['n']})\n"
            f"  pooled   undrifted : {_pct(pl['undrifted']['pass_rate'])} "
            f"(n={pl['undrifted']['n']})\n"
            f"  pooled   delta     : {pl['delta_pp']:+.1f}pp   "
            "(micro: size-weighted)\n"
            f"  within   delta     : {wi['delta_pp']:+.1f}pp   "
            f"(macro: mean over {wi['n_strata']} strata, sd={wi['sd_pp']:.1f}pp, "
            f"se={wi['sd_pp']/max(wi['n_strata'],1)**0.5:.1f}pp)\n"
            "  NOTE sd is spread ACROSS strata, not a standard error.  Strata\n"
            "  hold 1-3 preferences, so per-stratum rates are mostly 0%/100%\n"
            "  and a large sd is expected -- read the mean.\n"
        )
        ba = mp.get("by_action") or {}
        if ba:
            parts[-1] += "\n  by drift action:\n"
            for a, v in sorted(ba.items()):
                parts[-1] += (f"    {a:<10s} delta={v['delta_pp']:+6.1f}pp  "
                              f"strata={v['n_strata']:>4d}\n")
        bpp = mp.get("by_paradigm") or {}
        if bpp:
            parts[-1] += "\n  by paradigm:\n"
            for p_, v in sorted(bpp.items(), key=lambda kv: -kv[1]["n_strata"]):
                parts[-1] += (f"    {p_:<26s} delta={v['delta_pp']:+6.1f}pp  "
                              f"strata={v['n_strata']:>4d}\n")
        parts[-1] += "\n"

    # R-real -- realization measures: what the plan ENACTED, not what it scored.
    rr = x.get("R_realization_by_drift") or {}
    meas = (rr or {}).get("measures") or {}
    if meas:
        parts.append(
            "=== [R-real] Realization measures by OWN-drift action ===\n"
            "  Pass rate is gameable as a drift-impact instrument: vacuity\n"
            "  (Conditional gate drift) books a PASS over a real degradation,\n"
            "  and obligation-shrinking (Temporal subject drift) removes the\n"
            "  chances to fail without ever being flagged trivial.  A\n"
            "  REALIZATION measure asks instead how much the plan actually\n"
            "  ENACTED the preference, on a continuous scale with no pass\n"
            "  threshold: vacuity yields an undefined value (excluded, not a\n"
            "  free 1.0) and shrinking does not inflate it, because the\n"
            "  quantity is per-opportunity rather than per-preference.\n"
            "\n"
            "  Bucketed by whether THIS preference was itself drifted -- not by\n"
            "  the record's profile_drift label, which is diluted by\n"
            "  hard-constraint-only drift and by untouched sibling preferences.\n"
            "  'not_in' realization is omitted: it saturates at exactly 1.000\n"
            "  in both splits, so it has no variance to discriminate with.\n"
            "\n"
            f"  {'measure':<32s} {'bucket':<11s} {'n':>5s} {'mean':>8s} "
            f"{'median':>8s} {'se':>7s}  {'delta vs undrifted':>18s}")
        for mname in sorted(meas):
            per = meas[mname]
            base = (per.get("undrifted") or {}).get("mean")
            for b in ("undrifted", "drop", "invert", "mixed"):
                st_ = per.get(b)
                if not st_ or not st_.get("n"):
                    continue
                dl = ("" if b == "undrifted" or base is None
                      else f"{st_['mean'] - base:+.3f}")
                parts[-1] += (f"\n  {mname:<32s} {b:<11s} {st_['n']:>5d} "
                              f"{st_['mean']:>8.3f} {st_['median']:>8.3f} "
                              f"{st_['se']:>7.3f}  {dl:>18s}")
            parts[-1] += "\n"
        w = rr.get("numeric_within_bank") or {}
        if w.get("n_bank_entries"):
            parts[-1] += (
                "\n  numeric quantile_position, WITHIN-BANK paired delta\n"
                "  (each bank entry is its own control, so differing\n"
                "   attributes and candidate pools cancel):\n"
                f"    bank entries={w['n_bank_entries']}  "
                f"mean delta={w['mean_delta']:+.4f}  "
                f"sd={w['sd']:.4f}  se={w['se']:.4f}\n")
        parts[-1] += "\n"

    # S-supp -- Suppression Index (needs --plan-file; absent otherwise).
    sp = x.get("S_suppression_index") or {}
    if sp.get("n_leaves"):
        c = sp.get("counts") or {}
        parts.append(
            "=== [S-supp] Suppression Index: did the planner AVOID the "
            "drifted categories? ===\n"
            "  SI = mean realization(drifted members)\n"
            "     / mean realization(surviving members),\n"
            "  computed WITHIN one partial-coverage categorical leaf.\n"
            "     SI = 1.0  drifted and surviving members used equally\n"
            "     SI < 1.0  the planner avoided the drifted categories\n"
            "     SI > 1.0  it over-used them\n"
            "\n"
            "  Both arms come from the SAME plan, entity type, value set,\n"
            "  record and model, so every record-level confound (level, days,\n"
            "  scope mix, paradigm, trip difficulty) cancels exactly.  This is\n"
            "  a paired design needing no counterfactual, and unlike pass rate\n"
            "  it cannot be gamed by vacuity or obligation-shrinking: a\n"
            "  suppressed member simply fails to appear in the plan.\n"
            "\n"
            f"  leaves scored          : {sp['n_leaves']}\n"
            f"  mean SI                : {sp['mean_si']:.3f}"
            f"   (se {sp['se']:.3f})\n"
            f"  median SI              : {sp['median_si']:.3f}\n"
            f"  leaves with SI < 1     : {sp['n_below_1']}/{sp['n_leaves']}"
            f"  ({100.0*sp['n_below_1']/max(sp['n_leaves'],1):.1f}%)\n"
            f"  leaf accounting        : {c}\n")
        parts[-1] += (
            "\n  READ 'in' AND 'not_in' SEPARATELY -- they are opposite\n"
            "  predicates and SI means the reverse thing in each:\n"
            "    in      the drifted member is something the query WANTS.\n"
            "            SI < 1 = the planner followed the drifted PROFILE\n"
            "            and under-served the query.\n"
            "    not_in  the drifted member is something the query says to\n"
            "            AVOID.  SI > 1 = the planner followed the drifted\n"
            "            PROFILE and violated the query constraint.\n"
            "  In both cases, deviation from 1.0 IN THE DIRECTION THE DRIFTED\n"
            "  PROFILE SUGGESTS is profile-following over query-following;\n"
            "  pooling the two ops would cancel that signal.\n")
        for label, key in (("by op  (do NOT pool -- see note above)", "by_op"),
                           ("by op x drift action", "by_op_action"),
                           ("by drift action  (pooled over ops)", "by_action"),
                           ("by attribute", "by_attribute"),
                           ("by paradigm", "by_paradigm")):
            grp = sp.get(key) or {}
            if not grp:
                continue
            parts[-1] += f"\n  {label}:\n"
            parts[-1] += (f"    {'bucket':<26s} {'n':>5s} {'mean SI':>9s} "
                          f"{'median':>8s} {'se':>7s}\n")
            for k, a in sorted(grp.items(), key=lambda kv: -kv[1]["n"]):
                parts[-1] += (f"    {str(k):<26s} {a['n']:>5d} "
                              f"{a['mean']:>9.3f} {a['median']:>8.3f} "
                              f"{a['se']:>7.3f}\n")
        ex = sp.get("examples") or []
        if ex:
            parts[-1] += "\n  most-suppressed leaves (lowest SI):\n"
            for e in ex:
                parts[-1] += (
                    f"    id={e['id']:<5} {e['paradigm']}#{e['bank_id']}  "
                    f"{e['entity_type']}.{e['attribute']} {e['op']}  "
                    f"SI={e['si']:.2f}  ({e['action']})\n"
                    f"        drifted={e['drifted']} mean={e['drifted_mean']:.2f}"
                    f"   surviving={e['surviving']} mean={e['surviving_mean']:.2f}\n")
        parts[-1] += "\n"

    return parts


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def _attach_plan_days(records: list[dict], plan_file: Path,
                      dataset: str, split: str) -> None:
    """Populate ``_plan_days`` on each record from a structured plan JSONL.

    Reuses ``evaluate_preferences.transform_plan``, which is what resolves
    a plan's entity NAMES into the attribute views (cuisine, category,
    room_type, ...) the Suppression Index reads.  Re-deriving that here
    would risk drifting from the evaluator's own canonicalisation, so we
    call the same function the evaluation used.
    """
    import json as _json
    print(f"[in] plan-file = {plan_file}")
    if not plan_file.exists():
        print(f"[warn] plan file not found; [S-supp] section will be skipped",
              file=sys.stderr)
        return
    try:
        import evaluate_preferences as _EP
    except ImportError as e:                              # pragma: no cover
        print(f"[warn] cannot import evaluate_preferences ({e}); "
              f"[S-supp] skipped", file=sys.stderr)
        return

    plans: dict[int, Any] = {}
    with plan_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = _json.loads(line)
            if row.get("id") is None:
                continue
            plans[int(row["id"])] = row.get("plan")

    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, verification_mode="no_checks")
    ctx = {int(r["id"]): (r.get("date"), int(r.get("people_number") or 1))
           for r in ds}

    db = _EP._DB()
    n_ok = n_miss = 0
    for rec in _tqdm(records, total=len(records), desc="attach-plans"):
        rid = rec.get("id")
        raw = plans.get(int(rid)) if rid is not None else None
        if not raw:
            n_miss += 1
            continue
        date_seq, people = ctx.get(int(rid), (None, 1))
        try:
            rec["_plan_days"] = _EP.transform_plan(
                raw, db=db, date_seq=date_seq, people_number=people)
            n_ok += 1
        except Exception as e:                            # pragma: no cover
            print(f"[warn] transform_plan failed for id={rid}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            n_miss += 1
    print(f"[in] plans attached = {n_ok}   (missing/failed = {n_miss})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detailed", type=Path, required=True,
                    help="Detailed eval JSONL from `evaluation/eval.py --out`.")
    ap.add_argument("--dataset",  default="UKPLab/PreferTripPlan",
                    help="HF dataset (default: UKPLab/PreferTripPlan).")
    ap.add_argument("--split",    default="test",
                    help="HF split (default: test).")
    ap.add_argument("--out",      type=Path, default=None,
                    help="Optional plain-text destination for the report.")
    ap.add_argument("--top-bank-ids", type=int, default=20,
                    help="Number of bank_ids to show in the by-bank-id "
                         "preference table (default: 20).")
    ap.add_argument("--raw-out",  type=Path, default=None,
                    help="Optional JSON destination for the raw paradigm-"
                         "detail extraction (A1 / CD / L / N1 / CO / T4 / "
                         "T6 / T9 / T11 / T12 / X3).  Downstream plotting "
                         "scripts consume this directly.")
    ap.add_argument("--plan-file", type=Path, default=None,
                    help="Optional structured plan JSONL (from "
                         "evaluation/convert_plans.py).  Supplying it "
                         "enables the [S-supp] Suppression Index section, "
                         "which needs the PLAN -- not just the verdict -- to "
                         "see which categorical members the planner actually "
                         "realised.  Costs a TravelPlanner DB load plus one "
                         "transform_plan per record, so it is opt-in.")
    args = ap.parse_args()

    print(f"[in] detailed = {args.detailed}")
    rows = _load_detailed(args.detailed)
    print(f"[in] rows     = {len(rows)}")
    print(f"[in] dataset  = {args.dataset}:{args.split}")
    meta = _load_hf_meta(args.dataset, args.split)
    print(f"[in] hf meta  = {len(meta)}")

    records = [_record_metrics(r, meta)
                for r in _tqdm(rows, total=len(rows), desc="derive-metrics")]

    # Optional plan attachment.  Only the Suppression Index needs the plan
    # itself; everything else reads the evaluated verdict, so this stays
    # opt-in rather than making every run pay the DB load.
    if args.plan_file:
        _attach_plan_days(records, args.plan_file, args.dataset, args.split)

    parts: list[str] = []
    parts.append(_render_constraint_table(
        "Overall (single bucket)",
        {"all": _summarize_bucket(records)},
        order=["all"]))

    # 1. Per-record constraint metrics per axis --------------------------------
    axis_defs: list[tuple[str, str, Callable[[dict], Any], list | None]] = [
        ("days",             "days",            lambda r: r["days"],
         _order_from_data(records, "days")),
        ("level",            "level",           lambda r: r["level"],
         _order_from_data(records, "level", _STANDARD_ORDERS["level"])),
        ("profile_drift",    "profile_drift",   lambda r: r["profile_drift"],
         _order_from_data(records, "profile_drift", _STANDARD_ORDERS["profile_drift"])),
        ("pairing_type",     "pairing_type",    lambda r: r["pairing_type"],
         _order_from_data(records, "pairing_type", _STANDARD_ORDERS["pairing_type"])),
        ("pairing_subtype",  "pairing_subtype", lambda r: r["pairing_subtype"],
         _order_from_data(records, "pairing_subtype", _STANDARD_ORDERS["pairing_subtype"])),
    ]
    for label, _, key_fn, order in axis_defs:
        parts.append(_render_constraint_table(
            f"By {label}",
            _bucket_records(records, key_fn),
            order=order))

    # 1b. Drift × party-size joint bloc ---------------------------------------
    # The profile is single-person; the query can address a group.  So the
    # persona-intent-gap and domination-dilution-trap effects are only cleanly
    # observable on solo (people_number == 1) records, where the profile
    # speaks for the whole party.  On group records (>=2) drift is expected
    # to matter less because the query is speaking for many and the profile
    # is one member's voice.  This joint table splits every drift-mode row
    # into ``solo`` vs ``group`` so those two population effects can be read
    # off directly without inferring them from the marginal.
    drift_order = _order_from_data(records, "profile_drift",
                                    _STANDARD_ORDERS["profile_drift"])
    people_present = [b for b in ("solo", "group")
                       if any(r["people_bucket"] == b for r in records)]
    joint_order = [f"{d}+{b}" for d in drift_order for b in people_present]
    parts.append(_render_constraint_table(
        "By profile_drift × party-size  "
        "[persona-intent gap & domination-dilution trap: expect drift to "
        "hurt solo records more than group records]",
        _bucket_records(records, lambda r:
            f"{r['profile_drift']}+{r['people_bucket']}"
            if r["profile_drift"] and r["people_bucket"] else None),
        order=joint_order))

    # 2. Per-axis × paradigm preference pass-rate cross-tabs -----------------
    paradigm_order = _order_from_data(
        [{"_": p.get("paradigm")} for r in records for p in r["_prefs"]]
        if any(r["_prefs"] for r in records) else [{"_": None}],
        "_", _STANDARD_ORDERS["paradigm"])
    # ^ helper reused for a synthetic field; alternative is a small custom sort.

    for label, _, key_fn, order in axis_defs:
        cross = _pref_cross_tab(records, key_fn, _pref_paradigm)
        parts.append(_render_cross_tab(
            f"Preference pass_rate: {label} × paradigm",
            cross, axis_order=order, para_order=paradigm_order,
            axis_width=12, para_width=22))

    # 3. Per-axis × (paradigm, sub_paradigm) preference pass-rate cross-tabs -
    sub_labels = sorted({_pref_paradigm_sub(p)
                          for r in records for p in r["_prefs"]},
                         key=str)
    for label, _, key_fn, order in axis_defs:
        cross = _pref_cross_tab(records, key_fn, _pref_paradigm_sub)
        parts.append(_render_cross_tab(
            f"Preference pass_rate: {label} × (paradigm, sub_paradigm)",
            cross, axis_order=order, para_order=sub_labels,
            axis_width=12, para_width=28))

    # 4. Preference-only breakdowns not tied to any record axis --------------
    parts.append(_render_pref_table(
        "Preferences: by paradigm",
        _bucket_prefs(records, lambda p, r: _pref_paradigm(p)),
        order=[p for p in paradigm_order
                if p in _bucket_prefs(records, lambda p, r: _pref_paradigm(p))],
        label_width=30))
    parts.append(_render_pref_table(
        "Preferences: by (paradigm, sub_paradigm)",
        _bucket_prefs(records, lambda p, r: _pref_paradigm_sub(p)),
        label_width=40))
    parts.append(_render_pref_table(
        "Preferences: by trivial-vs-non-trivial",
        _bucket_prefs(records, lambda p, r: "trivial" if p.get("trivial") else "non-trivial"),
        order=["trivial", "non-trivial"],
        label_width=20))
    # A sub-100% trivial pass rate is expected, not a defect -- explain it
    # in the report itself so the reader is not sent to the source.
    triv_fail = sum(1 for r in records for p in r["_prefs"]
                    if p.get("trivial") and not p.get("passed"))
    if triv_fail:
        by_sub = Counter(p.get("sub_paradigm") or p.get("paradigm") or "?"
                         for r in records for p in r["_prefs"]
                         if p.get("trivial") and not p.get("passed"))
        parts.append(
            "  note: {n} preference(s) are flagged trivial yet did NOT pass "
            "({subs}).\n"
            "        Grouped Temporal preferences (scope=per_day/per_city) set\n"
            "        trivial = ANY group vacuous while passed = ALL groups pass,\n"
            "        so the flag means 'contains vacuous groups', not 'was vacuous\n"
            "        throughout' -- see preferences.py::TemporalPreference._aggregate.\n"
            "        Such rows count wholly in trivial_n, which is why triv_pass%\n"
            "        can read below 100% and why a genuine failure can be absent\n"
            "        from the ntriv_* columns.\n\n".format(
                n=triv_fail,
                subs=", ".join(f"{k}x{v}" for k, v in by_sub.most_common())))

    # 4b. Leaf-scope decomposition ------------------------------------------
    # Holds quantifier load fixed so cross-paradigm differences are
    # attributable to the combinator rather than to how many universally
    # -scoped leaves the paradigm happens to carry.  See the block
    # comment above ``_LEAF_CHILD_KEYS`` for why this is necessary.
    scope_cov = _bucket_prefs(
        records, lambda p, r: "resolved" if _pref_scope_tuple(p, r) is not None
        else "unresolved")
    n_res   = scope_cov.get("resolved",   {}).get("n", 0)
    n_unres = scope_cov.get("unresolved", {}).get("n", 0)
    if n_res == 0:
        parts.append(
            "=== Preferences: by (paradigm x leaf-scope tuple) ===\n"
            "  (no structural metadata -- rerun without --no-hf-meta)\n\n")
    else:
        cov_note = (f"  leaf-scope resolved for {n_res}/{n_res + n_unres} "
                    f"evaluated preferences"
                    + (f"  ({n_unres} unresolved)" if n_unres else "")
                    + "\n  tuple order: Composite=children, Conditional="
                      "(cond, then, else), Lexicographic=priority,\n"
                      "               Compensatory=(primary, margin, secondary), "
                      "Scoped=(filters, inner),\n"
                      "               Temporal=(subject, reference);  "
                      "'opt' = numeric optimization leaf (no quantifier)\n")

        para_scope = _bucket_prefs(records, _pref_paradigm_scope)
        # Order: canonical paradigm order, then quantifier load within it.
        def _paradigm_rank(label: str) -> int:
            for i, p in enumerate(paradigm_order):
                if label.startswith(p):
                    return i
            return len(paradigm_order)
        ps_order = sorted(
            para_scope,
            key=lambda k: (_paradigm_rank(k),
                           _scope_tuple_sort_key(
                               tuple(k[k.index("(") + 1:-1].split(","))
                               if "(" in k and k.endswith(")") and
                                  k[k.index("(") + 1:-1] else ())))
        table = _render_pref_table(
            "Preferences: by (paradigm x leaf-scope tuple)",
            para_scope, order=ps_order, label_width=46)
        # The legend goes AFTER the table, separated by the blank line
        # that already terminates it.  It must not sit between the
        # "=== title ===" banner and the column header: report parsers
        # (latex._parse_report, and every figure script built on it)
        # take the first non-empty line inside a section AS the header,
        # so a note there is silently consumed as column names and the
        # whole section stops being machine-readable.
        parts.append(table + cov_note + "\n")

        # Quantifier load alone, pooled across paradigms.  This is the
        # table that isolates the all-vs-any effect: read the marginal
        # cost of each additional universal leaf straight down the rows.
        # Sort by leaf count, then by universal load descending, so the
        # marginal cost of each added ``all`` reads down the rows.
        # Parses the compact "2leaf|1all,1any" form emitted by
        # ``_scope_signature``.
        def _sig_sort_key(s: str) -> tuple:
            head, _, tail = s.partition("leaf|")
            try:
                n_leaf = int(head)
            except ValueError:
                n_leaf = 0
            n_all = 0
            for chunk in tail.split(","):
                if chunk.endswith("all"):
                    try:
                        n_all = int(chunk[:-3])
                    except ValueError:
                        pass
            return (n_leaf, -n_all, s)

        sig_buckets = _bucket_prefs(records, _scope_signature)
        parts.append(_render_pref_table(
            "Preferences: by leaf-scope signature (quantifier load, all paradigms pooled)",
            sig_buckets,
            order=sorted(sig_buckets, key=_sig_sort_key),
            label_width=30))

        # Paradigm x scope-tuple matrix, non-trivial rates only.  The
        # column marginal answers "what does this quantifier load cost
        # regardless of paradigm"; the row marginal answers "what does
        # this paradigm cost regardless of load".  Reading a single
        # column ACROSS paradigms is the controlled comparison that the
        # headline per-paradigm table cannot give.
        matrix: dict[tuple, dict] = {}
        for key, bucket in _bucket_prefs(
                records,
                lambda p, r: ((_pref_paradigm(p), _pref_scope_tuple(p, r))
                              if _pref_scope_tuple(p, r) is not None else None)
        ).items():
            matrix[key] = bucket
        # Headline rate FIRST (every evaluated preference, trivial ones
        # included) -- that is the number the per-paradigm tables report,
        # so this is the like-for-like controlled view of it.  The
        # vacuous-excluded version follows as a second matrix rather than
        # replacing it, since which one is appropriate depends on the
        # claim being made and neither should be assumed.
        parts.append(_render_scope_matrix(
            "Preference pass rate (ALL, incl. trivial): paradigm x leaf-scope tuple",
            matrix, metric="pass_rate", count_key="n", pass_key="pass"))
        parts.append(_render_scope_matrix(
            "Preference NON-TRIVIAL pass rate: paradigm x leaf-scope tuple",
            matrix, metric="nontrivial_pass_rate",
            count_key="nontrivial_n", pass_key="nontrivial_pass"))

        # Temporal sub-paradigms x scope.  Temporal is the paradigm with
        # the most sub-operators (9 PDDL3 forms), and they differ in
        # arity -- unary forms (sometime / always / at-most-once) carry
        # one leaf, binary forms (within / sometime-before / ...) carry
        # two -- so the sub-paradigm and the scope tuple are entangled.
        # Splitting them apart is the only way to tell an operator that
        # is intrinsically hard from one that merely tends to appear with
        # universal leaves.
        temporal = _bucket_prefs(
            records,
            lambda p, r: (_pref_paradigm_sub_scope(p, r)
                          if p.get("paradigm") == "TemporalPreference" else None))
        if temporal:
            parts.append(_render_pref_table(
                "Preferences: Temporal sub-paradigm x leaf-scope tuple",
                temporal, label_width=52))

        # Same split for every other paradigm that has sub-paradigms,
        # so the entanglement check is not Temporal-only.
        sub_scope = _bucket_prefs(
            records,
            lambda p, r: (_pref_paradigm_sub_scope(p, r)
                          if p.get("sub_paradigm")
                          and p.get("paradigm") != "TemporalPreference" else None))
        if sub_scope:
            parts.append(_render_pref_table(
                "Preferences: non-Temporal sub-paradigm x leaf-scope tuple",
                sub_scope, label_width=52))

    # By preference bank_id (top-K only, ordered by prevalence).
    bank = _bucket_prefs(records, lambda p, r:
        f"{_pref_paradigm(p)}#{p.get('bank_id')}"
        if p.get("bank_id") is not None else None)
    if bank:
        top = sorted(bank.keys(), key=lambda k: -bank[k]["n"])[: args.top_bank_ids]
        parts.append(_render_pref_table(
            f"Preferences: top {len(top)} by bank_id (paradigm#bank_id)",
            {k: bank[k] for k in top},
            order=top,
            label_width=36))

    # 5. Paradigm-detail analyses (structured-details layer).  These
    # produce rich per-paradigm distributions in addition to their text
    # summaries; the raw data goes to ``--raw-out`` as JSON for
    # downstream plotting.
    try:
        from paradigm_details import (
            extract_A1_atomic_partial_credit,
            extract_CD_conditional,
            extract_L1_L2_lex_tiers,
            extract_N1_numeric_quantile_position,
            extract_CO_compensatory_tiers,
            extract_T4_temporal_within_first_pos,
            extract_T6_temporal_timed_day_curve,
            extract_T9_temporal_pair_strict,
            extract_T11_temporal_always_within,
            extract_T12_temporal_aggregate_by_scope,
            extract_X3_trivial_pass_audit,
            extract_drift_impact_numeric,
            extract_drift_impact_set,
            extract_gate_drift_triviality,
            extract_drift_matched_pairs,
            extract_realization_by_drift,
            extract_suppression_index,
        )
    except ImportError as e:
        print(f"[warn] paradigm_details helpers not importable ({e}); "
              f"skipping paradigm-detail analyses", file=sys.stderr)
        pd_extracts = None
    else:
        pd_extracts = {
            "A1_atomic_partial_credit":     extract_A1_atomic_partial_credit(records),
            "CD_conditional":               extract_CD_conditional(records),
            "L1_L2_lex_tiers":              extract_L1_L2_lex_tiers(records),
            "N1_numeric_quantile_position": extract_N1_numeric_quantile_position(records),
            "CO_compensatory_tiers":        extract_CO_compensatory_tiers(records),
            "T4_temporal_within_first_pos": extract_T4_temporal_within_first_pos(records),
            "T6_temporal_timed_day_curve":  extract_T6_temporal_timed_day_curve(records),
            "T9_temporal_pair_strict":      extract_T9_temporal_pair_strict(records),
            "T11_temporal_always_within":   extract_T11_temporal_always_within(records),
            "T12_temporal_aggregate_by_scope": extract_T12_temporal_aggregate_by_scope(records),
            "X3_trivial_pass_audit":        extract_X3_trivial_pass_audit(records),
            "D_numeric_drift_impact":       extract_drift_impact_numeric(records),
            "D_set_drift_impact":           extract_drift_impact_set(records),
            "G_gate_drift_triviality":      extract_gate_drift_triviality(records),
            "M_drift_matched_pairs":        extract_drift_matched_pairs(records),
            "R_realization_by_drift":       extract_realization_by_drift(records),
            "S_suppression_index":          extract_suppression_index(records),
        }
        parts.extend(_render_paradigm_detail_sections(pd_extracts))

    report = "\n".join(parts)
    print()
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"[out] report written to {args.out}")
    if args.raw_out and pd_extracts is not None:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_out.write_text(json.dumps(pd_extracts, indent=2))
        print(f"[out] raw paradigm-detail data → {args.raw_out}")


if __name__ == "__main__":
    main()
