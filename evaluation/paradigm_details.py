"""paradigm_details.py — extraction of paradigm-specific analysis data
from the detailed per-record eval JSONL emitted by
``evaluation/eval.py --out``.

Every function here is a PURE data extractor: it consumes the list of
per-record metric dicts produced by ``analyze_performance._record_metrics``
(so each row already carries a ``_prefs`` list plus per-record axes like
``profile_drift`` / ``people_bucket`` / ``level``) and returns a Python
structure of raw numbers / distributions / per-bucket counts.  No
rendering, no I/O.

The naming convention is ``extract_<code>_<what>`` where ``<code>``
mirrors the analysis-plan code (A1 / CO1 / L1 / T4 / ...).  Downstream
scripts consume these to (a) render text summaries and (b) dump raw
JSON for plotting.

Every value read from ``pref.details`` is defensively coerced so a
missing key produces a sensible zero rather than a crash: this module
is expected to work against any detailed JSONL produced by the current
``preferences.py`` schema.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _det(p: dict) -> dict:
    """Return ``p['details']`` as a dict, or an empty dict on missing /
    malformed entries.  Detailed JSONL rows produced before the
    ``details``-dict refactor carried a plain string here; treating
    those as ``{}`` lets this module work against both schemas without
    crashing (older records simply yield no data to the extractors)."""
    d = p.get("details")
    return d if isinstance(d, dict) else {}


def _iter_prefs(records: list[dict],
                 kind: str | None = None,
                 paradigm: str | None = None):
    """Yield every preference (as a dict) alongside its owning record.
    Optionally filter by exact ``details.kind`` and/or paradigm."""
    for r in records:
        for p in r.get("_prefs", []):
            if not isinstance(p, dict):
                continue
            if paradigm is not None and p.get("paradigm") != paradigm:
                continue
            d = _det(p)
            if kind is not None and d.get("kind") != kind:
                continue
            yield p, r


# --------------------------------------------------------------------------- #
# A1 — Atomic [all]-scope partial-credit distribution                         #
# --------------------------------------------------------------------------- #

def extract_A1_atomic_partial_credit(records: list[dict]) -> dict[str, Any]:
    """Values of ``details.passed_entities`` for every Atomic preference
    with ``details.scope == 'all'`` -- i.e. the universal-quantifier
    partial-credit signal.  Returns::

        {"values": [float, ...],       # one per Atomic-[all] preference
         "by_drift": {drift -> [float, ...]},
         "by_entity_type": {entity_type -> [float, ...]}}

    A histogram over ``values`` is the intended plot (A1 FIG).  Split
    versions (by drift, by entity_type) support the same plot as
    overlaid lines (X1 companion FIG).
    """
    values: list[float] = []
    by_drift: dict[str, list[float]] = defaultdict(list)
    by_entity: dict[str, list[float]] = defaultdict(list)
    for p, r in _iter_prefs(records, paradigm="AtomicPreference"):
        d = _det(p)
        if d.get("scope") != "all":
            continue
        v = d.get("passed_entities")
        if not isinstance(v, (int, float)):
            continue
        v = float(v)
        values.append(v)
        drift = r.get("profile_drift")
        if drift:
            by_drift[drift].append(v)
        et = d.get("entity_type")
        if et:
            by_entity[et].append(v)
    return {"values":         values,
            "by_drift":       dict(by_drift),
            "by_entity_type": dict(by_entity)}


# --------------------------------------------------------------------------- #
# CD1 / CD2 / CD3 — Conditional dynamics                                       #
# --------------------------------------------------------------------------- #

def extract_CD_conditional(records: list[dict]) -> dict[str, Any]:
    """Roll up Conditional preferences::

        {"n":                       int,
         "n_cond_fired":            int,   # CD1
         "n_cond_fired_active_pass":int,   # CD2 numerator
         "n_trivial_true_no_then":  int,   # kind == "conditional.trivial",
                                            # branch == "then"
         "n_trivial_false_no_else": int,   # kind == "conditional.trivial",
                                            # branch == "else"
         "n_active":                int,   # kind == "conditional"
         "n_active_pass":           int}   # kind == "conditional" and passed
    """
    out = {"n": 0, "n_cond_fired": 0, "n_cond_fired_active_pass": 0,
           "n_trivial_true_no_then": 0, "n_trivial_false_no_else": 0,
           "n_active": 0, "n_active_pass": 0}
    for p, r in _iter_prefs(records, paradigm="ConditionalPreference"):
        d = _det(p)
        out["n"] += 1
        if d.get("cond_passed") is True:
            out["n_cond_fired"] += 1
        kind = d.get("kind")
        branch = d.get("branch")
        if kind == "conditional.trivial":
            if branch == "then":
                out["n_trivial_true_no_then"] += 1
            elif branch == "else":
                out["n_trivial_false_no_else"] += 1
        elif kind == "conditional":
            out["n_active"] += 1
            if p.get("passed"):
                out["n_active_pass"] += 1
                if d.get("cond_passed") is True:
                    out["n_cond_fired_active_pass"] += 1
    return out


# --------------------------------------------------------------------------- #
# L1 / L2 — Lex per-tier and tier-gap                                          #
# --------------------------------------------------------------------------- #

def extract_L1_L2_lex_tiers(records: list[dict]) -> dict[str, Any]:
    """Per-tier pass counts and tier-1-only vs all-tiers-pass gap.

    Returns::

        {"n":                       int,         # total Lex evaluations
         "max_tier_index":          int,         # widest lex depth observed
         "tier_pass":               [int, ...],  # per-tier pass counts,
                                                  # tier_pass[i] = # records
                                                  # where tiers[i].passed
         "tier_total":              [int, ...],  # denominator per tier
         "n_tier1_pass":            int,
         "n_all_tiers_pass":        int,
         "tier_scalars":            [float, ...],# details.scalar for L3
         "tier_pass_matrix":        list[list[bool]],  # for L4 correlated-
                                                        # failure analysis
        }
    """
    out: dict[str, Any] = {"n": 0, "max_tier_index": 0,
                            "tier_pass": [], "tier_total": [],
                            "n_tier1_pass": 0, "n_all_tiers_pass": 0,
                            "tier_scalars": [],
                            "tier_pass_matrix": []}
    tier_pass = Counter()
    tier_total = Counter()
    for p, _ in _iter_prefs(records, paradigm="LexicographicPreference"):
        d = _det(p)
        tiers = d.get("tiers") or []
        if not tiers:
            continue
        out["n"] += 1
        out["max_tier_index"] = max(out["max_tier_index"], len(tiers) - 1)
        row = []
        for i, t in enumerate(tiers):
            if not isinstance(t, dict):
                row.append(False); continue
            tier_total[i] += 1
            passed = bool(t.get("passed"))
            row.append(passed)
            if passed:
                tier_pass[i] += 1
        out["tier_pass_matrix"].append(row)
        if row and row[0]:
            out["n_tier1_pass"] += 1
        if row and all(row):
            out["n_all_tiers_pass"] += 1
        sc = d.get("scalar")
        if isinstance(sc, (int, float)):
            out["tier_scalars"].append(float(sc))
    depth = out["max_tier_index"] + 1
    out["tier_pass"]  = [tier_pass[i]  for i in range(depth)]
    out["tier_total"] = [tier_total[i] for i in range(depth)]
    return out


# --------------------------------------------------------------------------- #
# N1 — Numeric quantile-position density, split by direction                  #
# --------------------------------------------------------------------------- #

def extract_N1_numeric_quantile_position(records: list[dict]) -> dict[str, Any]:
    """Return per-direction lists of ``details.quantile_position`` for
    NumericPreference records that had a valid threshold::

        {"max": [float, ...],
         "min": [float, ...]}

    Records with ``kind`` in {``numeric.no_threshold``,
    ``numeric.no_values``, ``numeric.bad_threshold``} are excluded --
    only the ``kind=="numeric"`` case exposes a valid
    ``quantile_position``.
    """
    out = {"max": [], "min": []}
    for p, _ in _iter_prefs(records, kind="numeric", paradigm="NumericPreference"):
        d = _det(p)
        v = d.get("quantile_position")
        if not isinstance(v, (int, float)):
            continue
        direction = d.get("direction")
        if direction in ("max", "min"):
            out[direction].append(float(v))
    return out


# --------------------------------------------------------------------------- #
# CO1 / CO2 / CO3 / CO4 — Compensatory tier story                              #
# --------------------------------------------------------------------------- #

_TIERS = ("tier1", "tier2_comp", "tier2_uncomp", "tier3")


def extract_CO_compensatory_tiers(records: list[dict]) -> dict[str, Any]:
    """Compensatory tier counts, both aggregated and split by axes::

        {"aggregate":       {tier -> int},          # CO1
         "aggregate_frac":  {tier -> float},        # CO1 normalised
         "n_entities_total": int,
         "n_evaluations":   int,
         "utilisation":     {"tier2_comp": int,     # CO2 numerator
                              "tier2_total": int},  # CO2 denominator
         "by_drift":        {drift -> {tier -> int}},           # CO4
         "by_drift_frac":   {drift -> {tier -> float}},
         "by_cross_entity": {"same" | "cross" -> {tier -> int}},# CO3
         "by_cross_entity_frac": {...},
         "per_record":      [{tier -> int}, ...]}   # CO5 raw profiles
    """
    agg = Counter()
    n_evals = 0
    n_ents_total = 0
    by_drift: dict[str, Counter] = defaultdict(Counter)
    by_cross: dict[str, Counter] = defaultdict(Counter)
    per_record: list[dict] = []
    for p, r in _iter_prefs(records, kind="compensatory",
                              paradigm="CompensatoryPreference"):
        d = _det(p)
        counts = d.get("tier_pass_counts") or {}
        if not isinstance(counts, dict) or not any(counts.get(t) for t in _TIERS):
            continue
        n_evals += 1
        drift = r.get("profile_drift") or "?"
        cross_key = "cross" if d.get("cross_entity") else "same"
        rec_profile = {}
        for t in _TIERS:
            v = int(counts.get(t, 0) or 0)
            agg[t] += v
            n_ents_total += v
            by_drift[drift][t] += v
            by_cross[cross_key][t] += v
            rec_profile[t] = v
        per_record.append(rec_profile)

    def _fracify(c: Counter | dict) -> dict:
        tot = sum(c.get(t, 0) for t in _TIERS)
        if tot == 0:
            return {t: 0.0 for t in _TIERS}
        return {t: c.get(t, 0) / tot for t in _TIERS}

    n_tier2 = int(agg["tier2_comp"] + agg["tier2_uncomp"])
    return {
        "aggregate":            {t: int(agg[t]) for t in _TIERS},
        "aggregate_frac":       _fracify(agg),
        "n_entities_total":     n_ents_total,
        "n_evaluations":        n_evals,
        "utilisation":          {"tier2_comp":  int(agg["tier2_comp"]),
                                  "tier2_total": n_tier2},
        "by_drift":             {k: {t: int(v[t]) for t in _TIERS}
                                  for k, v in by_drift.items()},
        "by_drift_frac":        {k: _fracify(v) for k, v in by_drift.items()},
        "by_cross_entity":      {k: {t: int(v[t]) for t in _TIERS}
                                  for k, v in by_cross.items()},
        "by_cross_entity_frac": {k: _fracify(v) for k, v in by_cross.items()},
        "per_record":           per_record,
    }


# --------------------------------------------------------------------------- #
# T4 — Temporal.within first-subject-position density                          #
# --------------------------------------------------------------------------- #

def extract_T4_temporal_within_first_pos(records: list[dict]
                                          ) -> dict[str, Any]:
    """Extract per-group ``temporal.within`` first-subject-position
    normalised to ``[0, 1]`` via ``first_subject_pos / (group_length - 1)``
    when ``group_length > 1``.  Groups with ``no_subject == True`` are
    reported separately.

    Returns::

        {"normalised":            [float, ...],    # 0..1, one per group
         "raw_positions":         [{"pos": int,
                                     "time_end": int,
                                     "group_length": int,
                                     "day_as_step": bool}, ...],
         "n_no_subject_groups":   int}
    """
    normalised: list[float] = []
    raw: list[dict] = []
    n_no_subject = 0
    # temporal.within lands on sub_results of the record-level
    # temporal.aggregate.  We walk pref -> sub_results -> details.
    for p, _ in _iter_prefs(records, paradigm="TemporalPreference"):
        for sub in _sub_details(p):
            if sub.get("kind") != "temporal.within":
                continue
            if sub.get("no_subject"):
                n_no_subject += 1
                continue
            pos = sub.get("first_subject_pos")
            L   = sub.get("group_length")
            if not isinstance(pos, int) or not isinstance(L, int) or L <= 1:
                continue
            normalised.append(pos / (L - 1))
            raw.append({"pos": int(pos), "time_end": int(sub.get("time_end") or 0),
                        "group_length": int(L),
                        "day_as_step": bool(sub.get("day_as_step"))})
    return {"normalised": normalised, "raw_positions": raw,
            "n_no_subject_groups": n_no_subject}


def _sub_details(pref: dict) -> Iterable[dict]:
    """Yield ``details`` dicts of the preference's sub_results (recursive
    for CheckResult trees serialised in the JSONL).  Present when the
    detailed JSONL retained sub_results (temporal aggregate keeps its
    per-group children as sub_results in Python but the detailed JSONL
    written by ``eval.py`` only carries the top-level details -- so we
    also probe ``details`` fields that themselves aggregate group data.
    This yields whatever we can find."""
    d = _det(pref)
    # Case 1: sub-results serialised as list under 'sub_results'.
    subs = pref.get("sub_results") or []
    if isinstance(subs, list):
        for s in subs:
            if isinstance(s, dict):
                sd = s.get("details") or {}
                if isinstance(sd, dict):
                    yield sd
                yield from _sub_details(s)
    # Case 2: the detailed JSONL might inline groups under the top-level
    # details as 'groups' -- if so, yield each.
    groups = d.get("groups") or []
    if isinstance(groups, list):
        for g in groups:
            if isinstance(g, dict):
                yield g


# --------------------------------------------------------------------------- #
# T6 — Temporal.timed per-window-day pass profile                              #
# --------------------------------------------------------------------------- #

def extract_T6_temporal_timed_day_curve(records: list[dict]
                                         ) -> dict[str, Any]:
    """Aggregate ``details.day_scores`` across every ``temporal.timed``
    group into a mean-per-window-position curve.  Windows are aligned
    at their starting position (index 0 = first window day, 1 = second,
    ...) so the curve reveals whether the LLM degrades at the start,
    middle, or end of the hold window.

    Returns::

        {"window_pos_curve": [float, ...],   # mean day_score at each
                                              # window position, length =
                                              # max observed window length
         "window_pos_counts":[int, ...],     # denominator per position
         "n_groups":         int,
         "n_no_overlap":     int}
    """
    pos_sum = Counter()
    pos_cnt = Counter()
    n_groups = 0
    n_no_overlap = 0
    for p, _ in _iter_prefs(records, paradigm="TemporalPreference"):
        for sub in _sub_details(p):
            if sub.get("kind") != "temporal.timed":
                continue
            if sub.get("overlap") is False:
                n_no_overlap += 1
                continue
            window = sub.get("window") or []
            day_scores = sub.get("day_scores") or {}
            if not window or not day_scores:
                continue
            n_groups += 1
            for i, d in enumerate(window):
                v = day_scores.get(str(d)) if str(d) in day_scores else day_scores.get(d)
                if isinstance(v, (int, float)):
                    pos_sum[i] += float(v)
                    pos_cnt[i] += 1
    max_pos = max(pos_cnt) if pos_cnt else -1
    curve = []
    counts = []
    for i in range(max_pos + 1):
        c = pos_cnt.get(i, 0)
        curve.append(pos_sum.get(i, 0) / c if c else 0.0)
        counts.append(c)
    return {"window_pos_curve":  curve,
            "window_pos_counts": counts,
            "n_groups":          n_groups,
            "n_no_overlap":      n_no_overlap}


# --------------------------------------------------------------------------- #
# T9 — Temporal.pair strict vs non-strict pass rate                            #
# --------------------------------------------------------------------------- #

def extract_T9_temporal_pair_strict(records: list[dict]) -> dict[str, Any]:
    """Aggregate per-group pass rates split by ``details.strict``.
    Returns::

        {"strict":     {"n": int, "pass": int, "pass_rate": float},
         "non_strict": {"n": int, "pass": int, "pass_rate": float}}
    """
    buckets = {"strict": {"n": 0, "pass": 0},
                "non_strict": {"n": 0, "pass": 0}}
    for p, _ in _iter_prefs(records, paradigm="TemporalPreference"):
        for sub in _sub_details(p):
            if sub.get("kind") != "temporal.pair":
                continue
            if sub.get("op") not in ("sometime_before", "sometime_after"):
                continue
            key = "strict" if sub.get("strict") else "non_strict"
            buckets[key]["n"] += 1
            n_ok = sub.get("pairs_ok")
            n_total = sub.get("pairs_total")
            passed = False
            if isinstance(n_ok, int) and isinstance(n_total, int) and n_total:
                if sub.get("strict"):
                    passed = (n_ok == n_total)
                else:
                    passed = (n_ok > 0)
            buckets[key]["pass"] += int(passed)
    for k, b in buckets.items():
        b["pass_rate"] = (b["pass"] / b["n"]) if b["n"] else 0.0
    return buckets


# --------------------------------------------------------------------------- #
# T11 — Temporal.pair always_within `passed_subjects` density                  #
# --------------------------------------------------------------------------- #

def extract_T11_temporal_always_within(records: list[dict]
                                        ) -> dict[str, Any]:
    """Extract per-group ``passed_subjects`` ratios for the
    ``always_within`` op.  Returns::

        {"values":  [float, ...],  # 0..1 ratio per group
         "n_no_subject_groups": int,
         "n_groups": int}
    """
    values: list[float] = []
    n_no_subject = 0
    n_groups = 0
    for p, _ in _iter_prefs(records, paradigm="TemporalPreference"):
        for sub in _sub_details(p):
            if sub.get("kind") != "temporal.pair" or sub.get("op") != "always_within":
                continue
            if sub.get("no_subject"):
                n_no_subject += 1; continue
            v = sub.get("passed_subjects")
            if isinstance(v, (int, float)):
                values.append(float(v))
                n_groups += 1
    return {"values": values, "n_no_subject_groups": n_no_subject,
            "n_groups": n_groups}


# --------------------------------------------------------------------------- #
# T12 — Temporal.aggregate passed_groups × scope                               #
# --------------------------------------------------------------------------- #

def extract_T12_temporal_aggregate_by_scope(records: list[dict]
                                             ) -> dict[str, Any]:
    """Aggregate root-level Temporal pass by ``details.scope``::

        {scope -> {"n": int, "pass": int, "pass_rate": float,
                   "mean_passed_groups": float}}
    """
    out: dict[str, dict] = {}
    for p, _ in _iter_prefs(records, kind="temporal.aggregate",
                              paradigm="TemporalPreference"):
        d = _det(p)
        scope = d.get("scope") or "?"
        b = out.setdefault(scope,
                            {"n": 0, "pass": 0, "passed_groups_sum": 0.0})
        b["n"] += 1
        b["pass"] += int(bool(p.get("passed")))
        pg = d.get("passed_groups")
        if isinstance(pg, (int, float)):
            b["passed_groups_sum"] += float(pg)
    for scope, b in out.items():
        n = b["n"] or 1
        b["pass_rate"] = b["pass"] / n
        b["mean_passed_groups"] = b["passed_groups_sum"] / n
        b.pop("passed_groups_sum", None)
    return out


# --------------------------------------------------------------------------- #
# X3 — trivial-pass audit                                                      #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# D-NUMERIC — drift impact on numeric-based (sub-)preferences                  #
# --------------------------------------------------------------------------- #
# Walks each record's drift_trace, follows every source's ``path`` into the
# matching preferences_json template AND into the corresponding sub_result
# in the evaluated preferences.  Filters to numeric-based leaves so the
# aggregate compares like-with-like across drift modes.
#
# Numeric-based leaves == either
#   (a) AtomicPreference whose op is one of {>=, <=, >, <, ==, !=, range}
#       AND the value is numeric; scoped by the leaf's ``scope`` (all / any),
#   (b) NumericPreference (top-level, optimization semantics).

_NUMERIC_OPS = frozenset({">=", "<=", ">", "<", "==", "!=", "range"})


def _is_numeric_leaf(template: dict) -> bool:
    """Return True when a preferences_json template block describes a
    numeric-based leaf (numeric-op Atomic OR standalone Numeric)."""
    if not isinstance(template, dict):
        return False
    # Nested-flat vs nested-wrapped disambiguation: a template that has a
    # ``class`` field and no explicit ``entity_type``/``attribute`` at
    # this level is a wrapper; recurse into ``template``.
    if "class" in template and "template" in template and \
            "entity_type" not in template:
        return _is_numeric_leaf(template["template"])
    # Direct AtomicPreference-shaped leaf.
    if template.get("op") in _NUMERIC_OPS and \
            "value" in template and \
            isinstance(template.get("value"), (int, float)):
        return True
    if template.get("op") == "range" and \
            isinstance(template.get("value"), (list, tuple)) and \
            len(template["value"]) == 2:
        return True
    # NumericPreference-shaped (optimization).  These live only at
    # top-level in the bank; the resolved template carries a
    # ``direction`` + a ``threshold`` list.
    if template.get("direction") in ("min", "max") and \
            template.get("aggregation") in ("avg", "sum", "min", "max"):
        return True
    return False


def _walk_pref_template(root_template: dict,
                         path: list) -> dict | None:
    """Follow the drift-source ``path`` from the top-level template of
    a preference into the specific sub-node it addresses.  Returns the
    leaf-template dict, or ``None`` if the path can't be resolved.

    Path elements interpreted:
      "inner"              -> ScopedPreference.inner
      "condition"          -> ConditionalPreference.condition
      "then_pref"          -> ConditionalPreference.then_pref
      "else_pref"          -> ConditionalPreference.else_pref
      "subject_ap"         -> TemporalPreference.subject_ap
      "reference_ap"       -> TemporalPreference.reference_ap
      "primary_ap"         -> CompensatoryPreference.primary_ap
      "margin_ap"          -> CompensatoryPreference.margin_ap
      "secondary_ap"       -> CompensatoryPreference.secondary_ap
      "children"           -> CompositePreference.children  (needs next idx)
      "preferences"        -> LexicographicPreference.preferences (needs next idx)
      "scope_filters"      -> ScopedPreference.scope_filters (needs next idx)
      integer index        -> pop the top of the list-key that came before
    """
    if not isinstance(root_template, dict):
        return None
    if not path:
        return root_template
    node = root_template
    i = 0
    while i < len(path):
        step = path[i]
        if not isinstance(node, dict):
            return None
        if isinstance(step, int):
            # A bare integer without a preceding container-key indicates
            # a malformed path -- skip defensively.
            return None
        if step in ("children", "preferences", "scope_filters"):
            container = node.get(step)
            if not isinstance(container, list):
                return None
            # Next path element MUST be an integer index.
            if i + 1 >= len(path) or not isinstance(path[i + 1], int):
                return None
            idx = path[i + 1]
            if idx < 0 or idx >= len(container):
                return None
            node = container[idx]
            i += 2
        else:
            # Named-key navigation.  Wrapper entries look like
            # ``{"class": ..., "template": {...}}`` OR directly carry
            # the sub-node keyed by ``step``.
            if step in node:
                node = node[step]
            elif "template" in node and isinstance(node["template"], dict) and \
                    step in node["template"]:
                node = node["template"][step]
            else:
                return None
            i += 1
        # Descend past any wrapper layers -- callers want the "template"
        # dict of the addressed sub-node, whether or not it carries a
        # ``class`` envelope.  Only unwrap when the current node isn't
        # already a numeric leaf (its own template block).
        if (isinstance(node, dict) and "template" in node
                and node.get("class") is not None
                and not _is_numeric_leaf(node)):
            node = node["template"]
    return node if isinstance(node, dict) else None


def _walk_sub_result(root_result: dict, path: list) -> dict | None:
    """Follow the drift-source ``path`` from the top-level evaluated
    preference into the corresponding sub_result (previously an inner
    CheckResult).  Returns the leaf dict, or ``None`` when the sub_result
    tree doesn't reflect the path (e.g. the evaluator collapsed groups
    or the eval was produced before ``sub_results`` were serialised).

    Path segments map to sub_result indices as follows (mirroring the
    order in which each Preference's ``evaluate`` builds its
    ``sub_results`` list):

      Composite   children[i]              -> sub_results[i]
      Conditional condition                -> sub_results[0]
                  then_pref                -> sub_results[1] (if cond passed)
      Lex         preferences[i]           -> sub_results[i]
      Scoped      inner                    -> sub_results[0]
      Temporal    subject_ap / reference_ap-> N/A (subject/reference feed
                  every group; there's no sub_result per predicate).
      Compensatory primary/margin/secondary-> N/A (sub_results are per
                  primary-entity).
    """
    if not isinstance(root_result, dict):
        return None
    if not path:
        return root_result
    node = root_result
    i = 0
    while i < len(path):
        step = path[i]
        subs = node.get("sub_results") if isinstance(node, dict) else None
        if step == "inner":
            if not subs:
                return None
            node = subs[0]; i += 1
        elif step == "condition":
            if not subs:
                return None
            node = subs[0]; i += 1
        elif step == "then_pref":
            # sub_results = [cond_result, active_result] when cond fired.
            if not subs or len(subs) < 2:
                return None
            node = subs[1]; i += 1
        elif step == "else_pref":
            if not subs or len(subs) < 2:
                return None
            node = subs[1]; i += 1
        elif step in ("children", "preferences"):
            if i + 1 >= len(path) or not isinstance(path[i + 1], int):
                return None
            idx = path[i + 1]
            if not subs or idx < 0 or idx >= len(subs):
                return None
            node = subs[idx]; i += 2
        else:
            # subject_ap / reference_ap / primary_ap / margin_ap /
            # secondary_ap / scope_filters -- no direct sub_result
            # counterpart today.  Bail cleanly.
            return None
    return node if isinstance(node, dict) else None


def extract_drift_impact_numeric(records: list[dict]) -> dict[str, Any]:
    """Path-aware drift-impact metric on numeric-based sub-preferences.

    For every record and every drift_trace source, walk its ``path`` into
    the preferences_json to reach a specific sub-node.  Restrict to
    numeric-based leaves (Atomic with a numeric op + numeric value, or
    NumericPreference optimization).  Then look up the matching sub-result
    (via the same path on the evaluated preference tree) and record its
    ``passed`` / ``score`` / structured-detail signal.

    Aggregate into::

        {"aligned":   {"n": int, "n_pass": int, "pass_rate": float,
                       "mean_score": float},
         "omission":  {...},
         "inversion": {...},
         "counts":    {"records_scanned": int,
                       "sources_scanned": int,
                       "numeric_hits":    int,
                       "sub_result_hits": int,
                       "fallback_top_level": int},
         "by_shape":  {(entity, attr, op, scope): {drift -> {...}}}}

    Baseline is "same-shape aligned" (source with ``drifted=False`` on
    the SAME preference-shape).  ``drift_action`` = ``"invert"`` /
    ``"drop"`` maps to ``inversion`` / ``omission`` buckets.
    """
    from collections import defaultdict

    # Six-bucket aggregate.  The three "drifted-*" buckets are direct;
    # the three aligned variants isolate the population where non-drifted
    # numeric sources live:
    #
    #  aligned_all              -- every non-drifted source, pooled
    #                              (the baseline used in the original
    #                              tabulation).
    #  aligned_in_aligned_records
    #                           -- sources from records whose overall
    #                              drift mode is ``aligned`` (no source
    #                              on the record was drifted).  Cleanest
    #                              baseline: profile fully agrees with query.
    #  aligned_in_omission_records
    #                           -- non-drifted sources from records whose
    #                              overall mode is ``omission`` (ONE
    #                              source was dropped, this one wasn't).
    #                              Captures the "residual" behavior of the
    #                              LLM on non-drifted parts of a drifted
    #                              record.
    #  aligned_in_inversion_records
    #                           -- same as above for inversion.
    #
    #  omission / inversion     -- the drifted sources themselves
    #                              (drifted=True and drift_action ==
    #                              "drop"/"invert" respectively).
    out_by_drift: dict[str, dict[str, Any]] = {
        "aligned_all":                   {"n": 0, "n_pass": 0, "score_sum": 0.0},
        "aligned_in_aligned_records":    {"n": 0, "n_pass": 0, "score_sum": 0.0},
        "aligned_in_omission_records":   {"n": 0, "n_pass": 0, "score_sum": 0.0},
        "aligned_in_inversion_records":  {"n": 0, "n_pass": 0, "score_sum": 0.0},
        "omission":                      {"n": 0, "n_pass": 0, "score_sum": 0.0},
        "inversion":                     {"n": 0, "n_pass": 0, "score_sum": 0.0},
    }
    _empty_bucket = lambda: {"n": 0, "n_pass": 0, "score_sum": 0.0}
    _empty_by_shape = lambda: {
        "aligned_all":                  _empty_bucket(),
        "aligned_in_aligned_records":   _empty_bucket(),
        "aligned_in_omission_records":  _empty_bucket(),
        "aligned_in_inversion_records": _empty_bucket(),
        "omission":                     _empty_bucket(),
        "inversion":                    _empty_bucket(),
    }
    by_shape: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(_empty_by_shape)
    counts = {"records_scanned": 0, "sources_scanned": 0,
               "numeric_hits":    0, "sub_result_hits":  0,
               "fallback_top_level": 0}

    for r in records:
        counts["records_scanned"] += 1
        prefs_json = r.get("_preferences_json") or []
        drift_tr   = r.get("_drift_trace")     or {}
        evals      = r.get("_prefs")           or []
        sources    = drift_tr.get("sources")   or []
        record_drift_mode = r.get("profile_drift") or "aligned"
        if not sources or not prefs_json:
            continue

        # Index preferences_json + evaluated preferences by bank_id
        # (paradigm + bank_id together disambiguate; but since a record
        # rarely has two preferences with the same bank_id, bank_id
        # alone is usually enough).
        pj_by_bid = {}
        for p in prefs_json:
            bid = p.get("bank_id")
            if bid is not None:
                pj_by_bid[(p.get("paradigm"), int(bid))] = p
        ev_by_bid = {}
        for p in evals:
            bid = p.get("bank_id")
            if bid is not None:
                ev_by_bid[(p.get("paradigm"), int(bid))] = p

        for src in sources:
            counts["sources_scanned"] += 1
            paradigm = src.get("paradigm")
            bid_raw  = src.get("bank_id")
            if paradigm is None or bid_raw is None:
                continue
            try:
                bid = int(bid_raw)
            except (TypeError, ValueError):
                continue
            path = src.get("path") or []
            drift_action = src.get("drift_action")   # invert / drop / None
            was_drifted  = bool(src.get("drifted"))

            pj_pref = pj_by_bid.get((paradigm, bid))
            ev_pref = ev_by_bid.get((paradigm, bid))
            if pj_pref is None or ev_pref is None:
                continue

            root_template = pj_pref.get("template") or {}
            leaf = _walk_pref_template(root_template, list(path))
            if leaf is None or not _is_numeric_leaf(leaf):
                continue
            # Some walker terminuses land on a ``{class, template}``
            # wrapper (e.g. Scoped.inner carries an AtomicPreference-
            # shaped inner as a wrapped block).  _is_numeric_leaf
            # recurses through the wrapper to decide numeric-ness, so
            # we unwrap here before extracting entity_type / attribute
            # / op / scope for the by-shape breakdown.
            if ("template" in leaf and leaf.get("class") is not None
                    and "entity_type" not in leaf):
                leaf = leaf["template"]
            counts["numeric_hits"] += 1

            # Fetch the sub_result along the same path.
            sub_result = _walk_sub_result(ev_pref, list(path))
            if sub_result is not None:
                counts["sub_result_hits"] += 1
                passed  = bool(sub_result.get("passed"))
                score   = float(sub_result.get("score") or 0.0)
            else:
                # No serialised sub_result (older JSONL, or a paradigm
                # whose sub_results don't map 1:1 to path).  Fall back
                # to the top-level pref outcome as a coarser proxy.
                counts["fallback_top_level"] += 1
                passed  = bool(ev_pref.get("passed"))
                score   = float(ev_pref.get("score") or 0.0)

            # Determine drift bucket(s).  Every source falls into either
            # one "aligned_*" bucket (record-mode-specific) AND the
            # pooled "aligned_all" bucket, OR into a drifted bucket
            # ("omission" / "inversion") when it was itself drifted.
            target_buckets: list[str] = []
            if was_drifted and drift_action == "invert":
                target_buckets.append("inversion")
            elif was_drifted and drift_action == "drop":
                target_buckets.append("omission")
            elif not was_drifted:
                target_buckets.append("aligned_all")
                if record_drift_mode == "aligned":
                    target_buckets.append("aligned_in_aligned_records")
                elif record_drift_mode == "omission":
                    target_buckets.append("aligned_in_omission_records")
                elif record_drift_mode == "inversion":
                    target_buckets.append("aligned_in_inversion_records")
            else:
                continue    # unknown action

            ent  = leaf.get("entity_type") or "?"
            attr = leaf.get("attribute")   or "?"
            op   = leaf.get("op")          or leaf.get("direction") or "?"
            sc   = leaf.get("scope")       or "-"
            shape = (ent, attr, op, sc)

            for bucket in target_buckets:
                slot = out_by_drift[bucket]
                slot["n"]         += 1
                slot["n_pass"]    += int(passed)
                slot["score_sum"] += float(score)
                sslot = by_shape[shape][bucket]
                sslot["n"]         += 1
                sslot["n_pass"]    += int(passed)
                sslot["score_sum"] += float(score)

    def _finalise(slot: dict[str, Any]) -> dict[str, Any]:
        n = slot["n"]
        return {
            "n":          n,
            "n_pass":     slot["n_pass"],
            "pass_rate":  (slot["n_pass"] / n) if n else 0.0,
            "mean_score": (slot["score_sum"] / n) if n else 0.0,
        }

    # by_shape keys are tuples; convert to "|"-joined strings so the
    # extract is JSON-encodable for --raw-out.  The original tuple is
    # preserved in a companion field for programmatic use inside the
    # same process (e.g. the text-renderer sorts / iterates on it).
    by_shape_str = {}
    for shape, inner in by_shape.items():
        key = "|".join(str(s) for s in shape)
        by_shape_str[key] = {k: _finalise(v) for k, v in inner.items()}

    return {
        "aligned_all":                   _finalise(out_by_drift["aligned_all"]),
        "aligned_in_aligned_records":    _finalise(out_by_drift["aligned_in_aligned_records"]),
        "aligned_in_omission_records":   _finalise(out_by_drift["aligned_in_omission_records"]),
        "aligned_in_inversion_records":  _finalise(out_by_drift["aligned_in_inversion_records"]),
        "omission":                      _finalise(out_by_drift["omission"]),
        "inversion":                     _finalise(out_by_drift["inversion"]),
        "counts":                        counts,
        "by_shape":                      by_shape_str,
    }


_TRIVIAL_KINDS = frozenset({
    "atomic.no_entities",
    "numeric.no_values", "numeric.no_threshold",
    "scoped.no_match_trivial",
    "compensatory.no_primary",
    "conditional.trivial",
    "temporal.empty_plan", "temporal.unknown_op",
})


def extract_X3_trivial_pass_audit(records: list[dict]) -> dict[str, Any]:
    """Audit preferences whose ``trivial`` flag is True OR whose
    ``details.kind`` is one of the paradigm-specific trivial sentinels.
    Returns::

        {"total":              int,   # all prefs
         "n_trivial_flagged":  int,   # p.trivial == True
         "n_trivial_by_kind":  {kind -> count},
         "trivial_pass_rate":  float, # pass rate on the trivial slice
         "by_paradigm":        {paradigm -> {"n_total": int,
                                              "n_trivial": int,
                                              "trivial_pass_rate": float}}}
    """
    n_total = 0
    n_trivial = 0
    n_trivial_pass = 0
    n_trivial_by_kind: Counter = Counter()
    by_paradigm: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_total": 0, "n_trivial": 0, "n_trivial_pass": 0})
    for p, _ in _iter_prefs(records):
        d = _det(p)
        n_total += 1
        para = p.get("paradigm") or "?"
        by_paradigm[para]["n_total"] += 1
        is_trivial = bool(p.get("trivial")) or (d.get("kind") in _TRIVIAL_KINDS)
        if is_trivial:
            n_trivial += 1
            by_paradigm[para]["n_trivial"] += 1
            n_trivial_by_kind[d.get("kind") or "-"] += 1
            if p.get("passed"):
                n_trivial_pass += 1
                by_paradigm[para]["n_trivial_pass"] += 1
    result_by_paradigm: dict[str, dict[str, float]] = {}
    for para, b in by_paradigm.items():
        result_by_paradigm[para] = {
            "n_total":           b["n_total"],
            "n_trivial":         b["n_trivial"],
            "trivial_pass_rate": (b["n_trivial_pass"] / b["n_trivial"])
                                   if b["n_trivial"] else 0.0,
        }
    return {
        "total":              n_total,
        "n_trivial_flagged":  n_trivial,
        "n_trivial_by_kind":  dict(n_trivial_by_kind),
        "trivial_pass_rate":  (n_trivial_pass / n_trivial) if n_trivial else 0.0,
        "by_paradigm":        result_by_paradigm,
    }


# --------------------------------------------------------------------------- #
# D-SET — drift impact on SET-VALUED CATEGORICAL (sub-)preferences             #
# --------------------------------------------------------------------------- #
# Companion to ``extract_drift_impact_numeric``.  That one handles leaves
# whose drifted quantity is a single number and which the generator always
# drifts INTACT -- verified: every ``numeric_top`` source is the sole
# source for its (paradigm, bank_id, path), 180/180 on test_large.  Scalar
# numeric-op Atomic leaves behave the same (rating 729/729, cost 375/375
# single-source), so numeric drift needs no regrouping.
#
# Categorical leaves are different.  When several members of one value set
# are drifted together, the generator combines them into a single natural
# profile sentence ("doesn't care for Chinese or American food"), but the
# trace still records ONE SOURCE PER ELEMENT.  Measured on test_large:
#
#     category      529 single-source groups, 180 pairs,  96 triples
#     cuisine        30 single-source groups,  35 pairs,  15 triples, 8 quads
#     house_rules    10 single-source groups,  10 pairs,  21 triples
#
# and every ``raw_value`` is a member of its leaf's value list (776/776
# category, 88/88 cuisine, 41/41 house_rules, 16/16 room_type, 6/6 mode).
#
# THE DISINTEGRATION THIS IMPLEMENTS.  Regrouping matters because drift is
# frequently PARTIAL within a leaf: a preference over
# ``[Museums, Sights & Landmarks, Nature & Parks]`` may have only
# ``Nature & Parks`` drifted, leaving the other two intact (test_large
# record id=23).  Treating that leaf as simply "drifted" overstates the
# perturbation; treating it as "aligned" understates it.  So each leaf is
# split into the drifted subset and the surviving remainder, and the
# leaf's drift COVERAGE (|drifted| / |value set|) becomes the graded
# quantity the impact is measured against.
#
# Grouping is per (record, paradigm, bank_id, path) -- i.e. WITHIN a
# preference, never across preferences.  Two preferences on the same
# record that mention the same cuisine are separate obligations and are
# kept separate.
#
# Coverage classes:
#   none     -- the leaf has trace sources but none drifted (the
#               like-for-like control: same attribute, op and scope,
#               same construction path, simply not perturbed)
#   partial  -- some members drifted, some survived
#   full     -- every member of the value set drifted

# Categorical attributes whose leaf value is a SET.  ``mode`` and
# ``room_type`` are categorical too but scalar in practice (all their
# groups are single-source), so they fall into the same machinery with
# coverage that is trivially 0 or 1.
_SET_CATEGORICAL_ATTRS = frozenset({
    "cuisine", "house_rules", "category", "room_type", "mode",
})

_SET_OPS = frozenset({"in", "not_in", "contains_all", "==", "!="})


def _coverage_class(n_drifted: int, n_values: int) -> str:
    if n_drifted <= 0:
        return "none"
    if n_values and n_drifted >= n_values:
        return "full"
    return "partial"


def extract_drift_impact_set(records: list[dict]) -> dict[str, Any]:
    """Path-aware drift-impact metric on set-valued categorical leaves.

    For every record, group drift_trace sources by
    ``(paradigm, bank_id, path)`` -- one group per addressed leaf -- then
    resolve the leaf template and its evaluated sub_result via the same
    path.  Keep leaves whose attribute is categorical and whose op is a
    set/equality op.  For each leaf record the value set, which members
    were drifted, the resulting coverage, and the outcome.

    Returns::

        {"by_coverage":  {"none"|"partial"|"full" ->
                              {n, n_pass, pass_rate, mean_score,
                               mean_coverage}},
         "by_action":    {"invert"|"drop"|"none" -> {...}},
         "by_attribute": {attr -> {coverage_class -> {...}}},
         "by_shape":     {"entity|attr|op|scope" -> {coverage_class -> {...}}},
         "coverage_hist":{rounded coverage -> n},
         "counts":       {...diagnostics...},
         "examples":     [ ...partial-coverage leaves, for the writeup... ]}

    ``pass_rate`` at ``coverage="none"`` is the within-shape baseline; the
    drift effect is the drop from that to ``partial`` and ``full``.
    """
    from collections import defaultdict

    def _blank():
        return {"n": 0, "n_pass": 0, "score_sum": 0.0, "cov_sum": 0.0}

    by_cov = defaultdict(_blank)
    by_action = defaultdict(_blank)
    by_attr = defaultdict(lambda: defaultdict(_blank))
    by_shape = defaultdict(lambda: defaultdict(_blank))
    cov_hist = Counter()
    examples: list[dict] = []
    counts = Counter()

    for rec in records:
        trace = rec.get("_drift_trace") or {}
        sources = trace.get("sources") or []
        if not sources:
            continue
        counts["records_scanned"] += 1

        prefs_json = {(p.get("paradigm"), p.get("bank_id")): p
                      for p in (rec.get("_preferences_json") or [])}
        evals = {(e.get("paradigm"), e.get("bank_id")): e
                 for e in (rec.get("_prefs") or [])}

        groups: dict[tuple, list[dict]] = defaultdict(list)
        for s in sources:
            if s.get("bank_id") is None:
                continue
            groups[(s.get("paradigm"), s.get("bank_id"),
                    tuple(s.get("path") or []))].append(s)

        for (paradigm, bank_id, path), srcs in groups.items():
            counts["groups_scanned"] += 1
            entry = prefs_json.get((paradigm, bank_id))
            if entry is None:
                counts["no_template"] += 1
                continue
            leaf = _walk_pref_template(entry.get("template") or {}, list(path))
            if not isinstance(leaf, dict):
                counts["path_unresolved"] += 1
                continue
            attr = leaf.get("attribute")
            op = leaf.get("op")
            if attr not in _SET_CATEGORICAL_ATTRS or op not in _SET_OPS:
                continue
            values = leaf.get("value")
            values = list(values) if isinstance(values, (list, tuple)) else [values]
            if not values:
                continue
            counts["set_leaves"] += 1

            drifted = {s.get("raw_value") for s in srcs if s.get("drifted")}
            # Only members of THIS leaf's value set count -- a source can
            # legitimately address a sibling leaf under the same path
            # prefix, and counting it here would inflate coverage.
            drifted &= set(values)
            n_values, n_drifted = len(values), len(drifted)
            coverage = n_drifted / n_values if n_values else 0.0
            klass = _coverage_class(n_drifted, n_values)

            actions = {s.get("drift_action") for s in srcs if s.get("drifted")}
            action = (next(iter(actions)) if len(actions) == 1
                      else ("mixed" if actions else "none"))

            ev = evals.get((paradigm, bank_id))
            if ev is None:
                counts["no_eval"] += 1
                continue
            sub = _walk_sub_result(ev, list(path))
            if isinstance(sub, dict):
                counts["sub_result_hits"] += 1
            else:
                # Fall back to the top-level preference outcome.  Recorded
                # separately so the reader can see how much of the
                # aggregate rests on the coarser signal.
                counts["fallback_top_level"] += 1
                sub = ev
            passed = bool(sub.get("passed"))
            score = float(sub.get("score") or 0.0)

            scope = leaf.get("scope") or "-"
            shape = f"{leaf.get('entity_type')}|{attr}|{op}|{scope}"
            for bucket in (by_cov[klass], by_action[action],
                           by_attr[attr][klass], by_shape[shape][klass]):
                bucket["n"] += 1
                bucket["n_pass"] += int(passed)
                bucket["score_sum"] += score
                bucket["cov_sum"] += coverage
            cov_hist[round(coverage, 2)] += 1

            if klass == "partial" and len(examples) < 40:
                examples.append({
                    "id": rec.get("id"), "paradigm": paradigm,
                    "bank_id": bank_id, "path": list(path),
                    "attribute": attr, "op": op, "scope": scope,
                    "values": values, "drifted": sorted(drifted),
                    "surviving": sorted(set(values) - drifted),
                    "coverage": round(coverage, 3),
                    "drift_action": action,
                    "passed": passed, "score": round(score, 3),
                    "record_drift": rec.get("profile_drift"),
                })

    def _norm(b):
        n = b["n"]
        return {"n": n, "n_pass": b["n_pass"],
                "pass_rate": (b["n_pass"] / n) if n else 0.0,
                "mean_score": (b["score_sum"] / n) if n else 0.0,
                "mean_coverage": (b["cov_sum"] / n) if n else 0.0}

    return {
        "by_coverage":  {k: _norm(v) for k, v in by_cov.items()},
        "by_action":    {k: _norm(v) for k, v in by_action.items()},
        "by_attribute": {a: {k: _norm(v) for k, v in d.items()}
                         for a, d in by_attr.items()},
        "by_shape":     {s: {k: _norm(v) for k, v in d.items()}
                         for s, d in by_shape.items()},
        "coverage_hist": {str(k): v for k, v in sorted(cov_hist.items())},
        "counts":       dict(counts),
        "examples":     examples,
    }


# --------------------------------------------------------------------------- #
# G-GATE — does drift on a GATING sub-node manufacture vacuous passes?         #
# --------------------------------------------------------------------------- #
# A "gate" is a sub-node whose non-firing makes the ENTIRE preference
# vacuously satisfied rather than failed:
#
#   ConditionalPreference.condition   -- condition false  -> nothing to check
#   TemporalPreference.subject_ap     -- no subject       -> nothing to trigger
#   ScopedPreference.scope_filters[*] -- filter matches 0 -> empty scope
#
# The hypothesis this measures: when drift lands ON the gate, the planner
# has been told (via the profile) to disregard the very trigger the
# preference hinges on.  If it complies, the gate never fires, the
# preference is satisfied VACUOUSLY, and it is booked as a PASS.  Drift
# would then *manufacture easy passes* rather than making the task harder
# -- which would explain why record-level ``profile_drift`` comparisons
# show inversion/omission matching or beating aligned.
#
# UNIT is one evaluated PREFERENCE (not a leaf, not a record), restricted
# to paradigms that have a gate.  The three classes partition them
# exhaustively, so the column total equals the number of gated
# preferences in the split.  Class membership is a DATASET property, so
# the n per class is identical across models; only triv%/pass% move.

# ScopedPreference is deliberately ABSENT.  It has a structural gate
# (``scope_filters``), but no Scoped scope_filter is ever drifted in
# either published split -- Scoped drift always targets ``inner`` -- so
# including it only contributed an all-zero GATE column.
_GATE_POSITIONS: dict[str, tuple[str, ...]] = {
    "ConditionalPreference": ("condition",),
    "TemporalPreference":    ("subject_ap",),
}


def extract_gate_drift_triviality(records: list[dict]) -> dict[str, Any]:
    """Triviality / pass rate by whether drift hit a GATING sub-node.

    Returns::

        {"by_class":     {"undrifted"|"nongate_drifted"|"gate_drifted" ->
                              {n, n_trivial, trivial_rate,
                               n_pass, pass_rate, mean_score}},
         "by_paradigm":  {paradigm -> {class -> {...}}},
         "gate_positions_seen": {"<paradigm>.<path head>" -> n},
         "counts":       {"gated_prefs": int, "ungated_prefs_skipped": int}}

    ``undrifted`` is the control: a gated preference on which no source
    was drifted at all.  ``nongate_drifted`` is the discriminating
    comparison -- drift landed on the preference but NOT on its gate, so
    any triviality jump seen only in ``gate_drifted`` cannot be a generic
    "this record was drifted" effect.
    """
    from collections import defaultdict

    def _blank():
        return {"n": 0, "n_trivial": 0, "n_pass": 0, "score_sum": 0.0,
                "triv_pass": 0, "ntriv_n": 0, "ntriv_pass": 0}

    by_class = defaultdict(_blank)
    by_para = defaultdict(lambda: defaultdict(_blank))
    seen_gates = Counter()
    counts = Counter()

    for rec in records:
        srcs_by_pref: dict[tuple, list[dict]] = defaultdict(list)
        for s in (rec.get("_drift_trace") or {}).get("sources") or []:
            bid = s.get("bank_id")
            if bid is None:
                continue
            srcs_by_pref[(s.get("paradigm"), int(bid))].append(s)

        for pref in (rec.get("_prefs") or []):
            paradigm = pref.get("paradigm")
            gates = _GATE_POSITIONS.get(paradigm)
            if gates is None:
                counts["ungated_prefs_skipped"] += 1
                continue
            bid = pref.get("bank_id")
            if bid is None:
                continue
            counts["gated_prefs"] += 1

            hit_gate = hit_other = False
            for s in srcs_by_pref.get((paradigm, int(bid)), []):
                if not s.get("drifted"):
                    continue
                path = s.get("path") or []
                head = str(path[0]) if path else ""
                if head in gates:
                    hit_gate = True
                    seen_gates[f"{paradigm}.{head}"] += 1
                else:
                    hit_other = True
            klass = ("gate_drifted" if hit_gate
                     else "nongate_drifted" if hit_other else "undrifted")

            trivial = bool(pref.get("trivial"))
            passed = bool(pref.get("passed"))
            score = float(pref.get("score") or 0.0)
            for slot in (by_class[klass], by_para[paradigm][klass]):
                slot["n"] += 1
                slot["n_trivial"] += int(trivial)
                slot["n_pass"] += int(passed)
                slot["score_sum"] += score
                # Decomposition is the point: the OVERALL pass rate mixes
                # vacuous passes with real ones, so a gate-drift effect can
                # be completely masked.  Conditional gate drift is exactly
                # that case -- overall pass rate stays flat while the
                # still-evaluated subset degrades ~10pp.
                if trivial:
                    slot["triv_pass"] += int(passed)
                else:
                    slot["ntriv_n"] += 1
                    slot["ntriv_pass"] += int(passed)

    def _norm(b):
        n, tn, nn = b["n"], b["n_trivial"], b["ntriv_n"]
        return {"n": n, "n_trivial": tn,
                "trivial_rate": (tn / n) if n else 0.0,
                "n_pass": b["n_pass"],
                "pass_rate": (b["n_pass"] / n) if n else 0.0,
                "mean_score": (b["score_sum"] / n) if n else 0.0,
                # pass rate WITHIN the trivial subset.  100% for genuine
                # full vacuity (Conditional); below 100% for Temporal,
                # whose ``trivial`` is the any(group vacuous) partial
                # marker and so contains real failures.
                "trivial_pass_rate": (b["triv_pass"] / tn) if tn else 0.0,
                "nontrivial_n": nn,
                "nontrivial_pass_rate": (b["ntriv_pass"] / nn) if nn else 0.0}

    return {
        "by_class":    {k: _norm(v) for k, v in by_class.items()},
        "by_paradigm": {p: {k: _norm(v) for k, v in d.items()}
                        for p, d in by_para.items()},
        "gate_positions_seen": dict(seen_gates),
        "counts":      dict(counts),
    }


# --------------------------------------------------------------------------- #
# M-PAIRS — matched-pairs drift contrast                                       #
# --------------------------------------------------------------------------- #
# The ideal design -- generate two profiles for ONE query, one drifted and
# one not, and compare -- is not available: each record carries a single
# profile.  This approximates it.
#
# ``bank_id`` identifies the exact bank entry a preference instantiates,
# so it pins the template shape (paradigm, structure, leaf scopes).  Drift
# mode is assigned by md5-hash rank stratified per level, INDEPENDENTLY of
# preference content, so within a stratum drifted-vs-undrifted is close to
# random assignment.  Stratifying additionally on ``level`` and ``days``
# holds trip difficulty fixed, which the pooled ``aligned_all`` baseline
# used by the D-num extractor does NOT do.
#
# Stratum key defaults to (paradigm, bank_id, days).  ``level`` is
# deliberately NOT in the default key: drift mode is assigned by md5-hash
# rank STRATIFIED PER LEVEL at fixed 30/35/35 ratios, so level is already
# balanced across drift modes by construction and stratifying on it only
# costs power.  ``days`` is not balanced that way (it mildly favours
# aligned), so it stays.
#
# Granularity is a real power trade-off, measured on gpt-5.6-terra /
# test_large:
#
#   (paradigm, bank_id, level, days)   214 strata,  778 prefs, se 2.1pp
#   (paradigm, bank_id, level)         179 strata, 1155 prefs, se 2.1pp
#   (paradigm, bank_id, days)          239 strata, 1241 prefs, se 1.7pp  <- default
#   (paradigm, bank_id)                105 strata, 1390 prefs, se 1.3pp
#
# Finer strata match better but discard the many cells that end up with
# only drifted or only undrifted instances.  Pass ``strata_keys`` to
# override.  Only strata containing at least one drifted AND one
# undrifted instance contribute; the rest are counted as
# ``strata_no_contrast`` rather than silently dropped.
#
# Two estimators are reported because they answer different questions:
#   pooled_*        -- micro: sum(pass)/sum(n) over contributing strata.
#                      Weights a stratum by its size.
#   mean_within_*   -- macro: mean over strata of (drifted_rate -
#                      undrifted_rate).  Weights every stratum equally, so
#                      one large stratum cannot carry the result.


def extract_drift_matched_pairs(records: list[dict],
                                strata_keys: tuple[str, ...] =
                                ("paradigm", "bank_id", "days")
                                ) -> dict[str, Any]:
    """Within-stratum drifted-vs-undrifted contrast on identical bank
    entries.

    Returns::

        {"pooled":   {"drifted": {...}, "undrifted": {...},
                      "delta_pp": float},
         "mean_within": {"delta_pp": float, "n_strata": int,
                         "sd_pp": float},
         "by_action": {"invert"|"drop" -> {"delta_pp", "n_strata", ...}},
         "by_paradigm": {paradigm -> {"delta_pp", "n_strata", ...}},
         "counts":   {"strata_total", "strata_contributing",
                      "strata_no_contrast", "prefs_used"},
         "strata_keys": tuple}

    ``mean_within.sd_pp`` is the spread ACROSS strata, not a standard
    error; divide by sqrt(n_strata) for that.  Individual strata hold
    only 1-3 preferences each, so per-stratum rates are mostly 0% or
    100% and the sd is necessarily large -- read the mean, not the sd,
    as the estimate.
    """
    from collections import defaultdict

    # stratum -> {"drifted": [(passed, action)], "undrifted": [passed]}
    strata: dict[tuple, dict[str, list]] = defaultdict(
        lambda: {"drifted": [], "undrifted": []})

    for rec in records:
        level, days = rec.get("level"), rec.get("days")
        srcs_by_pref: dict[tuple, list[dict]] = defaultdict(list)
        for s in (rec.get("_drift_trace") or {}).get("sources") or []:
            bid = s.get("bank_id")
            if bid is None:
                continue
            srcs_by_pref[(s.get("paradigm"), int(bid))].append(s)

        for pref in (rec.get("_prefs") or []):
            paradigm, bid = pref.get("paradigm"), pref.get("bank_id")
            if bid is None:
                continue
            srcs = srcs_by_pref.get((paradigm, int(bid)), [])
            actions = {s.get("drift_action") for s in srcs if s.get("drifted")}
            passed = bool(pref.get("passed"))
            avail = {"paradigm": paradigm, "bank_id": int(bid),
                     "level": level, "days": days}
            key = tuple(avail.get(k) for k in strata_keys)
            if actions:
                action = next(iter(actions)) if len(actions) == 1 else "mixed"
                strata[key]["drifted"].append((passed, action))
            else:
                strata[key]["undrifted"].append(passed)

    counts = Counter(strata_total=len(strata))
    deltas: list[float] = []
    by_action_d: dict[str, list[float]] = defaultdict(list)
    by_para_d: dict[str, list[float]] = defaultdict(list)
    pooled = {"drifted": [0, 0], "undrifted": [0, 0]}   # [n, n_pass]

    para_pos = strata_keys.index("paradigm") if "paradigm" in strata_keys else None
    for key, cell in strata.items():
        paradigm = key[para_pos] if para_pos is not None else "?"
        dr, un = cell["drifted"], cell["undrifted"]
        if not dr or not un:
            counts["strata_no_contrast"] += 1
            continue
        counts["strata_contributing"] += 1
        counts["prefs_used"] += len(dr) + len(un)
        d_rate = sum(1 for p, _a in dr if p) / len(dr)
        u_rate = sum(1 for p in un if p) / len(un)
        delta = 100.0 * (d_rate - u_rate)
        deltas.append(delta)
        by_para_d[paradigm].append(delta)
        acts = {a for _p, a in dr}
        if len(acts) == 1:
            by_action_d[next(iter(acts))].append(delta)
        pooled["drifted"][0] += len(dr)
        pooled["drifted"][1] += sum(1 for p, _a in dr if p)
        pooled["undrifted"][0] += len(un)
        pooled["undrifted"][1] += sum(1 for p in un if p)

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    def _sd(xs):
        if len(xs) < 2:
            return 0.0
        mu = _mean(xs)
        return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    def _rate(pair):
        n, p = pair
        return {"n": n, "n_pass": p, "pass_rate": (p / n) if n else 0.0}

    dr_r = _rate(pooled["drifted"])
    un_r = _rate(pooled["undrifted"])
    return {
        "pooled": {"drifted": dr_r, "undrifted": un_r,
                   "delta_pp": 100.0 * (dr_r["pass_rate"] - un_r["pass_rate"])},
        "mean_within": {"delta_pp": _mean(deltas), "n_strata": len(deltas),
                        "sd_pp": _sd(deltas)},
        "by_action": {a: {"delta_pp": _mean(v), "n_strata": len(v),
                          "sd_pp": _sd(v)}
                      for a, v in sorted(by_action_d.items())},
        "by_paradigm": {p: {"delta_pp": _mean(v), "n_strata": len(v),
                            "sd_pp": _sd(v)}
                        for p, v in sorted(by_para_d.items())},
        "counts": dict(counts),
        "strata_keys": list(strata_keys),
    }


# --------------------------------------------------------------------------- #
# R-REAL — realization measures: what the plan ENACTED, not what it scored     #
# --------------------------------------------------------------------------- #
# Pass rate is a poor instrument for drift impact because it is gameable in
# two directions we have measured:
#
#   * VACUITY      -- drift on a Conditional's ``condition`` makes the
#                     preference vacuously true; it books a PASS while the
#                     still-evaluated subset degrades ~10pp underneath.
#   * OBLIGATION   -- drift on a Temporal ``subject_ap`` halves the trigger
#     SHRINKING       count (2.94 -> 1.42 subjects), so there are simply
#                     fewer chances to fail.  Never flagged trivial, so no
#                     triviality correction catches it.
#
# A REALIZATION measure sidesteps both.  It asks "how much did the plan
# actually enact this preference", on a continuous scale, with no pass
# threshold.  Vacuity yields an UNDEFINED value (excluded, not a free
# 1.0); obligation-shrinking does not inflate it because the quantity is
# per-opportunity rather than per-preference.
#
# This extractor collects every realization signal ALREADY present in
# ``CheckResult.details`` -- no plan/DB join required:
#
#   numeric.quantile_position          where the realized aggregate sits in
#                                      the candidate quartile band (1.0 =
#                                      fully optimized)  -> NumericPreference
#   atomic.passed_entities             fraction of entities satisfying the
#                                      predicate, split by op
#   temporal.aggregate.passed_groups   fraction of groups satisfied
#   scoped n_post/n_pre                how much the scope filter narrowed
#
# Everything is bucketed by the preference's OWN drift action (was THIS
# preference drifted, and how), not by the record-level ``profile_drift``
# label -- the record label is diluted by hard-constraint-only drift and
# by untouched sibling preferences.
#
# NOT included, deliberately:
#   * ``not_in`` realization -- saturates at exactly 1.000 in both splits
#     (models never select an excluded category), so it carries no
#     variance and cannot discriminate.
#   * member-level categorical diversity / suppression index -- needs a
#     plan+DB join to know WHICH set member each chosen entity realized.


def _own_drift_action(record: dict, paradigm: str, bank_id: Any) -> str:
    """Was THIS preference drifted, and how?  ``undrifted`` / ``drop`` /
    ``invert`` / ``mixed``."""
    actions = set()
    for s in (record.get("_drift_trace") or {}).get("sources") or []:
        if s.get("bank_id") is None:
            continue
        try:
            same = (s.get("paradigm") == paradigm
                    and int(s["bank_id"]) == int(bank_id))
        except (TypeError, ValueError):
            continue
        if same and s.get("drifted"):
            actions.add(s.get("drift_action"))
    if not actions:
        return "undrifted"
    return next(iter(actions)) if len(actions) == 1 else "mixed"


def extract_realization_by_drift(records: list[dict]) -> dict[str, Any]:
    """Continuous realization measures bucketed by own-drift action.

    Returns ``{measure -> {bucket -> {n, mean, median, sd}}}`` plus a
    ``by_bank`` view for the numeric measure, which stratifies on
    ``bank_id`` so template heterogeneity (different attributes, different
    candidate pools) cannot masquerade as a drift effect.
    """
    from collections import defaultdict

    vals: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    by_bank: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))

    for rec in records:
        for pref in (rec.get("_prefs") or []):
            bid = pref.get("bank_id")
            if bid is None:
                continue
            paradigm = pref.get("paradigm")
            bucket = _own_drift_action(rec, paradigm, bid)
            d = _det(pref)
            kind = d.get("kind")

            if kind == "numeric":
                q = d.get("quantile_position")
                if isinstance(q, (int, float)):
                    vals["numeric.quantile_position"][bucket].append(float(q))
                    by_bank[(paradigm, int(bid))][bucket].append(float(q))
            elif kind == "atomic":
                v = d.get("passed_entities")
                op = d.get("op")
                # not_in is saturated at 1.0 -- no variance, so excluded.
                if isinstance(v, (int, float)) and op and op != "not_in":
                    vals[f"atomic.passed_entities[{op}]"][bucket].append(float(v))
            elif kind == "temporal.aggregate":
                v = d.get("passed_groups")
                if isinstance(v, (int, float)):
                    vals["temporal.passed_groups"][bucket].append(float(v))
            elif kind == "scoped":
                pre = d.get("n_entities_pre_scope_filter")
                post = d.get("n_entities_post_scope_filter")
                if isinstance(pre, (int, float)) and pre:
                    vals["scoped.narrowing_ratio"][bucket].append(
                        float(post or 0) / float(pre))

    def _stats(xs: list[float]) -> dict[str, Any]:
        n = len(xs)
        if n == 0:
            return {"n": 0, "mean": 0.0, "median": 0.0, "sd": 0.0, "se": 0.0}
        mu = sum(xs) / n
        srt = sorted(xs)
        med = (srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2)
        sd = ((sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5) if n > 1 else 0.0
        return {"n": n, "mean": mu, "median": med, "sd": sd,
                "se": sd / (n ** 0.5) if n else 0.0}

    # Within-bank paired delta for the numeric measure: mean over bank
    # entries of (drifted mean - undrifted mean).  Each bank entry is its
    # own control, so attribute/pool differences cancel.
    paired: list[float] = []
    for _key, per in by_bank.items():
        und = per.get("undrifted") or []
        dr = [x for b, xs in per.items() if b not in ("undrifted",) for x in xs]
        if und and dr:
            paired.append(sum(dr) / len(dr) - sum(und) / len(und))
    paired_stats = _stats(paired)

    return {
        "measures": {m: {b: _stats(xs) for b, xs in per.items()}
                     for m, per in vals.items()},
        "numeric_within_bank": {"n_bank_entries": paired_stats["n"],
                                "mean_delta": paired_stats["mean"],
                                "sd": paired_stats["sd"],
                                "se": paired_stats["se"]},
    }


# --------------------------------------------------------------------------- #
# S-SUPP — Suppression Index: did the planner AVOID the drifted categories?    #
# --------------------------------------------------------------------------- #
# Requires ``_plan_days`` on each record (populated only when
# ``analyze_performance.py`` is given ``--plan-file``); returns an empty
# result otherwise, so the section simply does not render.
#
# Given a categorical set predicate such as
# ``Restaurant.cuisine in [Chinese, Italian, Mexican]`` where the profile
# drifted only SOME members:
#
#     SI = mean realization(drifted members)
#        / mean realization(surviving members)
#
#     SI = 1.0  drifted and surviving members used equally (no effect)
#     SI < 1.0  the planner avoided the drifted categories
#     SI > 1.0  it over-used them
#
# WHY THIS IS THE STRONGEST DESIGN AVAILABLE.  The ideal experiment -- two
# profiles for one query, one drifted and one not -- does not exist here;
# each record carries a single profile.  But a PARTIAL-COVERAGE leaf
# supplies an internal control: the drifted and surviving members share
# the same plan, entity type, value set, record and model, so every
# record-level confound (level, days, scope mix, paradigm, trip
# difficulty) cancels EXACTLY.  It is a paired design needing no
# counterfactual, and unlike pass rate it cannot be gamed by vacuity or by
# obligation-shrinking -- a suppressed member simply fails to appear.
#
# Realization of member ``v`` = number of plan entities of the leaf's
# entity_type whose attribute includes ``v`` (case-insensitively, matching
# the evaluator's canonicalisation).  List-valued attributes such as a
# restaurant's ``cuisine`` count once per entity listing the member.

_SUPP_ATTRS = frozenset({"cuisine", "house_rules", "category"})
_SUPP_OPS = frozenset({"in", "not_in", "contains_all"})

_ENTITY_BUCKET = {"Restaurant": "restaurants", "Attraction": "attractions",
                  "Transportation": "transportation"}


def _entities_of_type(plan_days: list, entity_type: str) -> list[dict]:
    out: list[dict] = []
    for day in plan_days or []:
        if not isinstance(day, dict):
            continue
        if entity_type == "Accommodation":
            acc = day.get("accommodation")
            if isinstance(acc, dict):
                out.append(acc)
            continue
        bucket = _ENTITY_BUCKET.get(entity_type)
        for e in ((day.get(bucket) or []) if bucket else []):
            if isinstance(e, dict):
                out.append(e)
    return out


def extract_suppression_index(records: list[dict]) -> dict[str, Any]:
    """Within-leaf paired realization contrast on partial-coverage
    categorical leaves.

    Returns ``{"n_leaves", "mean_si", "median_si", "se", "n_below_1",
    "by_action", "by_attribute", "by_paradigm", "counts", "examples"}``,
    or ``{}`` when no record carries ``_plan_days``.
    """
    from collections import defaultdict

    if not any(r.get("_plan_days") for r in records):
        return {}

    def _norm(x):
        return str(x).strip().lower()

    leaves: list[dict] = []
    counts = Counter()
    for rec in records:
        plan_days = rec.get("_plan_days")
        if not plan_days:
            continue
        sources = (rec.get("_drift_trace") or {}).get("sources") or []
        if not sources:
            continue
        pj = {(p.get("paradigm"), p.get("bank_id")): p
              for p in (rec.get("_preferences_json") or [])}
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for s in sources:
            if s.get("bank_id") is None:
                continue
            groups[(s.get("paradigm"), int(s["bank_id"]),
                    tuple(s.get("path") or []))].append(s)

        for (para, bid, path), ss in groups.items():
            entry = pj.get((para, bid))
            if entry is None:
                continue
            leaf = _walk_pref_template(entry.get("template") or {}, list(path))
            if not isinstance(leaf, dict):
                continue
            attr, op = leaf.get("attribute"), leaf.get("op")
            if attr not in _SUPP_ATTRS or op not in _SUPP_OPS:
                continue
            vals = leaf.get("value")
            if not isinstance(vals, (list, tuple)) or len(vals) < 2:
                continue          # need >=2 members for a paired contrast
            values = list(vals)
            counts["set_leaves"] += 1
            drifted_raw = {s.get("raw_value") for s in ss if s.get("drifted")}
            drifted = [v for v in values if v in drifted_raw]
            surviving = [v for v in values if v not in drifted_raw]
            if not drifted or not surviving:
                counts["not_partial_coverage"] += 1
                continue          # full or zero coverage: no internal control
            counts["partial_leaves"] += 1

            ents = _entities_of_type(plan_days, leaf.get("entity_type"))
            if not ents:
                counts["no_entities_in_plan"] += 1
                continue
            real = {v: 0 for v in values}
            for e in ents:
                raw = e.get(attr)
                if raw is None:
                    continue
                have = {_norm(x) for x in
                        (raw if isinstance(raw, (list, tuple)) else [raw])}
                for v in values:
                    if _norm(v) in have:
                        real[v] += 1
            d_mean = sum(real[v] for v in drifted) / len(drifted)
            s_mean = sum(real[v] for v in surviving) / len(surviving)
            if s_mean == 0:
                # SI undefined (0 denominator).  Reported, never silently
                # dropped -- a plan realising NEITHER arm carries no signal.
                counts["undefined_si_zero_surviving"] += 1
                continue
            counts["scored"] += 1
            acts = {s.get("drift_action") for s in ss if s.get("drifted")}
            leaves.append({
                "id": rec.get("id"), "paradigm": para, "bank_id": bid,
                "attribute": attr, "op": op,
                "entity_type": leaf.get("entity_type"),
                "n_entities": len(ents), "values": values,
                "drifted": drifted, "surviving": surviving,
                "drifted_mean": d_mean, "surviving_mean": s_mean,
                "si": d_mean / s_mean,
                "action": (next(iter(acts)) if len(acts) == 1 else "mixed"),
            })

    def _agg(xs):
        n = len(xs)
        if not n:
            return {"n": 0, "mean": 0.0, "median": 0.0, "sd": 0.0, "se": 0.0}
        mu = sum(xs) / n
        srt = sorted(xs)
        med = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
        sd = ((sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5) if n > 1 else 0.0
        return {"n": n, "mean": mu, "median": med, "sd": sd, "se": sd / n ** 0.5}

    sis = [x["si"] for x in leaves]
    grp = lambda key: {k: _agg([x["si"] for x in leaves if x[key] == k])
                       for k in sorted({x[key] for x in leaves})}
    overall = _agg(sis)
    return {
        "n_leaves": overall["n"], "mean_si": overall["mean"],
        "median_si": overall["median"], "se": overall["se"],
        "n_below_1": sum(1 for s in sis if s < 1.0),
        "by_action": grp("action"),
        "by_op": grp("op"),
        # ``in`` and ``not_in`` mean OPPOSITE things and must never be
        # pooled -- see the rendering note in analyze_performance.py.
        "by_op_action": {f"{o}/{a}": _agg([x["si"] for x in leaves
                                           if x["op"] == o and x["action"] == a])
                         for o in sorted({x["op"] for x in leaves})
                         for a in sorted({x["action"] for x in leaves})
                         if any(x["op"] == o and x["action"] == a for x in leaves)},
        "by_attribute": grp("attribute"),
        "by_paradigm": grp("paradigm"),
        "counts": dict(counts),
        "examples": sorted(leaves, key=lambda x: x["si"])[:8],
    }
