#!/usr/bin/env python3
"""
eval.py -- Faithful port of TravelPlanner's ``evaluation/eval.py``,
adapted for the PreferTripPlan ``test`` and ``test_large`` splits, and
EXTENDED to fold in the preference-constraint evaluator alongside
commonsense + hard.

The upstream file lives at ``osunlp/TravelPlanner`` (also mirrored under
``travelplanner-ports/evaluation/eval.py`` in this repo).  The scoring
LOGIC is intentionally preserved -- the bucket-by-(level, day) tables
(``commonsenseConstraint_statistic``, ``hardConstraint_statistic``,
``count_record``, ``constraint_record``, ``mapping_constraint_record``,
``data_record``, ``constraint_dis_record``, ...), the ``statistics()``
aggregator, the ``paper_term_mapping()`` renamer, the gating that only
runs the hard evaluator when the commonsense plan is complete + in-
sandbox, and the final macro-pass tallying loop are all carried over
essentially verbatim so a diff against the upstream file stays short
and audit-able.

Preference evaluation (opt-out via ``--no-preferences``) runs on the
same structured plan file after the constraint loop; it reuses the
``evaluate_preferences.evaluate_records`` library API so both entry
points (this file and the standalone
``evaluate_preferences.py --plan ...``) share identical scoring.  The
extended top-line metrics::

    Preference Constraint Micro Pass Rate  -- prefs passed / total prefs
    Preference Constraint Macro Pass Rate  -- records where ALL prefs
                                              passed / records w/ prefs
    Final Pass Rate (incl. Preferences)    -- commonsense AND hard AND
                                              all-prefs-passed per rec
                                              (vacuous pass on no-prefs
                                              records)

The last figure lets you compare against the upstream ``Final Pass
Rate`` to see the marginal cost of the preference signal on top of the
existing constraint gates.

The substantive changes are limited to the four items you asked for:

  1. Data loading -- queries come from ``datasets.load_dataset`` (hub
     repo, local HF-formatted directory, or single JSONL file) instead
     of the upstream ``load_dataset('osunlp/TravelPlanner')`` /
     hard-coded HuggingFace name.  Plans come from
     ``evaluation/convert_plans.py``'s structured JSONL.
  2. Matching -- plans and queries are joined by 1-based ``id`` /
     ``idx`` rather than positional index.  Order and count of the two
     files can differ; missing plans evaluate as undelivered.
  3. Denominators -- the upstream ``420`` / ``105`` / ``2290`` magic
     constants are replaced by the same formula computed directly from
     the loaded queries::
         Delivery denom          = N
         Commonsense micro denom = 8 * N
         Hard        micro denom = N (one budget slot per instance)
                                 + Σ 1[local_constraint[k] not None]
                                     for every non-null slot k across
                                     every instance, regardless of level
     Nothing about which local_constraint keys "belong to" which level
     is hard-coded here -- the schema is discovered from the data.
  4. Paths -- ``commonsense_constraint`` / ``hard_constraint`` import
     the DB adapters via ``travelplanner-ports/`` (see the sys.path
     prep at the top of those files).

Usage
    python evaluation/eval.py \\
        --dataset UKPLab/PreferTripPlan --split test \\
        --plan-file plan-generation/structured_plans_openrouter_deepseek_deepseek-v4-flash.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_DEFAULT_DATASET = "UKPLab/PreferTripPlan"

sys.path.insert(0, str(_HERE))
from commonsense_constraint import evaluation as commonsense_eval   # noqa: E402
from hard_constraint       import evaluation as hard_eval           # noqa: E402
from evaluate_preferences  import (                                   # noqa: E402
    _DB               as _PrefDB,
    evaluate_records  as evaluate_preferences_bulk,
)


# --------------------------------------------------------------------------- #
# Helpers (kept as close to upstream ``eval.py`` as possible)                  #
# --------------------------------------------------------------------------- #
def load_line_json_data(filename: Path) -> list[dict]:
    data: list[dict] = []
    with Path(filename).open(encoding="utf-8") as f:
        for line in f.read().strip().split("\n"):
            if line:
                data.append(json.loads(line))
    return data


def load_queries(dataset: str, split: str) -> list[dict]:
    """Load queries via ``datasets.load_dataset(dataset, split=split)``.

    Accepts a HF hub repo, a local HF-formatted directory, or a single
    JSONL file (``split`` is ignored when a file path is given).
    ``verification_mode='no_checks'`` lets partially-populated splits
    (e.g. ``test_large`` before Phase 2) load without a cached
    split-size mismatch error."""
    p = Path(dataset)
    if p.is_file():
        ds = load_dataset("json", data_files=str(p), split="train",
                          verification_mode="no_checks")
    else:
        ds = load_dataset(str(dataset), split=split,
                          verification_mode="no_checks")
    return [dict(r) for r in ds]


def parse_local_constraint(raw: Any) -> dict:
    """The upstream file eagerly ``eval()``s / ``json.loads()`` the
    string form of ``local_constraint``; we do the same but a bit more
    defensively (HF-data stores it as a JSON string)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return json.loads(raw.replace("'", '"'))
    return {}


def count_true_false(data: list) -> tuple[int, int]:
    """Count the number of ``True`` and ``False`` values in a list.
    Verbatim from upstream."""
    return data.count(True), data.count(False)


def statistics(commonsense_statistic: dict) -> dict:
    """Generate statistics for each (level, day) and each constraint key.
    Verbatim from upstream: for each ``(level, day)`` cell that has
    populated eval outputs, tally ``true`` / ``false`` counts per key."""
    result = {level: {day: {} for day in commonsense_statistic[level]}
              for level in commonsense_statistic}
    for level, days in commonsense_statistic.items():
        for day, dicts in days.items():
            for dct in dicts:
                if dct:
                    for key, data in dct.items():
                        true_count, false_count = count_true_false(data)
                        if key not in result[level][day]:
                            result[level][day][key] = {"true": 0, "false": 0}
                        result[level][day][key]["true"] += true_count
                        result[level][day][key]["false"] += false_count
    return result


# The paper's user-facing labels for each constraint key.  Same map as
# upstream so downstream tooling (tables, plots) doesn't have to change.
_PAPER_TERM_MAP = {
    "is_valid_information_in_current_city": "Within Current City",
    "is_valid_information_in_sandbox":      "Within Sandbox",
    "is_reasonable_visiting_city":          "Reasonable City Route",
    "is_valid_restaurants":                 "Diverse Restaurants",
    "is_valid_transportation":              "Non-conf. Transportation",
    "is_valid_attractions":                 "Diverse Attractions",
    "is_valid_accommodation":               "Minimum Nights Stay",
    "is_not_absent":                        "Complete Information",
    "valid_cost":                           "Budget",
    "valid_room_rule":                      "Room Rule",
    "valid_cuisine":                        "Cuisine",
    "valid_room_type":                      "Room Type",
    "valid_transportation":                 "Transportation",
}


def paper_term_mapping(commonsense_constraint_record: dict,
                       hard_constraint_record: dict,
                       levels: list[str], days: list[int]) -> tuple[dict, dict]:
    """Rename raw eval-key names to paper-friendly labels.  Same as
    upstream but ``levels`` and ``days`` are computed from the loaded
    queries instead of the ``['easy','medium','hard']`` / ``[3,5,7]``
    hard-coded lists."""
    remap_c = {level: {day: {} for day in days} for level in levels}
    remap_h = {level: {day: {} for day in days} for level in levels}
    for level in commonsense_constraint_record:
        for day in commonsense_constraint_record[level]:
            remap_c[level][day] = {_PAPER_TERM_MAP[k]: v
                                   for k, v in commonsense_constraint_record[level][day].items()}
            remap_h[level][day] = {_PAPER_TERM_MAP[k]: v
                                   for k, v in hard_constraint_record[level][day].items()}
    return remap_c, remap_h


def _pyify(item: Any) -> Any:
    """Recursively convert numpy bools/ints so json.dumps works."""
    if isinstance(item, dict):
        return {k: _pyify(v) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return type(item)(_pyify(v) for v in item)
    if isinstance(item, np.bool_):
        return bool(item)
    if isinstance(item, (np.integer,)):
        return int(item)
    if isinstance(item, (np.floating,)):
        return float(item)
    return item


# --------------------------------------------------------------------------- #
# Score                                                                        #
# --------------------------------------------------------------------------- #
# Mapping between the upstream ``constraint_mapping`` and its inverse -- kept
# as data (not hardcoded per-level) so the schema is entirely discovered from
# the loaded queries in ``eval_score``.
_LOCAL_TO_EVAL = {
    "house rule":     "valid_room_rule",
    "cuisine":        "valid_cuisine",
    "room type":      "valid_room_type",
    "transportation": "valid_transportation",
}


# --------------------------------------------------------------------------- #
# Post-hoc audit: preference set ⊊ hard-constraint set                        #
# --------------------------------------------------------------------------- #
# Maps a preference-side ``(entity_type, attribute)`` to the corresponding
# ``local_constraint`` field for hard-vs-soft set comparison.  Only pairs
# where the LC field is list-valued are worth tagging (a scalar LC field
# has no notion of "strict subset").  Extend this map if the hard-
# constraint schema ever gains new list-valued fields.
_PREF_TO_HARD_LIST_FIELD: dict[tuple[str, str], str] = {
    ("Restaurant", "cuisine"): "cuisine",
}


def _walk_atomic_set_preds(preferences_json: list[dict] | None):
    """Yield every atomic set-op predicate anywhere inside the record's
    preferences (nested through Composite / Conditional / Lex /
    Compensatory / Scoped / Temporal wrappers), together with the
    ``scoped_depth`` (0 = trip-wide, >=1 = wrapped by a
    ScopedPreference) and a ``path`` string for provenance."""
    def visit(n, path, sd):
        if not isinstance(n, dict):
            return
        cls = n.get("class") or n.get("paradigm")
        t   = n.get("template") if "template" in n else n
        if isinstance(t, dict):
            et   = t.get("entity_type")
            attr = t.get("attribute")
            if et and attr:
                yield (t.get("op"), t.get("value"), t.get("scope"),
                       et, attr, path, sd)
        if cls == "CompositePreference":
            for i, c in enumerate(t.get("children") or []):
                yield from visit(c, path + f".children[{i}]", sd)
        elif cls == "ConditionalPreference":
            for k in ("condition", "then_pref", "else_pref"):
                if t.get(k):
                    yield from visit(t[k], path + f".{k}", sd)
        elif cls == "LexicographicPreference":
            for i, p in enumerate(t.get("preferences") or []):
                yield from visit(p, path + f".prefs[{i}]", sd)
        elif cls == "CompensatoryPreference":
            for k in ("primary_ap", "margin_ap", "secondary_ap"):
                if t.get(k):
                    yield from visit(t[k], path + f".{k}", sd)
        elif cls == "ScopedPreference":
            if t.get("inner"):
                yield from visit(t["inner"], path + ".inner", sd + 1)
            for i, sf in enumerate(t.get("scope_filters") or []):
                yield from visit(sf, path + f".scope_filters[{i}]", sd)
        elif cls == "TemporalPreference":
            for k in ("subject_ap", "reference_ap"):
                if t.get(k):
                    yield from visit(t[k], path + f".{k}", sd)

    for i, p in enumerate(preferences_json or []):
        yield from visit(p, f"p{i}", 0)


def detect_pref_strict_subset_of_hard(preferences_json: list[dict] | None,
                                       local_constraint: dict | None) -> list[dict]:
    """Return a list of tension instances where an atomic set-op
    preference's value set is a STRICT subset of the corresponding
    hard-constraint set on the same record.

    Each instance carries provenance so downstream analysis can filter
    to the specific pattern of interest -- e.g. only ``pref_scope=="all"``
    with ``scoped_depth==0`` isolates the trip-wide-``[all]``-vs-LC-superset
    tension that can force the LLM into a joint-reasoning bind (the plan
    must find multi-cuisine restaurants that both satisfy the pref's
    ``[all]``-scope and cover the extra LC cuisines).  Benign shapes
    (``[any]`` scope, or ``[all]`` wrapped inside a ScopedPreference)
    still surface here for completeness -- they aren't stripped, so the
    field can be used as raw provenance rather than only as a "failure
    warning" tag.

    Empty list => no strict-subset relationship anywhere in the record.
    """
    out: list[dict] = []
    lc = local_constraint or {}
    for op, val, scope, et, attr, path, sd in _walk_atomic_set_preds(preferences_json):
        if op not in ("in", "==", "∈"):
            continue
        hard_field = _PREF_TO_HARD_LIST_FIELD.get((et, attr))
        if hard_field is None:
            continue
        hard_val = lc.get(hard_field)
        if not isinstance(hard_val, list) or not hard_val:
            continue
        pref_set = set(val) if isinstance(val, list) else {val}
        hard_set = set(hard_val)
        if pref_set < hard_set:                       # strict subset
            out.append({
                "hard_field":        hard_field,
                "hard_set":          sorted(hard_set),
                "pref_entity":       et,
                "pref_attribute":    attr,
                "pref_op":           op,
                "pref_scope":        scope,
                "pref_set":          sorted(pref_set),
                "missing_from_pref": sorted(hard_set - pref_set),
                "scoped_depth":      int(sd),
                "path":              path,
            })
    return out


def eval_score(dataset: str, split: str, plan_file: Path, *,
               detailed_out: Path | None = None,
               with_preferences: bool = True) -> tuple[dict, dict]:
    """Score plans in ``plan_file`` against queries in
    ``load_dataset(dataset, split=split)``.  Returns
    ``(top_line_metrics, {"Commonsense Constraint":..., "Hard Constraint":...,
    "Preference Constraint": <_EvalSummary rendered tables>})`` --
    extending upstream's ``(result, detailed_scores)`` return shape with
    an optional preference-constraint block.

    When ``with_preferences=True`` (default), the same structured plan
    is passed through the preferences evaluator, and three additional
    top-line metrics are populated on ``result``::

        Preference Constraint Micro Pass Rate
        Preference Constraint Macro Pass Rate
        Final Pass Rate (incl. Preferences)

    The last metric AND's commonsense + hard + all-preferences-passed
    per record, so its ratio to ``Final Pass Rate`` quantifies the
    cost of the preference signal on top of the base constraints.

    When ``detailed_out`` is a path, per-row eval dicts are written to
    it (one JSON object per line) in addition to the top-line summary;
    each row also carries a ``preferences`` block when preference
    evaluation is enabled."""
    query_data_list = load_queries(dataset, split)
    tested_plans    = load_line_json_data(plan_file)

    N = len(query_data_list)
    if N == 0:
        raise SystemExit(f"[error] no queries loaded from {dataset}:{split}")

    # ----- discover (levels, days, local_constraint keys) from data -----
    # These SETS were hard-coded upstream (``['easy','medium','hard']``,
    # ``[3,5,7]``, and the per-level key allow-list).  Deriving them from
    # the loaded queries lets the eval work unchanged on any split whose
    # schema drifts (new levels, new day counts, new constraint slots).
    levels = sorted({q["level"] for q in query_data_list})
    days   = sorted({int(q["days"]) for q in query_data_list})
    lc_keys: set[str] = set()
    for q in query_data_list:
        lc_keys.update(parse_local_constraint(q.get("local_constraint")).keys())
    lc_keys_sorted = sorted(lc_keys)
    print(f"[in] {N} queries from {dataset} [{split}]  "
          f"levels={levels} days={days} local_constraint_keys={lc_keys_sorted}")
    print(f"[in] {len(tested_plans)} plans from {plan_file.name}")

    # Match plans to queries by 1-based ``id``/``idx`` (upstream used the
    # positional index into the file, but we can't rely on ordering here).
    # convert_plans.py emits ``id`` on freshly-produced structured files
    # (post the plan-generation id/source_id migration) but older files
    # still carry the 1-based ``idx`` key.  Auto-detect and use whichever
    # is present so the two flavours interop transparently.
    plan_id_key = ("id" if tested_plans and "id" in tested_plans[0]
                    else "idx")
    plans_by_id: dict[int, dict] = {int(p[plan_id_key]): p for p in tested_plans}
    unmatched = [q for q in query_data_list if int(q["id"]) not in plans_by_id]
    if unmatched:
        print(f"[warn] {len(unmatched)}/{N} queries have no plan record "
              f"(evaluated as empty / undelivered)")

    # ----- per-(level, day) buckets (same shape as upstream) ------------
    hardConstraint_statistic        = {lvl: {d: [] for d in days} for lvl in levels}
    commonsenseConstraint_statistic = {lvl: {d: [] for d in days} for lvl in levels}

    tested_plans_ordered: list[dict | None] = []
    plan_constraint_store: list[dict] = []
    delivery_cnt = 0
    per_row: list[dict] = []

    for query_data in tqdm(query_data_list, desc="evaluate"):
        qid = int(query_data["id"])
        query_data = dict(query_data)   # don't mutate the input
        query_data["local_constraint"] = parse_local_constraint(query_data.get("local_constraint"))
        query_data["days"] = int(query_data["days"])

        plan_rec  = plans_by_id.get(qid)
        plan_days = plan_rec["plan"] if plan_rec and plan_rec.get("plan") else None
        tested_plans_ordered.append(plan_rec)

        if plan_days:
            delivery_cnt += 1
            commonsense_info_box = commonsense_eval(query_data, plan_days)
        else:
            commonsense_info_box = None

        # Upstream gate (line 90 in the reference file): only score hard
        # constraints when the plan is COMPLETE and in-sandbox.
        if (commonsense_info_box
                and commonsense_info_box["is_not_absent"][0]
                and commonsense_info_box["is_valid_information_in_sandbox"][0]):
            hard_info_box = hard_eval(query_data, plan_days)
        else:
            hard_info_box = None

        plan_constraint_store.append({
            "commonsense_constraint": commonsense_info_box,
            "hard_constraint":        hard_info_box,
        })
        commonsenseConstraint_statistic[query_data["level"]][query_data["days"]].append(commonsense_info_box)
        hardConstraint_statistic[query_data["level"]][query_data["days"]].append(hard_info_box)

        # Post-hoc audit tag: strict-subset relationships between any
        # atomic set-op preference and the corresponding list-valued
        # hard-constraint field.  Empty list => no such relationship on
        # this record.  Downstream analysis can filter by ``pref_scope``
        # / ``scoped_depth`` to isolate the trip-wide-``[all]`` tension
        # from benign shapes.
        prefs_raw = query_data.get("preferences_json")
        if isinstance(prefs_raw, str):
            try:
                prefs_raw = json.loads(prefs_raw)
            except (TypeError, ValueError):
                prefs_raw = []
        subset_audit = detect_pref_strict_subset_of_hard(
            prefs_raw, query_data["local_constraint"])

        per_row.append({
            "id":         qid,
            "level":      query_data["level"],
            "days":       query_data["days"],
            "delivered":  plan_days is not None,
            "commonsense": _pyify(commonsense_info_box),
            "hard":        _pyify(hard_info_box),
            "pref_strict_subset_of_hard": subset_audit,
        })

    # ----- count non-null local_constraint slots per (level, day, key) --
    # Upstream hard-coded the 4-key list ``[house rule, cuisine, room
    # type, transportation]`` and the medium/hard-only filter.  Here we
    # discover the keys from the data (``lc_keys_sorted`` above) and
    # count every non-null slot regardless of level -- the level filter
    # was a defensive redundancy since the DATA already zeroed out
    # non-applicable slots.  If a future split assigns a
    # ``transportation`` slot to a ``medium`` row, we simply count it.
    constraint_record          = {lvl: {d: {k: 0 for k in lc_keys_sorted}
                                        for d in days} for lvl in levels}
    mapping_constraint_record  = {lvl: {d: {_LOCAL_TO_EVAL[k]: 0 for k in lc_keys_sorted
                                              if k in _LOCAL_TO_EVAL}
                                        for d in days} for lvl in levels}
    count_record               = {lvl: {d: 0 for d in days} for lvl in levels}

    for unit in query_data_list:
        lvl = unit["level"]
        day = int(unit["days"])
        count_record[lvl][day] += 1
        lc = parse_local_constraint(unit.get("local_constraint"))
        for k in lc_keys_sorted:
            if lc.get(k) is not None:
                constraint_record[lvl][day][k] += 1
                if k in _LOCAL_TO_EVAL:
                    mapping_constraint_record[lvl][day][_LOCAL_TO_EVAL[k]] += 1

    # ----- bucket-wise true/false tally (upstream ``statistics()``) -----
    commonsenseConstraint_statistic_processed = statistics(commonsenseConstraint_statistic)
    hardConstraint_statistic_processed        = statistics(hardConstraint_statistic)

    # ----- micro pass numerators AND denominators (same loop shape) ----
    data_record = {lvl: {d: [] for d in days} for lvl in levels}
    constraint_dis_record = {
        "commonsense": {"pass": 0, "total": 0},
        "hard":        {"pass": 0, "total": 0},
    }
    key_dict = {
        "commonsense": ["is_valid_information_in_current_city",
                        "is_valid_information_in_sandbox",
                        "is_reasonable_visiting_city",
                        "is_valid_restaurants",
                        "is_valid_transportation",
                        "is_valid_attractions",
                        "is_valid_accommodation",
                        "is_not_absent"],
        "hard":        ["valid_cost"] + [_LOCAL_TO_EVAL[k]
                                         for k in lc_keys_sorted
                                         if k in _LOCAL_TO_EVAL],
    }

    # Per-key numerator tallies -- used by the summary printer.  These are
    # NOT part of the metrics themselves; they're the natural break-outs of
    # ``constraint_dis_record[constraint]['pass']`` so a reader can see
    # which specific constraint types drove the aggregate score.
    commonsense_pass_by_key: dict[str, int] = {k: 0 for k in key_dict["commonsense"]}
    hard_pass_by_key:        dict[str, int] = {k: 0 for k in key_dict["hard"]}

    for constraint in ("commonsense", "hard"):
        cst = (commonsenseConstraint_statistic_processed if constraint == "commonsense"
               else hardConstraint_statistic_processed)
        pass_by_key = (commonsense_pass_by_key if constraint == "commonsense"
                       else hard_pass_by_key)
        for level in cst:
            for day in cst[level]:
                for key3 in key_dict[constraint]:
                    data_record[level][day].append("0/0")
                    if key3 in cst[level][day]:
                        constraint_dis_record[constraint]["pass"] += cst[level][day][key3]["true"]
                        pass_by_key[key3] += cst[level][day][key3]["true"]
                        # Denominator branch: matches upstream's semantics.
                        # * commonsense: one slot per instance for every commonsense key
                        # * hard 'valid_cost'/day/visitng_city: one per instance
                        # * hard local-constraint keys: one per non-null slot
                        # We use data-driven per-key counts throughout, so this
                        # applies uniformly whether a slot showed up under
                        # 'medium' or 'hard' or a hypothetical new level.
                        if constraint == "hard":
                            if key3 == "valid_cost":
                                constraint_dis_record[constraint]["total"] += count_record[level][day]
                                data_record[level][day][-1] = (
                                    f"{cst[level][day][key3]['true']}/{count_record[level][day]}")
                                hardConstraint_statistic_processed[level][day][key3]["total"] = count_record[level][day]
                            else:
                                slots = mapping_constraint_record[level][day].get(key3, 0)
                                constraint_dis_record[constraint]["total"] += slots
                                data_record[level][day][-1] = (
                                    f"{cst[level][day][key3]['true']}/{slots}")
                                hardConstraint_statistic_processed[level][day][key3]["total"] = slots
                        else:
                            constraint_dis_record[constraint]["total"] += count_record[level][day]
                            data_record[level][day][-1] = (
                                f"{cst[level][day][key3]['true']}/{count_record[level][day]}")
                            commonsenseConstraint_statistic_processed[level][day][key3]["total"] = count_record[level][day]

    # ----- macro pass tallying (verbatim from upstream) -----------------
    final_all_cnt          = 0
    final_commonsense_cnt  = 0
    final_hardConstraint_cnt = 0
    final_all_cnt_map      = {level: 0 for level in levels}

    for idx in range(N):
        if plan_constraint_store[idx]["commonsense_constraint"]:
            final_commonsense_pass  = True
            final_hardConstraint_pass = True
            for item in plan_constraint_store[idx]["commonsense_constraint"]:
                ok, _ = plan_constraint_store[idx]["commonsense_constraint"][item]
                if ok is not None and not ok:
                    final_commonsense_pass = False
                    break
            if plan_constraint_store[idx]["hard_constraint"] is None:
                continue
            for item in plan_constraint_store[idx]["hard_constraint"]:
                ok, _ = plan_constraint_store[idx]["hard_constraint"][item]
                if ok is not None and ok == False:
                    final_hardConstraint_pass = False
                    break
            if final_commonsense_pass:
                final_commonsense_cnt += 1
            if final_hardConstraint_pass:
                final_hardConstraint_cnt += 1
            if final_commonsense_pass and final_hardConstraint_pass:
                final_all_cnt += 1
                final_all_cnt_map[query_data_list[idx]["level"]] += 1

    # ----- preference evaluation (piggybacks on the structured plan file) --
    # Runs the exact same per-record evaluator the standalone
    # ``evaluate_preferences.py`` uses, but reuses this call's already-
    # loaded ``query_data_list`` + ``tested_plans`` so we don't re-read
    # the plan file or re-hydrate the HF split.  When
    # ``with_preferences=False`` we short-circuit and leave the
    # preference block off the result.
    pref_micro_total    = pref_micro_pass    = 0
    pref_macro_total    = pref_macro_pass    = 0    # denom = records w/ prefs
    final_incl_pref_cnt = 0                          # commonsense + hard + prefs
    final_incl_pref_by_level = {level: 0 for level in levels}
    pref_pass_by_paradigm: dict[str, dict[str, int]] = {}
    pref_result: dict = {}
    if with_preferences:
        hf_by_id = {int(q["id"]): q for q in query_data_list
                    if q.get("id") is not None}
        # Run preference eval bulk.  ``_PrefDB`` boots the four
        # TravelPlanner CSV pools lazily and can be re-used.
        pref_result = evaluate_preferences_bulk(
            tested_plans, hf_by_id,
            db        = _PrefDB(),
            plan_key  = "plan",
            prefs_key = "preferences_json",
            id_key    = plan_id_key,
            on_missing= "skip",
            progress_label = "eval-prefs",
        )
        per_id_prefs: dict[int, dict] = pref_result["per_id"]

        # Roll the per-record preference results into micro/macro tallies
        # AND merge them into the same ``per_row`` dicts so downstream
        # detailed_out carries the full picture.
        for idx in range(N):
            qid = int(query_data_list[idx]["id"])
            entry = per_id_prefs.get(qid)
            if entry is None or entry.get("evaluations") is None:
                per_row[idx]["preferences"] = None
                pref_pass_all = True   # pass-through: no prefs to fail
            else:
                evals = entry["evaluations"]
                pref_micro_total += len(evals)
                pref_micro_pass  += sum(1 for e in evals if e.get("passed"))
                per_row[idx]["preferences"]     = _pyify(evals)
                per_row[idx]["preference_summary"] = _pyify(entry["record_summary"])
                if evals:
                    pref_macro_total += 1
                    if entry["record_summary"]["all_passed"]:
                        pref_macro_pass += 1
                        pref_pass_all = True
                    else:
                        pref_pass_all = False
                    # per-paradigm micro tally
                    for e in evals:
                        pa = e.get("paradigm") or "?"
                        d = pref_pass_by_paradigm.setdefault(pa,
                            {"pass": 0, "total": 0})
                        d["total"] += 1
                        if e.get("passed"):
                            d["pass"] += 1
                else:
                    pref_pass_all = True   # record had no resolved prefs

            # Combined final pass: apply the same commonsense + hard gate
            # this row already went through, then intersect with prefs.
            cs_box   = plan_constraint_store[idx]["commonsense_constraint"]
            hard_box = plan_constraint_store[idx]["hard_constraint"]
            if not cs_box or hard_box is None:
                continue
            cs_pass   = all((v[0] is None or v[0])
                            for v in cs_box.values())
            hard_pass = all((v[0] is None or v[0])
                            for v in hard_box.values())
            if cs_pass and hard_pass and pref_pass_all:
                final_incl_pref_cnt += 1
                final_incl_pref_by_level[query_data_list[idx]["level"]] += 1

    # ----- assemble top-line metrics ------------------------------------
    result: dict = {}
    result["Dataset"]       = str(dataset)
    result["Split"]         = split
    result["N"]             = N
    result["Delivery Rate"] = delivery_cnt / N
    cs_denom   = 8 * N   # 8 commonsense keys per instance
    hard_denom = constraint_dis_record["hard"]["total"]
    result["Commonsense Constraint Micro Pass Rate"] = (
        constraint_dis_record["commonsense"]["pass"] / cs_denom)
    result["Commonsense Constraint Macro Pass Rate"] = final_commonsense_cnt / N
    result["Hard Constraint Micro Pass Rate"] = (
        constraint_dis_record["hard"]["pass"] / hard_denom) if hard_denom else 0.0
    result["Hard Constraint Macro Pass Rate"] = final_hardConstraint_cnt / N
    result["Final Pass Rate"]                 = final_all_cnt / N
    # ``hard`` denominator break-down by local_constraint slot (plus the
    # per-instance budget slot).  Derived from the already-computed
    # ``mapping_constraint_record``; no extra data pass.
    hard_denom_breakdown: dict[str, int] = {"budget": N}
    for k in lc_keys_sorted:
        if k in _LOCAL_TO_EVAL:
            eval_k = _LOCAL_TO_EVAL[k]
            hard_denom_breakdown[k] = sum(
                mapping_constraint_record[lvl][day][eval_k]
                for lvl in levels for day in days
            )

    result["_denominators"] = {
        "delivery":    N,
        "commonsense": cs_denom,
        "hard":        hard_denom,
    }
    result["_hard_denominator_breakdown"] = hard_denom_breakdown
    result["_commonsense_pass_by_key"]    = commonsense_pass_by_key
    result["_hard_pass_by_key"]           = hard_pass_by_key
    result["_final_pass_by_level"]        = final_all_cnt_map

    # Post-hoc audit roll-up: how many records have any strict-subset
    # tension between a preference set and a list-valued hard-constraint
    # field, and how many are in the "hardest" shape -- trip-wide (i.e.
    # not scope-restricted by a ScopedPreference wrapping) AND the
    # preference's own scope is ``all`` (universal on the pref-entity).
    n_records_with_subset  = sum(1 for r in per_row
                                  if r["pref_strict_subset_of_hard"])
    n_records_trip_wide_all = sum(
        1 for r in per_row
        if any(inst.get("scoped_depth") == 0 and inst.get("pref_scope") == "all"
                for inst in r["pref_strict_subset_of_hard"])
    )
    result["_pref_strict_subset_of_hard"] = {
        "records_any":              n_records_with_subset,
        "records_trip_wide_all":    n_records_trip_wide_all,
    }

    # ----- preference top-line metrics (only when the pass ran) --------
    if with_preferences:
        result["Preference Constraint Micro Pass Rate"] = (
            pref_micro_pass / pref_micro_total) if pref_micro_total else 0.0
        result["Preference Constraint Macro Pass Rate"] = (
            pref_macro_pass / pref_macro_total) if pref_macro_total else 0.0
        # Combined pass rate: commonsense + hard + all-preferences-passed.
        # Records with no preferences pass vacuously, so this metric
        # measures the ADDITIONAL cost of the preference signal on top of
        # the existing constraint gates -- compare against ``Final Pass
        # Rate`` above.
        result["Final Pass Rate (incl. Preferences)"] = final_incl_pref_cnt / N
        result["_denominators"]["preferences_micro"] = pref_micro_total
        result["_denominators"]["preferences_macro"] = pref_macro_total
        result["_preferences_pass_by_paradigm"]      = pref_pass_by_paradigm
        result["_final_incl_pref_by_level"]          = final_incl_pref_by_level

    remap_c, remap_h = paper_term_mapping(
        commonsenseConstraint_statistic_processed,
        hardConstraint_statistic_processed,
        levels=levels, days=days,
    )
    detailed_scores = {
        "Commonsense Constraint": remap_c,
        "Hard Constraint":        remap_h,
    }

    if detailed_out is not None:
        detailed_out.parent.mkdir(parents=True, exist_ok=True)
        with detailed_out.open("w") as f:
            for r in per_row:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[out] detailed per-row evaluation → {detailed_out}")

    return result, detailed_scores


# --------------------------------------------------------------------------- #
# Summary printer                                                              #
# --------------------------------------------------------------------------- #
def format_summary(result: dict, detailed: dict) -> str:
    lines: list[str] = []
    hdr = (f"===== PreferTripPlan eval :: {result['Dataset']} [{result['Split']}] "
           f"({result['N']} instances) =====")
    lines.append(hdr)
    # Existing constraint metrics (unchanged).
    for k in ("Delivery Rate",
              "Commonsense Constraint Micro Pass Rate",
              "Commonsense Constraint Macro Pass Rate",
              "Hard Constraint Micro Pass Rate",
              "Hard Constraint Macro Pass Rate",
              "Final Pass Rate"):
        lines.append(f"  {k:52s} = {result[k]:.4f}")
    # New preference block -- only when preference eval ran.
    has_prefs = "Preference Constraint Micro Pass Rate" in result
    if has_prefs:
        lines.append("")
        for k in ("Preference Constraint Micro Pass Rate",
                  "Preference Constraint Macro Pass Rate",
                  "Final Pass Rate (incl. Preferences)"):
            lines.append(f"  {k:52s} = {result[k]:.4f}")
        # Impact of preferences: how much does the combined-with-prefs
        # metric drop vs the base Final Pass Rate?  (0 = prefs didn't
        # cost anything on top of what commonsense + hard already
        # rejected; positive number = the "cost" of the preference
        # signal.)
        base   = result["Final Pass Rate"]
        combined = result["Final Pass Rate (incl. Preferences)"]
        drop   = base - combined
        ratio  = (combined / base) if base > 0 else 0.0
        lines.append(f"  {'Preference cost (Final - Final incl. Prefs)':52s} "
                     f"= {drop:.4f}")
        lines.append(f"  {'Preference retention (Final incl. Prefs / Final)':52s} "
                     f"= {ratio:.4f}")
    lines.append("")
    lines.append(f"  Denominators             : {result['_denominators']}")
    lines.append(f"  Hard denom  = {result['_denominators']['hard']}  "
                 f"({result['_hard_denominator_breakdown']})")
    lines.append(f"  Commonsense pass-by-key: {result['_commonsense_pass_by_key']}")
    lines.append(f"  Hard        pass-by-key: {result['_hard_pass_by_key']}")
    lines.append(f"  Final-pass by level    : {result['_final_pass_by_level']}")
    if "_pref_strict_subset_of_hard" in result:
        sub = result["_pref_strict_subset_of_hard"]
        lines.append(
            f"  Pref ⊊ hard (audit)    : "
            f"{sub['records_any']} records (any); "
            f"{sub['records_trip_wide_all']} records trip-wide + scope='all'")
    if has_prefs:
        lines.append(f"  Preferences pass-by-paradigm: "
                     f"{result['_preferences_pass_by_paradigm']}")
        lines.append(f"  Final-pass incl. Prefs by level: "
                     f"{result['_final_incl_pref_by_level']}")
    lines.append("=" * len(hdr))
    # The tabulated preference report (paradigm / sub-paradigm / level /
    # pairing / drift / triviality) is now produced separately by
    # ``evaluation/analyze_performance.py`` off the same detailed JSONL;
    # it's no longer appended here.
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Commonsense + hard constraint eval for PreferTripPlan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", default=_DEFAULT_DATASET,
                    help="HuggingFace dataset spec: hub repo, local HF-formatted "
                         "directory, or single JSONL file. "
                         f"Default: {_DEFAULT_DATASET}")
    ap.add_argument("--split", default="test",
                    help="Split name to load (ignored when --dataset is a "
                         "single JSONL file). Default: test.")
    ap.add_argument("--plan-file", type=Path, required=True,
                    help="Structured plan JSONL (from evaluation/convert_plans.py).")
    ap.add_argument("--out-summary", type=Path, default=None,
                    help="Optional plain-text summary destination.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional per-row detailed evaluation JSONL destination.")
    ap.add_argument("--no-preferences", action="store_true",
                    help="Skip the preference-constraint evaluation "
                         "(fall back to the upstream commonsense + hard "
                         "eval only).  Useful when the plan file has "
                         "no preferences or you want a strict apples-to-"
                         "apples upstream reproduction.")
    args = ap.parse_args()

    result, detailed = eval_score(
        dataset=args.dataset,
        split=args.split,
        plan_file=args.plan_file,
        detailed_out=args.out,
        with_preferences=not args.no_preferences,
    )
    summary = format_summary(result, detailed)
    print()
    print(summary)
    if args.out_summary is not None:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(summary + "\n")
        print(f"[out] summary → {args.out_summary}")


if __name__ == "__main__":
    main()
