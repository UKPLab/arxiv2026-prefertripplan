"""Export the HuggingFace-ready slices of ``data-generation/prefertripplan.jsonl``.

Maintainer-side utility.  NOT uploaded to the HF Hub; the Hub serves
the emitted JSONL files + a README.md under a chosen HF-data folder.

Reads the augmented + profile-rendered + LLM-rendered file inside this
data-generation directory and writes two files into ``HF-data/``:

  * ``prefertripplan_test_large.jsonl`` -- the full ``test_large`` split
    (one row per source record with an LLM-rendered query + a resolvable
    reference-information pool -- ~1000 rows).  Each row carries an
    ``id`` field (contiguous, starting at 1) as its first key.

  * ``prefertripplan_test.jsonl``        -- the curated ``test`` split
    (~225 rows), a balanced subset of ``test_large`` selected by
    ``data-generation/build_balanced_subset.py``.  Each row carries an
    ``id`` (contiguous, 1..N-of-subset) as its first key, immediately
    followed by ``source_id`` (the ``id`` of the same row in
    ``test_large``).

Which source records land in the ``test`` split is driven by
``--subset-ids`` (default ``data-generation/prefertripplan.subset.ids.txt``):
a plain-text file with one source ``query_id`` per line.

Emitted fields (per record):

  id                      -- contiguous integer, first key
  source_id               -- (test split only) id of the matching row
                             in test_large; second key immediately after id
  org                     -- origin city (from record)
  dest                    -- destination state or city (from record)
  days                    -- trip duration (int, from record)
  visiting_city_number    -- number of cities to visit (int, from record)
  date                    -- list of trip dates ["YYYY-MM-DD", ...] (from record)
  people_number           -- traveler count (int, from record)
  local_constraint        -- JSON-string of hard constraints
                             (house_rule / cuisine / room_type / transportation)
  budget                  -- emitted trip budget (int)
  level                   -- easy / medium / hard (from record)

  query                   -- LLM-rendered fluent trip request
                             (from ``llm_nl_query``)
  reference_information   -- JSON-string of the pool blocks the planner may
                             draw from.  Structure: a list of
                             { Description, Content } blocks.  The FIRST
                             block is always the date-day schedule
                             (``[[date, weekday], ...]``) so the planner
                             has the trip's temporal spine right at the
                             top; subsequent blocks are the per-city
                             entity pools (Accommodations / Restaurants /
                             Attractions) and per-leg flight / ground-
                             transport records inherited from
                             ``feasibility_metadata.reference_information``.

  profile                 -- LLM-rendered fluent traveler profile
                             (from ``llm_nl_profile``)
  profile_json            -- JSON-string of a trimmed profile dict:
                             { Location, Hobbies, Lifestyle, Travel Style,
                               Preferred Destinations,
                               Food and Dining Preferences, Dislikes }
  profile_drift           -- aligned / omission / inversion
                             (renamed from ``profile_drift_mode``)
  drift_trace             -- JSON-string of the full drift-source provenance
                             (renamed from ``profile_trace``)

  preferences_shorthand   -- JSON-string of a per-preference summary list:
                             [{ bank_id, preference, indicative_rationale }, ...]
                             derived from ``nl_render_log``
  preferences_json        -- JSON-string of the full ``preferences`` list

  preference_pair         -- JSON-string of pairing metadata:
                             { type, subtype }
                             derived from ``pairing_type`` / ``pairing_subtype``
  budget_update           -- JSON-string of budget-escalation info:
                             { original_value, multiplier }
                             derived from ``budget_original`` / ``budget_multiplier``

  trip_context            -- JSON-string of trip-context metadata
                             (phrase / category / people_number / etc.)

Every dict / list field carrying heterogeneous types across rows is
JSON-stringified so Arrow schema inference stays deterministic across
the whole split (the old-schema convention that HF's Hub loader
handles cleanly).  Downstream users deserialise with ``json.loads``.

Records missing ``llm_nl_query`` are skipped (typically none once the
LLM renderer has finished a full pass -- but skipping keeps partial
runs safe to export).

Usage:
    python3 data-generation/build_hf_dataset.py
    python3 data-generation/build_hf_dataset.py \\
        --in                data-generation/prefertripplan.jsonl \\
        --subset-ids        data-generation/prefertripplan.subset.ids.txt \\
        --out-test_large    data-generation/HF-data/prefertripplan_test_large.jsonl \\
        --out-test          data-generation/HF-data/prefertripplan_test.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

DEFAULT_SRC             = ROOT / "prefertripplan.jsonl"
DEFAULT_SUBSET_IDS      = ROOT / "prefertripplan.subset.ids.txt"
DEFAULT_OUT_TEST_LARGE  = ROOT / "HF-data" / "prefertripplan_test_large.jsonl"
DEFAULT_OUT_TEST        = ROOT / "HF-data" / "prefertripplan_test.jsonl"


PROFILE_JSON_INTERESTS_KEYS = (
    "Hobbies",
    "Lifestyle",
    "Travel Style",
    "Preferred Destinations",
    "Food and Dining Preferences",
    "Dislikes",
)


def _profile_json(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Trimmed profile view: Location + the 6 curated Interests fields."""
    p = profile or {}
    demo = p.get("Demographics") or {}
    interests = p.get("Interests") or {}
    out: dict[str, Any] = {
        "Location": demo.get("Location"),
    }
    for k in PROFILE_JSON_INTERESTS_KEYS:
        out[k] = list(interests.get(k) or [])
    return out


def _preferences_shorthand(nl_render_log: list[dict[str, Any]] | None,
                            preferences: list[dict[str, Any]] | None
                            ) -> list[dict[str, Any]]:
    """Compact per-preference summary: [{ bank_id, preference, rationale }, ...].

    Pulls `preference` from the log's ``resolved_source`` (the
    stringified resolved predicate) and `indicative_rationale` from the
    log's ``rationale``.  Also carries the ``bank_id`` from the
    preferences array; the log's ``preference`` key holds a
    ``ParadigmName#bank_id`` composite string but the numeric id itself
    is easier to consume as an int.
    """
    log = nl_render_log or []
    prefs = preferences or []
    # Positional zip -- generate_profiles.py emits nl_render_log in the
    # same order as the ``preferences`` list.
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(log):
        bank_id = None
        if i < len(prefs):
            bank_id = prefs[i].get("bank_id")
        out.append({
            "bank_id":              bank_id,
            "preference":           (entry.get("resolved_source") or "").strip(),
            "indicative_rationale": (entry.get("rationale")        or "").strip(),
        })
    return out


def _slim_flight_block(block: dict[str, Any]) -> dict[str, Any]:
    """Post-process a single reference_information block: for a flight
    leg whose Content is the augmenter's ``{available, count, price_stats,
    records}`` shape, replace Content with just the flat records list so
    the emitted pool shows only the raw flight rows.  All other block
    shapes (per-city entity pools, unavailable-flight diagnostics,
    ground-transport blocks, the date-day block prepended above) are
    returned untouched.
    """
    desc = block.get("Description") or ""
    if not desc.startswith("Flight from "):
        return block
    content = block.get("Content")
    if isinstance(content, dict) and isinstance(content.get("records"), list):
        return {"Description": desc, "Content": list(content["records"])}
    return block


def build_record(r: dict[str, Any]) -> dict[str, Any] | None:
    """Assemble one HF-emit row from a source record. Returns None if
    the source record can't be exported (missing query)."""
    query = r.get("llm_nl_query")
    if not query:
        return None

    fm = r.get("feasibility_metadata") or {}
    ref_info = fm.get("reference_information")
    if not ref_info:
        return None

    # Prepend the trip's date-day schedule as the first ref_info block
    # so the planner sees the temporal spine at the top of the pool.
    # Format matches ``feasibility_metadata`` conventions: one dict per
    # block with ``Description`` + ``Content`` keys.
    date_day = list(r.get("date_day") or [])
    ref_info = ([{"Description": "Trip dates and days-of-week",
                  "Content":     date_day}]
                + list(ref_info)) if date_day else list(ref_info)

    # HIDE flight metadata (count / price_stats / etc.) at HF-emit time:
    # a planner consuming ``reference_information`` should see the raw
    # flight records only, not the aggregate stats the augmenter keeps
    # around for feasibility auditing.  Unavailable-flight blocks stay
    # as-is (Content is a diagnostic dict, not a record list).
    ref_info = [_slim_flight_block(b) for b in ref_info]

    prefs = r.get("preferences") or []

    row: dict[str, Any] = {
        # trip-fact primitives (Arrow-safe)
        "org":                  r.get("org"),
        "dest":                 r.get("dest"),
        "days":                 r.get("days"),
        "visiting_city_number": r.get("visiting_city_number"),
        "date":                 list(r.get("date") or []),
        "people_number":        r.get("people_number"),
        "budget":               r.get("budget"),
        "level":                r.get("level"),

        # NL fields
        "query":   query,
        "profile": r.get("llm_nl_profile") or "",

        # simple string
        "profile_drift": r.get("profile_drift_mode"),
    }

    # Heterogeneous-shape fields -> JSON-stringify for Arrow stability.
    row["local_constraint"]      = json.dumps(r.get("local_constraint") or {},
                                              ensure_ascii=False)
    row["reference_information"] = json.dumps(ref_info, ensure_ascii=False)
    row["profile_json"]          = json.dumps(_profile_json(r.get("profile")),
                                              ensure_ascii=False)
    row["drift_trace"]           = json.dumps(r.get("profile_trace") or {},
                                              ensure_ascii=False)
    row["preferences_json"]      = json.dumps(prefs, ensure_ascii=False)
    row["preferences_shorthand"] = json.dumps(
        _preferences_shorthand(r.get("nl_render_log"), prefs),
        ensure_ascii=False)
    row["preference_pair"]       = json.dumps({
        "type":    r.get("pairing_type"),
        "subtype": r.get("pairing_subtype"),
    }, ensure_ascii=False)
    row["budget_update"]         = json.dumps({
        "original_value": r.get("budget_original"),
        "multiplier":     r.get("budget_multiplier"),
    }, ensure_ascii=False)
    row["trip_context"]          = json.dumps(r.get("trip_context") or {},
                                              ensure_ascii=False)

    return row


def _load_subset_ids(path: Path) -> set[int]:
    """Read one integer per non-empty line from ``path`` and return them
    as a set.  Comments (lines starting with ``#``) are ignored."""
    ids: set[int] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(int(line))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in",  dest="in_path",  type=Path, default=DEFAULT_SRC,
                    help="Source JSONL to export from.")
    ap.add_argument("--subset-ids", dest="subset_ids_path", type=Path,
                    default=DEFAULT_SUBSET_IDS,
                    help="Path to a text file listing source query_ids "
                         "(one per line) that belong to the curated "
                         "``test`` split.  Records not listed here go "
                         "only into ``test_large``.")
    ap.add_argument("--out-test_large", dest="out_large_path", type=Path,
                    default=DEFAULT_OUT_TEST_LARGE,
                    help="Output JSONL path for the full test_large split.")
    ap.add_argument("--out-test", dest="out_test_path", type=Path,
                    default=DEFAULT_OUT_TEST,
                    help="Output JSONL path for the curated test split.")
    args = ap.parse_args()

    subset_ids = _load_subset_ids(args.subset_ids_path)
    print(f"[subset] {len(subset_ids)} source query_ids from "
          f"{args.subset_ids_path}")

    args.out_large_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_test_path .parent.mkdir(parents=True, exist_ok=True)

    n_in = n_skipped = 0
    id_large = 0
    id_test  = 0
    unmatched_subset_ids = set(subset_ids)

    with args.in_path.open() as fin, \
         args.out_large_path.open("w") as f_large, \
         args.out_test_path .open("w") as f_test:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            r = json.loads(line)
            row = build_record(r)
            if row is None:
                n_skipped += 1
                continue
            # test_large: id (contiguous) is the first key.
            id_large += 1
            large_row = {"id": id_large, **row}
            f_large.write(json.dumps(large_row, ensure_ascii=False) + "\n")

            # test split: same underlying row, but with a contiguous id
            # and a source_id pointing back to the test_large id.
            qid = r.get("query_id")
            if qid in subset_ids:
                id_test += 1
                test_row = {"id": id_test, "source_id": id_large, **row}
                f_test.write(json.dumps(test_row, ensure_ascii=False) + "\n")
                unmatched_subset_ids.discard(qid)

    print(f"[test_large] wrote {id_large} rows to {args.out_large_path}")
    print(f"[test]       wrote {id_test} rows to {args.out_test_path}")
    print(f"[skip]       {n_skipped} source records skipped "
          f"(missing llm_nl_query or reference_information)")
    if unmatched_subset_ids:
        print(f"[warn]       {len(unmatched_subset_ids)} subset ids not "
              f"found in the source (missing NL or reference_information):"
              f" {sorted(unmatched_subset_ids)[:10]}"
              f"{' ...' if len(unmatched_subset_ids) > 10 else ''}")


if __name__ == "__main__":
    main()
