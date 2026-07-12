"""Export a HuggingFace-ready slice of ``data-generation/prefertripplan.jsonl``.

Maintainer-side utility.  NOT uploaded to the HF Hub; the Hub serves
the emitted JSONL + a README.md under a chosen HF-data folder.

Reads the augmented + profile-rendered + LLM-rendered file inside this
data-generation directory and writes ``prefertripplan.jsonl`` with a
richer per-record schema than the original (which only carried
persona / query / reference_information).

Emitted fields (per record):

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
    python3 data-generation/build_hf_dataset.py --in <path>.jsonl --out <out>.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

DEFAULT_SRC = ROOT / "prefertripplan.jsonl"
DEFAULT_OUT = ROOT / "HF-data" / "prefertripplan.jsonl"


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in",  dest="in_path",  type=Path, default=DEFAULT_SRC,
                    help="Source JSONL to export from.")
    ap.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT,
                    help="Output JSONL path (parent dir is created).")
    args = ap.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_skipped = 0
    with args.in_path.open() as fin, args.out_path.open("w") as fout:
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
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"read {n_in}, wrote {n_out}, skipped {n_skipped} -> {args.out_path}")


if __name__ == "__main__":
    main()
