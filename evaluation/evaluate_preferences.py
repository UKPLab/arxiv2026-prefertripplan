"""Evaluate PreferTripPlan preferences against a TravelPlanner-style plan.

Callable BOTH as a standalone CLI AND as a library imported by ``eval.py``
(alongside ``commonsense_constraint`` and ``hard_constraint``).

This module bridges two on-disk formats and the ``preferences.py`` evaluator:

1. **Structured plan JSONL** -- one row per record with an ``id``
   (1-indexed, matching the HF dataset ``id``) and a ``plan`` field
   holding a list of per-day dicts of the shape emitted by
   ``evaluation/convert_plans.py``::

       {"days":            <1-based int>,
        "current_city":    <str>,      "transportation":  <str>,
        "breakfast":       <str>,      "attraction":      <str>,
        "lunch":           <str>,      "dinner":          <str>,
        "accommodation":   <str>}

   The redundant text→structure pass (previously done here by parsing
   ``Day N:`` / ``<Field>:`` lines) is GONE -- this module now consumes
   the structured plan directly and only performs the DB-hydration step
   (entity name -> attribute-rich dict via the TravelPlanner CSV pools).

2. **PreferTripPlan structured preferences** -- the JSON-string
   ``preferences_json`` column on every HF-data row (an ordered list of
   preference dicts, one per resolved preference in the query).  Every
   dict carries a ``paradigm`` (one of the 8 supported classes) and a
   ``template`` block whose shape depends on the paradigm.

For each row we:

  a. Hydrate the per-day string fields into ``list[dict]`` -- one dict
     per day with pool-attribute-rich entities, in the exact shape
     ``preferences.py`` consumes.  (DB lookups only; no text parsing.)
  b. Parse each entry of ``preferences_json`` back into a concrete
     ``preferences.py`` class instance.
  c. Call ``pref.evaluate(plan_days)`` for each parsed preference and
     collect the ``CheckResult`` (name, passed, score, details, ...).

Plan-day shape emitted by ``transform_plan``
--------------------------------------------

    { day:            int,       # 1-based
      date:           str,       # "YYYY-MM-DD" (no time component)
      week_group:     str,       # "weekday" | "weekend"
      travel_phase:   str,       # "arrival" | "stay" | "departure"
      city:           str,
      city_index:     int,       # 1-based position of `city` in the
                                 # trip's ordered visit sequence
                                 # (matches the bank's Day.city_index)
      attractions:    list[dict],
      restaurants:    list[dict],
      transportation: list[dict],     # unified across flight / ground modes
      accommodation:  dict | None,
    }

Emitted entity fields::

  Attraction     : name, city, category (list), rating
  Restaurant     : name, city, meal_type, cost, cuisine (list), rating
  Accommodation  : name, city, room_type, price (per unit-night),
                    price_per_person (eval-derived), house_rules (list),
                    maximum_occupancy, minimum_nights, rating
  Transportation : mode, flight_number, origin, destination, date,
                    duration, distance_km, price,
                    "Departure Time"  (raw HH:MM, flight only),
                    "Arrival Time"    (raw HH:MM, flight only),
                    departure_time    (TIME_WINDOWS category, flight only),
                    arrival_time      (TIME_WINDOWS category, flight only)

Transportation is one list -- flights, self-driving, and taxi legs all
appear under ``day["transportation"]`` with the union of columns; fields
that don't apply to a given mode (e.g. ``flight_number`` on a taxi,
``distance_km`` on a flight) are ``None``.  This aligns with the
preference bank, whose Temporal predicates target ``Transportation`` as
a single entity type (never ``Flight``), and with ``preferences.py``'s
``_extract_sequence`` which iterates the ``transportation`` key with
``entity_type="transportation"``.

Bank predicates ``Accommodation.cost`` and ``Transportation.cost`` are
transparently rerouted to ``Accommodation.price_per_person`` and
``Transportation.price`` at parse time (see ``_BANK_ATTR_REMAP``), so
the bank stays untouched while the emitted entities keep the
user-facing names.  ``entity_type`` is NOT emitted -- ``_extract_sequence``
in ``preferences.py`` sets it itself from the containing key.

Sources -- CSV DBs, NOT reference_information
----------------------------------------------

Entity attributes are pulled from the TravelPlanner CSV DBs under
``database/`` -- adapted from TravelPlanner's ``Accommodations`` /
``Restaurants`` / ``Attractions`` / ``Flights`` / ``GoogleDistanceMatrix``
tool APIs.  The evaluator does NOT read the HF row's
``reference_information`` pool at all; that field is a maintainer-side
artefact of the augmenter and is bypassed here so evaluations remain
consistent with what a TravelPlanner-style tool-using agent would see.

Weekdays come from the row's first-class ``date`` field via
``datetime.date.strftime("%A")`` -- no lookup in ``reference_information``
is required.

Per-person cost adjustment (accommodation)
------------------------------------------

The TravelPlanner accommodation DB stores ``price`` per **unit** per
night (one apartment / one room, not per traveller) with a
``maximum occupancy`` cap.  Preferences of the form
``Accommodation.cost <= X`` are stated per-person, so we normalise::

    units_needed   = ceil(people_number / max_occupancy)
    total_cost     = price * units_needed
    per_person_cost = total_cost / people_number

and expose the result under ``cost``; the raw per-unit price is kept
under ``price_per_unit``.  Restaurant ``Average Cost`` is already
per-person, and flight ``price`` / ground ``cost`` follow TravelPlanner
conventions (per-seat and total-party respectively), so those columns
are passed through as-is.

Transportation rows also publish a normalised ``mode``
(``"flight"`` / ``"self-driving"`` / ``"taxi"``).

Parsing preferences_json
------------------------

``preferences_json`` entries come in three related shapes; the parser
normalises all three:

  * top-level::
        {"paradigm": "AtomicPreference",
         "bank_id": 2, "trace": "...", "rationale": "...",
         "template": { "entity_type": "Accommodation",
                       "attribute": "rating", "op": ">=", "value": 5,
                       "scope": "all" }}

  * nested-flat (Composite children / Conditional condition & then_pref
    / Compensatory primary_ap-margin_ap-secondary_ap / Temporal
    subject_ap-reference_ap / Lex preferences)::
        {"class": "AtomicPreference",
         "entity_type": "...", "attribute": "...", "op": "...",
         "value": ..., "scope": "..."}

  * nested-wrapped (Scoped inner / scope_filters)::
        {"class": "AtomicPreference",
         "template": { ...same fields as nested-flat }}

``parse_preference(entry)`` accepts any of the three and returns a
``preferences.py`` class instance.  ``parse_preferences(entries)``
accepts the whole list (or its JSON-string form) and returns a list.

Command-line usage
------------------

Two modes are supported.  ``--dataset`` accepts either an HF Hub repo id
(loaded via ``datasets.load_dataset``) or a local path -- default
``UKPLab/PreferTripPlan`` on the Hub.  ``--split`` picks the split
(default ``test``; use ``test_large`` for the full 1000-row pool)::

    # Batch: score every plan in a STRUCTURED plan file
    # (evaluation/convert_plans.py output) against the corresponding
    # HF row's preferences_json (looked up by id).
    python3 evaluation/evaluate_preferences.py \\
        --plan    plan-generation/structured_plans_<model>.jsonl \\
        --dataset UKPLab/PreferTripPlan \\
        --split   test \\
        --out     evaluation/eval_<model>.jsonl

    # Demo: pick one easy/single record per paradigm from the HF split,
    # transform its plan (from --plan) and evaluate.
    python3 evaluation/evaluate_preferences.py --demo \\
        --plan    plan-generation/structured_plans_<model>.jsonl \\
        --dataset UKPLab/PreferTripPlan \\
        --split   test

Alignment check
---------------

Both the structured plan JSONL (from ``convert_plans.py``) and the HF
splits use a 1-based ``id`` field, so the pairing invariant is simply::

    plan_row["id"] == hf_row["id"]

The CLI validates this at load time.  A mismatched pairing is reported
to stderr and the affected plan rows are written with ``plan=null`` /
``evaluations=null``.

Output shape (batch mode)
-------------------------

Each output row is a minimal 4-field object::

    id             : int          -- dataset id (1-indexed; matches HF)
    plan           : list[dict]   -- structured plan (echoed from input),
                                     or `null` on unmatched / null rows
    record_summary : dict         -- per-record aggregate of the record's
                                     preferences: n_preferences, n_passed,
                                     pass_rate, mean_score, all_passed,
                                     any_passed  (a macro data point).
                                     `null` when `plan` is `null`.
    evaluations    : list[dict]   -- one entry per preference (a micro
                                     data point):
                                     { name, paradigm, sub_paradigm,
                                       bank_id, passed, score,
                                       constraint_type, trivial, details }
                                     `null` when `plan` is `null`.

Library API (used by eval.py)
-----------------------------

The public entry point is::

    result = evaluate_records(plan_records, hf_rows_by_id,
                              db=db, on_missing="skip")

Returns ``{"per_id": {id: {...}}, "summary": _EvalSummary, "counts": {...}}``.
See the ``evaluate_records`` docstring for details.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# preferences.py sits at the project root; the TravelPlanner tool ports
# sit under travelplanner-ports/.  Add both to sys.path when this module
# is invoked directly.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_TP_PORTS = _ROOT / "travelplanner-ports"
for _p in (_ROOT, _TP_PORTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from preferences import (                                         # noqa: E402
    AtomicPreference,
    CompositePreference,
    ConditionalPreference,
    LexicographicPreference,
    NumericPreference,
    ScopedPreference,
    CompensatoryPreference,
    TemporalPreference,
    Preference,
    CheckResult,
    ConstraintType,
)

# TravelPlanner-ported tool APIs.  Each constructor lazily loads its CSV
# via pandas; we instantiate them once per evaluation run.
from tools.accommodations.apis      import Accommodations   # noqa: E402
from tools.restaurants.apis         import Restaurants      # noqa: E402
from tools.attractions.apis         import Attractions      # noqa: E402
from tools.flights.apis             import Flights          # noqa: E402
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix  # noqa: E402


# --------------------------------------------------------------------------- #
# Progress reporter                                                           #
# --------------------------------------------------------------------------- #
# Ported from ``plan-generation/generate_plans.py`` so the two CLIs share the
# same progress-bar semantics.  Uses ``tqdm`` when available, falls back to
# a plain ``\r``-updated stderr line otherwise -- no hard dependency added.

import time as _time  # local alias so we don't shadow anything downstream


class _ProgressReporter:
    def __init__(self, total: int, label: str = "evaluating"):
        self.total = total
        self.done  = 0
        self.label = label
        self.start = _time.time()
        self._last_line_len = 0
        self._tqdm = None
        try:
            from tqdm import tqdm as _tqdm
            self._tqdm = _tqdm(total=total, desc=label, unit="row",
                               dynamic_ncols=True, mininterval=0.3, leave=True)
        except Exception:
            self._fallback_write(f"[progress] {label}: 0/{total} (starting...)")

    @staticmethod
    def _fmt_dt(s: float) -> str:
        s = int(max(s, 0))
        h, rem = divmod(s, 3600); m, sec = divmod(rem, 60)
        if h: return f"{h}h{m:02d}m{sec:02d}s"
        if m: return f"{m}m{sec:02d}s"
        return f"{sec}s"

    def _fallback_write(self, s: str) -> None:
        pad = " " * max(0, self._last_line_len - len(s))
        sys.stderr.write("\r" + s + pad); sys.stderr.flush()
        self._last_line_len = len(s)

    def step(self, n: int = 1) -> None:
        self.done += n
        if self._tqdm is not None:
            self._tqdm.update(n); return
        elapsed = _time.time() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remain = (self.total - self.done) / rate if rate > 0 else 0.0
        pct = 100.0 * self.done / self.total if self.total else 100.0
        self._fallback_write(
            f"[progress] {self.label}: {self.done}/{self.total} "
            f"({pct:5.1f}%)  elapsed={self._fmt_dt(elapsed)}  "
            f"eta={self._fmt_dt(remain)}  rate={rate:.2f} row/s")

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close(); self._tqdm = None
        elif self._last_line_len:
            sys.stderr.write("\n"); sys.stderr.flush()
            self._last_line_len = 0


# --------------------------------------------------------------------------- #
# TravelPlanner CSV DBs                                                       #
# --------------------------------------------------------------------------- #

_WEEKEND = {"saturday", "sunday"}

# Time-window categories used by preference-bank Transportation predicates.
# Verbatim from ``data-generation/augment_preferences.py``'s TIME_WINDOWS.
# Values are (start_min, end_min) in minutes-since-midnight; the 'night'
# window wraps past midnight and uses the +24h convention (28*60 == 04:00
# of the next day).  A given HH:MM string maps to exactly one category.
TIME_WINDOWS: dict[str, tuple[int, int]] = {
    "morning":   ( 4 * 60, 12 * 60),   # [04:00, 12:00)
    "afternoon": (12 * 60, 17 * 60),   # [12:00, 17:00)
    "evening":   (17 * 60, 22 * 60),   # [17:00, 22:00)
    "night":     (22 * 60, 28 * 60),   # [22:00, 04:00 next day)
}


def _hhmm_minutes(s: Any) -> int | None:
    """Parse an ``HH:MM`` string into minutes-since-midnight."""
    if not isinstance(s, str) or ":" not in s:
        return None
    try:
        h_str, m_str = s.split(":", 1)
        h, m = int(h_str), int(m_str)
    except (ValueError, TypeError):
        return None
    if not (0 <= h < 24 and 0 <= m < 60):
        return None
    return h * 60 + m


def _time_category(hhmm: Any) -> str:
    """Map an ``HH:MM`` string to its TIME_WINDOWS category, or ``""``
    if the string is unparseable / doesn't fall in any window."""
    minutes = _hhmm_minutes(hhmm)
    if minutes is None:
        return ""
    for cat, (lo, hi) in TIME_WINDOWS.items():
        if cat == "night":
            # Night wraps past midnight: HH:MM in [22:00, 24:00) OR
            # [00:00, 04:00) counts as 'night'.
            if minutes >= lo or minutes < (hi - 24 * 60):
                return cat
        elif lo <= minutes < hi:
            return cat
    return ""


# Bank predicates on Accommodation.cost / Transportation.cost address the
# *evaluated* per-person / per-party derivations; the emitted entity dicts
# expose these under `price_per_person` / `price` respectively (matching
# the user-facing schema).  Remap at parse time so the bank stays
# untouched but the AtomicPreference reads the right key.
_BANK_ATTR_REMAP: dict[tuple[str, str], str] = {
    ("Accommodation",  "cost"): "price_per_person",
    ("Transportation", "cost"): "price",
}

# Ground-transport cost model, verbatim from TravelPlanner's
# ``GoogleDistanceMatrix.run``:  driving $0.05/km, taxi $1.0/km.  These
# are total-party costs; per-TravelPlanner convention we don't scale
# them by ``people_number``.  (Users who need per-person ground costs
# can divide downstream.)
_GROUND_COST_PER_KM = {"self-driving": 0.05, "taxi": 1.0}


def _norm_name(s: str) -> str:
    """Normalise an entity name for lenient matching: lowercase, strip
    surrounding whitespace and trailing punctuation ('.', ',', ';')."""
    if s is None:
        return ""
    return re.sub(r"[.,;]+$", "", str(s).strip()).lower()


def _parse_house_rules(raw: str) -> list[str]:
    """The accommodations CSV stores house rules as a comma-separated
    string (e.g. ``"No smoking, No parties"``) or Python-list-literal
    string (``"['No smoking', 'No parties']"``).  Return a clean list.
    Empty / missing / literal ``nan`` → ``[]``."""
    if not raw or not isinstance(raw, str):
        return []
    raw = raw.strip()
    if not raw or raw.lower() == "nan":
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            import ast
            v = ast.literal_eval(raw)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_listish(raw: Any) -> list[str]:
    """Best-effort conversion of a CSV cell that stores a list into a
    Python list of strings.  Accepts ``"['a', 'b']"``, ``"a, b"``, plain
    lists, or None."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [part.strip() for part in s.split(",") if part.strip()]


def _km_from_distance(distance_str: str) -> float:
    """Extract kilometres from the distance-matrix's ``"1,144 km"``
    string form.  Returns 0.0 if the cell is empty / unparseable."""
    if distance_str is None:
        return 0.0
    s = str(distance_str).lower().replace(",", "").replace("km", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # pandas turns missing CSV cells into NaN floats
    return None if (f != f) else f


def _safe_int(v: Any) -> int | None:
    f = _safe_float(v)
    return None if f is None else int(f)


def per_person_accommodation_cost(unit_price: float | None,
                                    max_occupancy: int | None,
                                    people_number: int) -> float:
    """Convert an accommodation's per-unit nightly price into a
    per-person nightly cost::

        units_needed    = ceil(people_number / max_occupancy)
        per_person_cost = unit_price * units_needed / people_number

    If ``max_occupancy`` is missing or non-positive we default to 1
    (worst case -- one unit per person).  If ``unit_price`` is missing
    or non-positive we return 0.0."""
    price = _safe_float(unit_price) or 0.0
    if price <= 0:
        return 0.0
    pn = max(int(people_number or 1), 1)
    mo = _safe_int(max_occupancy)
    if not mo or mo <= 0:
        mo = 1
    units = math.ceil(pn / mo)
    return price * units / pn


class _DB:
    """Thin wrapper around the TravelPlanner-ported tool APIs.

    Composes ``Accommodations`` / ``Restaurants`` / ``Attractions`` /
    ``Flights`` / ``GoogleDistanceMatrix`` (from
    ``travelplanner-ports/tools/*/apis.py``) and pre-computes lookup
    indices on their pandas DataFrames so per-day hydration is O(1)::

        (city_lower, name_norm) -> row-dict   (accom / rest / attr)
        flight_number           -> row-dict   (flights)
        (origin_lower, dest_lower) -> row-dict (distance matrix)

    Row dicts are normalised into the ``preferences.py`` schema names --
    e.g. accommodations expose both the raw ``price`` (per unit) and the
    per-person ``cost`` (computed via ``per_person_accommodation_cost``)."""

    def __init__(self):
        # Load raw tool APIs -- each reads its default CSV on construction.
        self.accommodations = Accommodations()
        self.restaurants    = Restaurants()
        self.attractions    = Attractions()
        self.flights        = Flights()
        self.distance       = GoogleDistanceMatrix()

        # --- Accommodations index (only bank-relevant cols) ---------------
        # The tool's own ``.dropna(subset=_REQUIRED_COLUMNS)`` keeps
        # rows whose only missing cell is `house_rules` / `minimum
        # nights`, so no CSV re-read is needed here.
        self._accom_ix: dict[tuple[str, str], dict] = {}
        for _, r in self.accommodations.data.iterrows():
            name = str(r.get("NAME") or "").strip()
            city = str(r.get("city") or "").strip()
            if not name:
                continue
            # ``price`` is per-unit; per-person ``cost`` is computed at
            # lookup time (needs the caller's people_number).
            self._accom_ix[(city.lower(), _norm_name(name))] = {
                "name":              name,
                "city":              city,
                "room_type":         str(r.get("room type") or "").strip(),
                "price":             _safe_float(r.get("price")),
                "rating":            _safe_float(r.get("review rate number")),
                "house_rules":       _parse_house_rules(r.get("house_rules")),
                "minimum_nights":    _safe_int(r.get("minimum nights")),
                "maximum_occupancy": _safe_int(r.get("maximum occupancy")),
            }


        # --- Restaurants index -------------------------------------------
        self._rest_ix: dict[tuple[str, str], dict] = {}
        for _, r in self.restaurants.data.iterrows():
            name = str(r.get("Name") or "").strip()
            city = str(r.get("City") or "").strip()
            if not name:
                continue
            self._rest_ix[(city.lower(), _norm_name(name))] = {
                "name":     name,
                "city":     city,
                "cost":     _safe_float(r.get("Average Cost")),
                "cuisine":  _parse_listish(r.get("Cuisines")),   # singular per bank
                "rating":   _safe_float(r.get("Aggregate Rating")),
            }

        # --- Attractions index -------------------------------------------
        self._attr_ix: dict[tuple[str, str], dict] = {}
        for _, r in self.attractions.data.iterrows():
            name = str(r.get("Name") or "").strip()
            city = str(r.get("City") or "").strip()
            if not name:
                continue
            self._attr_ix[(city.lower(), _norm_name(name))] = {
                "name":     name,
                "city":     city,
                "category": _parse_listish(r.get("categories")),  # singular per bank
                "rating":   _safe_float(r.get("rating")),
            }

        # --- Flights index (by flight number) ------------------------------
        # 4.5M rows -- use a MultiIndex-backed DataFrame for O(1) .loc[]
        # lookup without materialising every row into a Python dict.
        flights_df = self.flights.data.copy()
        flights_df["Flight Number"] = flights_df["Flight Number"].astype(str).str.strip()
        self._flights_by_num = flights_df.set_index("Flight Number", drop=True)

        # --- Distance matrix (by (origin, dest)) ---------------------------
        self._dist_ix: dict[tuple[str, str], dict] = {}
        for _, r in self.distance.data.iterrows():
            o = str(r.get("origin") or "").strip()
            d = str(r.get("destination") or "").strip()
            if not o or not d:
                continue
            self._dist_ix[(o.lower(), d.lower())] = {
                "origin":      o,
                "dest":        d,
                "duration":    "" if r.get("duration") is None else str(r.get("duration")),
                "distance_km": _km_from_distance(r.get("distance")),
            }

    # -- lookups -----------------------------------------------------------

    def lookup_attraction(self, name: str, city: str) -> dict | None:
        return self._city_named_lookup(self._attr_ix, name, city)

    def lookup_restaurant(self, name: str, city: str) -> dict | None:
        return self._city_named_lookup(self._rest_ix, name, city)

    def lookup_accommodation(self, name: str, city: str,
                              *, people_number: int) -> dict | None:
        """Return an accommodation dict carrying the TravelPlanner-named
        fields (``price`` / ``maximum_occupancy`` / ``minimum_nights`` /
        ``rating`` / ``room_type`` / ``house_rules``) verbatim, plus a
        ``price_per_person`` derivation computed via
        ``per_person_accommodation_cost``.  The bank's
        ``Accommodation.cost`` predicates are routed to
        ``price_per_person`` in ``_atomic_from_template``."""
        row = self._city_named_lookup(self._accom_ix, name, city)
        if row is None:
            return None
        out = dict(row)
        out["price_per_person"] = per_person_accommodation_cost(
            out.get("price"), out.get("maximum_occupancy"), people_number)
        return out

    def lookup_flight(self, number: str,
                       *, date: str | None = None) -> dict | None:
        """Look up a flight by its number.  When ``date`` is supplied,
        return the flight ONLY if its ``FlightDate`` matches -- an LLM
        that fabricates a flight-number / day pairing (real flight,
        wrong day) is caught here and the caller sees ``None``.

        The returned dict exposes the raw TravelPlanner columns the eval
        writes out (``flight_number``, ``elapsed``, ``date``, ``origin``,
        ``destination``, ``price``, ``Departure Time``, ``Arrival Time``)
        plus categorical bank attributes (``departure_time``,
        ``arrival_time``) derived from the HH:MM strings via
        ``TIME_WINDOWS``."""
        if not number:
            return None
        key = str(number).strip()
        try:
            hit = self._flights_by_num.loc[key]
        except KeyError:
            return None
        if hasattr(hit, "iloc") and hit.ndim > 1:
            if date:
                hit = hit[hit["FlightDate"].astype(str).str[:10]
                           == str(date)[:10]]
                if len(hit) == 0:
                    return None
                hit = hit.iloc[0]
            else:
                hit = hit.iloc[0]
        elif date:
            actual_date = str(hit.get("FlightDate") or "")[:10]
            if actual_date != str(date)[:10]:
                return None

        dep_hhmm = str(hit.get("DepTime") or "").strip()
        arr_hhmm = str(hit.get("ArrTime") or "").strip()
        return {
            "flight_number":  key,
            "elapsed":        str(hit.get("ActualElapsedTime") or "").strip(),
            "date":           str(hit.get("FlightDate") or "").strip()[:10],
            "origin":         str(hit.get("OriginCityName") or "").strip(),
            "destination":    str(hit.get("DestCityName") or "").strip(),
            "price":          _safe_float(hit.get("Price")),
            "Departure Time": dep_hhmm,
            "Arrival Time":   arr_hhmm,
            "departure_time": _time_category(dep_hhmm),
            "arrival_time":   _time_category(arr_hhmm),
        }

    def lookup_ground(self, mode: str, origin: str, dest: str) -> dict | None:
        """Look up a ``self-driving`` / ``taxi`` leg by (origin, dest).
        Returns origin / dest / distance_km / duration / price (the
        latter computed via the TravelPlanner rate table --
        ``$0.05/km`` driving, ``$1.0/km`` taxi -- total for the party)."""
        base = self._dist_ix.get(((origin or "").lower(),
                                    (dest or "").lower()))
        if base is None or not base.get("distance_km"):
            return None
        rate = _GROUND_COST_PER_KM.get(mode, 0.0)
        return {
            "mode":        mode,
            "origin":      base["origin"],
            "dest":        base["dest"],
            "duration":    base["duration"],
            "distance_km": base["distance_km"],
            "price":       base["distance_km"] * rate,
        }

    @staticmethod
    def _city_named_lookup(bucket: dict[tuple[str, str], dict],
                            name: str, city: str) -> dict | None:
        """Strict per-city lookup, matching TravelPlanner's
        ``commonsense_constraint.is_valid_information_in_sandbox``
        semantics (``Name.str.contains(re.escape(name)) & City == city``):

          * The city filter is HARD -- no cross-city fallback.  An LLM
            that names a real entity but pins it to the wrong city
            is treated the same as an entity that doesn't exist.
          * Match uses the LLM-supplied ``name`` as a substring of the
            DB name (equivalent to TravelPlanner's ``str.contains``).
            The reverse direction (DB name inside LLM name) and the
            exact-name-any-city fallback that earlier versions
            performed are NOT in TravelPlanner and are dropped here.

        Returns ``None`` when no row matches -- ``_build_day``'s
        ``on_missing`` policy then decides whether to warn/skip/raise
        and emits a stub carrying the LLM-stated ``name`` and ``city``
        verbatim (faithful to what the plan actually said)."""
        nn = _norm_name(name)
        if not nn:
            return None
        city_key = (city or "").lower()
        # Exact normalised match first.
        hit = bucket.get((city_key, nn))
        if hit:
            return hit
        # TravelPlanner-style substring: LLM name must appear inside the
        # DB name, within the same city.
        for (ck, nk), row in bucket.items():
            if ck == city_key and nn in nk:
                return row
        return None


def _weekday_of(date_str: str) -> str:
    """Return the English weekday name for a ``YYYY-MM-DD`` string, or
    an empty string if the date can't be parsed."""
    from datetime import date as _date
    try:
        return _date.fromisoformat(str(date_str)).strftime("%A")
    except (TypeError, ValueError):
        return ""


# --------------------------------------------------------------------------- #
# Per-day field parsing                                                        #
# --------------------------------------------------------------------------- #
# NOTE: text-plan parsing (``_parse_text`` + ``Day N:`` / ``<Field>:`` regexes)
# has been REMOVED.  The structured plan file emitted by
# ``evaluation/convert_plans.py`` already carries the per-day per-header string
# fields we need, so the redundant text-parsing pass is gone.  What remains
# below are the regexes / helpers that hydrate a single per-day field string
# (e.g. Transportation, Current City) into entity-attribute-rich dicts by
# looking up the TravelPlanner CSV DBs.  Those are still required.

_CITY_FROM_TO_RE = re.compile(r"^\s*from\s+(.+?)\s+to\s+(.+?)\s*$", re.IGNORECASE)
_FLIGHT_NUM_RE   = re.compile(r"Flight\s+Number\s*:\s*([A-Za-z0-9]+)", re.IGNORECASE)
_TRANSPORT_OD_RE = re.compile(r"from\s+(.+?)\s+to\s+(.+?)(?:,|$)", re.IGNORECASE)
_MODE_TOKENS     = (
    ("self-driving", "self-driving"),
    ("self driving", "self-driving"),
    ("selfdriving",  "self-driving"),
    ("driving",      "self-driving"),
    ("taxi",         "taxi"),
    ("flight",       "flight"),
    ("plane",        "flight"),
    ("air",          "flight"),
)


# --------------------------------------------------------------------------- #
# Template-literal canonicalisation                                            #
# --------------------------------------------------------------------------- #
#
# The plan side and the template side of a categorical comparison are
# produced by different pipelines, and only ONE of them is canonicalised.
# ``_parse_transportation`` folds every surface form of a transport line
# to the lower-case vocabulary above ("flight" / "self-driving" /
# "taxi"), so a plan-derived ``mode`` is always lower case.  The template
# side comes straight out of the HF ``preferences_json`` column and is
# never normalised -- and four bank entries spell the literal as
# ``"Flight"``:
#
#     LexicographicPreference#11   mode == "Flight"  [all]   tier 1
#     ScopedPreference#24          mode == "Flight"  [all]
#     ScopedPreference#27          mode == "Flight"  [any]   scope filter
#     ScopedPreference#28          mode == "Flight"  [any]   scope filter
#
# ``AtomicPreference._check_one`` compares with a plain ``==`` on strings,
# so those predicates are UNSATISFIABLE: they can never match "flight".
# Where the literal sits in a predicate that must hold this forces a
# guaranteed fail (Lexicographic#11 scored 0.0% across 28 evaluations and
# all four models); where it sits in a scope filter the filter matches
# nothing and the inner predicate passes VACUOUSLY instead.
#
# Fixing this in the bank would mean regenerating the dataset, and fixing
# it by changing the canonical token to "Flight" would mean touching the
# six internal sites that branch on ``mode == "flight"`` to decide
# flight-vs-ground field routing and cost lookup -- a silent-wrong-answer
# risk far worse than the bug.  So we normalise the TEMPLATE literal at
# the evaluation boundary instead, leaving ``preferences.py`` and the
# plan-side pipeline untouched.
#
# Deliberately restricted to ``mode``.  Every other categorical attribute
# (category / cuisine / house_rules / room_type / meal_type /
# travel_phase / week_group) was verified to agree in casing on both
# sides across all 225 test records, so widening the allowlist would be
# a no-op today; widen it only against fresh evidence of a mismatch.
_CASEFOLD_ATTRS = frozenset({"mode"})

_casefold_warned: set[tuple] = set()


def _casefold_literals(node: Any, *, trace: str | None = None) -> Any:
    """Return ``node`` with categorical literals folded to lower case.

    Recurses through the nested ``{"class": ..., "template": {...}}``
    structure and rewrites ``value`` wherever the sibling ``attribute``
    is in ``_CASEFOLD_ATTRS``.  Copy-on-write: nodes needing no change
    are returned as-is, so the common path allocates nothing and the
    caller's ``preferences_json`` is never mutated.

    A rewrite is reported once per (attribute, original literal) so a
    future bank-casing drift announces itself on stderr rather than
    silently scoring zero."""
    if isinstance(node, list):
        return [_casefold_literals(x, trace=trace) for x in node]
    if not isinstance(node, dict):
        return node

    out = node
    changed = False

    def _mark():
        nonlocal out, changed
        if not changed:
            out = dict(node)
            changed = True

    attribute = node.get("attribute")
    if attribute in _CASEFOLD_ATTRS and "value" in node:
        value = node["value"]
        if isinstance(value, str):
            folded = value.lower()
        elif isinstance(value, (list, tuple)):
            folded = [v.lower() if isinstance(v, str) else v for v in value]
        else:
            folded = value
        if folded != value:
            key = (attribute, str(value))
            if key not in _casefold_warned:
                _casefold_warned.add(key)
                print(f"[casefold] {trace or '?'}: {attribute} literal "
                      f"{value!r} -> {folded!r} (bank casing differs from the "
                      f"canonical plan-side vocabulary)",
                      file=sys.stderr, flush=True)
            _mark()
            out["value"] = folded

    for key, child in node.items():
        if key == "value" or not isinstance(child, (dict, list)):
            continue
        new_child = _casefold_literals(child, trace=trace)
        if new_child is not child:
            _mark()
            out[key] = new_child
    return out


def _parse_current_city(raw: str) -> tuple[str, str | None]:
    """Return (city, from_origin).  ``city`` is the destination if the
    line is ``from A to B``, otherwise the single city name.
    ``from_origin`` is non-None only for inter-city days."""
    raw = (raw or "").strip().rstrip(".;,")
    if not raw or raw == "-":
        return "", None
    m = _CITY_FROM_TO_RE.match(raw)
    if m:
        return m.group(2).strip().rstrip(".;,"), m.group(1).strip().rstrip(".;,")
    return raw, None


def _split_outside_parens(s: str, sep: str) -> list[str]:
    """Split ``s`` on every occurrence of ``sep`` that lies outside
    parentheses.  Real DB names contain both parens and ``;``/``,``
    (e.g. ``"Saipan & Tinian 2-DAY Tour (Day1: SPN; Day2: TIN)"``,
    ``"Thok (The House of Kakori)"``), and LLM asides contain the same
    punctuation inside their parenthetical -- so a naive ``re.split``
    corrupts both cases.  This walker tracks paren depth and only
    splits at depth 0."""
    out: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            out.append(s[start:i])
            start = i + 1
    out.append(s[start:])
    return out


def _rsplit_outside_parens(s: str, sep: str) -> tuple[str, str]:
    """Return ``(head, tail)`` at the last occurrence of ``sep`` outside
    parens.  Returns ``(s, "")`` if no such separator exists."""
    depth = 0
    last  = -1
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            last = i
    if last < 0:
        return s, ""
    return s[:last], s[last + 1:]


def _parse_named_line(raw: str) -> list[tuple[str, str]]:
    """Split a ``<name>, <city>`` (possibly ``;``-separated for the
    attraction line) into (name, city) pairs.  ``;`` and ``,`` inside
    parentheses are NOT treated as separators, so:

      - LLM asides like
        ``"Subway, Washington (assuming quick bite; using a placeholder)"``
        parse to a single item ``("Subway", "Washington")``.
      - Real DB names with parenthetical qualifiers containing punctuation
        (``"Thok (The House of Kakori)"``, ``"Gopal (Sindhi) Restaurant"``,
        ``"Saipan & Tinian 2-DAY Tour (Day1: SPN; Day2: TIN)"``) stay
        intact.

    Trailing ``.`` / ``,`` / ``;`` on each name are stripped."""
    if not raw or raw.strip() == "-":
        return []
    items: list[tuple[str, str]] = []
    for chunk in _split_outside_parens(raw, ";"):
        chunk = chunk.strip().rstrip(".;,")
        if not chunk:
            continue
        name, city = _rsplit_outside_parens(chunk, ",")
        if city:
            # City side may still trail an LLM aside like
            # "Washington (assuming quick bite; ...)".  US TravelPlanner
            # city names never contain parens, so a `(` in the city part
            # is always an LLM annotation -- trim from there onward.
            # The NAME side is left intact because real DB names DO
            # contain parens ("Thok (The House of Kakori)").
            paren_i = city.find("(")
            if paren_i >= 0:
                city = city[:paren_i]
            items.append((name.strip(), city.strip().rstrip(".;,")))
        else:
            items.append((chunk, ""))
    return items


def _parse_transportation(raw: str, day_od: tuple[str, str | None]
                           ) -> tuple[str | None, str, str, str]:
    """Classify a Transportation line.  Returns ``(mode, flight_num,
    origin, dest)`` where ``mode`` is ``None`` for the ``-`` no-transport
    day.  ``origin``/``dest`` fall back to the day's ``Current City``
    when the transport line doesn't spell them out."""
    if not raw or raw.strip() == "-":
        return None, "", "", ""

    lowered = raw.lower()
    mode: str | None = None
    for tok, canonical in _MODE_TOKENS:
        if tok in lowered:
            mode = canonical
            break

    m_num = _FLIGHT_NUM_RE.search(raw)
    flight_num = m_num.group(1) if m_num else ""
    if flight_num and mode is None:
        mode = "flight"

    origin, dest = "", ""
    m_od = _TRANSPORT_OD_RE.search(raw)
    if m_od:
        origin = m_od.group(1).strip().rstrip(".;,")
        dest   = m_od.group(2).strip().rstrip(".;,")
    else:
        current_city, from_origin = day_od
        origin, dest = (from_origin or "").strip(), current_city

    return mode, flight_num, origin, dest


# --------------------------------------------------------------------------- #
# Day construction                                                             #
# --------------------------------------------------------------------------- #

_TRANSPORT_UNION_FIELDS = (
    "mode",
    "flight_number",   # flight only
    "origin",
    "destination",
    "date",            # flight only
    "duration",        # elapsed (flight) / duration (ground)
    "distance_km",     # ground only
    "price",
    "Departure Time",  # flight only  (raw HH:MM)
    "Arrival Time",    # flight only  (raw HH:MM)
    "departure_time",  # flight only  (TIME_WINDOWS category)
    "arrival_time",    # flight only  (TIME_WINDOWS category)
)


def _transport_view(mode: str, row: dict | None,
                     *, flight_num: str = "") -> dict:
    """Unified transportation entity -- one shape for flight /
    self-driving / taxi.  Mode-inapplicable fields are ``None``.

    Union of columns:
      mode, flight_number, origin, destination, date, duration,
      distance_km, price, "Departure Time", "Arrival Time",
      departure_time, arrival_time

    Ground modes fill ``mode``, ``origin``, ``destination``, ``duration``,
    ``distance_km``, ``price``; leave the rest ``None``.  Flights fill
    everything except ``distance_km``."""
    out: dict = {k: None for k in _TRANSPORT_UNION_FIELDS}
    out["mode"] = mode
    if row is None:
        if mode == "flight":
            out["flight_number"] = flight_num or None
            out["price"] = 0.0
        else:
            out["price"] = 0.0
        return out

    if mode == "flight":
        out["flight_number"]  = row.get("flight_number") or flight_num
        out["origin"]         = row.get("origin", "")
        out["destination"]    = row.get("destination", "")
        out["date"]           = row.get("date", "")
        out["duration"]       = row.get("elapsed", "")   # flight elapsed → unified duration
        out["price"]          = row.get("price") or 0.0
        out["Departure Time"] = row.get("Departure Time", "")
        out["Arrival Time"]   = row.get("Arrival Time", "")
        out["departure_time"] = row.get("departure_time", "")
        out["arrival_time"]   = row.get("arrival_time", "")
    else:
        out["origin"]      = row.get("origin", "")
        out["destination"] = row.get("dest", "")
        out["duration"]    = row.get("duration", "")
        out["distance_km"] = row.get("distance_km") or 0.0
        out["price"]       = row.get("price") or 0.0
    return out


def _day_date(raw_date: Any) -> str:
    """Normalise a date value to ``YYYY-MM-DD``.  Handles str,
    pandas.Timestamp, and datasets-library datetime strings that carry a
    ``" 00:00:00"`` tail."""
    if raw_date is None:
        return ""
    s = str(raw_date).strip()
    # Take the first 10 chars if they look like an ISO date.
    return s[:10] if len(s) >= 10 and s[4] == "-" and s[7] == "-" else s


def _compute_city_indices(raw_days: list[dict]) -> list[int]:
    """For each day, return the **1-based** index of its city within
    the trip's ordered visit sequence (first appearance = 1, next
    distinct city = 2, ...).  Matches the preference bank's
    ``Day.city_index`` convention, which humans wrote 1-indexed."""
    order: dict[str, int] = {}
    out: list[int] = []
    for raw in raw_days:
        city, _ = _parse_current_city(str(raw.get("current_city") or ""))
        key = (city or "").lower()
        if key and key not in order:
            order[key] = len(order) + 1        # 1-indexed
        out.append(order.get(key, 1))
    return out


def _build_day(idx: int, total: int, raw: dict[str, Any],
                db: _DB, date_seq: list[str],
                city_index: int, people_number: int,
                on_missing: str) -> dict:
    """Assemble one plan-day dict from parsed raw fields + DB lookups.
    Only bank-relevant fields are emitted on each entity."""
    day_number = int(raw.get("_day_number") or (idx + 1))

    city, from_origin = _parse_current_city(str(raw.get("current_city") or ""))

    date_str = _day_date(date_seq[idx]) if idx < len(date_seq) else ""
    weekday  = _weekday_of(date_str)
    week_group = "weekend" if weekday.lower() in _WEEKEND else "weekday"

    if total <= 1:
        travel_phase = "arrival"        # single-day plan
    elif idx == 0:
        travel_phase = "arrival"
    elif idx == total - 1:
        travel_phase = "departure"
    else:
        travel_phase = "stay"

    # ---- Meals -----------------------------------------------------------
    restaurants: list[dict] = []
    for meal_key in ("breakfast", "lunch", "dinner"):
        for name, meal_city in _parse_named_line(str(raw.get(meal_key) or "")):
            row = db.lookup_restaurant(name, meal_city or city)
            if row is None:
                if on_missing == "raise":
                    raise LookupError(
                        f"Day {day_number}: restaurant '{name}' not in DB")
                if on_missing == "warn":
                    print(f"[evaluate] day {day_number} {meal_key}: "
                          f"restaurant '{name}' not in DB for city "
                          f"'{meal_city or city}'", file=sys.stderr)
                row = {"name": name, "city": meal_city or city,
                       "cost": None, "cuisine": [], "rating": None}
            # Emit only the bank-relevant fields, plus `meal_type` (the
            # attribute name Restaurant.meal_type in the bank).
            restaurants.append({
                "name":      row["name"],
                "city":      row["city"],
                "meal_type": meal_key,
                "cost":      row.get("cost"),
                "cuisine":   row.get("cuisine") or [],
                "rating":    row.get("rating"),
            })

    # ---- Attractions -----------------------------------------------------
    attractions: list[dict] = []
    for name, att_city in _parse_named_line(str(raw.get("attraction") or "")):
        row = db.lookup_attraction(name, att_city or city)
        if row is None:
            if on_missing == "raise":
                raise LookupError(
                    f"Day {day_number}: attraction '{name}' not in DB")
            if on_missing == "warn":
                print(f"[evaluate] day {day_number} attraction: "
                      f"'{name}' not in DB for city '{att_city or city}'",
                      file=sys.stderr)
            row = {"name": name, "city": att_city or city,
                   "category": [], "rating": None}
        attractions.append({
            "name":     row["name"],
            "city":     row["city"],
            "category": row.get("category") or [],
            "rating":   row.get("rating"),
        })

    # ---- Accommodation (per-person cost) ---------------------------------
    accommodation: dict | None = None
    accom_items = _parse_named_line(str(raw.get("accommodation") or ""))
    if accom_items:
        name, accom_city = accom_items[0]
        row = db.lookup_accommodation(name, accom_city or city,
                                        people_number=people_number)
        if row is None:
            if on_missing == "raise":
                raise LookupError(
                    f"Day {day_number}: accommodation '{name}' not in DB")
            if on_missing == "warn":
                print(f"[evaluate] day {day_number} accommodation: "
                      f"'{name}' not in DB for city "
                      f"'{accom_city or city}'", file=sys.stderr)
            row = {"name": name, "city": accom_city or city,
                   "room_type": "", "price": 0.0, "price_per_person": 0.0,
                   "house_rules": [], "maximum_occupancy": None,
                   "minimum_nights": None, "rating": None}
        accommodation = {
            "name":              row["name"],
            "city":              row["city"],
            "room_type":         row.get("room_type", ""),
            "price":             row.get("price"),
            "price_per_person":  row.get("price_per_person"),
            "house_rules":       row.get("house_rules") or [],
            "maximum_occupancy": row.get("maximum_occupancy"),
            "minimum_nights":    row.get("minimum_nights"),
            "rating":            row.get("rating"),
        }

    # ---- Transportation (unified: flight / self-driving / taxi) ----------
    transportation: list[dict] = []
    mode, flight_num, t_origin, t_dest = _parse_transportation(
        str(raw.get("transportation") or ""), (city, from_origin))

    if mode == "flight":
        # Filter by the day's date so a real flight-number reused across
        # days is not silently accepted on the wrong day.
        flight_row = db.lookup_flight(flight_num, date=date_str)
        if flight_row is None and on_missing == "warn":
            print(f"[evaluate] day {day_number} flight: "
                  f"flight_number '{flight_num}' not in DB on "
                  f"{date_str!r}", file=sys.stderr)
        elif flight_row is None and on_missing == "raise":
            raise LookupError(
                f"Day {day_number}: flight '{flight_num}' not in DB "
                f"on {date_str}")
        transportation.append(_transport_view("flight", flight_row,
                                                flight_num=flight_num))
    elif mode in ("self-driving", "taxi"):
        row = db.lookup_ground(mode, t_origin, t_dest)
        if row is None and on_missing == "warn":
            print(f"[evaluate] day {day_number} {mode}: "
                  f"({t_origin!r} → {t_dest!r}) not in DB",
                  file=sys.stderr)
        elif row is None and on_missing == "raise":
            raise LookupError(
                f"Day {day_number}: {mode} {t_origin}→{t_dest} not in DB")
        transportation.append(_transport_view(mode, row))
    # mode is None → "-" (no transport on this day); leave the list empty.

    return {
        "day":            day_number,
        "date":           date_str,
        "week_group":     week_group,
        "travel_phase":   travel_phase,
        "city":           city,
        "city_index":     city_index,
        "attractions":    attractions,
        "restaurants":    restaurants,
        "transportation": transportation,
        "accommodation":  accommodation,
    }


# --------------------------------------------------------------------------- #
# Plan-transform public API                                                    #
# --------------------------------------------------------------------------- #

_STRUCTURED_DAY_KEYS = (
    "current_city", "transportation", "breakfast",
    "attraction", "lunch", "dinner", "accommodation",
)


def transform_plan(plan_days: list[dict] | None,
                    *,
                    db: _DB,
                    date_seq: list[str] | None = None,
                    people_number: int = 1,
                    on_missing: str = "warn") -> list[dict]:
    """Hydrate a structured plan (list of per-day dicts, as emitted by
    ``evaluation/convert_plans.py``) into the ``preferences.py`` day-dict
    list.  Entity attributes come from the CSV DBs behind ``db``; the HF
    row's ``reference_information`` is NOT consulted.

    Input day-dict shape (from ``convert_plans.py`` / ``example_format``)::

        {"days":            <1-based int>,
         "current_city":    <str>,
         "transportation":  <str>,
         "breakfast":       <str>,
         "attraction":      <str>,
         "lunch":           <str>,
         "dinner":          <str>,
         "accommodation":   <str>}

    Missing / unset fields are treated as the empty string; a literal
    ``"-"`` placeholder is preserved.  Empty / missing inputs yield
    ``[]`` (undelivered plan).

    Parameters
    ----------
    plan_days : list[dict] | None
        The structured plan list.  ``None`` or empty -> ``[]``.
    db : _DB
        A pre-loaded DB instance -- reuse across many calls to amortise
        the CSV load.
    date_seq : list[str] | None
        Ordered ``["YYYY-MM-DD", ...]`` trip dates (one per day).  Comes
        from the HF row's first-class ``date`` field.  Used to compute
        each day's ``date`` and ``week_group``.
    people_number : int
        Party size; drives per-person accommodation cost.
    on_missing : {"warn", "skip", "raise"}
        Behaviour when an entity name isn't in the DB."""
    if not plan_days:
        return []
    date_seq = list(date_seq or [])
    # Normalise each incoming day-dict into the shape ``_build_day``
    # expects: ``_day_number`` (int; falls back to positional index+1)
    # plus every per-header string field.  We also cast ``"-"`` /
    # ``None`` to ``""`` uniformly so downstream parsers stay simple.
    raw_days: list[dict] = []
    for i, d in enumerate(plan_days or []):
        raw: dict = {}
        # Day number: the structured file uses ``days``; ``_day_number``
        # is also accepted so this stays interop with any caller that
        # already normalised.
        day_no = d.get("_day_number", d.get("days"))
        try:
            raw["_day_number"] = int(day_no)
        except (TypeError, ValueError):
            raw["_day_number"] = i + 1
        for k in _STRUCTURED_DAY_KEYS:
            v = d.get(k)
            raw[k] = "" if v is None else str(v)
        raw_days.append(raw)
    total = len(raw_days)
    city_indices = _compute_city_indices(raw_days)
    return [_build_day(i, total, raw, db, date_seq,
                        city_indices[i], int(people_number or 1),
                        on_missing)
            for i, raw in enumerate(raw_days)]


# --------------------------------------------------------------------------- #
# preferences_json → Preference instances                                     #
# --------------------------------------------------------------------------- #

def _remap_attr(entity_type: str, attribute: str) -> str:
    """Route bank attribute names to the eval-side field names emitted
    on the plan-day entities (see ``_BANK_ATTR_REMAP``).  Leaves any
    attribute not in the remap table unchanged."""
    return _BANK_ATTR_REMAP.get((entity_type, attribute), attribute)


def _atomic_from_template(t: dict) -> AtomicPreference:
    et = t["entity_type"]
    return AtomicPreference(
        entity_type = et,
        attribute   = _remap_attr(et, t["attribute"]),
        op          = t["op"],
        value       = t["value"],
        scope       = t.get("scope", "any"),
    )


def _resolve_entry(entry: dict) -> tuple[str, dict]:
    """Given a dict from ``preferences_json`` (top-level, nested-flat, or
    nested-wrapped), return ``(class_name, template_dict)`` in a
    consistent shape.  For nested-flat entries the pref fields ARE the
    template, so the returned template is a shallow view of the entry
    itself.  For nested-wrapped entries we unwrap the ``template`` key."""
    if not isinstance(entry, dict):
        raise TypeError(f"preferences_json entry must be a dict, got {type(entry).__name__}")

    # top-level shape: has both "paradigm" and "template"
    if "paradigm" in entry and "template" in entry:
        return entry["paradigm"], entry["template"]

    # nested-wrapped shape: has both "class" and "template"
    if "class" in entry and "template" in entry:
        return entry["class"], entry["template"]

    # nested-flat shape: has "class" only; pref fields live at top level
    if "class" in entry:
        cls = entry["class"]
        # drop the "class" key from the view so it looks like a template
        return cls, {k: v for k, v in entry.items() if k != "class"}

    raise ValueError(
        f"preferences_json entry missing paradigm/class: {sorted(entry.keys())}")


def parse_preference(entry: dict) -> Preference:
    """Convert one ``preferences_json`` entry into a concrete
    ``preferences.py`` class instance.  Recurses through nested prefs
    (Composite children, Conditional condition/then_pref, Compensatory
    primary/margin/secondary, Lex preferences, Scoped inner/scope_filters,
    Temporal subject_ap/reference_ap)."""
    cls, t = _resolve_entry(entry)

    if cls == "AtomicPreference":
        return _atomic_from_template(t)

    if cls == "CompositePreference":
        op = t.get("op") or t.get("operator") or "AND"
        children = [parse_preference(c) for c in (t.get("children") or [])]
        return CompositePreference(op, children)

    if cls == "ConditionalPreference":
        cond = parse_preference(t["condition"])
        then = parse_preference(t["then_pref"])
        els  = parse_preference(t["else_pref"]) if t.get("else_pref") else None
        return ConditionalPreference(cond, then, els)

    if cls == "LexicographicPreference":
        prefs = [parse_preference(p) for p in (t.get("preferences") or [])]
        return LexicographicPreference(prefs)

    if cls == "CompensatoryPreference":
        primary   = parse_preference(t["primary_ap"])
        margin    = parse_preference(t["margin_ap"])
        secondary = parse_preference(t["secondary_ap"])
        if not isinstance(primary, AtomicPreference) or \
           not isinstance(margin,  AtomicPreference) or \
           not isinstance(secondary, AtomicPreference):
            raise TypeError("CompensatoryPreference requires atomic "
                             "primary_ap / margin_ap / secondary_ap")
        return CompensatoryPreference(primary, margin, secondary)

    if cls == "NumericPreference":
        threshold = t.get("threshold")
        et = t["entity_type"]
        return NumericPreference(
            entity_type = et,
            attribute   = _remap_attr(et, t["attribute"]),
            direction   = t.get("direction", "min"),
            threshold   = tuple(threshold) if threshold is not None else None,
            aggregation = t.get("aggregation", "avg"),
        )

    if cls == "ScopedPreference":
        inner = parse_preference(t["inner"])
        filters_raw = t.get("scope_filters") or []
        filters: list[AtomicPreference] = []
        for f in filters_raw:
            pf = parse_preference(f)
            if not isinstance(pf, AtomicPreference):
                raise TypeError("ScopedPreference scope_filters must be "
                                 "AtomicPreference instances")
            filters.append(pf)
        return ScopedPreference(inner, scope_filters=filters)

    if cls == "TemporalPreference":
        subj = parse_preference(t["subject_ap"])
        ref  = parse_preference(t["reference_ap"]) if t.get("reference_ap") else None
        if not isinstance(subj, AtomicPreference):
            raise TypeError("TemporalPreference.subject_ap must be atomic")
        if ref is not None and not isinstance(ref, AtomicPreference):
            raise TypeError("TemporalPreference.reference_ap must be atomic")
        return TemporalPreference(
            subject_ap   = subj,
            reference_ap = ref,
            op           = t.get("op", "always"),
            time_start   = t.get("time_start"),
            time_end     = t.get("time_end"),
            scope        = t.get("scope", "global"),
            strict       = bool(t.get("strict", False)),
        )

    raise ValueError(f"Unknown preference class: {cls!r}")


def parse_preferences(entries: list[dict] | str) -> list[Preference]:
    """Parse a full ``preferences_json`` list (JSON-string or list) into
    a list of ``preferences.py`` instances -- one per resolved preference
    on the record.  Preserves order."""
    if isinstance(entries, str):
        entries = json.loads(entries)
    return [parse_preference(e) for e in (entries or [])]


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #

def _sub_paradigm_of(entry: dict) -> str | None:
    """Extract the paradigm-specific sub-tag used for tabulation:
       TemporalPreference   -> op (always / sometime / within / ...
                                    hold_during / hold_after / ...)
       CompositePreference  -> op (AND / OR / NOT)
       NumericPreference    -> direction (min / max)
       otherwise            -> None
    """
    paradigm = entry.get("paradigm") or entry.get("class") or ""
    t = entry.get("template") or {}
    if paradigm == "TemporalPreference":
        return t.get("op") or entry.get("subtype")
    if paradigm == "CompositePreference":
        return t.get("op") or t.get("operator")
    if paradigm == "NumericPreference":
        return t.get("direction")
    return None


def _sub_check_result_to_row(result: CheckResult) -> dict[str, Any]:
    """Recursively serialise a ``CheckResult`` (and its ``sub_results``
    tree) into a JSON-safe dict.  Used to record intermediate results
    for nested-preference paradigms (Composite / Conditional /
    Lexicographic / Scoped / Temporal / Compensatory) so downstream
    drift-impact analysis can attribute a sub-preference's pass / score
    back to a specific drift source via its ``path``."""
    ct = result.constraint_type
    out: dict[str, Any] = {
        "name":            result.name,
        "passed":          bool(result.passed),
        "score":           float(result.score),
        "constraint_type": (ct.value if isinstance(ct, ConstraintType) else str(ct)),
        "trivial":         bool(result.trivial),
        "details":         result.details,
    }
    if result.sub_results:
        out["sub_results"] = [_sub_check_result_to_row(s)
                               for s in result.sub_results]
    return out


def _result_to_row(entry: dict, result: CheckResult) -> dict[str, Any]:
    """Serialise a ``CheckResult`` alongside its paradigm / bank_id /
    sub_paradigm provenance into a JSON-safe dict.

    ``sub_results`` are serialised recursively so downstream analysis
    (``paradigm_details.py`` / ``analyze_performance.py``) can walk a
    drift source's ``path`` into the exact sub-preference outcome.  On
    paradigms without sub-results (top-level Atomic, standalone Numeric)
    the field is omitted so old / analytic code that keys off its
    presence works uniformly."""
    ct = result.constraint_type
    row: dict[str, Any] = {
        "name":            result.name,
        "paradigm":        entry.get("paradigm") or entry.get("class") or "?",
        "sub_paradigm":    _sub_paradigm_of(entry),
        "bank_id":         entry.get("bank_id"),
        "passed":          bool(result.passed),
        "score":           float(result.score),
        "constraint_type": (ct.value if isinstance(ct, ConstraintType) else str(ct)),
        "trivial":         bool(result.trivial),
        "details":         result.details,
    }
    if result.sub_results:
        row["sub_results"] = [_sub_check_result_to_row(s)
                               for s in result.sub_results]
    return row


# --------------------------------------------------------------------------- #
# Tabulated summary                                                           #
# --------------------------------------------------------------------------- #

def _hf_pairing(hf_row: dict) -> tuple[str, str | None]:
    """Return ``(pairing_type, pairing_subtype)`` from an HF row.  The
    ``preference_pair`` column is JSON-stringified on the Hub;
    tolerate both string and dict forms.  Pass-through records (no
    resolved preferences) come out as ``("pass_through", None)``."""
    pp = hf_row.get("preference_pair")
    if isinstance(pp, str):
        try:
            pp = json.loads(pp)
        except (TypeError, ValueError):
            pp = {}
    if not isinstance(pp, dict):
        pp = {}
    return (pp.get("type") or "pass_through", pp.get("subtype"))


def _aggregate_record(evals: list[dict]) -> dict[str, Any]:
    """Reduce a record's list of per-preference evaluations to a single
    record-level summary::

        n_preferences : int
        n_passed      : int   -- count of evals whose `passed == True`
        pass_rate     : float -- n_passed / n_preferences  (in [0, 1])
        mean_score    : float -- mean of the per-pref `score` values
        all_passed    : bool  -- every preference on the record passed
        any_passed    : bool  -- at least one preference passed

    Empty ``evals`` (pass-through record with no resolved preferences)
    yields zero counts and ``all_passed=False``."""
    n = len(evals)
    if n == 0:
        return {
            "n_preferences": 0,   "n_passed":   0,
            "pass_rate":     0.0, "mean_score": 0.0,
            "all_passed":    False, "any_passed": False,
        }
    n_passed = sum(1 for e in evals if e.get("passed"))
    mean_score = sum(float(e.get("score", 0.0)) for e in evals) / n
    return {
        "n_preferences": n,
        "n_passed":      n_passed,
        "pass_rate":     n_passed / n,
        "mean_score":    mean_score,
        "all_passed":    (n_passed == n),
        "any_passed":    (n_passed > 0),
    }


class _EvalSummary:
    """Accumulate per-preference AND per-record results, then render a
    tabulated report with both **micro** (each preference is one data
    point) and **macro** (each record is one data point) perspectives.

    Micro pass-rate = ``sum(passed) / total_preferences``.
    Macro pass-rate = ``mean(per_record_pass_rate)``.  Macro also
    reports ``all_passed`` -- the fraction of records where every
    preference passed.

    Aggregations::

      Overall             -- micro + macro summary
      By paradigm         -- micro only (records span multiple paradigms)
      By sub-paradigm     -- micro only
      By level            -- micro + macro (each record has one level)
      By pairing_type     -- micro + macro
      By pairing_subtype  -- micro + macro
      By triviality       -- micro only (triviality is a per-pref flag)
    """

    def __init__(self):
        self._pref_records: list[dict] = []     # per-preference
        self._row_records:  list[dict] = []     # per-record

    def add_record(self, hf_row: dict, evals: list[dict]) -> None:
        """Feed a single data record: adds each preference to the
        per-pref pool AND stores a per-record aggregate."""
        pt, ps = _hf_pairing(hf_row)
        drift  = hf_row.get("profile_drift")
        days   = hf_row.get("days")
        # per-preference rows
        for ev in evals:
            self._pref_records.append({
                "paradigm":        ev.get("paradigm"),
                "sub_paradigm":    ev.get("sub_paradigm"),
                "level":           hf_row.get("level"),
                "pairing_type":    pt,
                "pairing_subtype": ps,
                "profile_drift":   drift,
                "days":            days,
                "passed":          bool(ev.get("passed", False)),
                "score":           float(ev.get("score", 0.0)),
                "trivial":         bool(ev.get("trivial", False)),
            })
        # per-record aggregate (skipping pass-through with no prefs, to
        # avoid dragging the macro means toward 0)
        if evals:
            agg = _aggregate_record(evals)
            self._row_records.append({
                "id":               hf_row.get("id"),
                "level":            hf_row.get("level"),
                "pairing_type":     pt,
                "pairing_subtype":  ps,
                "profile_drift":    drift,
                "days":             days,
                **agg,
            })

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _fmt_table(title: str, header: tuple[str, ...],
                   rows: list[tuple]) -> str:
        """Format a plain-text table with left-aligned first col + right-
        aligned others.  Widths auto-fit the widest cell in each column."""
        widths = [len(h) for h in header]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        def _fmt(row):
            out = [str(row[0]).ljust(widths[0])]
            for i in range(1, len(row)):
                out.append(str(row[i]).rjust(widths[i]))
            return "  ".join(out)
        line = "─" * (sum(widths) + 2 * (len(header) - 1))
        lines = [f"=== {title} ===", _fmt(header), line]
        lines.extend(_fmt(r) for r in rows)
        return "\n".join(lines) + "\n"

    # -- micro grouping (each preference = one data point) ------------------
    def _group_micro(self, key_fn) -> list[tuple]:
        buckets: dict[Any, list[dict]] = {}
        for r in self._pref_records:
            buckets.setdefault(key_fn(r), []).append(r)
        rows = []
        for k, items in buckets.items():
            n = len(items)
            p = sum(1 for r in items if r["passed"])
            s = sum(r["score"] for r in items) / n if n else 0.0
            pct = f"{(100 * p / n) if n else 0:.1f}%"
            rows.append((str(k), n, p, pct, f"{s:.3f}"))
        rows.sort(key=lambda x: (-x[1], x[0]))
        return rows

    # -- macro grouping (TravelPlanner convention: each record = one data
    #    point, "macro pass" = the record satisfied ALL its preferences)
    def _group_macro(self, key_fn) -> list[tuple]:
        buckets: dict[Any, list[dict]] = {}
        for r in self._row_records:
            buckets.setdefault(key_fn(r), []).append(r)
        rows = []
        for k, items in buckets.items():
            n_rec = len(items)
            if n_rec == 0:
                continue
            macro_pass = sum(1 for r in items if r["all_passed"])
            mean_score = sum(r["mean_score"] for r in items) / n_rec
            pct        = f"{(100 * macro_pass / n_rec):.1f}%"
            rows.append((str(k), n_rec, macro_pass, pct, f"{mean_score:.3f}"))
        rows.sort(key=lambda x: (-x[1], x[0]))
        return rows

    # ---- rendering --------------------------------------------------------
    def render(self) -> str:
        parts: list[str] = []
        prefs = self._pref_records
        rows  = self._row_records

        # 1. Overall (micro + macro) --------------------------------------
        # micro: TravelPlanner-style pooled fraction across all preferences.
        n_pref  = len(prefs)
        p_pref  = sum(1 for r in prefs if r["passed"])
        s_pref  = sum(r["score"] for r in prefs) / n_pref if n_pref else 0.0
        trivial_passes = sum(1 for r in prefs if r["trivial"] and r["passed"])
        nontrivial     = sum(1 for r in prefs if not r["trivial"])
        # macro: TravelPlanner-style all-preferences-passed fraction across
        # records.  A record contributes 1 iff every preference on it
        # passed, else 0.
        n_rec        = len(rows)
        macro_pass   = sum(1 for r in rows if r["all_passed"])
        macro_score  = (sum(r["mean_score"] for r in rows) / n_rec) if n_rec else 0.0
        any_pass_rec = sum(1 for r in rows if r["any_passed"])
        avg_prefs    = (n_pref / n_rec) if n_rec else 0.0

        parts.append(
            "=== Overall ===\n"
            f"records evaluated  : {n_rec}\n"
            f"preferences total  : {n_pref}   "
            f"(≈{avg_prefs:.2f} per record; easy=1, medium/hard=2)\n"
            f"\n"
            f"[micro] each preference is one data point   (TravelPlanner)\n"
            f"  preferences passed : {p_pref}/{n_pref}   "
            f"({(100 * p_pref / n_pref if n_pref else 0):.1f}%)\n"
            f"  mean pref score    : {s_pref:.3f}\n"
            f"  trivial passes     : {trivial_passes}\n"
            f"  non-trivial prefs  : {nontrivial}\n"
            f"\n"
            f"[macro] each record is one data point       (TravelPlanner)\n"
            f"  records all-passed : {macro_pass}/{n_rec}   "
            f"({(100 * macro_pass / n_rec if n_rec else 0):.1f}%)  "
            f"[macro pass rate]\n"
            f"  macro mean score   : {macro_score:.3f}   "
            f"(mean of per-record mean scores)\n"
            f"  records any-passed : {any_pass_rec}/{n_rec}   "
            f"({(100 * any_pass_rec / n_rec if n_rec else 0):.1f}%)  "
            f"[lower bound]\n"
        )

        # Table headers -----------------------------------------------------
        header_micro = ("group", "count", "passed", "pass%", "mean_score")
        header_macro = ("group", "records", "macro_pass", "macro_pass%",
                         "macro_score")

        # 2. By paradigm  (micro only -- records span multiple paradigms) --
        parts.append(self._fmt_table(
            "By paradigm  [micro; count = # of preferences]", header_micro,
            self._group_micro(lambda r: r["paradigm"] or "-")))

        # 3. By sub-paradigm  (micro only) ---------------------------------
        parts.append(self._fmt_table(
            "By sub-paradigm  [micro]", header_micro,
            self._group_micro(
                lambda r: (f"{r['paradigm'] or '-'}.{r['sub_paradigm']}"
                            if r["sub_paradigm"]
                            else f"{r['paradigm'] or '-'}"))))

        # 4. By level  (micro + macro) -------------------------------------
        parts.append(self._fmt_table(
            "By level  [micro; count = # of preferences]", header_micro,
            self._group_micro(lambda r: r["level"] or "-")))
        parts.append(self._fmt_table(
            "By level  [macro; records = # of records]", header_macro,
            self._group_macro(lambda r: r["level"] or "-")))

        # 5. By pairing_type  (micro + macro) ------------------------------
        parts.append(self._fmt_table(
            "By pairing_type  [micro]", header_micro,
            self._group_micro(lambda r: r["pairing_type"] or "-")))
        parts.append(self._fmt_table(
            "By pairing_type  [macro]", header_macro,
            self._group_macro(lambda r: r["pairing_type"] or "-")))

        # 6. By pairing_subtype  (micro + macro) ---------------------------
        parts.append(self._fmt_table(
            "By pairing_subtype  [micro]", header_micro,
            self._group_micro(lambda r: r["pairing_subtype"] or "--")))
        parts.append(self._fmt_table(
            "By pairing_subtype  [macro]", header_macro,
            self._group_macro(lambda r: r["pairing_subtype"] or "--")))

        # 7. By profile_drift  (micro + macro) -----------------------------
        # Drift is per-record (aligned / omission / inversion), so both
        # perspectives make sense.
        parts.append(self._fmt_table(
            "By profile_drift  [micro]", header_micro,
            self._group_micro(lambda r: r["profile_drift"] or "-")))
        parts.append(self._fmt_table(
            "By profile_drift  [macro]", header_macro,
            self._group_macro(lambda r: r["profile_drift"] or "-")))

        # 8. By days  (trip length in days: 3 / 5 / 7)  (micro + macro) ---
        # A record has ONE `days` value, so both aggregations are valid.
        # Sort labels naturally as strings ("3" / "5" / "7").
        parts.append(self._fmt_table(
            "By days  [micro]", header_micro,
            self._group_micro(
                lambda r: f"{r['days']}-day" if r["days"] is not None else "-")))
        parts.append(self._fmt_table(
            "By days  [macro]", header_macro,
            self._group_macro(
                lambda r: f"{r['days']}-day" if r["days"] is not None else "-")))

        # 9. By triviality  (micro only -- flag is per-preference) ---------
        parts.append(self._fmt_table(
            "By triviality  [micro]", header_micro,
            self._group_micro(
                lambda r: "trivial" if r["trivial"] else "non-trivial")))

        return "\n".join(parts)


def evaluate_row(plan_days_raw: list[dict] | None,
                  preferences_json: list[dict] | str,
                  *,
                  db: _DB,
                  date_seq: list[str] | None = None,
                  people_number: int = 1,
                  on_missing: str = "skip") -> tuple[list[dict], list[dict]]:
    """One-shot: hydrate a structured plan, parse the preferences, run
    every preference against the hydrated plan.  Returns
    ``(plan_days, evaluations)`` where each evaluation dict follows
    ``_result_to_row``.  Malformed preference entries produce a stub
    evaluation with ``passed=False`` and a ``details`` string explaining
    the parse failure.

    Parameters
    ----------
    plan_days_raw : list[dict] | None
        The structured plan list (as emitted by ``convert_plans.py``),
        one dict per day carrying string-valued header fields.  ``None``
        or empty -> ``(plan_days=[], evaluations=[])``.  Entity
        attributes come from ``db`` (the TravelPlanner CSVs).
    date_seq : list[str] | None
        The HF row's ``date`` field (list of ``YYYY-MM-DD``).
    people_number : int
        The HF row's ``people_number`` (used for per-person
        accommodation cost).
    """
    plan_days = transform_plan(plan_days_raw, db=db, date_seq=date_seq,
                                 people_number=people_number,
                                 on_missing=on_missing)

    if isinstance(preferences_json, str):
        preferences_json = json.loads(preferences_json)
    entries = list(preferences_json or [])

    evaluations: list[dict] = []
    for entry in entries:
        paradigm     = entry.get("paradigm") or entry.get("class") or "?"
        bank_id      = entry.get("bank_id")
        sub_paradigm = _sub_paradigm_of(entry)
        entry        = _casefold_literals(entry, trace=entry.get("trace"))
        try:
            pref = parse_preference(entry)
        except Exception as e:
            evaluations.append({
                "name":            entry.get("trace", paradigm),
                "paradigm":        paradigm,
                "sub_paradigm":    sub_paradigm,
                "bank_id":         bank_id,
                "passed":          False,
                "score":           0.0,
                "constraint_type": "error",
                "trivial":         False,
                "details":         {"kind": "parse_error",
                                     "message": f"parse-error: {type(e).__name__}: {e}",
                                     "error_type": type(e).__name__},
            })
            continue
        try:
            res = pref.evaluate(plan_days)
        except Exception as e:
            evaluations.append({
                "name":            pref.name,
                "paradigm":        paradigm,
                "sub_paradigm":    sub_paradigm,
                "bank_id":         bank_id,
                "passed":          False,
                "score":           0.0,
                "constraint_type": "error",
                "trivial":         False,
                "details":         {"kind": "eval_error",
                                     "message": f"eval-error: {type(e).__name__}: {e}",
                                     "error_type": type(e).__name__},
            })
            continue
        evaluations.append(_result_to_row(entry, res))

    return plan_days, evaluations


# --------------------------------------------------------------------------- #
# Demo (one example per paradigm)                                             #
# --------------------------------------------------------------------------- #

_PARADIGMS = (
    "AtomicPreference",
    "CompositePreference",
    "ConditionalPreference",
    "LexicographicPreference",
    "CompensatoryPreference",
    "NumericPreference",
    "ScopedPreference",
    "TemporalPreference",
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _load_hf_split(name_or_path: str, split: str) -> list[dict]:
    """Load a HuggingFace split -- either from the online Hub or from a
    local file/directory.  Returns a plain ``list[dict]`` so the rest of
    the pipeline can index/iterate uniformly.

    Detection:
      * A local file  -> ``load_dataset("json", data_files=path, split="train")``
        (JSONL files carry no split concept; the ``--split`` CLI flag is
        used only to label the run.)
      * A local dir   -> ``load_dataset(path, split=split)`` (reads its
        ``README.md`` YAML config).
      * Anything else -> ``load_dataset(name_or_path, split=split)`` -- Hub id.
    """
    try:
        from datasets import load_dataset          # local import: optional dep
    except ImportError as e:
        raise SystemExit(
            "the `datasets` package is required (`pip install datasets`); "
            f"original error: {e}")

    p = Path(name_or_path)
    if p.is_file():
        ds = load_dataset("json", data_files=str(p), split="train")
    elif p.is_dir():
        ds = load_dataset(str(p), split=split)
    else:
        ds = load_dataset(name_or_path, split=split)
    return [dict(row) for row in ds]


def _select_demo_records(hf_rows: list[dict]) -> dict[str, dict]:
    """From easy/single-preference rows in the HF split, pick one row
    per paradigm.  Returns ``{paradigm: hf_row}``."""
    out: dict[str, dict] = {}
    for row in hf_rows:
        if row.get("level") != "easy":
            continue
        prefs_raw = row.get("preferences_json")
        prefs = json.loads(prefs_raw) if isinstance(prefs_raw, str) else prefs_raw
        if not isinstance(prefs, list) or len(prefs) != 1:
            continue
        pk = prefs[0].get("paradigm")
        if pk in _PARADIGMS and pk not in out:
            out[pk] = row
        if len(out) == len(_PARADIGMS):
            break
    return out


def _index_plans_by_id(rows: list[dict], id_key: str) -> dict[int, dict]:
    """Index plan rows by their 1-based ``id`` (or whatever ``--id-key``
    the CLI was invoked with).  Silently skips rows without the key --
    the batch loop reports those as unpaired."""
    ix: dict[int, dict] = {}
    for r in rows:
        rid = r.get(id_key)
        if rid is None:
            continue
        try:
            ix[int(rid)] = r
        except (TypeError, ValueError):
            continue
    return ix


def _verify_id_alignment(plan_ix: dict[int, dict],
                          hf_rows: list[dict],
                          id_key: str) -> tuple[int, int]:
    """Confirm the invariant ``plan[id_key] == hf["id"]`` on every HF row
    that has a matching plan (both are 1-indexed under the current
    schema).  Returns ``(matched, mismatched)`` and prints a bounded
    number of individual mismatches to stderr."""
    matched = mismatched = 0
    reported = 0
    for hf_row in hf_rows:
        hf_id = hf_row.get("id")
        if hf_id is None:
            continue
        expected_id = int(hf_id)
        plan_row = plan_ix.get(expected_id)
        if plan_row is None:
            continue
        if int(plan_row.get(id_key, -1)) != expected_id:
            mismatched += 1
            if reported < 5:
                print(f"[warn] alignment: hf id={hf_id} expects "
                      f"plan `{id_key}`={expected_id} but got "
                      f"{plan_row.get(id_key)!r}", file=sys.stderr)
                reported += 1
        else:
            matched += 1
    return matched, mismatched


def _hf_date_seq(hf_row: dict) -> list[str]:
    """Return the HF row's ordered date list.  The HF schema stores
    ``date`` as ``sequence<string>``; some producers pass through a
    JSON-string form so we handle both."""
    d = hf_row.get("date")
    if isinstance(d, str):
        try:
            v = json.loads(d)
            return [str(x) for x in (v or [])]
        except json.JSONDecodeError:
            return []
    return [str(x) for x in (d or [])]


def _hf_people_number(hf_row: dict) -> int:
    try:
        return max(int(hf_row.get("people_number") or 1), 1)
    except (TypeError, ValueError):
        return 1


def _run_demo(dataset: str, split: str, plans_path: Path,
                db: _DB,
                plan_key: str, id_key: str, on_missing: str) -> None:
    print(f"[demo] loading dataset={dataset!r} split={split!r}",
          file=sys.stderr)
    hf_rows   = _load_hf_split(dataset, split)
    plan_rows = _load_jsonl(plans_path)
    plan_ix   = _index_plans_by_id(plan_rows, id_key)
    demo      = _select_demo_records(hf_rows)

    matched, mismatched = _verify_id_alignment(plan_ix, hf_rows, id_key)
    print(f"[demo] {len(demo)}/{len(_PARADIGMS)} paradigms found in "
          f"easy/single of split={split!r}", file=sys.stderr)
    print(f"[demo] plan file: {plans_path}  ({len(plan_rows)} rows, "
          f"{len(plan_ix)} indexable by '{id_key}')  "
          f"| aligned={matched}  mismatched={mismatched}",
          file=sys.stderr)

    for paradigm in _PARADIGMS:
        row = demo.get(paradigm)
        if row is None:
            print(f"\n=== {paradigm}: no easy/single record available ===")
            continue

        hf_id = int(row["id"])
        plan_row = plan_ix.get(hf_id)
        plan_days_raw = plan_row.get(plan_key) if plan_row else None
        if not plan_days_raw:
            print(f"\n=== {paradigm} (hf.id={hf_id}) ===")
            print(f"  no plan row with {id_key}={hf_id} "
                  f"in {plans_path.name} under key '{plan_key}'")
            continue

        _plan_hydrated, evals = evaluate_row(
            plan_days_raw, row["preferences_json"],
            db            = db,
            date_seq      = _hf_date_seq(row),
            people_number = _hf_people_number(row),
            on_missing    = on_missing,
        )
        ev = evals[0] if evals else {}
        print(f"\n=== {paradigm} (hf.id={hf_id}, plan.{id_key}={hf_id}, "
              f"bank_id={ev.get('bank_id')}) ===")
        print(f"  name  : {ev.get('name')}")
        print(f"  passed: {ev.get('passed')}   score: {ev.get('score')}")
        det = ev.get('details') or {}
        det_msg = det.get('message') if isinstance(det, dict) else det
        print(f"  detail kind: {det.get('kind') if isinstance(det, dict) else '-'}")
        print(f"  detail msg : {det_msg}")


# --------------------------------------------------------------------------- #
# Batch CLI                                                                   #
# --------------------------------------------------------------------------- #

def evaluate_records(plan_records: list[dict],
                      hf_rows_by_id: dict[int, dict],
                      *,
                      db: _DB,
                      plan_key: str = "plan",
                      prefs_key: str = "preferences_json",
                      id_key: str = "id",
                      on_missing: str = "skip",
                      progress_label: str | None = None
                      ) -> dict[str, Any]:
    """Bulk preference evaluator -- programmatic entry point for eval.py.

    Runs one evaluation per plan record and returns::

        {
          "per_id":    {id: {"evaluations": [...],  # per-preference
                             "record_summary": {...},  # aggregate
                             "delivered":       bool}},
          "summary":   _EvalSummary,   # accumulator used to render
                                        # micro/macro tables
          "counts":    {"n_records": int,
                        "n_evaluated": int,
                        "n_null":     int,
                        "n_no_prefs": int},
        }

    ``plan_records`` is the structured plan JSONL (each row carries an
    ``id`` and a ``plan`` list, as emitted by
    ``evaluation/convert_plans.py``).  ``hf_rows_by_id`` maps each
    dataset ``id`` (1-indexed) to its full HF row -- the caller (eval.py)
    already loaded the queries, so we re-use the same view.

    Records with no ``plan`` (undelivered), no HF match, or no
    ``preferences_json`` on the HF row are recorded in ``per_id`` but
    with ``evaluations=None`` / ``record_summary=None`` and are NOT fed
    to the summary accumulator (so micro/macro tables aren't polluted
    by rows that never had a plan or a preference to evaluate).
    """
    summary = _EvalSummary()
    per_id: dict[int, dict] = {}
    n_evaluated = n_null = n_no_prefs = 0

    reporter = _ProgressReporter(len(plan_records),
                                   label=progress_label or "eval-prefs")
    try:
        for r in plan_records:
            plan_id_raw = r.get(id_key)
            plan_days_raw = r.get(plan_key)
            prefs_raw     = r.get(prefs_key)

            hf_row = None
            key: int | None = None
            if plan_id_raw is not None:
                try:
                    key = int(plan_id_raw)
                    hf_row = hf_rows_by_id.get(key)
                except (TypeError, ValueError):
                    hf_row = None

            if hf_row is not None and prefs_raw is None:
                prefs_raw = hf_row.get(prefs_key)

            delivered = bool(plan_days_raw)

            if not delivered or hf_row is None:
                if key is not None:
                    per_id[key] = {"evaluations": None,
                                    "record_summary": None,
                                    "delivered": delivered}
                n_null += 1
                reporter.step()
                continue

            # Pass-through record (no preferences resolved on the HF row):
            # log the delivery status but skip the eval loop.
            if not prefs_raw:
                per_id[key] = {"evaluations": [],
                                "record_summary": _aggregate_record([]),
                                "delivered": True}
                n_no_prefs += 1
                reporter.step()
                continue

            _plan_hydrated, evals = evaluate_row(
                plan_days_raw, prefs_raw,
                db            = db,
                date_seq      = _hf_date_seq(hf_row),
                people_number = _hf_people_number(hf_row),
                on_missing    = on_missing,
            )
            per_id[key] = {"evaluations":    evals,
                            "record_summary": _aggregate_record(evals),
                            "delivered":      True}
            summary.add_record(hf_row, evals)
            n_evaluated += 1
            reporter.step()
    finally:
        reporter.close()

    return {
        "per_id":  per_id,
        "summary": summary,
        "counts":  {"n_records":    len(plan_records),
                     "n_evaluated":  n_evaluated,
                     "n_null":       n_null,
                     "n_no_prefs":   n_no_prefs},
    }


def _run_batch(plan_path: Path, out_path: Path,
                dataset: str, split: str,
                db: _DB,
                plan_key: str, prefs_key: str,
                id_key: str, on_missing: str,
                summary_out: Path | None = None) -> None:
    plan_rows = _load_jsonl(plan_path)

    print(f"[evaluate] loading dataset={dataset!r} split={split!r}",
          file=sys.stderr)
    hf_rows = _load_hf_split(dataset, split)

    # Index HF rows by id (1-indexed under the current schema).
    hf_ix: dict[int, dict] = {}
    for hf_row in hf_rows:
        hid = hf_row.get("id")
        if hid is None:
            continue
        try:
            hf_ix[int(hid)] = hf_row
        except (TypeError, ValueError):
            continue

    aligned, mismatched = _verify_id_alignment(
        _index_plans_by_id(plan_rows, id_key), hf_rows, id_key)
    print(f"[evaluate] hf split loaded: {len(hf_ix)} rows keyed by id  "
          f"| aligned with plan: {aligned}  "
          f"mismatched: {mismatched}", file=sys.stderr)

    result = evaluate_records(plan_rows, hf_ix,
                              db=db,
                              plan_key=plan_key, prefs_key=prefs_key,
                              id_key=id_key, on_missing=on_missing,
                              progress_label="evaluating")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_null = 0
    with out_path.open("w") as fout:
        for r in plan_rows:
            key_raw = r.get(id_key)
            try:
                key = int(key_raw) if key_raw is not None else None
            except (TypeError, ValueError):
                key = None
            entry = result["per_id"].get(key) if key is not None else None
            if entry is None or entry.get("evaluations") is None:
                out_row = {id_key: key_raw, "plan": None,
                            "record_summary": None, "evaluations": None}
                n_null += 1
            else:
                out_row = {
                    id_key:           key_raw,
                    "plan":           r.get(plan_key),
                    "record_summary": entry["record_summary"],
                    "evaluations":    entry["evaluations"],
                }
                n_ok += 1
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"[evaluate] wrote {n_ok + n_null} rows to {out_path} "
          f"({n_ok} evaluated, {n_null} plan=null)", file=sys.stderr)

    # ---- tabulated summary ----------------------------------------------
    report = result["summary"].render()
    if summary_out is None:
        summary_out = out_path.with_suffix(".summary.txt")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(report)
    print(f"[evaluate] wrote summary to {summary_out}", file=sys.stderr)
    print("\n" + report, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Argparse driver                                                              #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", dest="plan_path", type=Path, required=True,
                    help="Input STRUCTURED plan JSONL (one row per record) as "
                         "emitted by evaluation/convert_plans.py.  Rows must "
                         "carry an `id` (1-indexed, matching the dataset id) "
                         "and a `plan` list of per-day dicts.  "
                         "`preferences_json` is read from the HF dataset row "
                         "on lookup.")
    ap.add_argument("--out", dest="out_path", type=Path,
                    help="Output JSONL (batch mode).  Each input row is "
                         "written with `plan` (echoed) and `evaluations`.")
    ap.add_argument("--summary-out", dest="summary_out", type=Path,
                    default=None,
                    help="Tabulated summary path (batch mode).  Aggregates "
                         "pass counts and mean scores overall + by "
                         "paradigm / sub-paradigm / level / pairing_type / "
                         "pairing_subtype / triviality.  Defaults to "
                         "`<out>.summary.txt` next to --out.")
    ap.add_argument("--dataset", default="UKPLab/PreferTripPlan",
                    help="HF dataset -- either a Hub repo id "
                         "(loaded via `datasets.load_dataset`) or a local "
                         "path (file or directory).  "
                         "Default: `UKPLab/PreferTripPlan`.")
    ap.add_argument("--split", default="test",
                    help="Split to load from --dataset "
                         "(default: `test`; use `test_large` for the "
                         "full 1000-row pool).")
    ap.add_argument("--plan-key",  default="plan",
                    help="Plan-row key holding the structured plan list "
                         "(default: `plan` -- convert_plans.py convention).")
    ap.add_argument("--prefs-key", default="preferences_json")
    ap.add_argument("--id-key",    default="id",
                    help="Plan-row key holding the 1-indexed dataset id "
                         "(default: `id` -- convert_plans.py / HF-data "
                         "convention).")
    ap.add_argument("--on-missing", choices=("warn", "skip", "raise"),
                    default="skip",
                    help="Behaviour when a plan entity isn't in the pool.")
    ap.add_argument("--demo", action="store_true",
                    help="Instead of writing an output file, print one "
                         "paradigm example from the dataset split -- "
                         "pick the first easy/single row per paradigm, "
                         "evaluate its preference against the "
                         "corresponding plan in --plan.")
    args = ap.parse_args()

    print(f"[evaluate] loading TravelPlanner DBs via ported tool APIs",
          file=sys.stderr)
    db = _DB()

    if args.demo:
        _run_demo(args.dataset, args.split, args.plan_path, db,
                   args.plan_key, args.id_key, args.on_missing)
        return

    if args.out_path is None:
        ap.error("--out is required in batch mode (or use --demo)")
    _run_batch(args.plan_path, args.out_path, args.dataset, args.split,
                db, args.plan_key, args.prefs_key,
                args.id_key, args.on_missing,
                summary_out=args.summary_out)


if __name__ == "__main__":
    main()
