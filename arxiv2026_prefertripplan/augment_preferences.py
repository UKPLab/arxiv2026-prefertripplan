#!/usr/bin/env python3
"""Preference-augment a TravelPlanner test JSONL using a human-curated bank.

8 paradigms (Atomic, Composite, Numeric, Conditional, Lexicographic,
Compensatory, Temporal, Scoped) are balanced equally across queries.
Temporal is treated as ONE paradigm in the top-level balance; its 9
modalities (always, sometime, within, atmost_once, sometime_before,
sometime_after, always_within, hold_during, hold_after) are balanced
only inside Temporal.

  easy   : 1 preference (any of the 8 paradigms).
  medium : 1 atomic + 1 complex (= any of the 6 non-atomic/non-composite).
  hard   : 1 composite + 1 complex.

For medium/hard, the pair is one of:

  independent        : no shared (entity, attribute).
  overlapping_non_competing : shared, ops aligned (no opposing semantics).
  overlapping_competing     : shared with opposing ops/dirs, joint must
                              be non-vacuous (both preferences exercised).

Vacuous overlaps (e.g. composite [all] rating>=4 paired with conditional
IF rating<=3 — the [all] forces the condition to never fire) are
rejected: the existential side of the second preference must have at
least one satisfying item under the joint pool reduced by all [all]-
scope constraints from both preferences.

Numeric placeholder values are resolved aggressively (per the appendix):
  - op >=: candidate values = [provided, stat, budget_heuristic]; pick
           the LARGEST that yields at least 10 satisfying items, falling
           back to the next-largest if needed.
  - op <=: pick the SMALLEST similarly.
  - rating         rounded to nearest 0.5
  - restaurant     cost rounded to nearest 5
  - accom / transp cost rounded to nearest 10

Queries whose destination cities are absent from attractions.csv are
emitted without preferences (about 16 such 3-day queries).

Output: prefertripplan.jsonl with extra fields:
  preferences        : list of resolved preference structures
  preference_traces  : list of "<paradigm>[.<subtype>]:<id>"
  pairing_type       : "single" | "independent" | "overlapping"
  pairing_subtype    : null | "competing" | "non_competing"
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Sequence


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

PARADIGMS = (
    "AtomicPreference",
    "CompositePreference",
    "NumericPreference",
    "ConditionalPreference",
    "LexicographicPreference",
    "CompensatoryPreference",
    "TemporalPreference",
    "ScopedPreference",
)
COMPLEX_PARADIGMS = (
    "NumericPreference",
    "ConditionalPreference",
    "LexicographicPreference",
    "CompensatoryPreference",
    "TemporalPreference",
    "ScopedPreference",
)
TEMPORAL_SUBTYPES = (
    "always", "sometime", "within", "atmost_once",
    "sometime_before", "sometime_after", "always_within",
    "hold_during", "hold_after",
)

STAT_KEY_MAP = {"Q0": 0.0, "Q1": 0.25, "Q2": 0.50, "Q3": 0.75, "Q4": 1.0}

# (entity, attribute) -> (kind, round-step).  kind in {"rating", "cost"}.
NUMERIC_ROUND = {
    ("Restaurant", "cost"): ("cost", 5),
    ("Restaurant", "rating"): ("rating", None),
    ("Accommodation", "cost"): ("cost", 10),
    ("Accommodation", "rating"): ("rating", None),
    ("Attraction", "rating"): ("rating", None),
    ("Transportation", "cost"): ("cost", 10),
}

# Cost-range fallbacks (used when stats give nothing).
COST_RANGE = {
    "Accommodation": (50.0, 1200.0),
    "Restaurant":    (10.0, 250.0),
    "Transportation": (5.0, 6800.0),
}

# Attribute aliases applied when reading from raw DB records.
ATTR_ALIAS = {
    ("Restaurant", "cuisine"): "cuisines",
    ("Attraction", "category"): "categories",
    ("Accommodation", "house_rules"): "house_rules_list",
    # Transportation time attributes: preferences use the semantic names
    # arrival_time / departure_time, flight records store the raw HH:MM
    # under arr_time / dep_time.
    ("Transportation", "arrival_time"):   "arr_time",
    ("Transportation", "departure_time"): "dep_time",
}

# ── Time-window categories for Transportation.arrival_time / departure_time ──
# Preferences reference times by category name; the flight record has an
# HH:MM string. Values are (start_min, end_min) in minutes-since-midnight;
# 'night' wraps past midnight and its end_min uses the +24h convention
# (28*60 == 04:00 of the next day).
TIME_WINDOWS: dict[str, tuple[int, int]] = {
    "morning":   ( 4 * 60, 12 * 60),   # [04:00, 12:00)
    "afternoon": (12 * 60, 17 * 60),   # [12:00, 17:00)
    "evening":   (17 * 60, 22 * 60),   # [17:00, 22:00)
    "night":     (22 * 60, 28 * 60),   # [22:00, 04:00 next day)
}


def _parse_hhmm(s: Any) -> int | None:
    """Parse an "HH:MM" string into minutes-since-midnight. None if invalid."""
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


def _time_in_category(hhmm: Any, category: Any) -> bool:
    """True iff the "HH:MM" time falls in the named TIME_WINDOWS category."""
    if not isinstance(category, str):
        return False
    win = TIME_WINDOWS.get(category)
    if win is None:
        return False
    minutes = _parse_hhmm(hhmm)
    if minutes is None:
        return False
    start, end = win
    if end <= 24 * 60:
        return start <= minutes < end
    # Wrap-around (night): [start, 24:00) ∪ [00:00, end - 24h).
    return minutes >= start or minutes < (end - 24 * 60)

# Slot-level attributes: not per-item fields, assigned when the planner
# places the entity into a specific plan slot. Any (entity, attribute)
# listed here must NOT be used to filter the entity pool -- every item
# is a valid candidate for every slot. Currently: Restaurant.meal_type
# (breakfast / lunch / dinner is a slot property, not a restaurant
# property).
SLOT_LEVEL_ATTRS: set[tuple[str, str]] = {
    ("Restaurant", "meal_type"),
}

HOUSE_RULE_TO_NEEDLE = {
    "pets": "no pets",
    "smoking": "no smoking",
    "parties": "no parties",
    "visitors": "no visitors",
    "children": "no children",
}

OPPOSING_OPS = (
    frozenset({">=", "<="}),
    frozenset({"==", "!="}),
    frozenset({"in", "not_in"}),
)

# Set-valued attributes whose `in` operator uses SUBSET semantics: every
# value listed in the predicate must be present in the candidate's value
# list. All other set-valued attributes (cuisine, category) use the
# intersection / any-match reading.  `not_in` uses disjoint semantics on
# both sides (candidate has NONE of the listed values).
SUBSET_IN_ATTRS = {("Accommodation", "house_rules")}

# Pair-aware tolerance for the adjustable side once a competing pair is
# being formed.  The values must stay within +/- TOL of the rigid value so
# the tension stays tight (otherwise either vacuous or resolvable too
# trivially).
TENSION_TOL = {
    ("Restaurant",     "rating"): 0.5,
    ("Attraction",     "rating"): 0.5,
    ("Accommodation",  "rating"): 1.0,
    ("Restaurant",     "cost"):  20,
    ("Accommodation",  "cost"):  50,
    ("Transportation", "cost"):  50,
}

# Per-query budget envelope multipliers (applied to the per-item budget
# heuristic).  A cost-attribute value V resolved from any preference must
# satisfy ENVELOPE_FLOOR_MULT * h <= V <= alpha * h, where alpha depends on
# the scope:
#   alpha_all = 1.0 -- every item must satisfy V; the cap has to leave
#                     headroom below the per-item budget.
#   alpha_any = 2.0 -- only at least one item must satisfy V; we tolerate
#                     up to twice the per-item budget for that splurge.
#   beta      = 0.5 -- below half the per-item budget the predicate is
#                     either trivially auto-satisfied (>=) or so tight no
#                     plan can hit it (<=).
ENVELOPE_FLOOR_MULT  = 0.5
ENVELOPE_CEIL_ANY    = 2.0
ENVELOPE_CEIL_ALL    = 1.5
# Per-query budget may be inflated up to this multiplier as a last-resort
# recovery before a preference is rejected and resampled.
BUDGET_ADJUST_CAP = 1.5

# Numerical attributes whose values are clipped by the envelope.
COST_ATTRS = {("Accommodation", "cost"),
              ("Restaurant",    "cost"),
              ("Transportation", "cost")}

# Roles whose [any]-style existence we additionally require NOT to be
# auto-satisfied by the joint [all]-scope pool.  Auto-satisfaction means
# every item in the reduced pool already meets the predicate -- the
# predicate then adds no real constraint and the pair degenerates.
AUTO_SATISFIED_CHECK_ROLES = frozenset({
    "main", "child", "primary", "margin", "secondary", "lex_0",
    "inner_main", "inner_child", "inner_primary", "inner_secondary",
})

MAX_PAIR_TRIALS = 400
MAX_SINGLE_TRIALS = 200

# Per-visiting-city pool floor for [all]-scope predicates.  The combined
# predicate filter (intersection of all [all]-scope predicates on the
# same entity plus the query constraints) must leave at least this many
# candidates in each city the planner could visit.  Strategy B (K-of-N):
# floor must clear in >= visiting_city_number cities.
PER_CITY_ALL_FLOOR = {
    "Attraction":     5,
    "Accommodation":  1,  # one valid lodging suffices; higher floor would
                          # spuriously drop queries where a room-type
                          # constraint narrows a city to 1 accommodation.
    "Restaurant":     6,
    "Transportation": 1,  # interpreted as "flight options per visiting city"
}
# [any]-scope predicates need a much smaller floor (one candidate per
# city suffices in principle; >1 gives the planner some choice).
PER_CITY_ANY_FLOOR = 1
# Compensatory paradigm uses a wider [any]-scope floor: one item per day
# (up to 3 days at a city even in multi-city trips) -> 3 affordable items.
# Exception: Accommodation is booked ONCE per city stay (not once per
# day), so its Compensatory floor stays at 1.
PER_CITY_ANY_FLOOR_COMPENSATORY: dict[str, int] = {
    "Accommodation": 1,
    "Restaurant":    3,
    "Attraction":    3,
    "Transportation": 1,
}

# Trip segments (arrival to next transit) never exceed 3 days in the
# 3/5/7-day design. An accommodation whose minimum_nights requirement
# exceeds 3 can therefore never satisfy the common-sense "min-nights
# compatibility" check inherited from original TravelPlanner. Filter
# such rows out of every downstream pool right at qdb-build time.
MAX_TRIP_SEGMENT_NIGHTS = 3


# --------------------------------------------------------------------------- #
# JSON loading (the bank uses trailing commas)                                #
# --------------------------------------------------------------------------- #

_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def load_jsonc(path: Path) -> Any:
    text = path.read_text()
    text = _TRAILING_COMMA.sub(r"\1", text)
    return json.loads(text)


# --------------------------------------------------------------------------- #
# Database loaders                                                            #
# --------------------------------------------------------------------------- #

def parse_house_rules(s: Any) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip() for x in s.split("&") if x.strip()]


def parse_attraction_category(s: Any) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return [str(x).strip() for x in v] if isinstance(v, list) else []


def parse_cuisines(s: Any) -> list[str]:
    if not isinstance(s, str):
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def load_city_state(db_dir: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    city_to_state: dict[str, str] = {}
    state_to_cities: dict[str, list[str]] = defaultdict(list)
    path = db_dir / "background" / "citySet_with_states.txt"
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                city, state = parts
                city_to_state[city] = state
                state_to_cities[state].append(city)
    return city_to_state, dict(state_to_cities)


def load_accommodations(db_dir: Path) -> list[dict[str, Any]]:
    """Load accommodations preserving every source CSV column so downstream
    metadata (fed to LLM agents) can quote the full row. The keys used by
    the augmenter's own filtering/scoring logic (name, city, room_type,
    cost, rating, house_rules_list) are kept exactly as before."""
    rows: list[dict[str, Any]] = []
    path = db_dir / "accommodations" / "clean_accommodations_2025.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                price = float(r["price"])
                rating = float(r["review rate number"])
            except (KeyError, ValueError, TypeError):
                continue
            try:
                min_nights = int(float(r["minimum nights"])) if r.get("minimum nights") else None
            except (ValueError, TypeError):
                min_nights = None
            try:
                max_occ = int(float(r["maximum occupancy"])) if r.get("maximum occupancy") else None
            except (ValueError, TypeError):
                max_occ = None
            rows.append({
                # Fields used by filtering logic (unchanged names/semantics):
                "name": (r.get("NAME") or "").strip(),
                "city": (r.get("city") or "").strip(),
                "room_type": (r.get("room type") or "").strip(),
                "cost": price,
                "rating": rating,
                "house_rules_list": parse_house_rules(r.get("house_rules", "")),
                # Extra columns kept for metadata completeness:
                "minimum_nights": min_nights,
                "maximum_occupancy": max_occ,
            })
    return rows


def load_restaurants(db_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = db_dir / "restaurants" / "clean_restaurant_2025.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                cost = float(r["Average Cost"])
                rating = float(r["Aggregate Rating"])
            except (KeyError, ValueError, TypeError):
                continue
            rows.append({
                "name": (r.get("Name") or "").strip(),
                "city": (r.get("City") or "").strip(),
                "cuisines": parse_cuisines(r.get("Cuisines", "")),
                "cost": cost,
                "rating": rating,
            })
    return rows


def load_attractions(db_dir: Path) -> list[dict[str, Any]]:
    """Load attractions preserving every source column (latitude, longitude,
    address, website) alongside the fields the augmenter uses."""
    rows: list[dict[str, Any]] = []
    path = db_dir / "attractions" / "attractions.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rating = float(r["rating"]) if r.get("rating") else None
            except ValueError:
                rating = None
            def _f(k):
                s = r.get(k)
                if s in (None, ""):
                    return None
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return None
            # attractions.csv now has "categories" column (was "category").
            cats_col = r.get("categories") or r.get("category", "")
            rows.append({
                # Fields used by filtering logic (unchanged names/semantics):
                "name": (r.get("Name") or "").strip(),
                "city": (r.get("City") or "").strip(),
                "categories": parse_attraction_category(cats_col),
                "rating": rating,
                # Extra columns kept for metadata completeness:
                "latitude":  _f("Latitude"),
                "longitude": _f("Longitude"),
                "address":   (r.get("Address") or "").strip() or None,
                "website":   (r.get("Website") or "").strip() or None,
            })
    return rows


def load_flight_prices(db_dir: Path) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Returns per-(origin, dest) list of (FlightDate, Price) tuples so
    downstream feasibility checks can filter to the query's date range."""
    out: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    path = db_dir / "flights" / "clean_Flights_2025.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                p = float(r["Price"])
            except (KeyError, ValueError, TypeError):
                continue
            date = (r.get("FlightDate") or "").strip()
            out[(r["OriginCityName"], r["DestCityName"])].append((date, p))
    return dict(out)


def _flight_prices_for_dates(
    pairs: dict[tuple[str, str], list[tuple[str, float]]],
    o: str, d: str, dates_set: set,
) -> list[float]:
    """Return the price list for (o, d) restricted to flights whose
    FlightDate is in `dates_set`."""
    return [price for (date, price) in pairs.get((o, d), []) if date in dates_set]


def _compute_day_meta(query: dict[str, Any],
                      prehoc_tour: list[str]
                      ) -> tuple[dict[int, str],
                                 dict[int, str],
                                 dict[int, str],
                                 dict[str, list[str]],
                                 dict[str, list[str]]]:
    """Tie each 1-indexed trip day to its city, travel_phase, and week_group.

    Returns:
        day_to_city         : dict[day_num -> city]
        day_to_phase        : dict[day_num -> "arrival" | "stay" | "departure"]
        day_to_week_group   : dict[day_num -> "weekday" | "weekend"]
        phase_to_cities     : dict[phase -> list of cities that have at least
                              one day of that phase; mid-transit days count
                              for BOTH the origin (departure) and destination
                              (arrival) cities so both scope filters resolve
                              correctly]
        week_group_to_cities: dict[week_group -> list of cities that have at
                              least one day of that group]

    Vocabulary aligned with `preferences.py::_extract_sequence.travel_type`
    and the ScopedPreference bank filter values.
    """
    dates = query.get("date") or []
    date_day = query.get("date_day") or []
    if not dates or not prehoc_tour:
        return {}, {}, {}, {}, {}

    seg_start, seg_mid, seg_end = segmentation_leg_dates(query)
    seg_mid_list: list[str] = list(seg_mid) if seg_mid else []

    tour = list(prehoc_tour)
    visit_n = len(tour)
    weekend_names = {"Saturday", "Sunday"}

    day_to_city: dict[int, str] = {}
    day_to_phase: dict[int, str] = {}
    day_to_week_group: dict[int, str] = {}
    # (day, phase, city) triples for phase_to_cities aggregation. A single
    # mid-transit day contributes TWO triples: departure from city_j and
    # arrival at city_{j+1}.
    day_phase_city: list[tuple[int, str, str]] = []

    current_city_idx = 0
    for i, d in enumerate(dates):
        day_num = i + 1
        if d == seg_start:
            # Flight org -> city_1: day belongs to city_1 as arrival.
            current_city_idx = 0
            city = tour[0] if visit_n > 0 else ""
            day_to_city[day_num] = city
            day_to_phase[day_num] = "arrival"
            if city:
                day_phase_city.append((day_num, "arrival", city))
        elif d in seg_mid_list:
            # Flight city_j -> city_{j+1}: primary label = arrival at
            # city_{j+1}, but the same day is also a departure from city_j.
            j = seg_mid_list.index(d)
            prev_city = tour[current_city_idx] if 0 <= current_city_idx < visit_n else ""
            current_city_idx = min(j + 1, visit_n - 1)
            dest_city = tour[current_city_idx] if 0 <= current_city_idx < visit_n else ""
            day_to_city[day_num] = dest_city
            day_to_phase[day_num] = "arrival"
            if prev_city:
                day_phase_city.append((day_num, "departure", prev_city))
            if dest_city:
                day_phase_city.append((day_num, "arrival", dest_city))
        elif d == seg_end:
            # Flight city_last -> org: day belongs to the last city as
            # departure.
            city = tour[current_city_idx] if 0 <= current_city_idx < visit_n else ""
            day_to_city[day_num] = city
            day_to_phase[day_num] = "departure"
            if city:
                day_phase_city.append((day_num, "departure", city))
        else:
            # Non-transit stay day in the current city.
            city = tour[current_city_idx] if 0 <= current_city_idx < visit_n else ""
            day_to_city[day_num] = city
            day_to_phase[day_num] = "stay"
            if city:
                day_phase_city.append((day_num, "stay", city))

    date_to_weekday: dict[str, str] = {}
    for pair in date_day:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            date_to_weekday[str(pair[0])] = str(pair[1])
    for i, d in enumerate(dates):
        wd = date_to_weekday.get(d, "")
        day_to_week_group[i + 1] = "weekend" if wd in weekend_names else "weekday"

    # Reverse maps: which cities appear in each phase / week_group.
    phase_to_cities: dict[str, list[str]] = {}
    for _, phase, c in day_phase_city:
        phase_to_cities.setdefault(phase, [])
        if c not in phase_to_cities[phase]:
            phase_to_cities[phase].append(c)

    week_group_to_cities: dict[str, list[str]] = {}
    for day_num, wg in day_to_week_group.items():
        c = day_to_city.get(day_num)
        if c is None:
            continue
        week_group_to_cities.setdefault(wg, [])
        if c not in week_group_to_cities[wg]:
            week_group_to_cities[wg].append(c)

    return (day_to_city, day_to_phase, day_to_week_group,
            phase_to_cities, week_group_to_cities)


def segmentation_leg_dates(query: dict[str, Any]) -> tuple[str | None, list[str], str | None]:
    """Return (start_date, mid_dates, end_date) for the trip-segmentation rule.

    A trip of D contiguous days visiting `visit_n = (D - 1) / 2` cities is
    modeled as `visit_n + 1` flight legs on the following dates:
        - org -> c_1                on start_date (= d_1)
        - c_i -> c_{i+1}            on d_{2i + 1}         for i in 1..visit_n - 1
        - c_{visit_n} -> org        on end_date  (= d_D)

    Concretely:
        D=3 (visit_n=1): start=d1, mid=[],       end=d3
        D=5 (visit_n=2): start=d1, mid=[d3],     end=d5
        D=7 (visit_n=3): start=d1, mid=[d3, d5], end=d7
    """
    dates = query.get("date") or []
    if not dates:
        return None, [], None
    start = dates[0]
    end = dates[-1]
    mid: list[str] = []
    i = 2                             # d_3 is at 0-indexed position 2
    while i < len(dates) - 1:
        mid.append(dates[i])
        i += 2
    return start, mid, end


def load_flight_records(db_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Like load_flight_prices but retains the full per-flight row (number,
    price, dep/arr time, date, distance) so each augmented record's
    metadata can quote the schedule alongside the price-only stat."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    path = db_dir / "flights" / "clean_Flights_2025.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                price = float(r["Price"])
            except (KeyError, ValueError, TypeError):
                continue
            try:
                dist = float(r.get("Distance") or "")
            except (ValueError, TypeError):
                dist = None
            out[(r["OriginCityName"], r["DestCityName"])].append({
                "flight_number": (r.get("Flight Number") or "").strip(),
                "price": price,
                "dep_time": (r.get("DepTime") or "").strip(),
                "arr_time": (r.get("ArrTime") or "").strip(),
                "elapsed": (r.get("ActualElapsedTime") or "").strip(),
                "date": (r.get("FlightDate") or "").strip(),
                "origin_city": (r.get("OriginCityName") or "").strip(),
                "dest_city":   (r.get("DestCityName") or "").strip(),
                "distance": dist,
            })
    return dict(out)


def parse_km(s: Any) -> float | None:
    if not isinstance(s, str):
        return None
    m = re.search(r"([\d,]+)\s*km", s)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def load_distance(db_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    path = db_dir / "googleDistanceMatrix" / "distance.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            km = parse_km(r.get("distance"))
            try:
                cost = float(r["cost"]) if r.get("cost") else None
            except ValueError:
                cost = None
            out[(r["origin"], r["destination"])] = {
                "cost": cost,
                "distance_km": km,
                "duration": (r.get("duration") or "").strip() or None,
            }
    return out


# --------------------------------------------------------------------------- #
# Rounding / stats                                                            #
# --------------------------------------------------------------------------- #

def round_rating(x: float) -> float:
    return round(x * 2) / 2


def round_cost(x: float, step: int) -> int:
    return int(round(x / step) * step)


def round_numeric(v: float | None, kind: str | None, step: int | None) -> Any:
    if v is None:
        return None
    if kind == "rating":
        return round_rating(float(v))
    if kind == "cost":
        return round_cost(float(v), step or 1)
    return v


def quantiles(values: list[float]) -> dict[float, float | None]:
    if not values:
        return {q: None for q in STAT_KEY_MAP.values()}
    s = sorted(values)
    n = len(s)
    out: dict[float, float | None] = {}
    for q in STAT_KEY_MAP.values():
        idx = q * (n - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, n - 1)
        out[q] = s[lo] + (s[hi] - s[lo]) * (idx - lo)
    return out


# --------------------------------------------------------------------------- #
# Query-scoped pools                                                          #
# --------------------------------------------------------------------------- #

def query_scope(query: dict[str, Any], state_to_cities: dict[str, list[str]]) -> tuple[list[str], str | None]:
    if query["days"] == 3:
        return [query["dest"]], None
    cities = state_to_cities.get(query["dest"], [])
    return list(cities), query["dest"]


def get_constraint(query: dict[str, Any], key: str) -> Any:
    return (query.get("local_constraint") or {}).get(key)


def accom_passes_constraints(item: dict[str, Any], query: dict[str, Any]) -> bool:
    hr = get_constraint(query, "house rule")
    if hr:
        needle = HOUSE_RULE_TO_NEEDLE.get(hr.strip().lower(), "no " + hr.strip().lower())
        for rule in item["house_rules_list"]:
            if needle in rule.lower():
                return False
    rt = get_constraint(query, "room type")
    if rt:
        rt_low = rt.strip().lower()
        room = item["room_type"].strip().lower()
        if rt_low.startswith("not "):
            forbidden = rt_low[4:].strip()
            if forbidden == "shared room" and room == "shared room":
                return False
            if forbidden == "private room" and room == "private room":
                return False
            if forbidden == "entire home/apt" and room == "entire home/apt":
                return False
        else:
            if rt_low == "entire home/apt" and room != "entire home/apt":
                return False
            if rt_low == "private room" and room != "private room":
                return False
            if rt_low == "shared room" and room != "shared room":
                return False
    return True


def rest_passes_constraints(item: dict[str, Any], query: dict[str, Any]) -> bool:
    c = get_constraint(query, "cuisine")
    if c and not any(cz in c for cz in item["cuisines"]):
        return False
    return True


def transportation_modes_allowed(query: dict[str, Any]) -> set[str]:
    tc = get_constraint(query, "transportation")
    modes = {"Flight", "self-driving", "taxi"}
    if not tc:
        return modes
    t = tc.strip().lower()
    if t == "no flight":
        modes.discard("Flight")
    elif t == "no self-driving":
        modes.discard("self-driving")
    return modes


def budget_heuristic(query: dict[str, Any]) -> dict[str, float]:
    budget = float(query["budget"])
    days = max(query["days"], 1)
    people = max(query["people_number"], 1)
    visit = max(query.get("visiting_city_number", 1), 1)
    return {
        "food_per_meal":    (0.25 * budget) / max(days * 3 * people, 1),
        "accom_per_night":  (0.40 * budget) / max(days * people, 1),
        "transport_per_leg": (0.30 * budget) / max((visit + 1) * people, 1),
        # Sentinel so qdb_with_budget can re-scale linearly.
        "_budget":          budget,
    }


def heuristic_for(entity: str, attribute: str, h: dict[str, float]) -> float | None:
    if attribute != "cost":
        return None
    return {
        "Restaurant":     h["food_per_meal"],
        "Accommodation":  h["accom_per_night"],
        "Transportation": h["transport_per_leg"],
    }.get(entity)


def build_query_db(query: dict[str, Any], raw: dict[str, Any],
                   city_to_state: dict[str, str],
                   state_to_cities: dict[str, list[str]],
                   flight_prices: dict[tuple[str, str], list[tuple[str, float]]],
                   distance: dict[tuple[str, str], dict[str, Any]],
                   flight_records: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
                   ) -> dict[str, Any]:
    cities, state = query_scope(query, state_to_cities)
    cities_set = set(cities)

    # Trip-segment min-nights filter: original TravelPlanner's common-
    # sense checks require an accommodation's minimum_nights <= the
    # segment stay length. Segments in the 3/5/7-day design are at most
    # 3 days, so any lodging demanding >3 nights up front can never be
    # booked and must be dropped from the pool up front -- before any
    # other constraint or feasibility computation reads the pool.
    accom = [a for a in raw["accommodations"]
             if a["city"] in cities_set
             and accom_passes_constraints(a, query)
             and (a.get("minimum_nights") is None
                  or a["minimum_nights"] <= MAX_TRIP_SEGMENT_NIGHTS)]
    rest = [r for r in raw["restaurants"]
            if r["city"] in cities_set and rest_passes_constraints(r, query)]
    attr = [a for a in raw["attractions"] if a["city"] in cities_set]

    org = query["org"]
    # Compute segmentation dates once — they gate every flight-specific
    # calculation (budget heuristic, Transportation.cost preference
    # validity, per-city flight pool count, flight_only_feasible).
    # Non-flight components (accommodation, restaurant, attraction,
    # driving-distance ground transport) remain date-agnostic.
    seg_start, seg_mid, seg_end = segmentation_leg_dates(query)
    seg_dates_set: set = set()
    if seg_start: seg_dates_set.add(seg_start)
    if seg_end:   seg_dates_set.add(seg_end)
    seg_dates_set.update(seg_mid)
    seg_start_set = {seg_start} if seg_start else set()
    seg_end_set   = {seg_end}   if seg_end   else set()
    seg_mid_set   = set(seg_mid)

    transp_costs: list[float] = []
    for c in cities:
        if c == org:
            continue
        # Outbound flights org -> c on start_date (leg 1)
        transp_costs.extend(_flight_prices_for_dates(flight_prices, org, c, seg_start_set))
        # Return flights c -> org on end_date (final leg)
        transp_costs.extend(_flight_prices_for_dates(flight_prices, c, org, seg_end_set))
        # Ground-transport distance proxy (date-agnostic).
        d = distance.get((org, c)) or distance.get((c, org))
        if d and d.get("distance_km") is not None:
            transp_costs.append(d["distance_km"])
    if len(cities) > 1:
        for c1 in cities:
            for c2 in cities:
                if c1 == c2:
                    continue
                # Intra-state flight legs on the mid segmentation dates
                # (multi-city trips only).
                if seg_mid_set:
                    transp_costs.extend(
                        _flight_prices_for_dates(flight_prices, c1, c2, seg_mid_set))
                # Driving distance between candidate cities (date-agnostic).
                d = distance.get((c1, c2))
                if d and d.get("distance_km") is not None:
                    transp_costs.append(d["distance_km"])

    stats: dict[tuple[str, str], dict[float, float | None]] = {}
    for entity, recs, attrs in (
        ("Accommodation", accom, ["cost", "rating"]),
        ("Restaurant",    rest,  ["cost", "rating"]),
        ("Attraction",    attr,  ["rating"]),
    ):
        for attribute in attrs:
            vals = [r[attribute] for r in recs if r.get(attribute) is not None]
            stats[(entity, attribute)] = quantiles(vals)
    stats[("Transportation", "cost")] = quantiles(transp_costs)

    # ---- Per-city pools ----
    #   pool_by_city      = filtered by hard local_constraint (used
    #                       everywhere inside sampling + solution_information).
    #   raw_pool_by_city  = every source row for the city, no hard-
    #                       constraint filter applied (used only when
    #                       emitting the raw `reference_information`).
    pool_by_city: dict[str, dict[str, list[dict[str, Any]]]] = {
        "Accommodation": defaultdict(list),
        "Restaurant":    defaultdict(list),
        "Attraction":    defaultdict(list),
    }
    raw_pool_by_city: dict[str, dict[str, list[dict[str, Any]]]] = {
        "Accommodation": defaultdict(list),
        "Restaurant":    defaultdict(list),
        "Attraction":    defaultdict(list),
    }
    for a in raw["accommodations"]:
        # NOTE: no min-nights filter here on purpose. `raw_pool_by_city`
        # feeds the emitted `reference_information` block -- the raw
        # source dump shown to the downstream planner LLM under test.
        # The min-nights common-sense check is part of what the LLM is
        # expected to enforce itself, so we must NOT pre-filter it here.
        # The internal working pool (`accom` above) is separately
        # filtered so augmentation / feasibility / stats never surface
        # infeasible lodgings.
        if a["city"] in cities_set:
            raw_pool_by_city["Accommodation"][a["city"]].append(a)
    for r in raw["restaurants"]:
        if r["city"] in cities_set:
            raw_pool_by_city["Restaurant"][r["city"]].append(r)
    for a in raw["attractions"]:
        if a["city"] in cities_set:
            raw_pool_by_city["Attraction"][a["city"]].append(a)
    for a in accom:
        pool_by_city["Accommodation"][a["city"]].append(a)
    for r in rest:
        pool_by_city["Restaurant"][r["city"]].append(r)
    for a in attr:
        pool_by_city["Attraction"][a["city"]].append(a)

    # Flight pool per city: round-trip count RESTRICTED to the trip-
    # segmentation dates (org->c on start_date, c->org on end_date).
    # Intra-state legs are checked in flight_only_feasible / metadata.
    # (Uses seg_start_set / seg_end_set computed above at the top of the
    # function.)
    flight_pool_by_city: dict[str, int] = {}
    for c in cities:
        if c == org:
            continue
        flight_pool_by_city[c] = (
            len(_flight_prices_for_dates(flight_prices, org, c, seg_start_set))
            + len(_flight_prices_for_dates(flight_prices, c, org, seg_end_set)))

    # Candidate cities: state cities (excluding origin) that clear the
    # PER_CITY_ALL_FLOOR minimum for EVERY visit-relevant entity type
    # (Accommodation, Restaurant, Attraction) *after* local-constraint
    # filtering. A city missing lodging under the query's room_type /
    # house_rule constraints, or with too few restaurants / attractions,
    # is not a viable visit target -- it must be dropped here so that
    # neither the preference sampler nor the tour picker can select it.
    _min_accom = PER_CITY_ALL_FLOOR["Accommodation"]
    _min_rest  = PER_CITY_ALL_FLOOR["Restaurant"]
    _min_attr  = PER_CITY_ALL_FLOOR["Attraction"]
    candidate_cities = sorted({
        c for c in cities
        if c != org
        and len(pool_by_city["Accommodation"].get(c, [])) >= _min_accom
        and len(pool_by_city["Restaurant"].get(c, []))    >= _min_rest
        and len(pool_by_city["Attraction"].get(c, []))    >= _min_attr
    })

    visit_n = max(query.get("visiting_city_number", 1), 1)
    flight_only_feasible = _compute_flight_only_feasible(
        org, candidate_cities, visit_n, flight_prices, query)

    # OPTION B: pre-select an ordered visit tour BEFORE preference
    # sampling. Every candidate city already clears PER_CITY_ALL_FLOOR
    # for every entity (accommodation / restaurant / attraction), so any
    # visit_n subset satisfies floors by construction. Rank candidates by
    # geometric-mean pool size (largest overall while penalizing any one
    # entity being at the floor edge -- keeps diversity).
    modes_allowed = transportation_modes_allowed(query)
    query_budget = float(query.get("budget", 0) or 0) or None
    query_people = max(int(query.get("people_number", 1) or 1), 1)
    prehoc_tour = _pick_prehoc_tour(
        candidate_cities, visit_n, org,
        seg_start, seg_mid, seg_end,
        pool_by_city, flight_records=None,
        flight_prices=flight_prices,
        modes_allowed=modes_allowed,
        distance=distance,
        people=query_people,
        budget=query_budget,
    )
    # Cheapest-mode-per-leg transport cost for the pinned tour. Feeds the
    # transport-affordability gate in combined_pool_ok and appears in the
    # emitted feasibility metadata for audit.
    prehoc_transport_cost = _tour_transport_cost(
        prehoc_tour, org, seg_start, seg_mid, seg_end,
        flight_prices, distance, set(modes_allowed), query_people)

    # Tie days to tour cities, travel_phase, and week_group ONCE here so
    # every temporal-scope feasibility check downstream reads a single
    # source of truth.
    day_to_city, day_to_phase, day_to_week_group, \
        phase_to_cities, week_group_to_cities = _compute_day_meta(
            query, prehoc_tour)

    return {
        "scope_cities": cities,
        "scope_cities_set": cities_set,
        "scope_state": state,
        "accom": accom,
        "rest": rest,
        "attr": attr,
        "stats": stats,
        "transport_modes": modes_allowed,
        "heuristics": budget_heuristic(query),
        "has_attractions": len(attr) > 0,
        "pool_by_city": pool_by_city,
        "raw_pool_by_city": raw_pool_by_city,
        "flight_pool_by_city": flight_pool_by_city,
        "candidate_cities": candidate_cities,
        "visiting_city_number": visit_n,
        "flight_only_feasible": flight_only_feasible,
        "prehoc_tour": prehoc_tour,     # ordered list of visit_n cities picked pre-sampling
        "origin": org,
        # Day-level metadata (1-indexed day numbers).
        "day_to_city":           day_to_city,
        "day_to_phase":          day_to_phase,
        "day_to_week_group":     day_to_week_group,
        "phase_to_cities":       phase_to_cities,        # {"stay":[...], "travelling":[...]}
        "week_group_to_cities":  week_group_to_cities,   # {"weekday":[...], "weekend":[...]}
        # Full flight records with dep_time/arr_time HH:MM strings. Used
        # by feasibility checks that reference Transportation.arrival_time
        # or departure_time by category (morning/afternoon/evening/night).
        "flight_records":        flight_records or {},
        # Segmentation dates for the trip's flight legs -- needed by
        # time-window feasibility to enumerate exactly the legs whose
        # flight schedule can carry the pref.
        "seg_start":             seg_start,
        "seg_mid":                list(seg_mid) if seg_mid else [],
        "seg_end":                seg_end,
        # Cheapest-mode-per-leg transport cost for the pinned tour, and
        # the cap (fraction of budget) beyond which combined_pool_ok
        # rejects the record as transport-infeasible.
        "prehoc_transport_cost":  prehoc_transport_cost,
        "transport_budget_cap":   (query_budget * TRANSPORT_BUDGET_CAP_FRAC
                                   if query_budget else None),
        "trip_people":            query_people,
        # Raw inputs kept here so combined_pool_ok / metadata can price
        # alternative tours if needed (e.g. under budget escalation).
        "flight_prices":          flight_prices,
        "distance":               distance,
    }


def _min_leg_cost(o: str, d: str, date: str | None,
                  flight_prices: dict[tuple[str, str], list[tuple[str, float]]],
                  distance: dict[tuple[str, str], dict[str, Any]],
                  modes_allowed: set[str], people: int
                  ) -> float | None:
    """Cheapest available cost for a single leg under `modes_allowed`.
    Flight cost scales with `people`; ground modes (self-driving / taxi)
    charge per vehicle. Returns None if no mode is feasible."""
    costs: list[float] = []
    if "Flight" in modes_allowed and date:
        prices = [p for (dd, p) in flight_prices.get((o, d), []) if dd == date]
        if prices:
            costs.append(min(prices) * max(people, 1))
    km = None
    d_row = distance.get((o, d)) or distance.get((d, o)) or {}
    km = d_row.get("distance_km")
    if km is not None:
        if "self-driving" in modes_allowed:
            costs.append(km * 0.05)   # osunlp cost model
        if "taxi" in modes_allowed:
            costs.append(km * 1.00)
    if not costs:
        return None
    return min(costs)


def _tour_transport_cost(tour: list[str], org: str,
                         seg_start: str | None, seg_mid: list[str], seg_end: str | None,
                         flight_prices, distance, modes_allowed: set[str],
                         people: int) -> float | None:
    """Total cheapest-mode cost across all legs of the ordered tour.
    Returns None if any leg has no feasible mode under `modes_allowed`.
    Every leg is priced independently -- no mode-exclusive constraint
    (matches the existing per-leg _pick_ordered_tour_prehoc scoring)."""
    if not tour:
        return None
    legs: list[tuple[str, str, str | None]] = [(org, tour[0], seg_start)]
    for i in range(len(tour) - 1):
        legs.append((tour[i], tour[i + 1],
                     seg_mid[i] if i < len(seg_mid) else None))
    legs.append((tour[-1], org, seg_end))
    total = 0.0
    for o, d, date in legs:
        c = _min_leg_cost(o, d, date, flight_prices, distance, modes_allowed, people)
        if c is None:
            return None
        total += c
    return total


# Cap on transport cost as a fraction of total budget. 30% is the design
# allocation (see budget_heuristic); we permit up to 2× that (=60% of
# budget) before rejecting the record as transport-infeasible, matching
# the escalation cap used by combined_pool_ok_with_budget_adjust.
TRANSPORT_BUDGET_CAP_FRAC = 0.60


def _pick_prehoc_tour(candidate_cities, visit_n, org, seg_start, seg_mid, seg_end,
                      pool_by_city, flight_records, flight_prices, modes_allowed,
                      distance=None, people=1, budget=None):
    """OPTION B tour picker: choose visit_n candidates such that
    (1) pool sizes are large across accommodation / restaurant /
    attraction, AND (2) the ordered tour's minimum-cost transport plan
    fits within TRANSPORT_BUDGET_CAP_FRAC × budget. Preference between
    feasible combinations goes to the LOWEST transport cost (a
    flight-only 3-city tour beats a 2-city tour that needs one cross-
    country taxi leg), with pool score as tiebreaker."""
    if visit_n <= 0 or not candidate_cities:
        return []

    def pool_score(c):
        a = len(pool_by_city["Accommodation"].get(c, []))
        r = len(pool_by_city["Restaurant"].get(c, []))
        t = len(pool_by_city["Attraction"].get(c, []))
        return a * r * t

    modes_set = set(modes_allowed)
    dist = distance or {}

    ranked = sorted(candidate_cities, key=lambda c: (-pool_score(c), c))
    # Shortlist: consider up to top-N by pool score so we don't O(n!) the
    # full state list. Enough headroom for the transport gate to override
    # pool-only choices while keeping cost bounded.
    top_k = ranked[:min(len(ranked), max(visit_n * 5, 15))]
    transport_cap = (budget * TRANSPORT_BUDGET_CAP_FRAC) if budget else None

    from itertools import combinations as _combinations
    best_key = None
    best_tour: list[str] = []
    for combo in _combinations(top_k, visit_n):
        # Best ordering of this combo by leg-mode preference (flight > drive
        # > taxi), reused so we don't reimplement it.
        ordered = _pick_ordered_tour_prehoc(
            list(combo), org, seg_start, seg_mid, seg_end,
            flight_prices, modes_set)
        if not ordered:
            continue
        total_cost = _tour_transport_cost(
            ordered, org, seg_start, seg_mid, seg_end,
            flight_prices, dist, modes_set, people)
        if total_cost is None:
            continue
        if transport_cap is not None and total_cost > transport_cap:
            continue
        combo_pool = sum(pool_score(c) for c in combo)
        # Sort key: LOWER cost wins, tiebreak by HIGHER combined pool.
        # `-combo_pool` in a tuple that we compare with < gives desired order.
        key = (total_cost, -combo_pool, tuple(ordered))
        if best_key is None or key < best_key:
            best_key = key
            best_tour = ordered

    # Fallback: if the transport cap knocked out every combination, pick
    # the cheapest-transport tour from the shortlist anyway (record will
    # still be rejected by the downstream transport gate, but we return
    # SOMETHING so the standard "no tour" branch isn't triggered
    # spuriously for records where no cap could be met).
    if not best_tour and transport_cap is not None:
        for combo in _combinations(top_k, visit_n):
            ordered = _pick_ordered_tour_prehoc(
                list(combo), org, seg_start, seg_mid, seg_end,
                flight_prices, modes_set)
            if not ordered:
                continue
            total_cost = _tour_transport_cost(
                ordered, org, seg_start, seg_mid, seg_end,
                flight_prices, dist, modes_set, people)
            if total_cost is None:
                continue
            combo_pool = sum(pool_score(c) for c in combo)
            key = (total_cost, -combo_pool, tuple(ordered))
            if best_key is None or key < best_key:
                best_key = key
                best_tour = ordered
    return best_tour


def _pick_ordered_tour_prehoc(cities, org, seg_start, seg_mid, seg_end,
                              flight_prices, modes_allowed):
    """Same idea as _pick_ordered_tour but uses flight_prices (dict of
    (o, d) -> [(date, price)]) instead of flight_records because build_query_db
    doesn't have flight_records in scope. Emits the best-scoring ordering
    of `cities`."""
    from itertools import permutations as _perm
    if not cities:
        return []
    if len(cities) == 1:
        return list(cities)
    flight_ok = "Flight" in modes_allowed
    drive_ok  = "self-driving" in modes_allowed

    def has_flight(o, d, date):
        if not date:
            return False
        return any(dd == date for (dd, _p) in flight_prices.get((o, d), []))

    def leg_score(o, d, date):
        if flight_ok and has_flight(o, d, date): return 2
        if drive_ok: return 1
        return 0

    best_score = None
    best_tour = list(cities)
    for tour in _perm(cities):
        legs = [(org, tour[0], seg_start)]
        for i in range(len(tour) - 1):
            legs.append((tour[i], tour[i + 1], seg_mid[i] if i < len(seg_mid) else None))
        legs.append((tour[-1], org, seg_end))
        s = sum(leg_score(o, d, dt) for (o, d, dt) in legs)
        key = (s, tuple(c for c in reversed(tour)))
        if best_score is None or key > best_score:
            best_score = key
            best_tour = list(tour)
    return best_tour


def _compute_flight_only_feasible(org: str, candidate_cities: list[str],
                                  visit_n: int,
                                  flight_prices: dict[tuple[str, str], list[tuple[str, float]]],
                                  query: dict[str, Any],
                                  ) -> bool:
    """True iff there exists an ordered visit tour of `visit_n` candidate
    cities such that every leg has at least one flight ON THE LEG'S
    SEGMENTATION DATE:
        - org -> c_1                on start_date
        - c_i -> c_{i+1}            on mid_dates[i-1]  (for i in 1..visit_n-1)
        - c_{visit_n} -> org        on end_date

    For 3-day queries (visit_n=1) this reduces to org<->dest on start/end
    dates.  For 5/7-day queries the intra-state legs also need coverage
    on their specific mid dates."""
    start, mid_dates, end = segmentation_leg_dates(query)
    if start is None:
        return False
    if len(mid_dates) != visit_n - 1:
        return False

    def flight_on(o: str, d: str, date: str) -> bool:
        return any(dd == date for (dd, _) in flight_prices.get((o, d), []))

    def has_flight(o: str, d: str) -> bool:
        # Reachability check used only for the initial pruning below --
        # a city is worth considering iff it can be reached from origin
        # on start_date or return to origin on end_date.
        return flight_on(o, d, start) or flight_on(o, d, end)

    if visit_n <= 0 or not candidate_cities:
        return False
    # Restrict to cities that could serve as either the first-leg
    # destination (org->c on start_date) OR the last-leg source
    # (c->org on end_date) to reduce the permutation search space.
    reachable = [c for c in candidate_cities if has_flight(org, c) or has_flight(c, org)]
    if len(reachable) < visit_n:
        return False
    # Bound: state cities for 7-day can be ~30; 30!/27! = 24360 permutations
    # in the worst case. Acceptable.
    for tour in permutations(reachable, visit_n):
        # Leg 1: org -> tour[0] on start_date
        if not flight_on(org, tour[0], start):
            continue
        # Middle legs: tour[i] -> tour[i+1] on mid_dates[i]
        ok = True
        for i in range(len(tour) - 1):
            if not flight_on(tour[i], tour[i + 1], mid_dates[i]):
                ok = False
                break
        if not ok:
            continue
        # Final leg: tour[-1] -> org on end_date
        if not flight_on(tour[-1], org, end):
            continue
        return True
    return False


def entity_records(entity: str, qdb: dict[str, Any]) -> list[dict[str, Any]]:
    return {
        "Accommodation": qdb["accom"],
        "Restaurant":    qdb["rest"],
        "Attraction":    qdb["attr"],
    }.get(entity, [])


# --------------------------------------------------------------------------- #
# Predicate evaluation                                                        #
# --------------------------------------------------------------------------- #

def get_item_attr(item: dict[str, Any], entity: str, attribute: str) -> Any:
    return item.get(ATTR_ALIAS.get((entity, attribute), attribute))


def predicate_holds(item: dict[str, Any], entity: str, attribute: str,
                    op: str, value: Any) -> bool:
    a = get_item_attr(item, entity, attribute)
    if a is None:
        return False
    # Transportation.{arrival_time, departure_time}: preferences reference
    # time-of-day by category (morning/afternoon/evening/night); the item
    # carries a raw "HH:MM" string. Route the comparison through the
    # TIME_WINDOWS table before falling through to generic op handling.
    if entity == "Transportation" and attribute in ("arrival_time", "departure_time"):
        if op == "==":
            return _time_in_category(a, value)
        if op == "!=":
            return not _time_in_category(a, value)
        if op == "in":
            vs = value if isinstance(value, list) else [value]
            return any(_time_in_category(a, v) for v in vs)
        if op == "not_in":
            vs = value if isinstance(value, list) else [value]
            return not any(_time_in_category(a, v) for v in vs)
        return False
    if op == ">=":
        return isinstance(a, (int, float)) and a >= value
    if op == "<=":
        return isinstance(a, (int, float)) and a <= value
    subset_in = (entity, attribute) in SUBSET_IN_ATTRS
    if op == "==":
        if isinstance(a, list):
            return value in a
        return a == value
    if op == "!=":
        if isinstance(a, list):
            return value not in a
        return a != value
    if op == "in":
        vs = value if isinstance(value, list) else [value]
        if isinstance(a, list):
            # house_rules: every listed rule must be present (subset);
            # cuisine / category: any listed value present is enough.
            if subset_in:
                return all(v in a for v in vs)
            return any(v in a for v in vs)
        return a in vs
    if op == "not_in":
        vs = value if isinstance(value, list) else [value]
        if isinstance(a, list):
            return not any(v in a for v in vs)
        return a not in vs
    return False


def count_matches(items: list[dict[str, Any]], entity: str, attribute: str,
                  op: str, value: Any) -> int:
    return sum(1 for it in items if predicate_holds(it, entity, attribute, op, value))


# --------------------------------------------------------------------------- #
# Bank flattening                                                             #
# --------------------------------------------------------------------------- #

def flatten_bank(bank: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in PARADIGMS:
        if p == "TemporalPreference":
            for sub, lst in (bank.get(p) or {}).items():
                for entry in lst:
                    e = dict(entry)
                    e["_paradigm"] = p
                    e["_subtype"] = sub
                    out.append(e)
        else:
            for entry in bank.get(p, []) or []:
                e = dict(entry)
                e["_paradigm"] = p
                e["_subtype"] = None
                out.append(e)
    return out


def trace(entry: dict[str, Any]) -> str:
    pk = paradigm_key(entry)
    return f"{pk}:{entry['id']}"


def paradigm_key(entry: dict[str, Any]) -> str:
    p = entry["_paradigm"]
    if p == "TemporalPreference":
        return f"{p}.{entry['_subtype']}"
    return p


# --------------------------------------------------------------------------- #
# Entity-attribute extraction & overlap                                       #
# --------------------------------------------------------------------------- #

def _atomic_ea(node: dict[str, Any]) -> tuple[str | None, str | None]:
    """Read entity/attr from either {entity_type, attribute, ...} or
    {class, template: {entity_type, attribute, ...}} variants."""
    if "template" in node and isinstance(node["template"], dict) and "entity_type" in node["template"]:
        t = node["template"]
        return t.get("entity_type"), t.get("attribute")
    return node.get("entity_type"), node.get("attribute")


def iter_template_eas(paradigm: str, template: dict[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield (entity, attribute) for every atomic-like node within a
    preference template, supporting all 8 paradigms."""
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
            yield from iter_template_eas(inner_cls, inner["template"])
        for sf in template.get("scope_filters", []) or []:
            yield from emit(sf)


def template_eas(entry: dict[str, Any]) -> set[tuple[str, str]]:
    return set(iter_template_eas(entry["_paradigm"], entry["template"]))


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return bool(template_eas(a) & template_eas(b))


# --------------------------------------------------------------------------- #
# Atomic-predicate extraction (entity, attr, op, value, scope, role)          #
# --------------------------------------------------------------------------- #

def _take_atomic(node: dict[str, Any]) -> tuple[str, str, str, str] | None:
    """Return (entity, attribute, op, scope) reading from either nesting style."""
    if "template" in node and isinstance(node["template"], dict) and "entity_type" in node["template"]:
        t = node["template"]
    else:
        t = node
    if not t.get("entity_type"):
        return None
    return (t["entity_type"], t["attribute"], t.get("op", "=="),
            t.get("scope", "all"))


def _read_value(node: dict[str, Any]) -> Any:
    if "template" in node and isinstance(node["template"], dict) and "value" in node["template"]:
        return node["template"].get("value")
    return node.get("value")


def _pred_from_node(node: dict[str, Any], role: str) -> dict[str, Any] | None:
    a = _take_atomic(node)
    if a is None:
        return None
    e, attr, op, scope = a
    return {"entity": e, "attribute": attr, "op": op, "scope": scope,
            "value": _read_value(node), "role": role}


def extract_atomic_preds(paradigm: str, template: dict[str, Any]
                         ) -> list[dict[str, Any]]:
    """Each predicate dict has keys: entity, attribute, op, value, scope, role.
    role is one of "main", "child", "primary", "margin", "secondary",
    "condition", "then", "subject", "reference", "inner", "filter",
    "lex_<i>".  value is populated from the (possibly resolved) template."""
    out: list[dict[str, Any]] = []

    def pack(node: dict[str, Any], role: str) -> dict[str, Any] | None:
        return _pred_from_node(node, role)

    if paradigm == "AtomicPreference":
        out.append({
            "entity":   template["entity_type"],
            "attribute": template["attribute"],
            "op":       template["op"],
            "scope":    template.get("scope", "all"),
            "value":    template.get("value"),
            "role":     "main",
        })
    elif paradigm == "CompositePreference":
        for c in template.get("children", []):
            p = pack(c, "child")
            if p:
                out.append(p)
    elif paradigm == "NumericPreference":
        out.append({
            "entity":   template["entity_type"],
            "attribute": template["attribute"],
            "op":       ">=" if template.get("direction") == "max" else "<=",
            "scope":    "any",
            "value":    template.get("threshold"),
            "role":     "numeric",
            "direction": template.get("direction"),
            "aggregation": template.get("aggregation"),
        })
    elif paradigm == "ConditionalPreference":
        c = pack(template["condition"], "condition")
        if c:
            out.append(c)
        t = pack(template["then_pref"], "then")
        if t:
            out.append(t)
    elif paradigm == "LexicographicPreference":
        for i, p in enumerate(template.get("preferences", [])):
            ap = pack(p, f"lex_{i}")
            if ap:
                out.append(ap)
    elif paradigm == "CompensatoryPreference":
        for k, role in (("primary_ap", "primary"),
                        ("margin_ap", "margin"),
                        ("secondary_ap", "secondary")):
            n = template.get(k)
            if n:
                p = pack(n, role)
                if p:
                    out.append(p)
    elif paradigm == "TemporalPreference":
        for k, role in (("subject_ap", "subject"), ("reference_ap", "reference")):
            n = template.get(k)
            if n:
                p = pack(n, role)
                if p:
                    out.append(p)
    elif paradigm == "ScopedPreference":
        inner = template.get("inner") or {}
        inner_cls = inner.get("class")
        if inner_cls and "template" in inner:
            inner_preds = extract_atomic_preds(inner_cls, inner["template"])
            for ip in inner_preds:
                ip = dict(ip)
                ip["role"] = "inner_" + ip["role"]
                out.append(ip)
        for sf in template.get("scope_filters", []) or []:
            p = pack(sf, "filter")
            if p:
                out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Value resolution (placeholder filling) with AGGRESSIVE numeric strategy     #
# --------------------------------------------------------------------------- #

def stat_value(entity: str, attribute: str, stat_key: str | None,
               qdb: dict[str, Any]) -> float | None:
    if not stat_key:
        return None
    q = STAT_KEY_MAP.get(stat_key)
    if q is None:
        return None
    return qdb["stats"].get((entity, attribute), {}).get(q)


def _candidates_for(entity: str, attribute: str, op: str, provided: Any,
                    stat_key: str | None, qdb: dict[str, Any],
                    include_heuristic: bool = True) -> list[tuple[str, float]]:
    """Return (tier_name, candidate) pairs in cascade order:
    stat -> seed -> heuristic.  Each entry is grounded as a specific tier
    so the multi-slot tier-lock can preserve the curator's relative
    ordering.  For multi-slot groups callers pass include_heuristic=False
    (the heuristic is a single scalar and would collapse two slots to the
    same value)."""
    if provided is None or isinstance(provided, (list, str)):
        return []
    cands: list[tuple[str, float]] = []
    sval = stat_value(entity, attribute, stat_key, qdb)
    if sval is not None:
        cands.append(("stat", float(sval)))
    cands.append(("seed", float(provided)))
    if include_heuristic:
        h = heuristic_for(entity, attribute, qdb["heuristics"])
        if h is not None:
            cands.append(("heuristic", float(h)))
    return cands


def _round_value(entity: str, attribute: str, v: float) -> Any:
    """Rounding rules:
       Accommodation.rating -> integer in [1, 5]
       Restaurant.rating    -> nearest 0.5, clamped to [0, 4.9]
       Attraction.rating    -> nearest 0.5, clamped to [1, 5]
       Restaurant.cost -> nearest 5
       Accommodation.cost, Transportation.cost -> nearest 10
    """
    if v is None:
        return None
    if attribute == "rating":
        if entity == "Accommodation":
            return max(1, min(5, int(round(float(v)))))
        r = round_rating(float(v))
        if entity == "Restaurant":
            return max(0.0, min(4.9, r))
        return max(1.0, min(5.0, r))
    kind, step = NUMERIC_ROUND.get((entity, attribute), (None, None))
    return round_numeric(v, kind, step)


def _envelope_ceil_mult(scope: str, paradigm: str | None = None) -> float:
    """Per-scope envelope ceiling multiplier.  Paradigm is accepted for
    forward-compatibility but currently has no effect -- per-paradigm
    differentiation is encoded in the FLOOR (see _floor_for), not the
    cost-affordability envelope."""
    if scope == "all":
        return ENVELOPE_CEIL_ALL
    return ENVELOPE_CEIL_ANY


def _cost_envelope(entity: str, attribute: str, scope: str,
                   qdb: dict[str, Any],
                   paradigm: str | None = None) -> tuple[float, float] | None:
    """Return (lo, hi) -- the per-query cost-envelope bounds for this
    (entity, attribute, scope) or None if no heuristic exists."""
    h = heuristic_for(entity, attribute, qdb["heuristics"])
    if h is None or h <= 0:
        return None
    lo = ENVELOPE_FLOOR_MULT * h
    hi = _envelope_ceil_mult(scope, paradigm) * h
    return lo, hi


def _try_one_value(entity: str, attribute: str, op: str, scope: str,
                   raw_v: float, qdb: dict[str, Any],
                   envelope: bool,
                   paradigm: str | None = None) -> Any | None:
    """Try to materialise a single numeric value: round and envelope-clip
    if requested.  Returns the rounded value or None.

    Pool-size feasibility is enforced later by `combined_pool_ok` (per-
    city floor across the joint filter), not here -- this function only
    cares that a single rounded value is realisable under the envelope.
    Transportation is intentionally never gated on entity_records; its
    feasibility is checked via stats / per-city flight pool downstream."""
    if raw_v is None:
        return None
    if envelope and (entity, attribute) in COST_ATTRS:
        env = _cost_envelope(entity, attribute, scope, qdb, paradigm)
        if env is not None:
            lo, hi = env
            if raw_v < lo or raw_v > hi:
                return None
    rv = _round_value(entity, attribute, raw_v)
    if rv is None:
        return None
    if envelope and (entity, attribute) in COST_ATTRS:
        env = _cost_envelope(entity, attribute, scope, qdb, paradigm)
        if env is not None:
            lo, hi = env
            if rv < lo or rv > hi:
                # Round-up to next step boundary at the floor if drift left us below.
                if rv < lo:
                    _, step = NUMERIC_ROUND.get((entity, attribute), (None, 1))
                    step = step or 1
                    rv = int(math.ceil(lo / step) * step)
                else:
                    return None
    return rv


def resolve_value(entity: str, attribute: str, op: str,
                  spec: dict[str, Any], qdb: dict[str, Any],
                  scope: str = "any",
                  pool_for_count: list[dict[str, Any]] | None = None,
                  include_heuristic: bool = True,
                  paradigm: str | None = None) -> Any:
    """Resolve a single (singleton) value placeholder.
    Categorical (list/string) -> returned as-is.
    Numeric -> cascade in order [stat, seed, heuristic]; the database stat
               is preferred (highest fidelity to local data); seed is the
               curator's anchor; heuristic is the budget-derived fallback.
               Cost predicates are envelope-clipped on every candidate;
               rating predicates have no envelope but are still
               MIN_POOL-checked.  Returns None when nothing in the cascade
               passes -- caller treats it as a rejected preference."""
    if "value" not in spec:
        return None
    provided = spec["value"]
    if isinstance(provided, list):
        return list(provided)
    if isinstance(provided, str):
        return provided

    cands = _candidates_for(entity, attribute, op, provided,
                            spec.get("stat_based_value"), qdb,
                            include_heuristic=include_heuristic)
    if not cands:
        return None
    is_cost = (entity, attribute) in COST_ATTRS
    for tier, v in cands:
        out = _try_one_value(entity, attribute, op, scope, v, qdb,
                             envelope=is_cost, paradigm=paradigm)
        if out is not None:
            return out
    return None


def _opposing(op1: str, op2: str) -> bool:
    return frozenset({op1, op2}) in OPPOSING_OPS


def _group_slots_by_ea(specs: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    """Group slot indices by their (entity, attribute) key."""
    g: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, s in enumerate(specs):
        g[(s["entity"], s["attribute"])].append(i)
    return dict(g)


def _categorical_value(ev: dict[str, Any]) -> Any | None:
    """Return the categorical seed (list / string) directly, or None if
    the spec doesn't carry one (numeric slot)."""
    v = ev.get("value")
    if isinstance(v, list):
        return list(v)
    if isinstance(v, str):
        return v
    return None


def _slot_group_strict_ordering_ok(specs: list[dict[str, Any]],
                                   vals: list[Any]) -> bool:
    """Validate strict ordering of resolved numeric values within a
    slot group sharing the same (entity, attribute, op).  specs[0]
    is the strictest (primary / tier_0) and the values must satisfy:
        op == '>='  ->  vals[0] > vals[1] > ... > vals[-1]
        op == '<='  ->  vals[0] < vals[1] < ... < vals[-1]
    Returns True if the group has fewer than 2 slots, or contains any
    non-numeric attempt (categorical), or has mixed ops (skip)."""
    if len(specs) < 2:
        return True
    ops = {s.get("op") for s in specs}
    if len(ops) != 1:
        return True
    op = next(iter(ops))
    if op not in (">=", "<=", ">", "<"):
        return True
    nums = []
    for v in vals:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return True
        nums.append(v)
    for i in range(len(nums) - 1):
        a, b = nums[i], nums[i + 1]
        if op in (">=", ">"):
            if a <= b: return False
        else:  # "<=", "<"
            if a >= b: return False
    return True


def resolve_group_tier_lock(specs: list[dict[str, Any]],
                            evs: list[dict[str, Any]],
                            qdb: dict[str, Any],
                            paradigm: str | None = None) -> list[Any] | None:
    """Tier-locked group resolver for slots sharing one (entity, attribute).

    For a multi-slot group every slot's resolved value must come from the
    SAME fidelity tier (so the curator's relative ordering encoded via
    stat-keys is preserved by construction).  Tier order is stat -> seed
    -> heuristic.  For multi-slot groups the heuristic tier is skipped
    (it produces the same value for every slot and collapses the ordering).

    Returns the list of resolved values aligned with `specs`, or None when
    no tier yields a valid value for every slot."""
    if not specs:
        return []

    # Categorical slots (value is a list / string in ev) are resolved as-is
    # at every tier -- there's no stat/seed/heuristic cascade for them.
    multi_slot = len(specs) > 1
    tiers: list[str] = ["stat", "seed"]
    if not multi_slot:
        tiers.append("heuristic")

    # Pre-classify slots: numeric or categorical (categorical bypasses cascade).
    cat_vals: list[Any] = []
    numeric_indices: list[int] = []
    for i, ev in enumerate(evs):
        cv = _categorical_value(ev)
        cat_vals.append(cv)
        if cv is None:
            numeric_indices.append(i)

    for tier in tiers:
        attempt: list[Any] = list(cat_vals)
        ok = True
        for idx in numeric_indices:
            s = specs[idx]; ev = evs[idx]
            entity, attribute, op = s["entity"], s["attribute"], s["op"]
            scope = s.get("scope", "any")
            provided = ev.get("value")
            stat_key = ev.get("stat_based_value")
            if tier == "stat":
                src = stat_value(entity, attribute, stat_key, qdb)
            elif tier == "seed":
                src = float(provided) if isinstance(provided, (int, float)) and not isinstance(provided, bool) else None
            else:  # heuristic
                src = heuristic_for(entity, attribute, qdb["heuristics"])
            if src is None:
                ok = False
                break
            out = _try_one_value(entity, attribute, op, scope, float(src), qdb,
                                 envelope=((entity, attribute) in COST_ATTRS),
                                 paradigm=paradigm)
            if out is None:
                ok = False
                break
            attempt[idx] = out
        if ok and all(v is not None for v in attempt):
            # When the tier resolves cleanly but the multi-slot group's
            # values collapse to violate strict ordering (e.g., primary
            # == margin after quartile rounding), fall through to the
            # next tier rather than emit a degenerate preference.
            if not _slot_group_strict_ordering_ok(specs, attempt):
                continue
            return attempt
    return None


def resolve_slots(specs: list[dict[str, Any]],
                  evs: list[dict[str, Any]],
                  qdb: dict[str, Any],
                  paradigm: str | None = None) -> list[Any] | None:
    """Top-level slot resolver: groups slots by (entity, attribute), runs
    the tier-locked group resolver on each group, returns the merged value
    list aligned with `specs`.  Returns None if any group fails."""
    groups = _group_slots_by_ea(specs)
    out: list[Any] = [None] * len(specs)
    for ea, idxs in groups.items():
        sub_specs = [specs[i] for i in idxs]
        sub_evs   = [evs[i]   for i in idxs]
        vals = resolve_group_tier_lock(sub_specs, sub_evs, qdb, paradigm=paradigm)
        if vals is None:
            return None
        for i, v in zip(idxs, vals):
            out[i] = v
    if any(v is None for v in out):
        return None
    return out


# --------------------------------------------------------------------------- #
# Constraint-aware set-predicate adjustment (cuisine / house_rules /          #
# room_type) and lexicographic cuisine partition                              #
# --------------------------------------------------------------------------- #

# Per accommodation house-rule keyword, the human-readable rule entry it
# corresponds to in the DB's house_rules list.
HOUSE_RULE_DB_FORM = {
    "pets":     "No pets",
    "smoking":  "No smoking",
    "parties":  "No parties",
    "visitors": "No visitors",
    "children": "No children",
}

# Canonical room-type strings as stored in the accommodations CSV.
ROOM_TYPE_DB = {"Entire home/apt", "Private room", "Shared room"}


def _allowed_room_types(constraint: str | None) -> set[str]:
    """Return the set of accommodation room_type values still legal under
    the query's room-type constraint.  None constraint -> all allowed."""
    if not constraint:
        return set(ROOM_TYPE_DB)
    c = constraint.strip().lower()
    mapping = {"entire home/apt": "Entire home/apt",
               "private room": "Private room",
               "shared room":  "Shared room"}
    if c.startswith("not "):
        forbidden = mapping.get(c[4:].strip())
        return ROOM_TYPE_DB - ({forbidden} if forbidden else set())
    fixed = mapping.get(c)
    return {fixed} if fixed else set(ROOM_TYPE_DB)


def _intersect_pref_set_with_constraint(entity: str, attribute: str,
                                        op: str, value: Any,
                                        query: dict[str, Any]) -> Any | None:
    """If the query has a constraint on this set-valued attribute,
    intersect the predicate's value list with the constraint-feasibility
    set (the values the constraint *allows*).

    Return semantics:
      - None constraint: pass through unchanged (None = fully permissive).
      - `op == in` (or `==`): adjusted intersection.  Empty -> return
        sentinel False (caller rejects the preference).
      - `op == not_in` (or `!=`): trim values that the constraint already
        forbids (they're redundant); empty result is allowed (returns
        an empty list to mean "no remaining forbidden values").
    """
    lc = query.get("local_constraint") or {}

    def normalise(v: Any) -> set[str]:
        if isinstance(v, list):
            return set(v)
        if isinstance(v, str):
            return {v}
        return set()

    if entity == "Restaurant" and attribute == "cuisine":
        c = lc.get("cuisine")
        if not c:
            return value
        constraint_set = set(c)
        pv = normalise(value)
        if op in ("in", "=="):
            inter = pv & constraint_set
            if not inter:
                return False
            return sorted(inter) if isinstance(value, list) else next(iter(inter))
        if op in ("not_in", "!="):
            # trim values already forbidden by the constraint's complement
            kept = pv & constraint_set
            return sorted(kept) if isinstance(value, list) else (
                next(iter(kept)) if kept else None)
        return value

    if entity == "Accommodation" and attribute == "house_rules":
        hr = lc.get("house rule")
        if not hr:
            return value
        # The constraint's keyword maps to a single forbidden rule entry
        # ("No <keyword>"); that's the value the augmented accommodation
        # pool can never carry.
        forbidden_rule = HOUSE_RULE_DB_FORM.get(hr.strip().lower(),
                                                "No " + hr.strip().lower())
        pv = normalise(value)
        contains_forbidden = any(v.lower() == forbidden_rule.lower() for v in pv)
        if op in ("in", "=="):
            # Subset semantics: a pref `in [..., forbidden, ...]` would
            # demand every accommodation in the pool carry the forbidden
            # rule -- which the constraint has just excluded.  Reject the
            # whole pref rather than silently dropping a value (that would
            # weaken the curator's intent).
            if contains_forbidden:
                return False
            return value
        if op in ("not_in", "!="):
            # The constraint already removes accoms carrying the
            # forbidden rule; pref's `not_in [forbidden, ...]` is therefore
            # redundant on that value.  Trim it; if all pref values were
            # redundant the predicate adds no constraint -- treat as
            # vacuous and reject.
            kept = {v for v in pv if v.lower() != forbidden_rule.lower()}
            if not kept:
                return False
            return sorted(kept) if isinstance(value, list) else next(iter(kept))
        return value

    if entity == "Transportation" and attribute == "mode":
        allowed = transportation_modes_allowed(query)
        pv = normalise(value)
        if op in ("in", "=="):
            inter = pv & allowed
            if not inter:
                return False
            return sorted(inter) if isinstance(value, list) else next(iter(inter))
        if op in ("not_in", "!="):
            # if pref forbids every legal mode, the predicate is unsatisfiable
            if allowed.issubset(pv):
                return False
            return value
        return value

    if entity == "Accommodation" and attribute == "room_type":
        rt = lc.get("room type")
        if not rt:
            return value
        allowed = _allowed_room_types(rt)
        pv = normalise(value)
        if op in ("in", "=="):
            kept = pv & allowed
            if not kept:
                return False
            return sorted(kept) if isinstance(value, list) else next(iter(kept))
        if op in ("not_in", "!="):
            kept = pv & allowed
            return sorted(kept) if isinstance(value, list) else (
                next(iter(kept)) if kept else None)
        return value

    return value


def _apply_constraint_feasibility(paradigm: str, template: dict[str, Any],
                                  query: dict[str, Any]) -> bool:
    """Walk every set-valued predicate in the template and intersect its
    value list with the query's constraint.  Mutates `template` in place.
    Returns False (reject preference) if any `in`/`==` predicate ends up
    with an empty intersection."""
    nodes_to_visit: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    def visit(node: dict[str, Any]) -> None:
        nodes_to_visit.append((node, None))

    if paradigm == "AtomicPreference":
        nodes_to_visit.append((template, "raw"))
    elif paradigm == "CompositePreference":
        for c in template.get("children", []):
            visit(c)
    elif paradigm == "ConditionalPreference":
        for k in ("condition", "then_pref"):
            n = template.get(k)
            if n: visit(n)
    elif paradigm == "LexicographicPreference":
        for p in template.get("preferences", []):
            visit(p)
    elif paradigm == "CompensatoryPreference":
        for k in ("primary_ap", "margin_ap", "secondary_ap"):
            n = template.get(k)
            if n: visit(n)
    elif paradigm == "TemporalPreference":
        for k in ("subject_ap", "reference_ap"):
            n = template.get(k)
            if n: visit(n)
    elif paradigm == "ScopedPreference":
        inner = template.get("inner") or {}
        inner_cls = inner.get("class")
        inner_t = inner.get("template")
        if inner_cls and inner_t is not None:
            if not _apply_constraint_feasibility(inner_cls, inner_t, query):
                return False
        for sf in template.get("scope_filters", []) or []:
            visit(sf)

    for node, hint in nodes_to_visit:
        a = _take_atomic(node)
        if a is None:
            continue
        e, attr, op, _scope = a
        if (e, attr) not in {("Restaurant", "cuisine"),
                             ("Accommodation", "house_rules"),
                             ("Accommodation", "room_type"),
                             ("Transportation", "mode")}:
            continue
        cur_val = _read_value(node)
        if cur_val is None:
            continue
        new_val = _intersect_pref_set_with_constraint(e, attr, op, cur_val, query)
        if new_val is False:
            return False
        if hint == "raw":
            template["value"] = new_val
        else:
            _set_atomic_node_value(node, new_val)
    return True


def _try_lex_cuisine_partition(prefs: list[dict[str, Any]],
                               query: dict[str, Any],
                               rng: random.Random) -> list[Any] | None:
    """If the lexicographic template's two slots are both on
    Restaurant.cuisine AND the query's cuisine constraint lists >=2
    items, partition that constraint into the two slots (TeX paper's
    pattern).  Returns the two value lists, or None when not applicable
    (caller falls back to the standard cascade)."""
    if len(prefs) != 2:
        return None
    if not all(p.get("entity_type") == "Restaurant"
               and p.get("attribute") == "cuisine"
               and p.get("op") in ("in", "==")
               for p in prefs):
        return None
    c = (query.get("local_constraint") or {}).get("cuisine")
    if not c or len(c) < 2:
        return None
    pool = sorted(set(c))
    rng.shuffle(pool)
    half = max(1, len(pool) // 2)
    primary = sorted(pool[:half])
    secondary = sorted(pool[half:])
    if not primary or not secondary:
        return None
    return [primary, secondary]


# --------------------------------------------------------------------------- #
# Intra-preference invariant post-check                                       #
# --------------------------------------------------------------------------- #

def _slot_numeric_value(node: dict[str, Any]) -> Any:
    v = _read_value(node)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def intra_pref_invariants_ok(paradigm: str, template: dict[str, Any]) -> bool:
    """Belt-and-braces validation that the curator's slot-vs-slot ordering
    encoded via stat keys survived resolution.  Tier-lock guarantees this
    by construction for the current bank, but this check protects us if a
    future bank entry uses inconsistent stat keys."""
    def check_same_dir(node_a: dict[str, Any], op_a: str,
                       node_b: dict[str, Any], op_b: str) -> bool:
        a = _take_atomic(node_a); b = _take_atomic(node_b)
        if a is None or b is None: return True
        if (a[0], a[1]) != (b[0], b[1]): return True
        if op_a != op_b: return True
        va = _slot_numeric_value(node_a); vb = _slot_numeric_value(node_b)
        if va is None or vb is None: return True
        # primary should be STRICTLY tighter than the fallback slot in
        # the curator's encoding -- equality collapses the tolerance band
        # and degenerates the paradigm semantic (e.g., Compensatory primary
        # == margin means there is no tolerance to compensate for).
        if op_a == "<=" and va >= vb: return False
        if op_a == ">=" and va <= vb: return False
        return True

    if paradigm == "CompensatoryPreference":
        prim = template["primary_ap"]; marg = template["margin_ap"]
        ap = _take_atomic(prim); am = _take_atomic(marg)
        if ap and am and (ap[0], ap[1]) == (am[0], am[1]):
            if not check_same_dir(prim, ap[2], marg, am[2]):
                return False
            # Categorical vacuity guard: if primary and margin share the
            # SAME (entity, attribute) and their categorical value sets
            # are identical, the compensatory tradeoff has collapsed --
            # margin is supposed to name a DIFFERENT acceptable value
            # ("prefer French; Mediterranean is the acceptable margin"),
            # so primary == margin is degenerate.  This can happen after
            # _pair_adjust re-samples both slots independently from the
            # same rigid intersection.
            if not _is_numerical_attribute(ap[0], ap[1]):
                pv = _read_value(prim); mv = _read_value(marg)
                def _as_set(v: Any) -> frozenset:
                    if isinstance(v, list): return frozenset(v)
                    if v is None: return frozenset()
                    return frozenset({v})
                pv_s, mv_s = _as_set(pv), _as_set(mv)
                if pv_s and mv_s and pv_s == mv_s:
                    return False
    elif paradigm == "LexicographicPreference":
        prefs = template.get("preferences", [])
        if len(prefs) >= 2:
            a = _take_atomic(prefs[0]); b = _take_atomic(prefs[1])
            if a and b and (a[0], a[1]) == (b[0], b[1]) and a[2] == b[2]:
                if not check_same_dir(prefs[0], a[2], prefs[1], b[2]):
                    return False
    elif paradigm == "ConditionalPreference":
        cond = template["condition"]; then_p = template["then_pref"]
        a = _take_atomic(cond); b = _take_atomic(then_p)
        if a and b and (a[0], a[1]) == (b[0], b[1]) and _opposing(a[2], b[2]):
            va = _slot_numeric_value(cond); vb = _slot_numeric_value(then_p)
            if va is not None and vb is not None:
                # Conditional with opposing same-attribute ops encodes
                # cond >= V_high THEN then <= V_low (or symmetric):
                # require V_high > V_low so the conditional is non-trivial.
                if a[2] == ">=" and b[2] == "<=" and va <= vb: return False
                if a[2] == "<=" and b[2] == ">=" and va >= vb: return False
    elif paradigm == "TemporalPreference":
        subj = template.get("subject_ap")
        ref = template.get("reference_ap")
        if subj and ref:
            a = _take_atomic(subj); b = _take_atomic(ref)
            if a and b and (a[0], a[1]) == (b[0], b[1]):
                if a[2] == b[2]:
                    if not check_same_dir(subj, a[2], ref, b[2]):
                        return False
    return True


# --------------------------------------------------------------------------- #
# Placeholder resolution for whole preferences                                #
# --------------------------------------------------------------------------- #

def select_example_block(entry: dict[str, Any], days: int) -> dict[str, Any] | None:
    ev_root = entry.get("example_values") or {}
    key = str(days)
    if key in ev_root and ev_root[key] is None:
        return None
    if key in ev_root and isinstance(ev_root[key], dict):
        return ev_root[key]
    return ev_root.get("default")


def sample_cities(scope_cities: list[str], origin: str, usable: set[str],
                  n: int, rng: random.Random) -> list[str] | None:
    cands = [c for c in scope_cities if c != origin and c in usable]
    if len(cands) < n:
        return None
    return rng.sample(cands, n)


def _set_value(node: dict[str, Any], v: Any) -> None:
    if "template" in node and isinstance(node["template"], dict) and "value" in node["template"]:
        node["template"]["value"] = v
    elif "template" in node and isinstance(node["template"], dict):
        node["template"]["value"] = v
    else:
        node["value"] = v


def _set_threshold(node: dict[str, Any], v: Any) -> None:
    if "template" in node and isinstance(node["template"], dict):
        node["template"]["threshold"] = v
    else:
        node["threshold"] = v


def resolve_atomic_dict(node: dict[str, Any], ev: dict[str, Any],
                        qdb: dict[str, Any]) -> bool:
    """Mutate `node` (an atomic-like dict in any nesting style) by writing
    `value` from `ev`. Returns False if the resulting value is None."""
    a = _take_atomic(node)
    if a is None:
        return False
    e, attr, op, scope = a
    v = resolve_value(e, attr, op, ev, qdb, scope=scope)
    if v is None:
        return False
    _set_value(node, v)
    return True


def resolve_numeric_dict(node: dict[str, Any], ev: dict[str, Any],
                         qdb: dict[str, Any]) -> bool:
    if "template" in node and isinstance(node["template"], dict):
        t = node["template"]
    else:
        t = node
    th = ev.get("threshold")
    sk = ev.get("stat_based_threshold")
    e = t["entity_type"]; a = t["attribute"]
    # Pick the [lo, hi] tuple aggressively, falling back to global range / stats.
    if isinstance(th, list) and len(th) == 2:
        lo, hi = float(th[0]), float(th[1])
    else:
        lo, hi = COST_RANGE.get(e, (1.0, 5.0))
    if isinstance(sk, list) and len(sk) == 2:
        s0 = stat_value(e, a, sk[0], qdb)
        s1 = stat_value(e, a, sk[1], qdb)
        if s0 is not None and s1 is not None:
            lo = max(lo, float(s0))
            hi = min(hi, float(s1)) if hi > 0 else float(s1)
    if hi < lo:
        hi = lo
    lo = _round_value(e, a, lo)
    hi = _round_value(e, a, hi)
    _set_threshold(node, [lo, hi])
    return True


def resolve_template(entry: dict[str, Any], query: dict[str, Any],
                     qdb: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    ev = select_example_block(entry, query["days"])
    if ev is None:
        return None
    template = deepcopy(entry["template"])
    paradigm = entry["_paradigm"]

    if paradigm == "AtomicPreference":
        v = resolve_value(template["entity_type"], template["attribute"],
                          template["op"], ev, qdb,
                          scope=template.get("scope", "all"),
                          paradigm=paradigm)
        if v is None:
            return None
        template["value"] = v

    elif paradigm == "CompositePreference":
        children = template["children"]
        cevs = ev.get("children", [])
        if len(children) != len(cevs):
            return None
        specs = [{"entity": c["entity_type"], "attribute": c["attribute"],
                  "op": c["op"], "scope": c.get("scope", "all")}
                 for c in children]
        vals = resolve_slots(specs, cevs, qdb, paradigm=paradigm)
        if vals is None:
            return None
        for c, v in zip(children, vals):
            c["value"] = v

    elif paradigm == "NumericPreference":
        th = ev.get("threshold")
        sk = ev.get("stat_based_threshold")
        e = template["entity_type"]; a = template["attribute"]
        if isinstance(th, list) and len(th) == 2:
            lo, hi = float(th[0]), float(th[1])
        else:
            lo, hi = COST_RANGE.get(e, (1.0, 5.0))
        if isinstance(sk, list) and len(sk) == 2:
            s0 = stat_value(e, a, sk[0], qdb)
            s1 = stat_value(e, a, sk[1], qdb)
            if s0 is not None and s1 is not None:
                lo = max(lo, float(s0))
                hi = min(hi, float(s1)) if hi >= float(s1) else float(s1)
        if hi < lo:
            hi = lo
        # Apply the per-query cost envelope to a Numeric paradigm's threshold
        # (the direction's aspiration shouldn't aspire beyond budget reach).
        if (e, a) in COST_ATTRS:
            env = _cost_envelope(e, a, "any", qdb, paradigm)  # numeric is existential
            if env is not None:
                env_lo, env_hi = env
                lo = max(lo, env_lo)
                hi = min(hi, env_hi)
                if hi < lo:
                    return None  # envelope cannot be satisfied
        template["threshold"] = [_round_value(e, a, lo), _round_value(e, a, hi)]

    elif paradigm == "ConditionalPreference":
        cond, then_p = template["condition"], template["then_pref"]
        cev, tev = ev["condition"], ev["then_pref"]
        if cond["entity_type"] == "Day" and cond["attribute"] == "city":
            if query["days"] == 3:
                return None
            # Under Option B, the tour is decided BEFORE preference
            # resolution. Draw the Conditional's city pair EXCLUSIVELY
            # from prehoc_tour so both cond and then_pref reference
            # visit cities -- the preference is meaningfully active
            # (condition can fire) instead of vacuously true.
            tour = qdb.get("prehoc_tour") or []
            if len(tour) < 2:
                return None
            # Two distinct tour cities, deterministic-ish via rng.
            pair = list(tour)
            rng.shuffle(pair)
            pair = pair[:2]
            cond["value"], then_p["value"] = pair
        elif then_p.get("entity_type") == "Day" and then_p.get("attribute") == "city":
            # then_pref alone is Day.city (cond is on some other entity):
            # bind then_p to a tour city so the conditional actually
            # forces something achievable.
            tour = qdb.get("prehoc_tour") or []
            if not tour:
                return None
            then_p["value"] = rng.choice(tour)
            # Fall through to the general resolution below for the
            # non-Day condition side.
            specs = [{"entity": cond["entity_type"], "attribute": cond["attribute"],
                      "op": cond["op"], "scope": cond.get("scope", "any")}]
            cvals = resolve_slots(specs, [cev], qdb, paradigm=paradigm)
            if cvals is None or cvals[0] is None:
                return None
            cond["value"] = cvals[0]
        else:
            specs = [{"entity": cond["entity_type"], "attribute": cond["attribute"],
                      "op": cond["op"], "scope": cond.get("scope", "any")},
                     {"entity": then_p["entity_type"], "attribute": then_p["attribute"],
                      "op": then_p["op"], "scope": then_p.get("scope", "any")}]
            vals = resolve_slots(specs, [cev, tev], qdb, paradigm=paradigm)
            if vals is None:
                return None
            cond["value"], then_p["value"] = vals

    elif paradigm == "LexicographicPreference":
        prefs = template["preferences"]
        pevs = ev.get("preferences", [])
        if len(prefs) != len(pevs):
            return None
        # Special: lexicographic over Restaurant.cuisine -- when the query has
        # a cuisine constraint with >=2 entries, partition that constraint
        # list into the two slots instead of using the bank's literal values
        # (cleanly aligns with the TeX paper's partitioning prescription).
        partitioned = _try_lex_cuisine_partition(prefs, query, rng)
        if partitioned is not None:
            for p, v in zip(prefs, partitioned):
                p["value"] = v
        else:
            specs = [{"entity": p["entity_type"], "attribute": p["attribute"],
                      "op": p["op"], "scope": p.get("scope", "all")}
                     for p in prefs]
            vals = resolve_slots(specs, pevs, qdb, paradigm=paradigm)
            if vals is None:
                return None
            for p, v in zip(prefs, vals):
                p["value"] = v

    elif paradigm == "CompensatoryPreference":
        slot_keys = ("primary_ap", "margin_ap", "secondary_ap")
        nodes = [template.get(k) for k in slot_keys]
        slot_evs = [ev.get(k) for k in slot_keys]
        if any(n is None or e is None for n, e in zip(nodes, slot_evs)):
            return None
        specs = []
        for n in nodes:
            a = _take_atomic(n)
            specs.append({"entity": a[0], "attribute": a[1], "op": a[2],
                          "scope": a[3] if a[3] is not None else "any"})
        vals = resolve_slots(specs, slot_evs, qdb, paradigm=paradigm)
        if vals is None:
            return None
        for n, v in zip(nodes, vals):
            _set_atomic_node_value(n, v)

    elif paradigm == "TemporalPreference":
        subj = template["subject_ap"]; sev = ev["subject_ap"]
        ref = template.get("reference_ap"); rev = ev.get("reference_ap")
        if ref is not None and rev is not None:
            specs = [{"entity": subj["entity_type"], "attribute": subj["attribute"],
                      "op": subj["op"], "scope": subj.get("scope", "any")},
                     {"entity": ref["entity_type"], "attribute": ref["attribute"],
                      "op": ref["op"], "scope": ref.get("scope", "any")}]
            vals = resolve_slots(specs, [sev, rev], qdb, paradigm=paradigm)
            if vals is None:
                return None
            subj["value"], ref["value"] = vals
        else:
            v = resolve_value(subj["entity_type"], subj["attribute"], subj["op"],
                              sev, qdb, scope=subj.get("scope", "any"),
                              paradigm=paradigm)
            if v is None:
                return None
            subj["value"] = v
        # Propagate any per-days time_start / time_end / step_count overrides.
        for k in ("time_start", "time_end", "step_count"):
            if k in ev:
                template[k] = ev[k]

    elif paradigm == "ScopedPreference":
        inner = template["inner"]; inner_ev = ev["inner"]
        inner_cls = inner.get("class")
        inner_t = inner["template"]
        if inner_cls == "AtomicPreference":
            ok = resolve_atomic_dict(inner, inner_ev, qdb)
            if not ok:
                return None
        elif inner_cls == "NumericPreference":
            resolve_numeric_dict(inner, inner_ev, qdb)
        elif inner_cls == "CompositePreference":
            cevs = inner_ev.get("children", [])
            for child, cev in zip(inner_t["children"], cevs):
                ok = resolve_atomic_dict(child, cev, qdb)
                if not ok:
                    return None
        elif inner_cls == "TemporalPreference":
            sub = inner_t["subject_ap"]; sev = inner_ev["subject_ap"]
            ref = inner_t.get("reference_ap"); rev = inner_ev.get("reference_ap")
            ok1 = resolve_atomic_dict(sub, sev, qdb)
            if not ok1:
                return None
            if ref is not None and rev is not None:
                ok2 = resolve_atomic_dict(ref, rev, qdb)
                if not ok2:
                    return None
        else:
            return None
        # scope_filters: ev maps to a single dict OR list depending on entry style.
        sf_ev = ev.get("scope_filters")
        filters = template.get("scope_filters", [])
        if isinstance(sf_ev, list):
            for sf, fev in zip(filters, sf_ev):
                resolve_atomic_dict(sf, fev, qdb)
        elif isinstance(sf_ev, dict) and len(filters) == 1:
            resolve_atomic_dict(filters[0], sf_ev, qdb)
        else:
            return None

    else:
        return None

    return template


# --------------------------------------------------------------------------- #
# Single-preference feasibility                                               #
# --------------------------------------------------------------------------- #

def query_pool(entity: str, qdb: dict[str, Any]) -> list[dict[str, Any]]:
    return entity_records(entity, qdb)


def _existence_preds(paradigm: str, template: dict[str, Any]) -> list[dict[str, Any]]:
    """Predicates the resolved template needs to satisfy by existence
    (scope=any-ish) under the joint pool.  Used for non-vacuity check."""
    preds = extract_atomic_preds(paradigm, template)
    out: list[dict[str, Any]] = []
    # Inject resolved value for each pred by re-reading the template tree.
    if paradigm == "AtomicPreference":
        if template.get("scope") != "all":
            preds[0]["value"] = template.get("value")
            out.append(preds[0])
        return out
    if paradigm == "CompositePreference":
        op = template.get("op")
        cs = template.get("children", [])
        for i, c in enumerate(cs):
            pred = preds[i]
            pred["value"] = c.get("value")
            if c.get("scope") != "all":
                out.append(pred)
        return out
    if paradigm == "NumericPreference":
        return out  # treated as soft optimization
    if paradigm == "ConditionalPreference":
        cond = template["condition"]
        pred = preds[0]
        pred["value"] = cond.get("value")
        out.append(pred)  # condition existence required for the pref to fire
        return out
    if paradigm == "LexicographicPreference":
        primary = template["preferences"][0]
        pred = preds[0]
        pred["value"] = primary.get("value")
        out.append(pred)
        return out
    if paradigm == "CompensatoryPreference":
        for k, p in zip(("primary_ap", "margin_ap", "secondary_ap"), preds[:3]):
            n = template[k]
            p["value"] = (n.get("template") or n).get("value")
            out.append(p)
        return out
    if paradigm == "TemporalPreference":
        # atmost_once used to be treated as "0 occurrences ok" and its
        # subject was skipped. Per the design decision to always require
        # >=1 satisfying item (leaving the 0-occurrence semantic to the
        # planner), we now include the subject_ap unconditionally.
        subj = template["subject_ap"]
        pred = preds[0]
        pred["value"] = subj.get("value")
        out.append(pred)
        ref = template.get("reference_ap")
        if ref:
            pred2 = preds[1]
            pred2["value"] = ref.get("value")
            out.append(pred2)
        return out
    if paradigm == "ScopedPreference":
        inner = template["inner"]
        inner_cls = inner["class"]
        inner_t = inner["template"]
        if inner_cls == "AtomicPreference":
            p = _pred_from_node(inner, "inner_main")
            if p and p["value"] is not None:
                out.append(p)
        # NumericPreference inner: treated as soft optimization, no existential.
        elif inner_cls == "CompositePreference":
            for c in inner_t.get("children", []):
                p = _pred_from_node(c, "inner_child")
                if p and p["value"] is not None:
                    out.append(p)
        elif inner_cls == "TemporalPreference":
            # atmost_once subject is now included (require >=1 satisfying
            # item for feasibility; planner decides whether to fire it).
            sub = inner_t.get("subject_ap")
            if sub:
                p = _pred_from_node(sub, "inner_subject")
                if p and p["value"] is not None:
                    out.append(p)
            ref = inner_t.get("reference_ap")
            if ref:
                p = _pred_from_node(ref, "inner_reference")
                if p and p["value"] is not None:
                    out.append(p)
        return out
    return out


def _all_scope_preds(paradigm: str, template: dict[str, Any]) -> list[dict[str, Any]]:
    """Atomic preds that must hold over EVERY selected item (scope=all)."""
    preds = extract_atomic_preds(paradigm, template)
    out: list[dict[str, Any]] = []
    if paradigm == "AtomicPreference":
        if template.get("scope") == "all":
            preds[0]["value"] = template.get("value")
            out.append(preds[0])
    elif paradigm == "CompositePreference":
        if template.get("op") != "OR":
            for c, p in zip(template.get("children", []), preds):
                if c.get("scope") == "all":
                    p["value"] = c.get("value")
                    out.append(p)
    elif paradigm == "LexicographicPreference":
        primary = template["preferences"][0]
        if primary.get("scope") == "all":
            preds[0]["value"] = primary.get("value")
            out.append(preds[0])
    elif paradigm == "ConditionalPreference":
        # then_pref does NOT impose universal constraint until condition fires;
        # skip for [all] pool reduction.
        return out
    elif paradigm == "CompensatoryPreference":
        return out  # primary/margin act on existence
    elif paradigm == "TemporalPreference":
        return out
    elif paradigm == "ScopedPreference":
        return out
    return out


def _tour_flight_leg_records(qdb: dict[str, Any]
                             ) -> list[list[dict[str, Any]]]:
    """Return the per-leg list of candidate flight records for the pinned
    prehoc_tour. Each entry is the flights available on that leg's
    segmentation date, restricted to the (origin, destination) pair. An
    empty inner list indicates the leg has no flights on its seg date."""
    tour = list(qdb.get("prehoc_tour") or [])
    org = qdb.get("origin")
    seg_start = qdb.get("seg_start")
    seg_mid   = list(qdb.get("seg_mid") or [])
    seg_end   = qdb.get("seg_end")
    fr = qdb.get("flight_records") or {}
    if not tour or org is None or seg_start is None or seg_end is None:
        return []
    legs: list[tuple[str, str, str]] = []
    legs.append((org, tour[0], seg_start))
    for i in range(len(tour) - 1):
        # For visit_n > 1, len(seg_mid) == visit_n - 1
        date = seg_mid[i] if i < len(seg_mid) else None
        if date is None:
            return []
        legs.append((tour[i], tour[i + 1], date))
    legs.append((tour[-1], org, seg_end))
    out: list[list[dict[str, Any]]] = []
    for o, d, date in legs:
        rows = [r for r in fr.get((o, d), []) if r.get("date") == date]
        out.append(rows)
    return out


def predicate_feasible_alone(pred: dict[str, Any], qdb: dict[str, Any]) -> bool:
    entity = pred["entity"]; attr = pred["attribute"]
    op = pred["op"]; val = pred["value"]
    if val is None:
        return False
    if entity == "Day":
        if attr in ("city",):
            return val in qdb["scope_cities_set"]
        return True
    if entity == "Transportation":
        if attr == "mode":
            if op == "==":
                return val in qdb["transport_modes"]
            if op == "in":
                vs = val if isinstance(val, list) else [val]
                return any(v in qdb["transport_modes"] for v in vs)
            if op == "!=":
                return any(m != val for m in qdb["transport_modes"])
            if op == "not_in":
                vs = val if isinstance(val, list) else [val]
                return any(m not in vs for m in qdb["transport_modes"])
            return True
        if attr == "cost":
            stats = qdb["stats"].get(("Transportation", "cost"), {})
            mn = stats.get(0.0); mx = stats.get(1.0)
            if mn is None and mx is None:
                return True
            if op == "<=":
                return mn is None or mn <= val
            if op == ">=":
                return mx is None or mx >= val
            return True
        if attr in ("arrival_time", "departure_time"):
            # Existence gate: at least one flight across the pinned tour's
            # seg-date legs must satisfy the time-category predicate. This
            # is a coarse necessary condition -- combined_pool_ok enforces
            # the stricter per-leg check when a [all] Composite AND pairs
            # this with mode==Flight.
            legs = _tour_flight_leg_records(qdb)
            if not legs:
                return False
            for leg in legs:
                for r in leg:
                    if predicate_holds(r, entity, attr, op, val):
                        return True
            return False
        return True
    items = query_pool(entity, qdb)
    return any(predicate_holds(it, entity, attr, op, val) for it in items)


def entity_joint_feasible(entity: str, preds: list[dict[str, Any]],
                          qdb: dict[str, Any]) -> bool:
    if entity in ("Day", "Transportation"):
        return all(predicate_feasible_alone(p, qdb) for p in preds)
    pool = query_pool(entity, qdb)
    for p in preds:
        if p["scope"] == "all":
            pool = [it for it in pool
                    if predicate_holds(it, entity, p["attribute"], p["op"], p["value"])]
            if not pool:
                return False
    for p in preds:
        if p["scope"] != "all":
            if not any(predicate_holds(it, entity, p["attribute"], p["op"], p["value"])
                       for it in pool):
                return False
    return True


_RATING_CLAMP = {
    ("Accommodation", "rating"): (1, 5),
    ("Restaurant",    "rating"): (0, 4.9),
    ("Attraction",    "rating"): (1, 5),
}


def _is_rating_cap_trivial(entity: str, attribute: str, op: str,
                           value: Any) -> bool:
    """True iff a rating predicate's resolved value sits at the attribute's
    natural cap (for op='<=') or floor (for op='>='), making the predicate
    trivially satisfied by every item in the rating-valid range."""
    if attribute != "rating":
        return False
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    bounds = _RATING_CLAMP.get((entity, attribute))
    if bounds is None:
        return False
    mn, mx = bounds
    if op == "<=" and value >= mx:
        return True
    if op == ">=" and value <= mn:
        return True
    return False


def _pred_is_trivially_satisfied(pred: dict[str, Any],
                                 qdb: dict[str, Any]) -> bool:
    """True iff EVERY item in the trip-scope (constraint-filtered) entity
    pool satisfies the predicate -- the preference adds no real constraint.
    Catches rating-at-cap saturations from pair-adjust as well as any other
    accidental no-op preds.  Day / Transportation handled via dedicated
    feasibility paths so this returns False for them.  Numeric paradigm
    aggregation targets (value is [lo, hi] list) are skipped -- they're
    soft optimization objectives, not per-item filters."""
    entity = pred.get("entity")
    attr = pred.get("attribute")
    op = pred.get("op")
    val = pred.get("value")
    if val is None:
        return False
    if pred.get("role") == "numeric":
        return False  # NumericPreference aggregation target, not a filter
    if isinstance(val, list) and op in ("<=", ">=", "==", "!="):
        return False  # threshold-band style; not a per-item filter
    if entity in ("Day", "Transportation"):
        return False
    if _is_rating_cap_trivial(entity, attr, op, val):
        return True
    items = query_pool(entity, qdb)
    if not items:
        return False  # empty pool is a different problem (caught elsewhere)
    return all(predicate_holds(it, entity, attr, op, val) for it in items)


def _not_in_inverse_exists(pred: dict[str, Any], qdb: dict[str, Any]) -> bool:
    """For a `not_in` atomic predicate, return True iff at least one item
    in the trip-scope entity pool DOES match the inverse `in` filter --
    i.e. the predicate is non-trivial because the excluded values actually
    appear in the pool.  Without this guard, e.g. `Attraction.category
    not_in [Casinos & Gambling]` in a city with no casinos is trivially
    satisfied by every plan, which is unhelpful as a preference."""
    if pred.get("op") != "not_in":
        return True
    entity = pred["entity"]
    attr = pred["attribute"]
    val = pred["value"]
    if entity == "Day":
        if attr != "city":
            return True
        vs = val if isinstance(val, list) else [val]
        return any(v in qdb["scope_cities_set"] for v in vs)
    if entity == "Transportation":
        if attr == "mode":
            vs = val if isinstance(val, list) else [val]
            return any(v in qdb["transport_modes"] for v in vs)
        return True
    items = query_pool(entity, qdb)
    return any(predicate_holds(it, entity, attr, "in", val) for it in items)


def _template_bound_cities(paradigm: str,
                           template: dict[str, Any]) -> list[str]:
    """Return every city name referenced by a Day.city binding ANYWHERE
    inside the template -- exhaustively, at any nesting depth. Catches
    Day.city in:
      - ScopedPreference.scope_filters, ScopedPreference.inner (nested)
      - ConditionalPreference.condition AND .then_pref
      - LexicographicPreference.preferences[*]
      - CompensatoryPreference.primary_ap / margin_ap / secondary_ap
      - TemporalPreference.subject_ap / reference_ap
      - CompositePreference.children[*]
      - any future paradigm-path a bank entry may introduce.
    Any city that isn't in qdb['scope_cities_set'] must upstream-reject
    the record (a Minnesota query cannot legitimately name San Francisco
    in any preference slot). `paradigm` is retained for API symmetry but
    the walk is paradigm-agnostic."""
    out: list[str] = []
    def _absorb(v: Any) -> None:
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out.extend(x for x in v if isinstance(x, str))
    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            # Detect the atomic-node shape at this level (Day.city). Both
            # the flat form (entity_type/attribute at top) and the nested
            # form (under 'template') appear in the bank.
            for shape in (node, node.get("template") if isinstance(node.get("template"), dict) else None):
                if not isinstance(shape, dict):
                    continue
                if shape.get("entity_type") == "Day" and shape.get("attribute") == "city":
                    _absorb(shape.get("value"))
                    return  # atomic leaf; don't recurse into its fields
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for x in node:
                _walk(x)
    _walk(template)
    return out


def is_single_feasible(entry: dict[str, Any], template: dict[str, Any],
                       qdb: dict[str, Any]) -> bool:
    paradigm = entry["_paradigm"]
    # Cross-state bound-city guard: reject records where a ScopedPreference
    # scope_filter or ConditionalPreference condition names a city that
    # is not in the query's state (e.g. Day.city == "San Francisco" on a
    # Minnesota query). The per-city breakdown otherwise silently skips
    # the check for a candidate-city != bound_city, letting the record
    # admit with a preference that references a city the plan will never
    # visit.
    scope_cities_set = qdb.get("scope_cities_set") or set()
    for city in _template_bound_cities(paradigm, template):
        if city not in scope_cities_set:
            return False
    all_p = _all_scope_preds(paradigm, template)
    exist_p = _existence_preds(paradigm, template)
    for p in all_p + exist_p:
        if not predicate_feasible_alone(p, qdb):
            return False
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in all_p:
        by_entity[p["entity"]].append(p)
    for entity, plist in by_entity.items():
        if not entity_joint_feasible(entity, plist, qdb):
            return False
    # not_in trivial-satisfaction guard: for every atomic sub-pred whose
    # op is not_in, require >=1 item in the trip-scope pool that DOES
    # match the inverse `in` filter.  Otherwise the pred is satisfied
    # vacuously by every plan and provides no preference signal.
    for p in extract_atomic_preds(paradigm, template):
        if not _not_in_inverse_exists(p, qdb):
            return False
    # Rating-cap / general trivial-satisfaction guard: a pred whose value
    # sits at the attribute's natural cap (e.g. Accommodation.rating <= 5)
    # or whose filter is passed by every item in the pool is no constraint.
    for p in extract_atomic_preds(paradigm, template):
        if _pred_is_trivially_satisfied(p, qdb):
            return False
    return True


# --------------------------------------------------------------------------- #
# Per-city pool floor: combined-filter feasibility (Strategy B)               #
# --------------------------------------------------------------------------- #

def _items_pass_all(items: list[dict[str, Any]], entity: str,
                    preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter items by intersecting every predicate in preds."""
    out = items
    for p in preds:
        out = [it for it in out
               if predicate_holds(it, entity, p["attribute"], p["op"], p["value"])]
        if not out:
            break
    return out


def _walk_atomic_nodes(paradigm: str, template: dict[str, Any]
                       ) -> Iterable[tuple[dict[str, Any], str]]:
    """Yield (node, role) for every atomic-like sub-node in the template.
    Used by floor-check helpers that need access to the original node
    (value + scope) rather than the extracted pred dict."""
    if paradigm == "AtomicPreference":
        yield template, "main"
    elif paradigm == "CompositePreference":
        for c in template.get("children", []):
            yield c, "child"
    elif paradigm == "NumericPreference":
        yield template, "numeric"
    elif paradigm == "ConditionalPreference":
        for k, r in (("condition", "condition"), ("then_pref", "then")):
            n = template.get(k)
            if n: yield n, r
    elif paradigm == "LexicographicPreference":
        for i, p in enumerate(template.get("preferences", []) or []):
            yield p, f"lex_{i}"
    elif paradigm == "CompensatoryPreference":
        for k, r in (("primary_ap", "primary"), ("margin_ap", "margin"),
                     ("secondary_ap", "secondary")):
            n = template.get(k)
            if n: yield n, r
    elif paradigm == "TemporalPreference":
        for k, r in (("subject_ap", "subject"), ("reference_ap", "reference")):
            n = template.get(k)
            if n: yield n, r
    elif paradigm == "ScopedPreference":
        inner = template.get("inner", {}) or {}
        inner_cls = inner.get("class")
        inner_t = inner.get("template")
        if inner_cls and inner_t is not None:
            for node, role in _walk_atomic_nodes(inner_cls, inner_t):
                yield node, "inner_" + role
        for sf in template.get("scope_filters", []) or []:
            yield sf, "filter"


def _is_flight_demanding_pred(p: dict[str, Any]) -> bool:
    """True if the predicate effectively requires flight transport.
    Per the design directive: cost predicates on Transportation, and any
    mode predicate that mandates Flight, require flight_only_feasible."""
    if p["entity"] != "Transportation":
        return False
    attr = p["attribute"]
    if attr in ("cost", "departure_time", "arrival_time"):
        return True
    if attr == "mode":
        op = p["op"]; val = p["value"]
        if op == "==" and val == "Flight":
            return True
        if op == "in":
            vs = val if isinstance(val, list) else [val]
            return "Flight" in vs and len(vs) == 1  # mandates flight
    return False


def _make_pred(node: dict[str, Any] | None, role: str) -> dict[str, Any] | None:
    """Build a predicate dict from any atomic-like node, regardless of
    nesting style.  Returns None if the node lacks an entity_type."""
    if node is None:
        return None
    a = _take_atomic(node)
    if a is None:
        return None
    return {"entity": a[0], "attribute": a[1], "op": a[2],
            "scope": (a[3] or "any"), "value": _read_value(node),
            "role": role}


def _scoped_inner_preds(template: dict[str, Any]
                        ) -> list[dict[str, Any]]:
    """Extract atomic predicates from a ScopedPreference inner template,
    respecting per-atomic scope.  Returns predicates with role prefixed
    by 'inner_'.  TemporalPreference inner: subject is skipped iff op ==
    atmost_once (matches the rest of the codebase)."""
    inner = template.get("inner", {}) or {}
    inner_cls = inner.get("class")
    inner_t = inner.get("template")
    out: list[dict[str, Any]] = []
    if not (inner_cls and inner_t is not None):
        return out
    if inner_cls == "AtomicPreference":
        # template-style: entity_type at the inner_t level
        p = _make_pred(inner_t, "inner_main")
        if p:
            out.append(p)
    elif inner_cls == "CompositePreference":
        for c in inner_t.get("children", []) or []:
            p = _make_pred(c, "inner_child")
            if p:
                out.append(p)
    elif inner_cls == "NumericPreference":
        pass  # soft pref, no pool check
    elif inner_cls == "TemporalPreference":
        op = inner_t.get("op")
        for k, role in (("subject_ap", "inner_subject"),
                        ("reference_ap", "inner_reference")):
            n = inner_t.get(k)
            if n is None:
                continue
            # atmost_once inner_subject is now included -- see comments in
            # _existence_preds / _collect_record_preds.
            p = _make_pred(n, role)
            if p:
                out.append(p)
    return out


def _collect_record_preds(specs: list[tuple[str, dict[str, Any]]]
                          ) -> dict[str, Any]:
    """Walk every atomic sub-predicate in the record and split into:

      "all_by_entity": dict[entity -> list[pred]]
          Trip-wide UNCONDITIONALLY [all]-scope predicates.  Contribute
          to the combined filter for every other floor check.  Only
          AtomicPreference.main, AND-Composite.children, and Lex
          primary (lex_0) with scope=='all' enter this bucket.  Sub-
          preds that are scope-conditioned (Conditional.then_pref,
          Temporal.subject/reference, Compensatory.{primary,margin,
          secondary}, Scoped.inner) NEVER enter this bucket -- their
          [all] semantics applies only within their scope, so they
          float as independent checks.

      "floor_checks": list[(pred, bound_city_or_None, paradigm_or_None)]
          Every other sub-predicate that needs an existence-style
          floor check.  Each check is run against the combined [all]
          filter + this pred.  Floor selected by pred['scope']:
          [all] -> PER_CITY_ALL_FLOOR[entity];  [any] -> PER_CITY_ANY_FLOOR.
          paradigm tag is consumed by combined_pool_ok so Compensatory
          [any] uses ENVELOPE_CEIL_ANY_COMPENSATORY instead of the default.

      "flight_demanding_all", "flight_demanding_any": bool flags for
          any Transportation predicate that demands flight transport
          (mode==Flight, cost, departure_time, arrival_time).

    Day-entity preds are routed to feasibility via the bound-city
    mechanism (Conditional.cond on Day.city, Scoped.scope_filter on
    Day.city); they never enter all_by_entity or floor_checks as
    entity-pool preds.  Transportation preds are surfaced only via the
    flight-demanding flags."""
    all_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    floor_checks: list[tuple[dict[str, Any], str | None, str | None]] = []
    flight_demanding_all = False
    flight_demanding_any = False
    # Time-window predicates on Transportation.{arrival_time,
    # departure_time}. Collected separately so combined_pool_ok can run
    # a per-leg feasibility check against the pinned tour's actual
    # flight schedule.
    time_window_checks: list[dict[str, Any]] = []

    def flag_flight(pred: dict[str, Any] | None) -> None:
        nonlocal flight_demanding_all, flight_demanding_any
        if pred is None or not _is_flight_demanding_pred(pred):
            return
        if pred["scope"] == "all":
            flight_demanding_all = True
        else:
            flight_demanding_any = True
        if (pred.get("entity") == "Transportation"
                and pred.get("attribute") in ("arrival_time", "departure_time")):
            time_window_checks.append(pred)

    def add_all_filter(pred: dict[str, Any]) -> None:
        # Trip-wide [all] filter contributor.  Day/Transportation are
        # handled via dedicated paths and never enter this bucket.
        if pred["entity"] in ("Day", "Transportation"):
            return
        all_by_entity[pred["entity"]].append(pred)

    def add_check(pred: dict[str, Any],
                  paradigm_tag: str | None,
                  bound_city: str | None = None) -> None:
        # Day handled via Day.city scope check; Transportation handled
        # via flight-demanding flags.  Skip both from per-entity floor.
        if pred["entity"] in ("Day", "Transportation"):
            return
        floor_checks.append((pred, bound_city, paradigm_tag))

    for paradigm, template in specs:
        if paradigm == "AtomicPreference":
            p = _make_pred(template, "main")
            if p is None:
                continue
            flag_flight(p)
            if p["scope"] == "all":
                add_all_filter(p)
            else:
                add_check(p, paradigm)

        elif paradigm == "CompositePreference":
            comp_op = template.get("op", "AND")
            for c in template.get("children", []) or []:
                p = _make_pred(c, "child")
                if p is None:
                    continue
                flag_flight(p)
                # AND composite with scope=all: hard universal demand on
                # the entity -> contributes to combined filter.
                if comp_op != "OR" and p["scope"] == "all":
                    add_all_filter(p)
                else:
                    add_check(p, paradigm)

        elif paradigm == "NumericPreference":
            # Soft pref (max/min direction or [lo, hi] band).  No floor.
            # Flag flight-demanding still.
            p = _make_pred(template, "numeric")
            flag_flight(p)

        elif paradigm == "ConditionalPreference":
            cond = template.get("condition")
            then_p = template.get("then_pref")
            c_a = _take_atomic(cond) if cond else None
            t_a = _take_atomic(then_p) if then_p else None
            # Case: Day.city condition -> then_pref bound to named city.
            if (c_a and c_a[0] == "Day" and c_a[1] == "city"
                    and t_a and t_a[0] != "Day"):
                city = _read_value(cond)
                p_then = _make_pred(then_p, "then")
                flag_flight(p_then)
                if isinstance(city, str):
                    add_check(p_then, paradigm, bound_city=city)
                else:
                    add_check(p_then, paradigm)
                # condition is Day.city; no entity-pool floor check.
                continue
            # General: each side gets its own existential floor check.
            p_cond = _make_pred(cond, "condition")
            p_then = _make_pred(then_p, "then")
            flag_flight(p_cond); flag_flight(p_then)
            if p_cond:
                add_check(p_cond, paradigm)
            if p_then:
                add_check(p_then, paradigm)

        elif paradigm == "LexicographicPreference":
            for i, n in enumerate(template.get("preferences", []) or []):
                p = _make_pred(n, f"lex_{i}")
                if p is None:
                    continue
                flag_flight(p)
                # Primary (lex_0) [all] is a trip-wide universal demand;
                # fallback slots (lex_1+) and any non-[all] primary are
                # existential checks.
                if i == 0 and p["scope"] == "all":
                    add_all_filter(p)
                else:
                    add_check(p, paradigm)

        elif paradigm == "CompensatoryPreference":
            for k, role in (("primary_ap", "primary"),
                            ("margin_ap", "margin"),
                            ("secondary_ap", "secondary")):
                n = template.get(k)
                p = _make_pred(n, role)
                if p is None:
                    continue
                flag_flight(p)
                # All three slots are existential trade-offs; none is a
                # hard trip-wide [all] filter even if marked scope=all.
                add_check(p, paradigm)

        elif paradigm == "TemporalPreference":
            op = template.get("op")
            temporal_scope = template.get("scope") or "global"
            # Semantic day-window for hold_during / hold_after (absolute
            # 1-indexed day indices in the plan).
            ts_day = template.get("time_start") if op in ("hold_during", "hold_after") else None
            te_day = template.get("time_end")   if op == "hold_during" else None
            for k, role in (("subject_ap", "subject"),
                            ("reference_ap", "reference")):
                n = template.get(k)
                if n is None:
                    continue
                p = _make_pred(n, role)
                if p is None:
                    continue
                flag_flight(p)
                # Propagate the OPERATOR's scope + op onto the pred so
                # combined_pool_ok / build_feasibility_metadata can route
                # per_city / per_day / week_group / travel_phase checks
                # correctly instead of collapsing them to a union.
                p["temporal_scope"] = temporal_scope
                p["temporal_op"]    = op
                if op in ("hold_during", "hold_after"):
                    p["hold_time_start"] = ts_day
                    p["hold_time_end"]   = te_day
                # atmost_once subject IS floor-checked now: we require
                # >=1 satisfying item exist in the joint pool so the
                # "at most once" event is at least reachable. The
                # planner can still choose 0 occurrences at plan time.
                add_check(p, paradigm)

        elif paradigm == "ScopedPreference":
            scope_filters = template.get("scope_filters", []) or []
            # Split scope filters into (a) Day-axis partition-scopes that
            # restrict WHICH days/cities the inner pred applies to, and
            # (b) non-Day inner filters that further restrict WHICH items
            # in the entity pool count toward the check.
            bound_cities: list[str] = []
            scoped_group_axis: str | None = None
            scoped_group_values: list[Any] = []
            inner_filters_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for sf in scope_filters:
                sf_a = _take_atomic(sf)
                # Flight-demanding flags from scope filters too:
                p_sf = _make_pred(sf, "filter")
                flag_flight(p_sf)
                if not sf_a:
                    continue
                sf_ent, sf_attr, sf_op, _ = sf_a
                sf_val = _read_value(sf)
                if sf_ent == "Day":
                    if sf_attr == "city":
                        if isinstance(sf_val, str):
                            bound_cities.append(sf_val)
                        elif isinstance(sf_val, list):
                            bound_cities.extend(x for x in sf_val if isinstance(x, str))
                    elif sf_attr in ("week_group", "travel_phase", "city_index"):
                        # Partition-scope on a day-level axis: enforce the
                        # inner pred on the union of cities that have at
                        # least one day matching the filter's value(s).
                        scoped_group_axis = sf_attr
                        if isinstance(sf_val, list):
                            scoped_group_values.extend(sf_val)
                        else:
                            scoped_group_values.append(sf_val)
                    # Other Day attributes (date, etc.) are ignored for
                    # pool-count feasibility -- planner enforces them at
                    # plan time.
                elif p_sf is not None and p_sf["entity"] not in ("Day", "Transportation"):
                    # Non-Day scope filter: restrict WHICH entity items
                    # count. Attached below as inner_filters on the pred.
                    # Skip slot-level attributes (e.g. Restaurant.meal_type):
                    # they are not per-item fields, so filtering the pool
                    # by them would spuriously zero the count.
                    if (p_sf["entity"], p_sf["attribute"]) in SLOT_LEVEL_ATTRS:
                        continue
                    inner_filters_by_entity[p_sf["entity"]].append(p_sf)
            inner_preds = _scoped_inner_preds(template)
            for ip in inner_preds:
                flag_flight(ip)
                # Intersect the entity pool with same-entity scope filters
                # before the floor check runs.
                extra = inner_filters_by_entity.get(ip["entity"], [])
                if extra:
                    ip["inner_filters"] = extra
                # Attach Day-axis partition-scope so combined_pool_ok can
                # route to a union-across-group-cities enforcement.
                if scoped_group_axis and scoped_group_values:
                    ip["scoped_axis"] = scoped_group_axis
                    ip["scoped_values"] = list(scoped_group_values)
                if bound_cities:
                    for city in bound_cities:
                        add_check(ip, paradigm, bound_city=city)
                else:
                    add_check(ip, paradigm)

    return {
        "all_by_entity":         dict(all_by_entity),
        "floor_checks":          floor_checks,
        "flight_demanding_all":  flight_demanding_all,
        "flight_demanding_any":  flight_demanding_any,
        "time_window_checks":    time_window_checks,
    }


def _floor_for(pred: dict[str, Any], paradigm: str | None = None) -> int:
    """Per-city floor for a single predicate.  [all]-scope uses the entity-
    specific PER_CITY_ALL_FLOOR; [any]-scope uses PER_CITY_ANY_FLOOR, or
    the entity-specific Compensatory floor for Compensatory sub-preds
    (per-day scaling for Restaurant/Attraction; still 1 for Accommodation
    since a stay is booked once per city, not once per day)."""
    if pred["scope"] == "all":
        return PER_CITY_ALL_FLOOR.get(pred["entity"], PER_CITY_ANY_FLOOR)
    if paradigm == "CompensatoryPreference":
        return PER_CITY_ANY_FLOOR_COMPENSATORY.get(pred["entity"], PER_CITY_ANY_FLOOR)
    return PER_CITY_ANY_FLOOR


def _affordable_subset(items: list[dict[str, Any]], entity: str,
                       scope: str, paradigm: str | None,
                       qdb: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter items to the affordable subset under the per-entity envelope
    ceiling at the given scope/paradigm.  Entities with no cost dimension
    (Attraction) are returned unchanged."""
    if entity not in ("Restaurant", "Accommodation"):
        return items
    h = heuristic_for(entity, "cost", qdb["heuristics"])
    if h is None or h <= 0:
        return items
    ceil = _envelope_ceil_mult(scope, paradigm) * h
    return [it for it in items
            if it.get("cost") is not None and it["cost"] <= ceil]


def combined_pool_ok(specs: list[tuple[str, dict[str, Any]]],
                     qdb: dict[str, Any]) -> bool:
    """STRICT subset-feasibility check (option 2): admit iff there exists
    a single `visit_n`-subset of candidate cities such that
      - each picked city individually clears every [all]-scope joint
        check (Step 1) and every [all]-scope or bound per-pred check;
      - the union of picked cities' affordable items clears every
        [any]-scope per-pred check's floor;
      - for OR-composite children with shared entity, the union of items
        satisfying any OR child clears the per-city floor (per picked
        city if [all]-scope, or by union if [any]-scope);
      - Step 3 flight feasibility holds against the same picked subset.

    This is stricter than the prior per-check K-of-N rule and admits
    only queries where one coherent itinerary exists across `visit_n`
    cities to satisfy every preference simultaneously."""
    candidate = qdb["candidate_cities"]
    visit_n = qdb["visiting_city_number"]
    if not candidate or len(candidate) < visit_n:
        return False
    pool_by_city = qdb["pool_by_city"]
    # Defensive cross-state guard: reject if any preference names a
    # bound city not in this query's state. Redundant with the same
    # check in is_single_feasible, but kept here so combined_pool_ok
    # can never admit a record with a nonsense city reference.
    scope_cities_set = qdb.get("scope_cities_set") or set()
    for paradigm, template in specs:
        for city in _template_bound_cities(paradigm, template):
            if city not in scope_cities_set:
                return False

    buckets = _collect_record_preds(specs)

    # ---- Per-city per-check breakdown (same as build_feasibility_metadata) ----
    per_city: dict[str, dict[str, dict[str, Any]]] = {}
    for c in candidate:
        checks: dict[str, dict[str, Any]] = {}
        # Step 1
        for entity, preds in buckets["all_by_entity"].items():
            floor = PER_CITY_ALL_FLOOR.get(entity)
            if floor is None:
                continue
            items = _items_pass_all(
                pool_by_city.get(entity, {}).get(c, []), entity, preds)
            aff = _affordable_subset(items, entity, "all", None, qdb)
            checks[f"step1_all[{entity}]"] = {
                "affordable_count": len(aff), "floor": floor,
                "passes": len(aff) >= floor, "scope": "all", "bound": False,
            }
        # Step 2 per-pred
        for i, fc in enumerate(buckets["floor_checks"]):
            pred, bound_city, paradigm_tag = fc
            entity = pred["entity"]
            if entity in ("Day", "Transportation"):
                continue
            if bound_city is not None and bound_city != c:
                continue
            all_preds = buckets["all_by_entity"].get(entity, [])
            floor = _floor_for(pred, paradigm_tag)
            scope = pred.get("scope", "any")
            # Same-entity scope filters (from ScopedPreference, e.g.
            # Restaurant.meal_type == 'breakfast') must also gate the
            # pool -- otherwise the inner-pred floor over-admits.
            inner_filters = pred.get("inner_filters") or []
            combined = all_preds + list(inner_filters) + [pred]
            items = _items_pass_all(
                pool_by_city.get(entity, {}).get(c, []), entity, combined)
            aff = _affordable_subset(items, entity, scope, paradigm_tag, qdb)
            label = f"step2[{paradigm_tag}:{entity}.{pred.get('attribute')}:{scope}:idx{i}]"
            checks[label] = {
                "affordable_count": len(aff), "floor": floor,
                "passes": len(aff) >= floor, "scope": scope,
                "bound": bound_city is not None,
            }
        per_city[c] = checks

    # OR-composite union check
    or_groups: list[tuple[int, str, list[dict[str, Any]]]] = []
    for pi, (paradigm, template) in enumerate(specs):
        if paradigm != "CompositePreference" or template.get("op") != "OR":
            continue
        by_ent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for child in template.get("children", []) or []:
            ent = child.get("entity_type")
            if ent and ent not in ("Day", "Transportation"):
                by_ent[ent].append(child)
        for ent, kids in by_ent.items():
            or_groups.append((pi, ent, kids))

    for pi, ent, kids in or_groups:
        scopes = {k.get("scope", "any") for k in kids}
        or_scope = "all" if "all" in scopes else "any"
        floor = (PER_CITY_ALL_FLOOR.get(ent) if or_scope == "all"
                 else PER_CITY_ANY_FLOOR)
        if floor is None:
            continue
        label = f"step2_or[CompositePreference#{pi}:{ent}:{or_scope}]"
        for c in candidate:
            pool = pool_by_city.get(ent, {}).get(c, [])
            baseline = _items_pass_all(pool, ent,
                                       buckets["all_by_entity"].get(ent, []))
            seen: set[int] = set()
            union_items: list[dict[str, Any]] = []
            for kid in kids:
                attr = kid.get("attribute"); op = kid.get("op"); val = kid.get("value")
                for it in baseline:
                    if id(it) in seen:
                        continue
                    if predicate_holds(it, ent, attr, op, val):
                        union_items.append(it); seen.add(id(it))
            aff = _affordable_subset(union_items, ent, or_scope,
                                     "CompositePreference", qdb)
            per_city[c][label] = {
                "affordable_count": len(aff), "floor": floor,
                "passes": len(aff) >= floor, "scope": or_scope,
                "bound": False, "or_union": True,
            }
        # Supersede individual OR children
        for c in candidate:
            for chk_label, d in per_city[c].items():
                if d.get("or_union"):
                    continue
                if d.get("superseded_by"):
                    continue
                if not chk_label.startswith("step2["):
                    continue
                try:
                    inner = chk_label[len("step2["):-1]
                    parts = inner.split(":")
                    if parts[0] != "CompositePreference":
                        continue
                    ea = parts[1].split(".")
                    if len(ea) != 2 or ea[0] != ent:
                        continue
                    if any(k.get("attribute") == ea[1] for k in kids):
                        d["superseded_by"] = label
                except Exception:
                    continue

    # ---- Classify checks by scope (excluding superseded children) ----
    # Buckets:
    #   all_labels        -> each selected city individually passes
    #                        (scope='all' or day-city-bound preds).
    #   per_city_labels   -> temporal ops with per_city / per_day scope:
    #                        each selected city individually must have
    #                        affordable_count >= floor of that pred.
    #   phase_labels      -> temporal ops with travel_phase scope: for
    #                        each phase present in the tour's day map,
    #                        the union of affordable_count across the
    #                        cities visited in that phase must clear
    #                        the pred's floor.
    #   week_group_labels -> same as phase_labels but keyed on
    #                        weekday / weekend.
    #   hold_window_labels-> hold_during / hold_after: each city visited
    #                        during the absolute day-index window
    #                        [time_start, time_end] individually passes.
    #   scoped_group_labels -> ScopedPreference with Day-axis partition
    #                        (week_group / travel_phase / city_index):
    #                        union of affordable_count across the cities
    #                        that have at least one day matching the
    #                        scope filter's value(s) clears the floor.
    #   any_labels        -> everything else (global temporal scope OR
    #                        non-temporal [any]-scope with no partition):
    #                        union across selected cities clears floor.
    all_labels: set[str] = set()
    per_city_labels: set[str] = set()
    phase_labels: set[str] = set()
    week_group_labels: set[str] = set()
    hold_window_labels: set[str] = set()
    scoped_group_labels: set[str] = set()
    any_labels: set[str] = set()
    floor_of: dict[str, int] = {}
    label_pred: dict[str, dict[str, Any]] = {}
    # ALL temporal ops route to a per-scope-group pool check -- the
    # benchmark rule is "no vacuous satisfaction anywhere". Even
    # atmost_once (an upper bound) must be included: a scope group with
    # zero matching items would trivially satisfy `at most one
    # occurrence` and contribute no signal for the LLM to reason about.
    # The bucket the label lands in (per_city / phase / week_group /
    # hold_window / any) then decides whether the ≥1 check is per-city
    # individual or union-across-group-cities.
    _EXISTENCE_TEMPORAL_OPS = {"sometime", "always", "atmost_once",
                               "within", "always_within",
                               "sometime_before", "sometime_after",
                               "hold_during", "hold_after"}
    for c in candidate:
        for lbl, d in per_city.get(c, {}).items():
            if d.get("superseded_by"):
                continue
            floor_of[lbl] = d["floor"]
    # Second pass: pick a representative pred per label so we can inspect
    # its temporal_scope / temporal_op (all cities share the same pred
    # object underneath).
    for i, fc in enumerate(buckets["floor_checks"]):
        pred, bound_city, paradigm_tag = fc
        entity = pred["entity"]
        if entity in ("Day", "Transportation"):
            continue
        scope = pred.get("scope", "any")
        lbl = f"step2[{paradigm_tag}:{entity}.{pred.get('attribute')}:{scope}:idx{i}]"
        label_pred[lbl] = pred
    for c in candidate:
        for lbl, d in per_city.get(c, {}).items():
            if d.get("superseded_by"):
                continue
            if lbl in all_labels or lbl in per_city_labels or \
               lbl in phase_labels or lbl in week_group_labels or \
               lbl in hold_window_labels or lbl in scoped_group_labels or \
               lbl in any_labels:
                continue
            pred = label_pred.get(lbl, {})
            t_scope = pred.get("temporal_scope")
            t_op    = pred.get("temporal_op")
            s_axis  = pred.get("scoped_axis")
            # Priority order: a ScopedPreference Day-axis partition wins
            # over the plain inner scope=all bucket, otherwise a check
            # "always on weekends" would collapse to trip-wide [all] and
            # silently pass when the trip contains no weekend days.
            if s_axis in ("week_group", "travel_phase", "city_index"):
                scoped_group_labels.add(lbl)
            elif d.get("scope") == "all" or d.get("bound"):
                all_labels.add(lbl)
            # Only enforce scope-aware bucketing when the temporal op is
            # existence-style (sometime / always / hold_during /
            # hold_after). Ordering / at-most-once ops don't need a
            # per-group lower bound -- their subjects can be legitimately
            # missing from a scope group.
            elif t_op in _EXISTENCE_TEMPORAL_OPS and t_scope in ("per_city", "per_day"):
                per_city_labels.add(lbl)
            elif t_op in _EXISTENCE_TEMPORAL_OPS and t_scope == "travel_phase":
                phase_labels.add(lbl)
            elif t_op in _EXISTENCE_TEMPORAL_OPS and t_scope == "week_group":
                week_group_labels.add(lbl)
            elif t_op in ("hold_during", "hold_after"):
                hold_window_labels.add(lbl)
            else:
                any_labels.add(lbl)

    # OPTION B: use the tour pre-selected by build_query_db instead of a
    # subset-existence search. Preferences must be feasible on THIS
    # specific ordered tour (largest-pool candidates picked before any
    # preference sampling), not merely on some hypothetical subset.
    selected = list(qdb.get("prehoc_tour") or [])
    if len(selected) < visit_n:
        return False

    # Each selected city must clear every [all]/bound floor check.
    for c in selected:
        for lbl, d in per_city.get(c, {}).items():
            if d.get("superseded_by"):
                continue
            if lbl in all_labels and not d["passes"]:
                return False

    # Per-city temporal existence: each selected city must individually
    # meet the pred's floor (this fixes qid 996-style admissions where
    # a per_city sometime was collapsing to a union).
    for lbl in per_city_labels:
        need = floor_of[lbl]
        for c in selected:
            d = per_city.get(c, {}).get(lbl)
            if d is None or d["affordable_count"] < need:
                return False

    # Travel-phase temporal existence: for each phase present in the
    # tour's day map, the union of affordable_counts across cities in
    # that phase must clear the pred's floor. Uses the qdb day-metadata
    # tied at build_query_db time.
    phase_to_cities = qdb.get("phase_to_cities") or {}
    for lbl in phase_labels:
        need = floor_of[lbl]
        for phase, cities_in_phase in phase_to_cities.items():
            total = 0
            for c in cities_in_phase:
                d = per_city.get(c, {}).get(lbl)
                if d is not None:
                    total += d["affordable_count"]
            if total < need:
                return False

    # Week-group temporal existence: same as phase but keyed on
    # weekday / weekend.
    week_group_to_cities = qdb.get("week_group_to_cities") or {}
    for lbl in week_group_labels:
        need = floor_of[lbl]
        for wg, cities_in_wg in week_group_to_cities.items():
            total = 0
            for c in cities_in_wg:
                d = per_city.get(c, {}).get(lbl)
                if d is not None:
                    total += d["affordable_count"]
            if total < need:
                return False

    # hold_during / hold_after: each city visited during the absolute
    # day-index window must individually clear the pred's floor.
    day_to_city = qdb.get("day_to_city") or {}
    for lbl in hold_window_labels:
        need = floor_of[lbl]
        pred = label_pred.get(lbl, {})
        ts = pred.get("hold_time_start")
        te = pred.get("hold_time_end")
        if ts is None:
            continue
        # hold_after has no time_end; window runs to the last plan day.
        if te is None:
            te = max(day_to_city.keys()) if day_to_city else ts
        window_cities: list[str] = []
        for day_num in range(int(ts), int(te) + 1):
            c = day_to_city.get(day_num)
            if c is not None and c not in window_cities:
                window_cities.append(c)
        for c in window_cities:
            d = per_city.get(c, {}).get(lbl)
            if d is None or d["affordable_count"] < need:
                return False

    # ScopedPreference partition-scope: restrict the check to cities
    # that have at least one day matching the scope filter's value(s).
    # Uses qdb.week_group_to_cities / phase_to_cities for week_group /
    # travel_phase, and the tour position for city_index.
    week_group_to_cities = qdb.get("week_group_to_cities") or {}
    phase_to_cities      = qdb.get("phase_to_cities") or {}
    for lbl in scoped_group_labels:
        need = floor_of[lbl]
        pred = label_pred.get(lbl, {})
        inner_scope = pred.get("scope", "any")
        axis = pred.get("scoped_axis")
        vals = pred.get("scoped_values") or []
        group_cities: list[str] = []
        if axis == "week_group":
            for v in vals:
                for c in week_group_to_cities.get(v, []) or []:
                    if c not in group_cities:
                        group_cities.append(c)
        elif axis == "travel_phase":
            for v in vals:
                for c in phase_to_cities.get(v, []) or []:
                    if c not in group_cities:
                        group_cities.append(c)
        elif axis == "city_index":
            for v in vals:
                try:
                    idx = int(v)
                except (TypeError, ValueError):
                    continue
                if 1 <= idx <= len(selected):
                    c = selected[idx - 1]
                    if c not in group_cities:
                        group_cities.append(c)
        else:
            group_cities = list(selected)
        # Restrict to cities that are actually part of the selected tour.
        group_cities = [c for c in group_cities if c in selected]
        # Empty scope -> the preference cannot be exercised on this plan
        # at all. Reject regardless of inner any/all: a preference that
        # cannot be tested contributes no signal to the benchmark.
        if not group_cities:
            return False
        if inner_scope == "all":
            # Universal: every scope city individually must clear the floor.
            for c in group_cities:
                d = per_city.get(c, {}).get(lbl)
                if d is None or not d.get("passes"):
                    return False
        else:
            # Existential: union across scope cities clears the floor.
            total = 0
            for c in group_cities:
                d = per_city.get(c, {}).get(lbl)
                if d is not None:
                    total += d["affordable_count"]
            if total < need:
                return False

    # Union of [any]-scope checks across the selected cities must clear
    # every [any] floor.
    for lbl in any_labels:
        need = floor_of[lbl]
        total = 0
        for c in selected:
            d = per_city.get(c, {}).get(lbl)
            if d is not None:
                total += d["affordable_count"]
        if total < need:
            return False

    # ---- Step 3: flight feasibility against the selected subset ----
    if buckets["flight_demanding_all"] or buckets["flight_demanding_any"]:
        if not qdb.get("flight_only_feasible"):
            return False
        if buckets["flight_demanding_all"]:
            floor = PER_CITY_ALL_FLOOR["Transportation"]
            for c in selected:
                if qdb["flight_pool_by_city"].get(c, 0) < floor:
                    return False
        # flight_demanding_any: flight_only_feasible alone is sufficient.

    # ---- Step 3b: Transport-cost affordability on the pinned tour ----
    # The picker only guarantees a cheapest-mode leg exists; the total
    # may still blow the transport budget when a required leg has no
    # flight and self-driving is banned (e.g. cross-country taxi). Under
    # budget escalation the cap scales with the escalated budget.
    tc = qdb.get("prehoc_transport_cost")
    budget = qdb.get("heuristics", {}).get("_budget")
    if tc is not None and budget:
        cap = budget * TRANSPORT_BUDGET_CAP_FRAC
        if tc > cap:
            return False

    # ---- Step 4: Transportation time-window feasibility ----
    # Each pred on Transportation.{arrival_time, departure_time} carries a
    # time category (morning / afternoon / evening / night). Enforce
    # against the actual flight_records for the pinned tour's seg-date
    # legs. Semantics:
    #   scope == "all" : EVERY leg must have at least one flight with a
    #                    time in the requested category. Under a Composite
    #                    AND(mode==Flight, time in [...]), this makes the
    #                    "all flights are in the category" claim reachable.
    #   scope == "any" : at least ONE leg has a flight matching. Standard
    #                    existence semantics for temporal/existence ops
    #                    like within/atmost_once/sometime referencing time.
    time_window_checks = buckets.get("time_window_checks") or []
    if time_window_checks:
        legs = _tour_flight_leg_records(qdb)
        if not legs:
            return False
        for pred in time_window_checks:
            if pred.get("scope") == "all":
                for leg in legs:
                    if not any(predicate_holds(r, "Transportation",
                                               pred["attribute"],
                                               pred["op"], pred["value"])
                               for r in leg):
                        return False
            else:
                if not any(predicate_holds(r, "Transportation",
                                           pred["attribute"],
                                           pred["op"], pred["value"])
                           for leg in legs for r in leg):
                    return False

    return True


def qdb_with_budget(qdb: dict[str, Any],
                    new_budget: float) -> dict[str, Any]:
    """Return a shallow-copy qdb whose heuristics reflect `new_budget`.
    Everything else (pools, stats, flight feasibility) is budget-
    independent and shared with the original qdb."""
    out = dict(qdb)
    base = qdb["heuristics"]
    # Rebuild only the cost-relative heuristics; they all scale linearly
    # with the budget under the current 25/40/30/5 split.
    cur_budget = base.get("_budget", None)
    if cur_budget is None:
        # Reconstruct the original budget from any of the heuristics.
        # food_per_meal = (0.25 * B) / (days * 3 * people) -> B = h * days * 3 * people / 0.25
        # We don't have days/people here; the caller must supply new_budget
        # as an absolute amount, and we re-scale proportionally.
        out["heuristics"] = {k: (v if not isinstance(v, (int, float)) else v * 1.0)
                             for k, v in base.items()}
    scale = None
    if "_budget" in base:
        scale = float(new_budget) / float(base["_budget"])
    out_h = {k: (v * scale if scale is not None and isinstance(v, (int, float))
                 and not isinstance(v, bool) and k != "_budget" else v)
             for k, v in base.items()}
    out_h["_budget"] = float(new_budget)
    out["heuristics"] = out_h
    return out


def combined_pool_ok_with_budget_adjust(
        specs: list[tuple[str, dict[str, Any]]],
        qdb: dict[str, Any],
        cap_mult: float = BUDGET_ADJUST_CAP
        ) -> tuple[bool, float, dict[str, Any]]:
    """Try combined_pool_ok at the current budget; if it fails, escalate
    the budget toward `cap_mult` × original and binary-search for the
    smallest multiplier in [1.0, cap_mult] that passes.

    Returns (passed, applied_mult, qdb_to_use).  When passed is False the
    final two fields are (1.0, original qdb)."""
    if combined_pool_ok(specs, qdb):
        return (True, 1.0, qdb)
    if cap_mult <= 1.0:
        return (False, 1.0, qdb)
    base_budget = qdb["heuristics"].get("_budget")
    if base_budget is None:
        return (False, 1.0, qdb)
    cap_qdb = qdb_with_budget(qdb, base_budget * cap_mult)
    if not combined_pool_ok(specs, cap_qdb):
        return (False, 1.0, qdb)
    # Binary-search the smallest multiplier in (1.0, cap_mult] that passes.
    lo, hi = 1.0, cap_mult
    for _ in range(8):
        mid = (lo + hi) / 2
        if combined_pool_ok(specs, qdb_with_budget(qdb, base_budget * mid)):
            hi = mid
        else:
            lo = mid
    return (True, hi, qdb_with_budget(qdb, base_budget * hi))


# --------------------------------------------------------------------------- #
# Pair feasibility + non-vacuity + competing classification                   #
# --------------------------------------------------------------------------- #

def pair_joint_feasible(t1: dict[str, Any], paradigm1: str,
                        t2: dict[str, Any], paradigm2: str,
                        qdb: dict[str, Any]) -> bool:
    p1_all = _all_scope_preds(paradigm1, t1)
    p2_all = _all_scope_preds(paradigm2, t2)
    p1_ex = _existence_preds(paradigm1, t1)
    p2_ex = _existence_preds(paradigm2, t2)

    by_entity_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in p1_all + p2_all:
        by_entity_all[p["entity"]].append(p)
    for entity, plist in by_entity_all.items():
        if not entity_joint_feasible(entity, plist, qdb):
            return False

    # Apply joint [all] reductions and check existentials inside that pool.
    for p in p1_ex + p2_ex:
        e = p["entity"]
        if e in ("Day", "Transportation"):
            if not predicate_feasible_alone(p, qdb):
                return False
            continue
        pool = query_pool(e, qdb)
        for ap in by_entity_all.get(e, []):
            pool = [it for it in pool
                    if predicate_holds(it, e, ap["attribute"], ap["op"], ap["value"])]
        if not any(predicate_holds(it, e, p["attribute"], p["op"], p["value"])
                   for it in pool):
            return False
    return True


def _is_active_for_classification(p: dict[str, Any]) -> bool:
    """Predicates that exert an active constraint contributing to tension.
    Lex fallbacks (lex_1, lex_2, ...) are excluded; scope filters are
    slice selectors, not value constraints."""
    role = p.get("role", "")
    if role.startswith("lex_") and role != "lex_0":
        return False
    if role in ("filter", "inner_filter"):
        return False
    return True


def _is_numerical_attribute(entity: str, attribute: str) -> bool:
    return (entity, attribute) in NUMERIC_ROUND


def _value_sets_overlap(v1: Any, v2: Any) -> bool:
    """For categorical predicates only: do the value sets share any element?
    If either side is unresolved (None), we cannot prove non-overlap, so we
    conservatively say the sets do NOT overlap (i.e. trigger no false
    competing tag from missing data)."""
    if v1 is None or v2 is None:
        return False
    s1 = set(v1) if isinstance(v1, list) else {v1}
    s2 = set(v2) if isinstance(v2, list) else {v2}
    return bool(s1 & s2)


def classify_overlap(t1: dict[str, Any], paradigm1: str,
                     t2: dict[str, Any], paradigm2: str) -> str:
    """Return 'competing' if the active predicates from the two
    preferences exert opposing tension on a shared (entity, attribute),
    else 'non_competing'.

    Per prefertrip.tex, competing requires a *non-trivial* trade-off:
      - Numerical attribute (cost/rating): opposing ops (>= vs <=) or
        numeric direction vs opposing explicit bound creates real
        planner allocation tension regardless of specific values.
      - Categorical attribute (category/cuisine/etc.): opposing ops
        (in vs not_in, == vs !=) only matter when the VALUE SETS
        OVERLAP -- disjoint sets coexist trivially with no planner
        trade-off.
      - Compensatory or Numeric paradigm inherently couples a numerical
        axis with an opposing demand; if its NUMERICAL axis is shared
        with the other preference, classify as competing.
    """
    preds1 = [p for p in extract_atomic_preds(paradigm1, t1)
              if _is_active_for_classification(p)]
    preds2 = [p for p in extract_atomic_preds(paradigm2, t2)
              if _is_active_for_classification(p)]
    shared_attrs: set[tuple[str, str]] = set()
    for p1 in preds1:
        for p2 in preds2:
            if (p1["entity"], p1["attribute"]) != (p2["entity"], p2["attribute"]):
                continue
            shared_attrs.add((p1["entity"], p1["attribute"]))
            numerical = _is_numerical_attribute(p1["entity"], p1["attribute"])
            d1 = p1.get("direction"); d2 = p2.get("direction")
            # Direction vs explicit opposing bound -> competing.
            if d1 == "max" and p2["op"] == "<=":
                return "competing"
            if d1 == "min" and p2["op"] == ">=":
                return "competing"
            if d2 == "max" and p1["op"] == "<=":
                return "competing"
            if d2 == "min" and p1["op"] == ">=":
                return "competing"
            # Two Numerics with opposing directions -> competing.
            if d1 and d2 and d1 != d2:
                return "competing"
            # Opposing ops -> competing for numerical, or for categorical
            # only when the value sets actually overlap.
            if _opposing(p1["op"], p2["op"]):
                if numerical:
                    return "competing"
                if _value_sets_overlap(p1.get("value"), p2.get("value")):
                    return "competing"
    # Compensatory has an internal primary <-> secondary tension; pairing
    # with another preference on the SECONDARY's axis externalises that
    # tension regardless of operator alignment (planner must allocate
    # items balancing both the primary's demand and the secondary's
    # compensating bound).  Pairing on the primary/margin axis is just a
    # reinforcing constraint, so we do NOT tag it competing on that
    # basis alone.
    for paradigm, template in ((paradigm1, t1), (paradigm2, t2)):
        if paradigm != "CompensatoryPreference":
            continue
        sec = template.get("secondary_ap") or {}
        e, a = _atomic_ea(sec)
        if e and a and (e, a) in shared_attrs and _is_numerical_attribute(e, a):
            return "competing"
    return "non_competing"


# --------------------------------------------------------------------------- #
# Pair-aware value adjustment + value-level non-vacuity                       #
# --------------------------------------------------------------------------- #

def _available_categorical_values(entity: str, attribute: str,
                                  qdb: dict[str, Any]) -> set[Any]:
    """Distinct categorical values observed in the entity records."""
    out: set[Any] = set()
    for it in entity_records(entity, qdb):
        v = get_item_attr(it, entity, attribute)
        if isinstance(v, list):
            out.update(v)
        elif v is not None:
            out.add(v)
    return out


def _sample_outside(entity: str, attribute: str, exclude: set[Any], n: int,
                    qdb: dict[str, Any], rng: random.Random | None = None) -> set[Any] | None:
    """Sample n distinct categorical values from those NOT in `exclude`."""
    avail = _available_categorical_values(entity, attribute, qdb) - exclude
    if not avail:
        return None
    n = max(1, min(n, len(avail)))
    items = sorted(avail)
    if rng is not None:
        rng.shuffle(items)
    return set(items[:n])


def _sample_intersection(entity: str, attribute: str, allowed: set[Any], n: int,
                         qdb: dict[str, Any], rng: random.Random | None = None) -> set[Any] | None:
    """Sample n distinct categorical values from those IN `allowed` AND present in DB."""
    avail = _available_categorical_values(entity, attribute, qdb) & allowed
    if not avail:
        return None
    n = max(1, min(n, len(avail)))
    items = sorted(avail)
    if rng is not None:
        rng.shuffle(items)
    return set(items[:n])


def _set_atomic_node_value(node: dict[str, Any], value: Any) -> None:
    """Write `value` into an atomic-like node regardless of nesting style."""
    if "template" in node and isinstance(node["template"], dict):
        node["template"]["value"] = value
    else:
        node["value"] = value


def _set_value_by_role(template: dict[str, Any], paradigm: str, role: str,
                       value: Any) -> bool:
    """Apply an adjusted value back into `template` at the slot identified
    by `role`. Returns True on success.  Composite is intentionally not
    supported (composite is the rigid side in our pair pipeline)."""
    if paradigm == "AtomicPreference" and role == "main":
        template["value"] = value
        return True
    if paradigm == "NumericPreference" and role == "numeric":
        template["threshold"] = value  # value is [lo, hi]
        return True
    if paradigm == "ConditionalPreference":
        if role == "condition":
            _set_atomic_node_value(template["condition"], value)
            return True
        if role == "then":
            _set_atomic_node_value(template["then_pref"], value)
            return True
    if paradigm == "LexicographicPreference" and role.startswith("lex_"):
        try:
            i = int(role[4:])
        except ValueError:
            return False
        prefs = template.get("preferences", [])
        if 0 <= i < len(prefs):
            _set_atomic_node_value(prefs[i], value)
            return True
    if paradigm == "CompensatoryPreference":
        for slot, r in (("primary_ap", "primary"),
                        ("margin_ap", "margin"),
                        ("secondary_ap", "secondary")):
            if role == r and template.get(slot) is not None:
                _set_atomic_node_value(template[slot], value)
                return True
    if paradigm == "TemporalPreference":
        if role == "subject" and template.get("subject_ap") is not None:
            _set_atomic_node_value(template["subject_ap"], value)
            return True
        if role == "reference" and template.get("reference_ap") is not None:
            _set_atomic_node_value(template["reference_ap"], value)
            return True
    if paradigm == "ScopedPreference":
        inner = template.get("inner") or {}
        inner_cls = inner.get("class")
        inner_t = inner.get("template")
        if role.startswith("inner_") and inner_cls and inner_t is not None:
            inner_role = role[6:]
            return _set_value_by_role(inner_t, inner_cls, inner_role, value)
        if role == "filter":
            filters = template.get("scope_filters", []) or []
            if filters:
                _set_atomic_node_value(filters[0], value)
                return True
    return False


def _round_pair_value(entity: str, attribute: str, op: str, target: float,
                      qdb: dict[str, Any]) -> Any | None:
    """Round the pair-aware target value.  Pool-feasibility is enforced
    later by `combined_pool_ok` on the joint filter."""
    return _round_value(entity, attribute, target)


def _adjust_numeric_atomic_pred(rigid_list: list[dict[str, Any]],
                                ap: dict[str, Any], t2: dict[str, Any],
                                paradigm2: str, qdb: dict[str, Any]) -> bool:
    """Tighten the adjustable scalar value `ap.value` so it stays within
    TENSION_TOL of the matching rigid value(s)."""
    entity, attribute = ap["entity"], ap["attribute"]
    ea = (entity, attribute)
    tol = TENSION_TOL.get(ea)
    if tol is None:
        return True
    v_adj = ap["value"]
    if not isinstance(v_adj, (int, float)) or isinstance(v_adj, bool):
        return True
    op_adj = ap["op"]
    candidate_targets: list[float] = []
    for rp in rigid_list:
        v_rigid = rp["value"]
        if not isinstance(v_rigid, (int, float)) or isinstance(v_rigid, bool):
            continue
        op_rigid = rp["op"]
        # Same direction tightening
        if op_rigid == "<=" and op_adj == "<=":
            candidate_targets.append(float(v_rigid) - tol)
        elif op_rigid == ">=" and op_adj == ">=":
            candidate_targets.append(float(v_rigid) + tol)
        # Opposing -> align within tolerance below/above the rigid threshold
        elif op_rigid == "<=" and op_adj == ">=":
            candidate_targets.append(float(v_rigid) - tol)
        elif op_rigid == ">=" and op_adj == "<=":
            candidate_targets.append(float(v_rigid) + tol)
        # == / != on numeric: keep within tolerance
        elif op_rigid == "==" and op_adj in ("==", "!="):
            candidate_targets.append(float(v_rigid))
    if not candidate_targets:
        return True
    # For <= keep the tightest (smallest); for >= keep the highest (tightest).
    if op_adj == "<=":
        target = min(candidate_targets)
    elif op_adj == ">=":
        target = max(candidate_targets)
    else:
        target = candidate_targets[0]
    # Clamp to the per-query envelope so the adjusted value doesn't slip
    # below the affordability floor.
    if (entity, attribute) in COST_ATTRS:
        env = _cost_envelope(entity, attribute,
                             ap.get("scope", "any"), qdb)
        if env is not None:
            env_lo, env_hi = env
            target = max(env_lo, min(env_hi, target))
    rounded = _round_pair_value(entity, attribute, op_adj, target, qdb)
    if rounded is None:
        return False
    # Rounding to the cost step (5 or 10) can drift below the floor by up
    # to step/2; bump up to the next step boundary if so.
    if (entity, attribute) in COST_ATTRS:
        env = _cost_envelope(entity, attribute,
                             ap.get("scope", "any"), qdb)
        if env is not None and isinstance(rounded, (int, float)) and rounded < env[0]:
            _, step = NUMERIC_ROUND.get((entity, attribute), (None, 1))
            step = step or 1
            rounded = int(math.ceil(env[0] / step) * step)
    # Cap-trivial guard: if the adjusted value saturates the attribute's
    # natural rating bound (e.g. Accommodation.rating <= 5), the adjusted
    # predicate becomes vacuously satisfied by every item -- reject so the
    # caller resamples a different pair.
    if _is_rating_cap_trivial(entity, attribute, op_adj, rounded):
        return False
    return _set_value_by_role(t2, paradigm2, ap["role"], rounded)


def _adjust_numeric_paradigm_threshold(rigid_list: list[dict[str, Any]],
                                       t2: dict[str, Any],
                                       ea: tuple[str, str]) -> bool:
    """For a NumericPreference at p2, shape `threshold = [lo, hi]` so the
    aggregate direction can actually contest the rigid side."""
    entity, attribute = ea
    tol = TENSION_TOL.get(ea)
    if tol is None:
        return True
    th = t2.get("threshold")
    if not (isinstance(th, list) and len(th) == 2):
        return True
    lo, hi = float(th[0]), float(th[1])
    direction = t2.get("direction")
    for rp in rigid_list:
        v_rigid = rp["value"]
        if not isinstance(v_rigid, (int, float)) or isinstance(v_rigid, bool):
            continue
        op_rigid = rp["op"]
        if direction == "max":
            if op_rigid == "<=":
                # Want hi just above v_rigid so the numeric's pursuit hits the cap.
                hi = max(hi, float(v_rigid) + tol)
                lo = max(lo, float(v_rigid) - tol)
            elif op_rigid == ">=":
                hi = max(hi, float(v_rigid) + tol)
        elif direction == "min":
            if op_rigid == ">=":
                lo = min(lo, float(v_rigid) - tol)
                hi = min(hi, float(v_rigid) + tol)
            elif op_rigid == "<=":
                lo = min(lo, float(v_rigid) - tol)
    if hi < lo:
        hi = lo
    lo_r = _round_value(entity, attribute, lo)
    hi_r = _round_value(entity, attribute, hi)
    t2["threshold"] = [lo_r, hi_r]
    return True


def _adjust_categorical_pred(rigid_list: list[dict[str, Any]],
                             ap: dict[str, Any], t2: dict[str, Any],
                             paradigm2: str, qdb: dict[str, Any],
                             rng: random.Random | None = None) -> bool:
    """Set-based inter-preference adjustment on shared (entity, attr).

    Rules (rigid = t1, adjustable = t2):
      * Opposing ops (`in` vs `not_in`, `==` vs `!=`): enforce MUTUAL
        EXCLUSION of the two value sets -- there must be no value
        appearing in both (otherwise one side's claim of "include X"
        directly contradicts the other side's "exclude X").  Trim the
        adjustable side by removing the offending values; if the adj
        side becomes empty, reject the pair.
      * Synergistic ops (both `in`, both `not_in`): overlap is fine and
        usually desirable; only enforce non-empty under the narrow case
        where rigid is [all] and adj is [any] on `in` (then adj must
        share at least one value with rigid)."""
    entity, attribute = ap["entity"], ap["attribute"]
    v_adj = ap["value"]
    if not isinstance(v_adj, (list, str)):
        return True
    is_list = isinstance(v_adj, list)
    s_adj = set(v_adj) if is_list else {v_adj}
    op_adj = ap["op"]

    for rp in rigid_list:
        v_rigid = rp["value"]
        if not isinstance(v_rigid, (list, str)):
            continue
        s_rigid = set(v_rigid) if isinstance(v_rigid, list) else {v_rigid}
        op_rigid = rp["op"]

        # ---- Opposing set ops: enforce mutual exclusion. ----
        if (op_rigid in ("in", "==") and op_adj in ("not_in", "!=")) or \
           (op_rigid in ("not_in", "!=") and op_adj in ("in", "==")):
            # Trim the adjustable side of any value that appears on the
            # rigid side (the contradiction).
            trimmed = s_adj - s_rigid
            if not trimmed:
                # If adjustable is the `in/==` side, try to sample fresh
                # values from outside the rigid set so the adj still has
                # content.
                if op_adj in ("in", "=="):
                    replacement = _sample_outside(entity, attribute, s_rigid,
                                                   max(1, len(s_adj)), qdb, rng)
                    if not replacement:
                        return False
                    s_adj = replacement
                else:
                    # `not_in/!=` side fully forbidden by rigid -- empty
                    # not_in trivially holds, but it's also vacuous.
                    return False
            else:
                s_adj = trimmed

        # ---- Synergistic in vs in: overlap allowed; ensure non-empty
        # intersection when rigid=[all] AND adj=[any]. ----
        elif op_rigid == "in" and op_adj == "in":
            if rp.get("scope") == "all" and ap.get("scope") == "any":
                if not (s_adj & s_rigid):
                    replacement = _sample_intersection(entity, attribute, s_rigid,
                                                       max(1, len(s_adj)), qdb, rng)
                    if not replacement:
                        return False
                    s_adj = replacement

        # ---- `==` vs `==`: same value or trivially incompatible. ----
        elif op_rigid == "==" and op_adj == "==":
            if v_rigid != v_adj:
                s_adj = {v_rigid}

    if not s_adj:
        return False
    new_value = sorted(s_adj) if is_list else next(iter(s_adj))
    return _set_value_by_role(t2, paradigm2, ap["role"], new_value)


def _pair_adjust(t1: dict[str, Any], paradigm1: str,
                 t2: dict[str, Any], paradigm2: str,
                 qdb: dict[str, Any],
                 rng: random.Random | None = None) -> bool:
    """Adjust t2's adjustable predicate values so shared-attr predicates
    maintain genuine tension within TENSION_TOL of t1's rigid values.
    Returns True if the adjustment succeeded (or wasn't needed), False if
    the pair is irreconcilable."""
    preds1 = [p for p in extract_atomic_preds(paradigm1, t1)
              if _is_active_for_classification(p)]
    preds2 = [p for p in extract_atomic_preds(paradigm2, t2)
              if _is_active_for_classification(p)]
    rigid_by_attr: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in preds1:
        rigid_by_attr[(p["entity"], p["attribute"])].append(p)

    # 1. Numeric paradigm threshold first (handled once per shared attr).
    if paradigm2 == "NumericPreference":
        for ea, rigid_list in rigid_by_attr.items():
            num_preds = [p for p in preds2
                         if (p["entity"], p["attribute"]) == ea
                         and p.get("role") == "numeric"]
            if not num_preds:
                continue
            if not _adjust_numeric_paradigm_threshold(rigid_list, t2, ea):
                return False

    # 2. Per-pred adjustments for atomic-like values.
    for ap in preds2:
        ea = (ap["entity"], ap["attribute"])
        rigid_list = rigid_by_attr.get(ea)
        if not rigid_list:
            continue
        if ap.get("role") == "numeric":
            continue  # handled above
        if _is_numerical_attribute(*ea):
            if not _adjust_numeric_atomic_pred(rigid_list, ap, t2, paradigm2, qdb):
                return False
        else:
            if not _adjust_categorical_pred(rigid_list, ap, t2, paradigm2, qdb, rng):
                return False

    # 3. Compensatory sibling-distinctness enforcement.
    # When primary_ap and margin_ap share the same (entity, attribute) and
    # both had to be re-sampled from the rigid set above, they can land on
    # the SAME categorical value -- collapsing the tradeoff. Post-fix:
    # if margin's value set equals primary's, re-sample margin from the
    # rigid intersection minus primary's set. Reject if no distinct
    # value is available.
    if paradigm2 == "CompensatoryPreference":
        prim = t2.get("primary_ap"); marg = t2.get("margin_ap")
        pa = _take_atomic(prim) if prim else None
        ma = _take_atomic(marg) if marg else None
        if (pa and ma and (pa[0], pa[1]) == (ma[0], ma[1])
                and not _is_numerical_attribute(pa[0], pa[1])):
            entity, attribute = pa[0], pa[1]
            pv = _read_value(prim); mv = _read_value(marg)
            def _to_set(v: Any) -> set:
                if isinstance(v, list): return set(v)
                if v is None: return set()
                return {v}
            p_set, m_set = _to_set(pv), _to_set(mv)
            if p_set and m_set and p_set == m_set:
                # Constrain replacement so it (a) differs from primary
                # and (b) doesn't reintroduce any value the paired
                # predicate excludes. Two branches, mirroring
                # _adjust_categorical_pred:
                #   - rigid has `in`  -> sample from (rigid_in_union - p_set)
                #   - rigid has `not_in` -> sample from (available - p_set
                #     - rigid_not_in_union) so we can't land on a value
                #     the rigid predicate forbids (e.g. rigid says
                #     `not_in {V2}` and the pool is {V1, V2} with primary
                #     already on V1 -- picking V2 would directly
                #     contradict the rigid predicate).
                rigid_list = rigid_by_attr.get((entity, attribute)) or []
                rigid_in_union: set = set()
                rigid_not_in_union: set = set()
                for rp in rigid_list:
                    rv = rp.get("value")
                    vals = rv if isinstance(rv, list) else ([rv] if rv is not None else [])
                    if rp.get("op") == "in":
                        rigid_in_union.update(vals)
                    elif rp.get("op") == "not_in":
                        rigid_not_in_union.update(vals)
                if rigid_in_union:
                    allowed = rigid_in_union - p_set - rigid_not_in_union
                    replacement = _sample_intersection(entity, attribute,
                                                       allowed,
                                                       max(1, len(m_set)),
                                                       qdb, rng)
                else:
                    replacement = _sample_outside(entity, attribute,
                                                   p_set | rigid_not_in_union,
                                                   max(1, len(m_set)), qdb, rng)
                if not replacement:
                    return False
                new_val = (sorted(replacement) if isinstance(mv, list)
                           else next(iter(replacement)))
                if not _set_value_by_role(t2, paradigm2, "margin", new_val):
                    return False
    return True


def pair_genuinely_active(t1: dict[str, Any], paradigm1: str,
                          t2: dict[str, Any], paradigm2: str,
                          qdb: dict[str, Any]) -> bool:
    """Value-level non-vacuity: for each existence-style predicate whose
    role is in AUTO_SATISFIED_CHECK_ROLES, the joint [all]-scope reduced
    pool must contain BOTH some items satisfying the predicate (existence)
    AND some items violating it (the predicate is not auto-satisfied)."""
    p1_all = _all_scope_preds(paradigm1, t1)
    p2_all = _all_scope_preds(paradigm2, t2)
    by_entity_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in p1_all + p2_all:
        by_entity_all[p["entity"]].append(p)

    def check_side(paradigm: str, template: dict[str, Any]) -> bool:
        ex_preds = _existence_preds(paradigm, template)
        # Skip temporal modalities whose semantics permit/require auto-satisfied subjects.
        skip_temporal_subject = (
            paradigm == "TemporalPreference"
            and template.get("op") in {"atmost_once", "hold_during",
                                       "hold_after", "within",
                                       "sometime_before", "sometime_after",
                                       "always_within"})
        for p in ex_preds:
            role = p.get("role", "")
            if role not in AUTO_SATISFIED_CHECK_ROLES:
                continue
            if skip_temporal_subject and role in ("subject", "reference"):
                continue
            entity = p["entity"]
            if entity in ("Day", "Transportation"):
                continue
            pool = query_pool(entity, qdb)
            for ap in by_entity_all.get(entity, []):
                pool = [it for it in pool
                        if predicate_holds(it, entity, ap["attribute"],
                                           ap["op"], ap["value"])]
            if not pool:
                return False
            sat = sum(1 for it in pool
                      if predicate_holds(it, entity, p["attribute"],
                                         p["op"], p["value"]))
            # Existence (>=1) and active (<all) constraint.
            if sat == 0 or sat >= len(pool):
                return False
        return True

    return check_side(paradigm1, t1) and check_side(paradigm2, t2)


# --------------------------------------------------------------------------- #
# Secondary cost-feasibility (rating / cuisine / house_rules / room_type)    #
# --------------------------------------------------------------------------- #

# Attributes that don't themselves carry cost but can constrain the entity
# pool such that the AFFORDABLE subset is destroyed (cost-rating
# correlation, premium-cuisine pools, restrictive room types, etc.).
SECONDARY_FILTER_ATTRS = {
    ("Accommodation", "rating"),
    ("Accommodation", "house_rules"),
    ("Accommodation", "room_type"),
    ("Restaurant",    "rating"),
    ("Restaurant",    "cuisine"),
}


def _secondary_predicate_passes(entity: str, attribute: str, op: str,
                                value: Any, scope: str,
                                qdb: dict[str, Any],
                                paradigm: str | None = None) -> bool:
    """Filter the entity pool by the predicate and check that at least one
    surviving item has cost within the per-query budget envelope ceiling.

    For scope='all' we use the tighter alpha_all (the whole plan would be
    drawn from this pool, so even the cheapest must leave headroom).
    For scope='any' we use alpha_any (only one item needs to fit).
    Compensatory [any] uses a wider ceiling (one item per day, up to 3 days)."""
    if (entity, attribute) not in SECONDARY_FILTER_ATTRS:
        return True
    if value is None:
        return True
    pool = entity_records(entity, qdb)
    if not pool:
        return True
    filtered = [it for it in pool
                if predicate_holds(it, entity, attribute, op, value)]
    if not filtered:
        return False  # nothing survives the secondary filter itself
    h = heuristic_for(entity, "cost", qdb["heuristics"])
    if h is None or h <= 0:
        return True  # no cost dimension to check (defensive)
    ceiling = _envelope_ceil_mult(scope, paradigm) * h
    return any((it.get("cost") is not None and it["cost"] <= ceiling)
               for it in filtered)


def secondary_budget_feasible(entry: dict[str, Any], template: dict[str, Any],
                              qdb: dict[str, Any]) -> bool:
    """For every secondary-filter predicate in this preference's template,
    verify that the predicate-filtered entity pool contains at least one
    cost-affordable item (cost <= alpha * heuristic).  Returns True if all
    predicates pass, False if any does not (caller rejects + resamples)."""
    paradigm = entry["_paradigm"]
    for p in extract_atomic_preds(paradigm, template):
        if not _is_active_for_classification(p):
            continue
        # Numeric paradigm's "value" is a [lo, hi] aggregation target, not a
        # per-item filter -- it doesn't restrict the entity pool, so skip.
        if p.get("role") == "numeric":
            continue
        ea = (p["entity"], p["attribute"])
        if ea not in SECONDARY_FILTER_ATTRS:
            continue
        if not _secondary_predicate_passes(p["entity"], p["attribute"],
                                           p["op"], p["value"],
                                           p.get("scope", "all"), qdb,
                                           paradigm=paradigm):
            return False
    return True


# --------------------------------------------------------------------------- #
# Output records                                                              #
# --------------------------------------------------------------------------- #

def make_resolved_record(entry: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    # NOTE: the bank's `raw_source` string is intentionally NOT emitted
    # on the resolved record. It refers to the bank template's original
    # unresolved values (e.g. hold_after time_start=6 or a fixed cuisine
    # set) and does not reflect post-resolution / constraint-filter /
    # quartile-grounding adjustments. Downstream consumers should use
    # `template` (the resolved values) as the source of truth, and the
    # `nl_render_log[i].resolved_source` as its stringified form.
    rec: dict[str, Any] = {
        "paradigm":   entry["_paradigm"],
        "bank_id":    entry["id"],
        "trace":      trace(entry),
        "template":   template,
        "rationale":  entry.get("rationale", ""),
    }
    if entry.get("_subtype"):
        rec["subtype"] = entry["_subtype"]
    return rec


# --------------------------------------------------------------------------- #
# Feasibility metadata (for manual verification of the augmented record)      #
# --------------------------------------------------------------------------- #

def build_feasibility_metadata(
        query: dict[str, Any],
        qdb_emit: dict[str, Any],
        preferences: list[dict[str, Any]],
        flight_records: dict[tuple[str, str], list[dict[str, Any]]],
        distance: dict[tuple[str, str], dict[str, Any]],
        ) -> dict[str, Any]:
    """Build a self-contained feasibility metadata block for the emitted
    record, enabling manual verification.  Includes:
      - the heuristics + envelope + floor constants used for the check;
      - per-candidate-city pool counts (raw and affordable) under the
        combined filter for every floor check;
      - the constraint-filtered pools (accom/rest/attr) for every visit-
        candidate city, plus full flight records and driving distances
        between origin and each city (and inter-city for multi-city);
      - `selected_cities`: a `visit_n`-sized subset whose every floor
        check clears (the planner could realize the trip on these
        cities).  Empty list if no single subset clears every check.
    """
    org = query["org"]
    candidate = qdb_emit["candidate_cities"]
    visit_n = qdb_emit["visiting_city_number"]
    pool_by_city = qdb_emit["pool_by_city"]
    specs = [(p["paradigm"], p["template"]) for p in preferences]
    buckets = _collect_record_preds(specs)

    # ---- Per-candidate-city pass/fail per floor check ----
    per_city_breakdown: dict[str, dict[str, Any]] = {}
    for c in candidate:
        checks: dict[str, Any] = {}
        # Step 1: per-entity joint [all] filter
        for entity, preds in buckets["all_by_entity"].items():
            floor = PER_CITY_ALL_FLOOR.get(entity)
            if floor is None:
                continue
            items = _items_pass_all(pool_by_city.get(entity, {}).get(c, []),
                                    entity, preds)
            aff = _affordable_subset(items, entity, "all", None, qdb_emit)
            checks[f"step1_all[{entity}]"] = {
                "raw_count": len(items),
                "affordable_count": len(aff),
                "floor": floor,
                "passes": len(aff) >= floor,
                "scope": "all",
                "bound": False,
            }
        # Step 2: per-pred floor on combined filter
        for i, fc in enumerate(buckets["floor_checks"]):
            pred, bound_city, paradigm_tag = fc
            entity = pred["entity"]
            if entity in ("Day", "Transportation"):
                continue
            if bound_city is not None and bound_city != c:
                continue  # bound check applies only to its specific city
            all_preds = buckets["all_by_entity"].get(entity, [])
            floor = _floor_for(pred, paradigm_tag)
            scope = pred.get("scope", "any")
            inner_filters = pred.get("inner_filters") or []
            combined = all_preds + list(inner_filters) + [pred]
            items = _items_pass_all(pool_by_city.get(entity, {}).get(c, []),
                                    entity, combined)
            aff = _affordable_subset(items, entity, scope, paradigm_tag, qdb_emit)
            label = f"step2[{paradigm_tag}:{entity}.{pred.get('attribute')}:{scope}:idx{i}]"
            checks[label] = {
                "raw_count": len(items),
                "affordable_count": len(aff),
                "floor": floor,
                "passes": len(aff) >= floor,
                "scope": scope,
                "bound": bound_city is not None,
                # Surface the temporal operator's scope AND the
                # ScopedPreference Day-axis partition scope so downstream
                # audit tools can read the enforced routing (per_city,
                # week_group, travel_phase, hold window, scoped group,
                # non-Day inner filters) from the emitted breakdown itself.
                "temporal_scope": pred.get("temporal_scope"),
                "temporal_op":    pred.get("temporal_op"),
                "scoped_axis":    pred.get("scoped_axis"),
                "scoped_values":  pred.get("scoped_values"),
                "inner_filter_count": len(inner_filters),
            }
        per_city_breakdown[c] = checks

    # ---- OR-composite union check (per-city) ----
    # CompositePreference with op=OR has children whose [all]/[any] scope
    # is meaningfully a UNION semantic across children, not an intersection
    # nor independent per-child clearance.  An accommodation satisfies an
    # OR composite if it matches ANY child; the planner draws per-day items
    # from each visited city's OR-union pool.  Add a per-city `step2_or`
    # entry computed as the union of items matching any OR child, and
    # mark individual child checks as superseded so the selection logic
    # uses the union instead of the per-child counts.
    or_groups: list[tuple[int, str, list[dict[str, Any]]]] = []
    for pi, (paradigm, template) in enumerate(specs):
        if paradigm != "CompositePreference":
            continue
        if template.get("op") != "OR":
            continue
        by_ent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for child in template.get("children", []) or []:
            ent = child.get("entity_type")
            if ent and ent not in ("Day", "Transportation"):
                by_ent[ent].append(child)
        for ent, kids in by_ent.items():
            or_groups.append((pi, ent, kids))

    for pi, ent, kids in or_groups:
        scopes = {k.get("scope", "any") for k in kids}
        or_scope = "all" if "all" in scopes else "any"
        floor = (PER_CITY_ALL_FLOOR.get(ent) if or_scope == "all"
                 else PER_CITY_ANY_FLOOR)
        if floor is None:
            continue
        label = f"step2_or[CompositePreference#{pi}:{ent}:{or_scope}]"
        for c in candidate:
            pool = pool_by_city.get(ent, {}).get(c, [])
            baseline = _items_pass_all(
                pool, ent, buckets["all_by_entity"].get(ent, []))
            seen: set[int] = set()
            union_items: list[dict[str, Any]] = []
            for kid in kids:
                attr = kid.get("attribute")
                op   = kid.get("op")
                val  = kid.get("value")
                for it in baseline:
                    if id(it) in seen:
                        continue
                    if predicate_holds(it, ent, attr, op, val):
                        union_items.append(it)
                        seen.add(id(it))
            aff = _affordable_subset(union_items, ent, or_scope,
                                     "CompositePreference", qdb_emit)
            per_city_breakdown[c][label] = {
                "raw_count":        len(union_items),
                "affordable_count": len(aff),
                "floor":            floor,
                "passes":           len(aff) >= floor,
                "scope":            or_scope,
                "bound":            False,
                "or_union":         True,
            }
        # Mark individual OR children as superseded by the union check so
        # the selection logic ignores their independent pass/fail status.
        kid_keys = {(k.get("entity_type"), k.get("attribute"), k.get("op"))
                    for k in kids}
        for c in candidate:
            for chk_label, d in per_city_breakdown[c].items():
                if d.get("or_union"):
                    continue
                if d.get("superseded_by"):
                    continue
                if not chk_label.startswith("step2["):
                    continue
                try:
                    inner = chk_label[len("step2["):-1]
                    parts = inner.split(":")
                    if len(parts) < 4:
                        continue
                    if parts[0] != "CompositePreference":
                        continue
                    ea = parts[1].split(".")
                    if len(ea) != 2 or ea[0] != ent:
                        continue
                    # Match (entity, attribute, op); op isn't in the label,
                    # so match by entity+attribute and trust that all OR
                    # children on the same EA are part of this group.
                    if (ent, ea[1], None) in {(k[0], k[1], None) for k in kid_keys}:
                        d["superseded_by"] = label
                except Exception:
                    continue

    # Select `visit_n` cities such that:
    #   - Each picked city individually clears every [all]-scope and bound
    #     check applied to it (the planner draws from each city's pool for
    #     [all] demands).
    #   - For each [any]-scope check, the union of affordable counts across
    #     the picked cities clears that check's floor.  An [any]-scope pref
    #     only needs >=1 satisfying item somewhere in the trip plan, so
    #     contributions sum across visited cities.
    # Greedy: drop ineligible cities (those failing any [all]/bound check)
    # first, then pick `visit_n` from the remainder by maximum coverage of
    # uncovered [any]-check floors.
    # OPTION B: selected_cities is the pre-selected tour that was picked
    # in build_query_db (largest-pool candidates that clear PER_CITY_ALL_FLOOR
    # under hard constraints). Preferences were sampled against THIS tour,
    # so no re-selection is required or desired.
    selected_cities: list[str] = list(qdb_emit.get("prehoc_tour") or [])

    # ---- Constraint-filtered pools per candidate city ----
    def flight_block(recs: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize the flight pool for a single leg: total count + price
        stats + the FULL list of flight records (no cap). Records are
        already restricted to a single segmentation date at the caller,
        so the list is small (typically < 100)."""
        if not recs:
            return {"count": 0, "records": [], "price_stats": None}
        prices = sorted(r["price"] for r in recs if r.get("price") is not None)
        n = len(prices)
        stats = {
            "min":    prices[0] if n else None,
            "p25":    prices[max(0, n // 4 - 1)] if n else None,
            "median": prices[n // 2] if n else None,
            "p75":    prices[min(n - 1, 3 * n // 4)] if n else None,
            "max":    prices[-1] if n else None,
        }
        return {"count": len(recs), "price_stats": stats, "records": recs}

    # Restrict flight records to the trip-segmentation dates so the
    # metadata mirrors the flight legs an LLM planner would actually use:
    #   - org -> c on start_date        (leg 1)
    #   - c -> org on end_date          (leg visit_n + 1)
    #   - c_i -> c_j on mid_dates[k]    (intra-state legs, k in 0..visit_n-2)
    seg_start, seg_mid, seg_end = segmentation_leg_dates(query)

    constraint_pools: dict[str, dict[str, Any]] = {}
    pool_counts_per_city: dict[str, dict[str, int]] = {}
    for c in candidate:
        accom = pool_by_city.get("Accommodation", {}).get(c, [])
        rest  = pool_by_city.get("Restaurant",    {}).get(c, [])
        attr  = pool_by_city.get("Attraction",    {}).get(c, [])
        ffo   = ([r for r in flight_records.get((org, c), []) if r.get("date") == seg_start]
                 if seg_start else [])
        fto   = ([r for r in flight_records.get((c, org), []) if r.get("date") == seg_end]
                 if seg_end else [])
        constraint_pools[c] = {
            "accommodation": accom,
            "restaurant":    rest,
            "attraction":    attr,
            "flights_from_origin": flight_block(ffo),
            "flights_to_origin":   flight_block(fto),
            "drive_from_origin":   distance.get((org, c)),
            "drive_to_origin":     distance.get((c, org)),
        }
        pool_counts_per_city[c] = {
            "accommodation":       len(accom),
            "restaurant":          len(rest),
            "attraction":          len(attr),
            "flights_from_origin": len(ffo),
            "flights_to_origin":   len(fto),
        }

    # ---- Inter-city driving (multi-city only) ----
    inter_city: dict[str, Any] = {}
    if visit_n > 1:
        for c1 in candidate:
            for c2 in candidate:
                if c1 == c2:
                    continue
                d = distance.get((c1, c2))
                if d is not None:
                    inter_city[f"{c1}->{c2}"] = d

    # ---- Inter-city flights (only for multi-city trips: visit_n > 1) ----
    inter_city_flights: list[dict[str, Any]] = []
    if visit_n > 1 and seg_mid:
        # Segmentation places an intra-state flight on each of seg_mid
        # dates. For every ordered pair of candidate cities we list the
        # flights on that date so an LLM planner can pick a routing.
        for leg_idx, date in enumerate(seg_mid, start=1):
            for c1 in candidate:
                for c2 in candidate:
                    if c1 == c2:
                        continue
                    recs = [r for r in flight_records.get((c1, c2), [])
                            if r.get("date") == date]
                    if not recs:
                        continue
                    inter_city_flights.append({
                        "leg_index": leg_idx,     # 1-based, position in mid segmentation
                        "leg_date":  date,
                        "origin":    c1,
                        "dest":      c2,
                        **flight_block(recs),
                    })

    heuristics = {k: v for k, v in qdb_emit["heuristics"].items()
                  if not k.startswith("_")}

    trip_segmentation = {
        "start_date": seg_start,
        "mid_dates":  seg_mid,
        "end_date":   seg_end,
        "n_legs":     (visit_n + 1) if seg_start else 0,
        "leg_pattern": (
            [f"org -> visit_1 on {seg_start}"]
            + [f"visit_{i} -> visit_{i+1} on {d}"
               for i, d in enumerate(seg_mid, start=1)]
            + [f"visit_{visit_n} -> org on {seg_end}"]
        ) if seg_start else [],
    }

    # ---- Ordered visit-tour (plan-ready) ----
    # OPTION B: the pre-selected `prehoc_tour` was ordered in
    # build_query_db already; selected_cities IS that ordered tour.
    modes_allowed_q = qdb_emit["transport_modes"]  # subset of {"Flight","self-driving","taxi"}
    selected_tour = list(selected_cities)

    # ---- solution_information (feasibility-filtered, plan-ready) ----
    # Apply the SAME post-hoc feasibility filter that try_resolve accepted
    # the preference against: [all]-scope preference predicates + envelope
    # affordability under the (budget_multiplier-adjusted) heuristics.
    # Result: every item in the pool passes hard constraints AND every
    # [all]-scope preference AND fits the trip's affordability envelope.
    #   - [any]-scope prefs: NOT applied at item level (their semantics
    #     say "some item across the trip matches"; whole pool is kept).
    #   - Composite / Conditional / Lexicographic / Temporal / Scoped:
    #     evaluated at planning time by the LLM; not applied here.
    solution_pool_source: dict[str, dict[str, list]] = {}
    solution_pool_shortfalls: list[dict[str, Any]] = []
    specs = [(p["paradigm"], p["template"])
             for p in (preferences or [])]
    buckets = _collect_record_preds(specs) if specs else {"all_by_entity": {}, "floor_checks": []}
    entity_key_map = (("accommodation", "Accommodation"),
                      ("restaurant",    "Restaurant"),
                      ("attraction",    "Attraction"))
    for city in selected_tour:
        entity_pools: dict[str, list] = {}
        for ek, entity in entity_key_map:
            base = pool_by_city.get(entity, {}).get(city, [])
            # Only `all_by_entity` predicates are trip-wide UNCONDITIONAL
            # [all]-filters (per _collect_record_preds's contract). Sub-
            # preds nested under Scoped / Conditional / Temporal /
            # Compensatory / Lex-secondary appear in `floor_checks` and
            # apply only within their scope -- they must NOT be treated
            # as pool-wide filters here.
            preds_all = list(buckets["all_by_entity"].get(entity, []))
            if preds_all:
                # This entity is constrained by at least one preference:
                # apply the [all]-scope filter AND the affordability
                # envelope (mirrors combined_pool_ok / try_resolve).
                items = _items_pass_all(base, entity, preds_all)
                aff = _affordable_subset(items, entity, "all", None, qdb_emit)
            else:
                # No preference touches this entity -> keep the
                # constraint-filtered pool unchanged (try_resolve does
                # not affordability-trim entities the prefs don't
                # constrain, so neither do we here).
                aff = base
            entity_pools[ek] = aff
            floor = PER_CITY_ALL_FLOOR.get(entity, 0)
            if len(aff) < floor:
                solution_pool_shortfalls.append({
                    "city": city, "entity": entity,
                    "n_after_filter": len(aff), "floor": floor,
                })
        solution_pool_source[city] = entity_pools

    # Filter flight records for solution: any [all]-scope Transportation
    # predicate acts as a per-record filter. Also drop records above the
    # affordability envelope × current transport heuristic. We ONLY touch
    # the leg pairs that appear in the tour — no global scan.
    transp_all_preds = buckets["all_by_entity"].get("Transportation", [])
    _transp_h = heuristic_for("Transportation", "cost", qdb_emit["heuristics"])
    _transp_ceil = (ENVELOPE_CEIL_ALL * _transp_h) if (_transp_h and _transp_h > 0) else None

    def _flight_passes(rec):
        for p in transp_all_preds:
            attr = p.get("attribute"); op = p.get("op"); val = p.get("value")
            v = rec.get("price") if attr == "cost" else rec.get(attr)
            if v is None: return False
            if op == ">=" and not (v >= val): return False
            if op == "<=" and not (v <= val): return False
            if op == "==" and not (v == val): return False
        if _transp_ceil is not None:
            if rec.get("price") is None or rec["price"] > _transp_ceil:
                return False
        return True

    def _tour_leg_pairs():
        if not selected_tour: return []
        pairs = [(org, selected_tour[0])]
        for i in range(len(selected_tour) - 1):
            pairs.append((selected_tour[i], selected_tour[i + 1]))
        pairs.append((selected_tour[-1], org))
        return pairs

    # Build a filtered flight_records dict that overrides ONLY the tour's
    # leg pairs; other pairs pass through unchanged (they're never emitted).
    solution_flight_records: dict[tuple[str, str], list[dict[str, Any]]] = flight_records
    if transp_all_preds or _transp_ceil is not None:
        _overrides = {}
        for (o, d) in _tour_leg_pairs():
            base = flight_records.get((o, d), [])
            _overrides[(o, d)] = [r for r in base if _flight_passes(r)]
        # ChainMap-like: check overrides first, fall through to flight_records.
        class _FRWrapper(dict):
            def __init__(self, ov, base): self.ov = ov; self.base = base
            def get(self, k, default=None):
                if k in self.ov: return self.ov[k]
                return self.base.get(k, default)
            def __getitem__(self, k):
                if k in self.ov: return self.ov[k]
                return self.base[k]
            def __contains__(self, k): return k in self.ov or k in self.base
        solution_flight_records = _FRWrapper(_overrides, flight_records)

    solution_information = _build_reference_information(
        org=org, selected_tour=selected_tour,
        seg_start=seg_start, seg_mid=seg_mid, seg_end=seg_end,
        pool_source=solution_pool_source,   # feasibility-filtered pools
        flight_records=solution_flight_records,
        distance=distance,
        modes_allowed=modes_allowed_q,
        raw_mode=False,                     # respect local_constraint modes
    )
    # Build a per-city raw-pool projection matching the constraint_pools
    # shape so the same emitter can render an unfiltered variant.
    raw_pool_bc = qdb_emit.get("raw_pool_by_city") or {}
    raw_pools_shaped: dict[str, dict[str, list]] = {}
    for city in selected_tour:
        raw_pools_shaped[city] = {
            "accommodation": raw_pool_bc.get("Accommodation", {}).get(city, []),
            "restaurant":    raw_pool_bc.get("Restaurant",    {}).get(city, []),
            "attraction":    raw_pool_bc.get("Attraction",    {}).get(city, []),
        }
    reference_information = _build_reference_information(
        org=org, selected_tour=selected_tour,
        seg_start=seg_start, seg_mid=seg_mid, seg_end=seg_end,
        pool_source=raw_pools_shaped,   # raw source rows, no filter
        flight_records=flight_records,
        distance=distance,
        modes_allowed=modes_allowed_q,
        raw_mode=True,                  # ignore local_constraint modes
    )

    return {
        "heuristics": heuristics,
        "budget_used": qdb_emit["heuristics"].get("_budget"),
        "envelope_ceil": {
            "all": ENVELOPE_CEIL_ALL,
            "any_default": ENVELOPE_CEIL_ANY,
        },
        "per_city_floor": {
            "Accommodation_all":   PER_CITY_ALL_FLOOR["Accommodation"],
            "Restaurant_all":      PER_CITY_ALL_FLOOR["Restaurant"],
            "Attraction_all":      PER_CITY_ALL_FLOOR["Attraction"],
            "Transportation_all":  PER_CITY_ALL_FLOOR["Transportation"],
            "any_default":         PER_CITY_ANY_FLOOR,
            "any_compensatory":    PER_CITY_ANY_FLOOR_COMPENSATORY,
        },
        "visit_n": visit_n,
        "scope_state": qdb_emit.get("scope_state"),
        "scope_cities": qdb_emit["scope_cities"],
        "candidate_cities": candidate,
        "selected_cities": selected_cities,
        "selected_tour":  selected_tour,
        "modes_allowed":  sorted(modes_allowed_q),
        "flight_only_feasible": qdb_emit.get("flight_only_feasible"),
        "flight_pool_by_city": qdb_emit.get("flight_pool_by_city", {}),
        "pool_counts_per_city": pool_counts_per_city,
        "per_city_check_breakdown": per_city_breakdown,
        "constraint_filtered_pools": constraint_pools,
        "inter_city_driving": inter_city,
        "inter_city_flights": inter_city_flights,
        "trip_segmentation": trip_segmentation,
        # Cheapest-mode-per-leg transport cost estimate for the pinned
        # tour under the query's modes_allowed. The record is admitted
        # only if this fits within `transport_budget_cap`
        # (TRANSPORT_BUDGET_CAP_FRAC × applied_budget); see
        # combined_pool_ok. `transport_budget_cap` here is against the
        # APPLIED budget (after any budget escalation), matching the
        # cap the gate actually enforced.
        "prehoc_transport_cost":  qdb_emit.get("prehoc_transport_cost"),
        "transport_budget_cap":   (qdb_emit["heuristics"].get("_budget", 0)
                                   * TRANSPORT_BUDGET_CAP_FRAC
                                   if qdb_emit.get("heuristics", {}).get("_budget")
                                   else None),
        "transport_budget_frac":  TRANSPORT_BUDGET_CAP_FRAC,
        # Day-level metadata (1-indexed day numbers). Ties each trip day
        # to its city, travel_phase, and week_group -- read by temporal-
        # scope-aware feasibility checks and available for downstream
        # planning validation.
        "day_to_city":          qdb_emit.get("day_to_city", {}),
        "day_to_phase":         qdb_emit.get("day_to_phase", {}),
        "day_to_week_group":    qdb_emit.get("day_to_week_group", {}),
        "phase_to_cities":      qdb_emit.get("phase_to_cities", {}),
        "week_group_to_cities": qdb_emit.get("week_group_to_cities", {}),
        "solution_information":  solution_information,
        "reference_information": reference_information,
        "solution_pool_shortfalls": solution_pool_shortfalls,
    }


# --------------------------------------------------------------------------- #
# Ordered-tour picker + reference_information builder                         #
# --------------------------------------------------------------------------- #

def _has_flight_on(flight_records: dict[tuple[str, str], list[dict[str, Any]]],
                   o: str, d: str, date: str | None) -> bool:
    if not date:
        return False
    return any(r.get("date") == date for r in flight_records.get((o, d), []))


def _pick_ordered_tour(selected_cities: list[str], org: str,
                       seg_start: str | None, seg_mid: list[str], seg_end: str | None,
                       flight_records: dict[tuple[str, str], list[dict[str, Any]]],
                       modes_allowed: set[str]) -> list[str]:
    """Pick an ordered permutation of `selected_cities` that maximizes the
    number of segmentation legs served by the higher-priority transport
    mode. Priority: Flight > self-driving > taxi. Flight only counts as
    served when it exists on that leg's date AND is in `modes_allowed`.
    Deterministic tie-break by lexicographic city order."""
    if not selected_cities:
        return []
    if len(selected_cities) == 1:
        return list(selected_cities)

    flight_ok  = "Flight"        in modes_allowed
    drive_ok   = "self-driving"  in modes_allowed
    # taxi always OK; every leg is at least taxi-served.

    def leg_priority_score(o: str, d: str, date: str | None) -> int:
        # 2 = flight, 1 = self-driving, 0 = taxi
        if flight_ok and _has_flight_on(flight_records, o, d, date):
            return 2
        if drive_ok:
            return 1
        return 0

    best_score = None
    best_tour: list[str] = list(selected_cities)
    for tour in permutations(selected_cities):
        legs = [(org, tour[0], seg_start)]
        for i in range(len(tour) - 1):
            legs.append((tour[i], tour[i + 1], seg_mid[i] if i < len(seg_mid) else None))
        legs.append((tour[-1], org, seg_end))
        score = sum(leg_priority_score(o, d, dt) for (o, d, dt) in legs)
        # Tie-break on lexicographic tour to keep determinism.
        key = (score, tuple(c for c in reversed(tour)))
        if best_score is None or key > best_score:
            best_score = key
            best_tour = list(tour)
    return best_tour


def _entity_block(desc: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"Description": desc, "Content": records}


def _flight_leg_block(o: str, d: str, date: str | None,
                      flight_records: dict[tuple[str, str], list[dict[str, Any]]],
                      allowed: bool) -> dict[str, Any]:
    date_s = date or "unspecified date"
    desc = f"Flight from {o} to {d} on {date_s}"
    if not allowed:
        return {"Description": desc,
                "Content": {"available": False,
                            "reason": "flight disallowed by local_constraint"}}
    if not date:
        return {"Description": desc,
                "Content": {"available": False,
                            "reason": "no segmentation date"}}
    recs = [r for r in flight_records.get((o, d), []) if r.get("date") == date]
    if not recs:
        return {"Description": desc,
                "Content": {"available": False,
                            "reason": "no matching flight on the segmentation date"}}
    prices = sorted(r["price"] for r in recs if r.get("price") is not None)
    n = len(prices)
    stats = {
        "min":    prices[0] if n else None,
        "median": prices[n // 2] if n else None,
        "max":    prices[-1] if n else None,
    }
    return {"Description": desc,
            "Content": {"available": True,
                        "count": len(recs),
                        "price_stats": stats,
                        "records": recs}}


def _ground_leg_block(mode: str, o: str, d: str,
                      distance: dict[tuple[str, str], dict[str, Any]],
                      allowed: bool) -> dict[str, Any]:
    """mode is 'self-driving' or 'taxi'. Ground legs have no date filter."""
    desc = f"{mode.capitalize()} from {o} to {d}"
    if not allowed:
        return {"Description": desc,
                "Content": {"available": False,
                            "reason": f"{mode} disallowed by local_constraint"}}
    d_row = distance.get((o, d)) or distance.get((d, o)) or {}
    km = d_row.get("distance_km")
    if km is None:
        return {"Description": desc,
                "Content": {"available": False,
                            "reason": "no ground-transport distance data"}}
    # osunlp cost model: self-driving ≈ $0.05/km, taxi ≈ $1.00/km.
    if mode == "self-driving":
        cost = round(km * 0.05, 2)
    else:  # taxi
        cost = round(km * 1.00, 2)
    return {"Description": desc,
            "Content": {"available": True,
                        "origin": o, "dest": d,
                        "distance_km": km,
                        "duration": d_row.get("duration"),
                        "cost": cost}}


def _build_reference_information(*, org: str, selected_tour: list[str],
                                 seg_start: str | None, seg_mid: list[str],
                                 seg_end: str | None,
                                 pool_source: dict[str, dict[str, Any]],
                                 flight_records: dict[tuple[str, str], list[dict[str, Any]]],
                                 distance: dict[tuple[str, str], dict[str, Any]],
                                 modes_allowed: set[str],
                                 raw_mode: bool = False,
                                 ) -> list[dict[str, Any]]:
    """Emit a list of {Description, Content} blocks matching the osunlp
    reference_information structure. Block layout:
      per selected_tour city (in visit order):
        - Attractions in <city>
        - Restaurants in <city>
        - Accommodations in <city>
      per segmentation leg (in visit order):
        - Flight from X to Y on <date>
        - Self-driving from X to Y
        - Taxi from X to Y
    Block count = 3 * visit_n + 3 * (visit_n + 1) = 9 / 15 / 21 for
    visit_n = 1 / 2 / 3.

    Parameters
    ----------
    pool_source : dict city -> {"accommodation": [...], "restaurant": [...],
        "attraction": [...]}. The caller decides whether to pass
        constraint-filtered pools (solution_information) or raw pools
        (reference_information).
    raw_mode : when True, all three transport-mode blocks are emitted
        even if the query's local_constraint disallows a mode -- the raw
        reference just reports availability; the planner enforces the
        constraint itself.
    """
    if raw_mode:
        flight_ok = drive_ok = taxi_ok = True
    else:
        flight_ok = "Flight"       in modes_allowed
        drive_ok  = "self-driving" in modes_allowed
        taxi_ok   = "taxi"         in modes_allowed

    blocks: list[dict[str, Any]] = []
    for c in selected_tour:
        pool = pool_source.get(c, {}) or {}
        blocks.append(_entity_block(f"Attractions in {c}",   pool.get("attraction")    or []))
        blocks.append(_entity_block(f"Restaurants in {c}",   pool.get("restaurant")    or []))
        blocks.append(_entity_block(f"Accommodations in {c}", pool.get("accommodation") or []))

    if selected_tour:
        legs: list[tuple[str, str, str | None]] = [(org, selected_tour[0], seg_start)]
        for i in range(len(selected_tour) - 1):
            legs.append((selected_tour[i], selected_tour[i + 1],
                         seg_mid[i] if i < len(seg_mid) else None))
        legs.append((selected_tour[-1], org, seg_end))
        for (o, d, date) in legs:
            blocks.append(_flight_leg_block(o, d, date, flight_records, flight_ok))
            blocks.append(_ground_leg_block("self-driving", o, d, distance, drive_ok))
            blocks.append(_ground_leg_block("taxi", o, d, distance, taxi_ok))
    return blocks


# --------------------------------------------------------------------------- #
# Balanced sampler                                                            #
# --------------------------------------------------------------------------- #

class Tracker:
    def __init__(self) -> None:
        # Per-level paradigm count (Temporal counted as a single paradigm).
        self.paradigm_counts: Counter[tuple[str, str]] = Counter()
        # Per-level temporal subtype balance (only when paradigm = temporal).
        self.temporal_sub_counts: Counter[tuple[str, str]] = Counter()
        # Per-level complex-paradigm balance (medium / hard second slot).
        self.complex_counts: Counter[tuple[str, str]] = Counter()
        # Pairing balance.
        self.pair_counts: Counter[tuple[str, str]] = Counter()
        self.pair_sub_counts: Counter[tuple[str, str]] = Counter()
        # Per-(level, paradigm, sub_key, bank_id) usage count so we can
        # round-robin bank entries within a paradigm — and, where a
        # paradigm has internal partitions (currently only Temporal, via
        # its subtype field), within each sub-partition independently.
        # sub_key = entry["_subtype"] for Temporal (e.g. "atmost_once",
        # "sometime", "hold_after", ...); None for every other paradigm.
        # Namespacing by sub_key keeps e.g. atmost_once id=3 separate
        # from sometime id=3 in the counter — they are not interchangeable.
        self.bank_id_counts: Counter[tuple[str, str, Any, int]] = Counter()
        # Hierarchical pair-type / sub-type balance PER complex bank id.
        # Instead of forcing a global competing/non_competing 50/50 (which
        # concentrates whole-pair repetition on the few competing-capable
        # bank entries), we balance pair-type per (paradigm, bank_id).
        # Each bank entry gets a fair split of independent vs overlapping
        # and, within overlapping, competing vs non_competing — matched
        # to what THAT entry can feasibly produce. Global sub totals
        # drift to reflect the bank's actual composition instead of a
        # forced target.
        # Keys include `sub_key` (entry["_subtype"] for Temporal, None
        # otherwise) so that e.g. always_within/id=11 and
        # sometime_after/id=11 have SEPARATE counters — they are not
        # interchangeable bank entries. Matches the namespacing of
        # bank_id_counts above.
        self.pair_type_counts: Counter[tuple[str, str, Any, int, str]] = Counter()
        # (level, paradigm, sub_key, bank_id, pairing_type) → count
        self.pair_sub_type_counts: Counter[tuple[str, str, Any, int, str]] = Counter()
        # (level, paradigm, sub_key, bank_id, sub_type) → count

    def order_paradigms(self, level: str, options: Sequence[str]) -> list[str]:
        return sorted(options,
                      key=lambda o: (self.paradigm_counts[(level, o)], random.random()))

    def order_temporal_subs(self, level: str) -> list[str]:
        return sorted(TEMPORAL_SUBTYPES,
                      key=lambda o: (self.temporal_sub_counts[(level, o)], random.random()))

    def order_complex(self, level: str) -> list[str]:
        return sorted(COMPLEX_PARADIGMS,
                      key=lambda o: (self.complex_counts[(level, o)], random.random()))

    def order_bank_entries(self, level: str, paradigm: str,
                           entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Order bank entries within their (level, paradigm, sub_key)
        bucket. Two layered mechanisms:

        (a) Namespace by sub_key. For Temporal, sub_key = entry["_subtype"]
            so atmost_once id=3 does NOT share counts with sometime id=3.
            For every other paradigm sub_key is None.

        (b) 1.5x-uniform-share cap. Within the entries being ordered, an
            entry whose pick count exceeds 1.5 x uniform-share (where
            uniform = total_bucket_picks / n_distinct_ids_in_this_pool)
            is deprioritized — pushed to the back of the ordering so it
            only serves as fallback when nothing else is feasible.

        Under-cap entries are still tie-broken by ascending count and
        rng noise so the least-picked entry is tried first."""
        if not entries:
            return []
        # Group the current pool by sub_key so we can compute a cap
        # scoped to each partition. n_ids uses only ids present in
        # THIS pool -- if an id isn't a candidate right now, it doesn't
        # count against the uniform share.
        pool_ids_by_sub: dict[Any, set] = {}
        for e in entries:
            sub_key = e.get("_subtype")
            pool_ids_by_sub.setdefault(sub_key, set()).add(e.get("id", -1))
        caps: dict[Any, float] = {}
        for sub_key, ids in pool_ids_by_sub.items():
            n_ids = max(1, len(ids))
            total = sum(self.bank_id_counts.get(
                            (level, paradigm, sub_key, bid), 0)
                        for bid in ids)
            # If nothing has been picked yet in this bucket, don't cap.
            caps[sub_key] = (1.5 * total / n_ids) if total > 0 else float("inf")

        def key(e: dict[str, Any]) -> tuple:
            bid = e.get("id", -1)
            sub_key = e.get("_subtype")
            cnt = self.bank_id_counts[(level, paradigm, sub_key, bid)]
            over_cap = 1 if cnt > caps[sub_key] else 0
            return (over_cap, cnt, random.random(), bid)
        return sorted(entries, key=key)

    def record_bank_pick(self, level: str, paradigm: str, entry: dict[str, Any]) -> None:
        bid = entry.get("id")
        if bid is not None:
            sub_key = entry.get("_subtype")
            self.bank_id_counts[(level, paradigm, sub_key, bid)] += 1

    def pick_pairing(self, level: str) -> str:
        return min(("independent", "overlapping"),
                   key=lambda o: (self.pair_counts[(level, o)], random.random()))

    def pick_overlap_subtype(self, level: str) -> str:
        return min(("competing", "non_competing"),
                   key=lambda o: (self.pair_sub_counts[(level, o)], random.random()))

    def competing_under_quota(self, level: str) -> bool:
        """True if the global competing share at this level is below the
        COMPETING_SHARE_FLOOR quota. Checked at the DISPATCHER level to
        add a competing-first pass before the normal per-bank
        hierarchical search."""
        n_comp = self.pair_sub_counts[(level, "competing")]
        n_non  = self.pair_sub_counts[(level, "non_competing")]
        total  = n_comp + n_non
        # Use quota only after a small warm-up so early-run noise doesn't
        # trigger it before any pairs exist.
        return total >= 4 and (n_comp / total) < self.COMPETING_SHARE_FLOOR

    def order_pairings_for_bank(self, level: str, paradigm: str,
                                sub_key: Any, bank_id: int) -> list[str]:
        """Order (independent, overlapping) by per-(paradigm, sub_key,
        bank_id) pair-type balance. sub_key namespaces Temporal sub-ops
        so always_within/id=11 and sometime_after/id=11 do NOT share
        the same counter."""
        return sorted(("independent", "overlapping"),
                      key=lambda p: (self.pair_type_counts[
                                        (level, paradigm, sub_key, bank_id, p)],
                                     random.random()))

    # Minimum share of overlapping pairs that should be competing. When
    # the global competing count at a level is below this floor, we
    # force competing first regardless of per-bank counters — bank_ids
    # that CAN produce competing get their attempts pushed early. Bank
    # ids that structurally can't produce competing fall through to
    # non_competing as usual (no wasted picks).
    COMPETING_SHARE_FLOOR = 0.25

    def order_overlap_subs_for_bank(self, level: str, paradigm: str,
                                    sub_key: Any, bank_id: int) -> list[str]:
        """Order (competing, non_competing) inside the overlapping branch.

        Two-mode selector:
        1. If global competing share at `level` is below
           COMPETING_SHARE_FLOOR (25% of overlapping pairs), FORCE
           competing first regardless of per-(paradigm, bank_id)
           balance. This injects competing pairs into the benchmark
           when they'd otherwise be starved by the bank's natural
           non_competing bias.
        2. Otherwise fall back to per-(paradigm, bank_id) balance so
           each bank entry gets a fair split of competing / non_competing
           based on what IT can feasibly produce.

        Order per (paradigm, sub_key, bank_id) so each bank entry gets
        a fair competing/non_competing split matched to what it can
        feasibly produce. sub_key namespaces Temporal sub-ops so
        e.g. always_within/id=11 competing count does not leak into
        sometime_after/id=11. The global competing-share quota is
        enforced at the DISPATCHER level (a competing-first pass) not
        here — this selector stays pure per-bank balance."""
        return sorted(("competing", "non_competing"),
                      key=lambda s: (self.pair_sub_type_counts[
                                        (level, paradigm, sub_key, bank_id, s)],
                                     random.random()))

    def record_pair_bank_context(self, level: str, cx_paradigm: str,
                                 cx_sub_key: Any, cx_bank_id: int,
                                 pairing: str, sub: str | None) -> None:
        """Update per-(paradigm, sub_key, bank_id) pair-type / sub-type
        counters on a successful pair pick. Call once per pair after
        record_pair. sub_key namespaces Temporal sub-ops so their
        counters don't collide."""
        self.pair_type_counts[
            (level, cx_paradigm, cx_sub_key, cx_bank_id, pairing)] += 1
        if sub is not None:
            self.pair_sub_type_counts[
                (level, cx_paradigm, cx_sub_key, cx_bank_id, sub)] += 1

    def record_single(self, level: str, paradigm: str, subtype: str | None) -> None:
        self.paradigm_counts[(level, paradigm)] += 1
        if subtype:
            self.temporal_sub_counts[(level, subtype)] += 1

    def record_pair(self, level: str, first_paradigm: str, second_paradigm: str,
                    second_subtype: str | None, pairing: str,
                    sub: str | None) -> None:
        self.paradigm_counts[(level, first_paradigm)] += 1
        self.paradigm_counts[(level, second_paradigm)] += 1
        self.complex_counts[(level, second_paradigm)] += 1
        if second_subtype:
            self.temporal_sub_counts[(level, second_subtype)] += 1
        self.pair_counts[(level, pairing)] += 1
        if sub:
            self.pair_sub_counts[(level, sub)] += 1

    def absorb_record(self, record: dict[str, Any]) -> None:
        """Update counters from an already-augmented record (used during
        targeted-regeneration so the sampler stays balanced relative to
        whatever wasn't regenerated)."""
        level = record.get("level", "?")
        prefs = record.get("preferences") or []
        if not prefs:
            return
        pairing = record.get("pairing_type") or "single"
        sub = record.get("pairing_subtype")
        if pairing == "single":
            p = prefs[0]
            self.record_single(level, p["paradigm"], p.get("subtype"))
        elif len(prefs) >= 2:
            p1, p2 = prefs[0], prefs[1]
            self.record_pair(level, p1["paradigm"], p2["paradigm"],
                             p2.get("subtype"), pairing, sub)


def _template_uses_unavailable_entity(entry: dict[str, Any],
                                      qdb: dict[str, Any]) -> bool:
    """True if the bank entry's template references any entity that has
    no rows in this query's filtered DB (e.g. Attraction when the city has
    zero attraction records).  Day and Transportation are never marked
    unavailable -- they're feasibility-checked via stats / scope helpers,
    not entity_records."""
    unavailable: set[str] = set()
    if not qdb.get("has_attractions", True):
        unavailable.add("Attraction")
    if not unavailable:
        return False
    for e, _a in iter_template_eas(entry["_paradigm"], entry["template"]):
        if e in unavailable:
            return True
    return False


def order_by_paradigm(flat: list[dict[str, Any]], paradigm: str,
                      subtype: str | None,
                      qdb: dict[str, Any] | None = None
                      ) -> list[dict[str, Any]]:
    out = []
    for e in flat:
        if e["_paradigm"] != paradigm:
            continue
        if paradigm == "TemporalPreference" and subtype is not None and e["_subtype"] != subtype:
            continue
        if qdb is not None and _template_uses_unavailable_entity(e, qdb):
            continue
        out.append(e)
    return out


def try_resolve(entry: dict[str, Any], query: dict[str, Any],
                qdb: dict[str, Any], rng: random.Random
                ) -> tuple[dict[str, Any], float] | None:
    """Resolve `entry` against `query`+`qdb` and return (template,
    applied_budget_multiplier).  The multiplier is 1.0 if the preference
    passes the per-city pool floor at the original budget, or up to
    BUDGET_ADJUST_CAP if budget escalation was required to clear the
    affordable-count floor.  Returns None on rejection."""
    tmpl = resolve_template(entry, query, qdb, rng)
    if tmpl is None:
        return None
    # Adjust set-valued predicates against query hard constraints
    # (cuisine / house_rules / room_type / transportation.mode).  Empty
    # intersection on `in` rejects the preference; `not_in` trims silently.
    if not _apply_constraint_feasibility(entry["_paradigm"], tmpl, query):
        return None
    # Intra-preference invariants encoded via stat-keys must survive
    # resolution (defensive belt-and-braces over tier-lock).
    if not intra_pref_invariants_ok(entry["_paradigm"], tmpl):
        return None
    if not is_single_feasible(entry, tmpl, qdb):
        return None
    # Secondary cost-feasibility: rating / cuisine / house_rules / room_type
    # predicates filtered against the entity pool must still leave at least
    # one item whose cost is within the per-query budget envelope.
    if not secondary_budget_feasible(entry, tmpl, qdb):
        return None
    # Per-city pool floor (Strategy B) on the combined filter with up-to-
    # 1.5x budget escalation as the final recovery before rejection.
    ok, mult, _ = combined_pool_ok_with_budget_adjust(
        [(entry["_paradigm"], tmpl)], qdb)
    if not ok:
        return None
    return (tmpl, mult)


def pick_single(level: str, query: dict[str, Any], qdb: dict[str, Any],
                rng: random.Random, flat: list[dict[str, Any]],
                tracker: Tracker) -> dict[str, Any] | None:
    paradigm_order = tracker.order_paradigms(level, PARADIGMS)
    for paradigm in paradigm_order:
        if paradigm == "TemporalPreference":
            sub_order = tracker.order_temporal_subs(level)
            for sub in sub_order:
                cands = order_by_paradigm(flat, paradigm, sub, qdb)
                cands = tracker.order_bank_entries(level, paradigm, cands)
                for entry in cands[:MAX_SINGLE_TRIALS]:
                    res = try_resolve(entry, query, qdb, rng)
                    if res is None:
                        continue
                    tmpl, applied_mult = res
                    tracker.record_single(level, paradigm, sub)
                    tracker.record_bank_pick(level, paradigm, entry)
                    return {
                        "preferences":       [make_resolved_record(entry, tmpl)],
                        "preference_traces": [trace(entry)],
                        "pairing_type":      "single",
                        "pairing_subtype":   None,
                        "budget_multiplier": applied_mult,
                    }
        else:
            cands = order_by_paradigm(flat, paradigm, None, qdb)
            cands = tracker.order_bank_entries(level, paradigm, cands)
            for entry in cands[:MAX_SINGLE_TRIALS]:
                res = try_resolve(entry, query, qdb, rng)
                if res is None:
                    continue
                tmpl, applied_mult = res
                tracker.record_single(level, paradigm, None)
                tracker.record_bank_pick(level, paradigm, entry)
                return {
                    "preferences":       [make_resolved_record(entry, tmpl)],
                    "preference_traces": [trace(entry)],
                    "pairing_type":      "single",
                    "pairing_subtype":   None,
                    "budget_multiplier": applied_mult,
                }
    return None


def _try_pair_pc_anchor(level: str, pc: dict[str, Any], tc: dict[str, Any],
                        mult_c: float, anchor_cands: list[dict[str, Any]],
                        pairing: str, sub: str | None,
                        query: dict[str, Any], qdb: dict[str, Any],
                        rng: random.Random, tracker: Tracker
                        ) -> dict[str, Any] | None:
    """Given a resolved complex preference (pc, tc) and a target (pairing,
    sub), search the anchor bank for a compatible partner. Returns the
    emitted-record dict on success, None if no anchor works.

    Preserves all anchor-first _pair_adjust semantics and the emit
    convention (anchor at preferences[0], complex at [1])."""
    for pa in anchor_cands:
        is_overlap = overlaps(pc, pa)
        if pairing == "overlapping" and not is_overlap:
            continue
        if pairing == "independent" and is_overlap:
            continue
        res_a = try_resolve(pa, query, qdb, rng)
        if res_a is None:
            continue
        ta, mult_a = res_a
        if is_overlap:
            # Anchor (Atomic/Composite) side stays as t1 for
            # _pair_adjust regardless of search-order shift.
            if not _pair_adjust(ta, pa["_paradigm"],
                                tc, pc["_paradigm"], qdb, rng):
                continue
            if not intra_pref_invariants_ok(pa["_paradigm"], ta):
                continue
            if not intra_pref_invariants_ok(pc["_paradigm"], tc):
                continue
            if not pair_genuinely_active(
                    ta, pa["_paradigm"], tc, pc["_paradigm"], qdb):
                continue
            if not is_single_feasible(pa, ta, qdb) \
                    or not is_single_feasible(pc, tc, qdb):
                continue
            if not secondary_budget_feasible(pa, ta, qdb) \
                    or not secondary_budget_feasible(pc, tc, qdb):
                continue
        if not pair_joint_feasible(
                ta, pa["_paradigm"], tc, pc["_paradigm"], qdb):
            continue
        ok_pair, mult_pair, _ = combined_pool_ok_with_budget_adjust(
            [(pa["_paradigm"], ta), (pc["_paradigm"], tc)], qdb)
        if not ok_pair:
            continue
        applied_mult = max(mult_c, mult_a, mult_pair)
        if pairing == "overlapping":
            cls = classify_overlap(
                ta, pa["_paradigm"], tc, pc["_paradigm"])
            if sub is not None and cls != sub:
                continue
            actual_sub = cls
        else:
            actual_sub = None
        tracker.record_pair(
            level, pa["_paradigm"], pc["_paradigm"],
            pc.get("_subtype"), pairing, actual_sub)
        tracker.record_bank_pick(level, pa["_paradigm"], pa)
        tracker.record_bank_pick(level, pc["_paradigm"], pc)
        # Per-(paradigm, bank_id) pair-type / sub-type counters. The
        # hierarchical balance target lives HERE — global pair_counts
        # and pair_sub_counts still track for reporting but no longer
        # drive the choice.
        tracker.record_pair_bank_context(
            level, pc["_paradigm"], pc.get("_subtype"),
            pc.get("id", -1), pairing, actual_sub)
        return {
            "preferences": [
                make_resolved_record(pa, ta),
                make_resolved_record(pc, tc),
            ],
            "preference_traces": [trace(pa), trace(pc)],
            "pairing_type":      pairing,
            "pairing_subtype":   actual_sub,
            "budget_multiplier": applied_mult,
        }
    return None


def pick_pair(level: str, complex_paradigm: str, anchor_paradigm: str,
              query: dict[str, Any],
              qdb: dict[str, Any], rng: random.Random,
              flat: list[dict[str, Any]], tracker: Tracker,
              force_pairing: str | None = None,
              force_subtype: str | None = None,
              ) -> dict[str, Any] | None:
    """Search a pair with HIERARCHICAL BALANCE (per-(paradigm, bank_id)):

    - OUTER: bank entries of the complex paradigm (ordered by cap-aware
      bank_id_counts). Each bank entry gets its round-robin turn as
      the search primary.
    - PER-BANK-ID: for the current bank entry, order (independent,
      overlapping) by that entry's own pair_type_counts. Inside
      overlapping, order (competing, non_competing) by that entry's
      pair_sub_type_counts.
    - INNER: anchor (Atomic/Composite) bank entries, searched for the
      chosen (pairing, sub) target.
    - _pair_adjust anchor: Atomic/Composite side is t1 (rigid). Preserved.
    - EMIT: preferences[0] = anchor, preferences[1] = complex.

    This differs from a global-target search: instead of forcing a
    single competing/non_competing ratio across the whole dataset (which
    concentrates the few competing-capable bank entries into whole-pair
    repetition), each bank entry gets a fair distribution of pair-types
    matched to what IT can feasibly produce. The global sub ratio
    becomes a natural outcome of the bank's composition.

    force_pairing / force_subtype (targeted-regeneration) still short-
    circuit the per-bank-id choice."""
    # Complex bank entries as SEARCH PRIMARY (outer axis in the
    # hierarchical balance). For Temporal, iterate sub-ops in tracker
    # order and concatenate their bank entries.
    if complex_paradigm == "TemporalPreference":
        sub_temporal_order = tracker.order_temporal_subs(level)
        complex_cands: list[dict[str, Any]] = []
        for sub_temp in sub_temporal_order:
            sub_cands = order_by_paradigm(flat, complex_paradigm, sub_temp, qdb)
            sub_cands = tracker.order_bank_entries(level, complex_paradigm, sub_cands)
            complex_cands.extend(sub_cands)
    else:
        complex_cands = order_by_paradigm(flat, complex_paradigm, None, qdb)
        complex_cands = tracker.order_bank_entries(level, complex_paradigm, complex_cands)

    anchor_cands = order_by_paradigm(flat, anchor_paradigm, None, qdb)
    anchor_cands = tracker.order_bank_entries(level, anchor_paradigm, anchor_cands)

    for pc in complex_cands:
        res_c = try_resolve(pc, query, qdb, rng)
        if res_c is None:
            continue
        tc, mult_c = res_c
        pc_bid = pc.get("id", -1)
        pc_sub = pc.get("_subtype")   # None for non-Temporal

        # Decide (pairing, sub) order PER THIS (paradigm, sub_key, bank_id).
        if force_pairing in ("independent", "overlapping"):
            pairings = [force_pairing]
        else:
            pairings = tracker.order_pairings_for_bank(
                level, complex_paradigm, pc_sub, pc_bid)

        for pairing in pairings:
            if pairing == "overlapping":
                if force_subtype in ("competing", "non_competing"):
                    sub_order: list[str | None] = [force_subtype]
                else:
                    sub_order = list(tracker.order_overlap_subs_for_bank(
                        level, complex_paradigm, pc_sub, pc_bid))
            else:
                sub_order = [None]

            for sub in sub_order:
                result = _try_pair_pc_anchor(
                    level, pc, tc, mult_c, anchor_cands,
                    pairing, sub, query, qdb, rng, tracker)
                if result is not None:
                    return result
    return None


def augment_one(query: dict[str, Any], qdb: dict[str, Any],
                rng: random.Random, flat: list[dict[str, Any]],
                tracker: Tracker,
                force_pairing: str | None = None,
                force_subtype: str | None = None) -> dict[str, Any] | None:
    # Note: queries with empty Attraction pool are NOT skipped here -- the
    # template sampler filters out Attraction-touching bank entries via
    # the entity-availability mask in order_by_paradigm, so the query can
    # still receive Restaurant / Accommodation / Transportation / Day
    # preferences.
    level = query["level"]
    if level == "easy":
        return pick_single(level, query, qdb, rng, flat, tracker)
    if level in ("medium", "hard"):
        anchor = "AtomicPreference" if level == "medium" else "CompositePreference"
        # Competing-share quota (25% of overlapping): if global competing
        # share is under the floor, run a competing-first pass BEFORE
        # the normal hierarchical search. This exhausts competing search
        # across all complex paradigms and all their bank ids for THIS
        # query before falling back to per-bank hierarchical balance.
        # Skipped when force_pairing/force_subtype are set (targeted
        # regeneration overrides).
        if (force_pairing is None and force_subtype is None
                and tracker.competing_under_quota(level)):
            for cx in tracker.order_paradigms(level, list(COMPLEX_PARADIGMS)):
                result = pick_pair(level, cx, anchor, query, qdb, rng, flat, tracker,
                                   force_pairing="overlapping",
                                   force_subtype="competing")
                if result is not None:
                    return result
            # Competing was not achievable this query; fall through to
            # normal hierarchical search below so we still emit a pair.
        # Normal hierarchical search: bank_id outer, per-bank pair-type
        # / sub-type inner.
        for cx in tracker.order_paradigms(level, list(COMPLEX_PARADIGMS)):
            result = pick_pair(level, cx, anchor, query, qdb, rng, flat, tracker,
                               force_pairing=force_pairing,
                               force_subtype=force_subtype)
            if result is not None:
                return result
        return None
    return None


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def _parse_regen_spec(spec: str | None) -> dict[int, tuple[str | None, str | None]]:
    """Parse a --regen string into {query_id: (forced_pairing, forced_sub)}.

    Format: comma-separated entries of the form
       id[:pairing_type[:pairing_subtype]]
    e.g. "360,361:overlapping:competing,362:independent".  Forced fields
    must be "independent" / "overlapping" / "competing" / "non_competing"
    or omitted.  An "_" placeholder is accepted to skip a field, e.g.
    "362:_:competing" means force subtype only."""
    if not spec:
        return {}
    out: dict[int, tuple[str | None, str | None]] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        try:
            qid = int(parts[0])
        except ValueError:
            raise SystemExit(f"--regen: invalid query id in entry {entry!r}")
        pairing = parts[1].strip() if len(parts) > 1 and parts[1].strip() not in ("", "_") else None
        sub = parts[2].strip() if len(parts) > 2 and parts[2].strip() not in ("", "_") else None
        if pairing and pairing not in ("independent", "overlapping"):
            raise SystemExit(f"--regen: invalid pairing {pairing!r} in entry {entry!r}")
        if sub and sub not in ("competing", "non_competing"):
            raise SystemExit(f"--regen: invalid pairing_subtype {sub!r} in entry {entry!r}")
        out[qid] = (pairing, sub)
    return out


def main() -> None:
    project = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", type=Path,
                    default=project / "preference_bank.json")
    ap.add_argument("--queries", type=Path,
                    default=project / "travelplanner-test.jsonl")
    ap.add_argument("--db", type=Path,
                    default=project / "database")
    ap.add_argument("--out", type=Path,
                    default=project / "prefertripplan.jsonl")
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--regen",
                    help=("Targeted regeneration spec: comma-separated "
                          "id[:pairing[:subtype]] entries.  "
                          "Example: '360,361:overlapping:competing,362:_:competing'.  "
                          "Listed queries are regenerated (with optional "
                          "forced pairing/subtype); the rest are read back "
                          "from --out and preserved unchanged."))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)

    bank_raw = load_jsonc(args.bank)
    flat = flatten_bank(bank_raw)

    city_to_state, state_to_cities = load_city_state(args.db)
    raw = {
        "accommodations": load_accommodations(args.db),
        "restaurants":    load_restaurants(args.db),
        "attractions":    load_attractions(args.db),
    }
    flight_prices = load_flight_prices(args.db)
    flight_records = load_flight_records(args.db)
    distance = load_distance(args.db)

    regen_spec = _parse_regen_spec(args.regen)
    existing: dict[int, dict[str, Any]] = {}
    if regen_spec and args.out.exists():
        with open(args.out) as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing[rec["query_id"]] = rec
        print(f"--regen: loaded {len(existing)} existing records from {args.out}")
        print(f"--regen: will regenerate {len(regen_spec)} queries: "
              f"{sorted(regen_spec)}")

    tracker = Tracker()
    # When regenerating, pre-populate counters from records we'll keep so
    # the balanced sampler stays consistent with the wider distribution.
    if regen_spec:
        for qid, rec in existing.items():
            if qid in regen_spec:
                continue
            tracker.absorb_record(rec)

    n_total = n_with_prefs = n_skipped_no_attr = n_regen = 0
    out_records: list[dict[str, Any]] = []
    with open(args.queries) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            query = json.loads(line)
            qid = query["query_id"]
            n_total += 1

            # Targeted-regeneration: pass through existing record verbatim
            # if this query isn't in the regen spec AND we have it on disk.
            if regen_spec and qid not in regen_spec and qid in existing:
                out_records.append(existing[qid])
                if existing[qid].get("preferences"):
                    n_with_prefs += 1
                continue

            qdb = build_query_db(query, raw, city_to_state, state_to_cities,
                                 flight_prices, distance,
                                 flight_records=flight_records)
            out_rec = dict(query)
            if not qdb["has_attractions"]:
                n_skipped_no_attr += 1
            force_pairing, force_subtype = (
                regen_spec.get(qid, (None, None))
                if regen_spec else (None, None))
            result = augment_one(query, qdb, rng, flat, tracker,
                                 force_pairing=force_pairing,
                                 force_subtype=force_subtype)
            if result is None:
                out_rec["preferences"] = []
                out_rec["preference_traces"] = []
                out_rec["pairing_type"] = None
                out_rec["pairing_subtype"] = None
                out_rec["budget_original"] = query["budget"]
                out_rec["budget_multiplier"] = 1.0
            else:
                n_with_prefs += 1
                if regen_spec and qid in regen_spec:
                    n_regen += 1
                # Capture (and remove) the budget multiplier from result so
                # it doesn't collide with the explicit field below; then
                # apply it to the query's emitted budget.  Adjusted budgets
                # are rounded UP to the nearest 10 to stay feasible while
                # matching the dataset's typical budget granularity.
                mult = float(result.pop("budget_multiplier", 1.0))
                original_budget = float(query["budget"])
                if mult > 1.0:
                    adjusted_budget = int(math.ceil(original_budget * mult / 10.0)) * 10
                else:
                    adjusted_budget = int(original_budget)
                out_rec.update(result)
                out_rec["budget_original"] = int(original_budget)
                out_rec["budget"] = adjusted_budget
                out_rec["budget_multiplier"] = mult
                # Build feasibility metadata at the EMITTED budget so the
                # block reflects what the planner / evaluator will work
                # with downstream.
                if adjusted_budget != int(original_budget):
                    qdb_emit = qdb_with_budget(qdb, adjusted_budget)
                else:
                    qdb_emit = qdb
                out_rec["feasibility_metadata"] = build_feasibility_metadata(
                    query, qdb_emit, result.get("preferences", []),
                    flight_records, distance)
            out_records.append(out_rec)

    with open(args.out, "w") as fout:
        for rec in out_records:
            fout.write(json.dumps(rec) + "\n")

    print(f"Augmented {n_with_prefs}/{n_total} queries; "
          f"{n_skipped_no_attr} skipped due to missing attractions.")
    if regen_spec:
        print(f"  ...of which {n_regen} were targeted regenerations.")
    print("Per-level paradigm distribution (Temporal counted as 1):")
    for (lv, p), c in sorted(tracker.paradigm_counts.items()):
        print(f"  {lv:6s} {p:25s} {c}")
    print("Temporal subtype distribution per level:")
    for (lv, s), c in sorted(tracker.temporal_sub_counts.items()):
        print(f"  {lv:6s} {s:20s} {c}")
    print("Pairing distribution (medium/hard):")
    for (lv, pt), c in sorted(tracker.pair_counts.items()):
        print(f"  {lv:6s} {pt:14s} {c}")
    print("Overlap sub-tag distribution:")
    for (lv, st), c in sorted(tracker.pair_sub_counts.items()):
        print(f"  {lv:6s} {st:14s} {c}")


if __name__ == "__main__":
    main()
