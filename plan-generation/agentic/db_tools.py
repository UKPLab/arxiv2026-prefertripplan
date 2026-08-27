"""DB-backed tool layer for the agentic planner.

SELF-CONTAINED. Reads ``database/*.csv`` directly and imports nothing from
``data-generation/`` or ``preferences.py``. The only external code this package
is permitted to use is the TravelPlanner-ported tool APIs under
``travelplanner-ports/tools`` (used by the verification scripts, not here).

Named ``db_tools`` rather than ``tools`` so it does not shadow that package.

Design constraint (DESIGN.md, "Information ladder"): every record returned here
must be byte-identical to the corresponding ``reference_information`` block, so
the agentic-vs-direct contrast measures retrieval and not serialisation. Because this
module no longer shares code with the generator, that parity is now an
INDEPENDENT REIMPLEMENTATION checked by ``verify_pool_parity.py`` -- a stronger
guarantee than the earlier shared-loader version, and one that will catch any
future drift on either side.

Two deliberate divergences from ``reference_information``, both so the agent
sees what the grader will actually do (an agent must never be penalised for a
fact no tool would show it):

  * over-a-day ground legs are reported UNAVAILABLE (the grader refuses them);
  * distances are parsed correctly and duplicate CSV rows resolve first-wins,
    matching ``GoogleDistanceMatrix``.

``ground_transport_raw`` reproduces the generator's behaviour, bugs included,
so the parity test can still compare against shipped blocks.
"""
from __future__ import annotations

import ast
import csv
import re
import threading
from collections import defaultdict
from datetime import date as _date
from typing import Any

import pandas as pd

import _paths  # noqa: F401  (sys.path shim / repo paths)

DB = _paths.DB_DIR

# --------------------------------------------------------------------------- #
# Field parsers. Reimplemented to match the shapes in reference_information.   #
# --------------------------------------------------------------------------- #
def _num(s: Any) -> float | None:
    if s in (None, ""):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _house_rules(s: Any) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip() for x in s.split("&") if x.strip()]


def _cuisines(s: Any) -> list[str]:
    if not isinstance(s, str):
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _categories(s: Any) -> list[str]:
    """attractions.csv stores a Python list literal, e.g. "['Museums', ...]"."""
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return [str(x).strip() for x in v] if isinstance(v, list) else []


def _km(s: Any) -> float | None:
    """Parse "1,972 km" / "94.0 km".

    Note the decimal group: the generator's own parser omitted it and so read
    "94.0 km" as 4.0 (DESIGN.md fact 4). Empty -> None; a non-empty value that
    cannot be parsed raises, so format drift is loud rather than silent.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*km", s)
    if not m:
        raise ValueError(f"_km: cannot parse a distance from {s!r}")
    return float(m.group(1).replace(",", ""))


# --------------------------------------------------------------------------- #
# Process-wide singletons: built once under a lock, read-only thereafter.      #
# --------------------------------------------------------------------------- #
_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _get(key: str):
    if key in _CACHE:
        return _CACHE[key]
    with _LOCK:
        if key in _CACHE:                       # re-check inside the lock
            return _CACHE[key]
        _CACHE[key] = _BUILDERS[key]()
        return _CACHE[key]


def _by_city(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r.get("city", "")].append(r)
    return dict(out)


def _build_attractions() -> dict[str, list[dict]]:
    rows = []
    with open(DB / "attractions" / "attractions.csv") as f:
        for r in csv.DictReader(f):
            rows.append({
                "name": (r.get("Name") or "").strip(),
                "city": (r.get("City") or "").strip(),
                "categories": _categories(r.get("categories") or r.get("category", "")),
                "rating": _num(r.get("rating")),
                "latitude": _num(r.get("Latitude")),
                "longitude": _num(r.get("Longitude")),
                "address": (r.get("Address") or "").strip() or None,
                "website": (r.get("Website") or "").strip() or None,
            })
    return _by_city(rows)


def _build_restaurants() -> dict[str, list[dict]]:
    rows = []
    with open(DB / "restaurants" / "clean_restaurant_2025.csv") as f:
        for r in csv.DictReader(f):
            cost, rating = _num(r.get("Average Cost")), _num(r.get("Aggregate Rating"))
            if cost is None or rating is None:   # unpriced/unrated rows are excluded
                continue
            rows.append({
                "name": (r.get("Name") or "").strip(),
                "city": (r.get("City") or "").strip(),
                "cuisines": _cuisines(r.get("Cuisines", "")),
                "cost": cost,
                "rating": rating,
            })
    return _by_city(rows)


def _build_accommodations() -> dict[str, list[dict]]:
    rows = []
    with open(DB / "accommodations" / "clean_accommodations_2025.csv") as f:
        for r in csv.DictReader(f):
            price, rating = _num(r.get("price")), _num(r.get("review rate number"))
            if price is None or rating is None:
                continue
            mn, mo = _num(r.get("minimum nights")), _num(r.get("maximum occupancy"))
            rows.append({
                "name": (r.get("NAME") or "").strip(),
                "city": (r.get("city") or "").strip(),
                "room_type": (r.get("room type") or "").strip(),
                "cost": price,
                "rating": rating,
                "house_rules_list": _house_rules(r.get("house_rules", "")),
                "minimum_nights": None if mn is None else int(mn),
                "maximum_occupancy": None if mo is None else int(mo),
            })
    return _by_city(rows)


def _build_distance(correct: bool) -> dict[tuple[str, str], dict[str, Any]]:
    """``correct=True``  -> proper decimal parse, FIRST duplicate row wins,
    matching ``GoogleDistanceMatrix`` and therefore the grader.
    ``correct=False`` -> the generator's behaviour (regex without the decimal
    group, LAST duplicate row wins), used only to reproduce shipped blocks.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with open(DB / "googleDistanceMatrix" / "distance.csv") as f:
        for r in csv.DictReader(f):
            key = (r["origin"], r["destination"])
            if correct and key in out:
                continue                        # first-wins
            raw = r.get("distance")
            if correct:
                km = _km(raw)
            else:
                m = re.search(r"([\d,]+)\s*km", raw or "")
                km = float(m.group(1).replace(",", "")) if m else None
            out[key] = {"distance_km": km,
                        "duration": (r.get("duration") or "").strip() or None}
    return out


def _build_cities() -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    with open(DB / "background" / "citySet_with_states.txt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                out[parts[1]].append(parts[0])
    return dict(out)


def _build_flights() -> pd.DataFrame:
    """Indexed on (origin, dest, date) so a lookup is O(log n) rather than a
    scan of 4.5M rows. Categorical keys keep RSS to a fraction of a
    dict-of-lists build."""
    df = pd.read_csv(
        DB / "flights" / "clean_Flights_2025.csv",
        dtype={"OriginCityName": "category", "DestCityName": "category",
               "FlightDate": "category", "Flight Number": "string",
               "DepTime": "string", "ArrTime": "string",
               "ActualElapsedTime": "string"},
    )
    df["price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df[df["price"].notna()]
    df["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    df = df.rename(columns={
        "Flight Number": "flight_number", "DepTime": "dep_time",
        "ArrTime": "arr_time", "ActualElapsedTime": "elapsed",
        "FlightDate": "date", "OriginCityName": "origin_city",
        "DestCityName": "dest_city"})
    df = df[["flight_number", "price", "dep_time", "arr_time", "elapsed",
             "date", "origin_city", "dest_city", "distance"]]
    return df.set_index(["origin_city", "dest_city", "date"], drop=False).sort_index()


_BUILDERS = {
    "attractions": _build_attractions,
    "restaurants": _build_restaurants,
    "accommodations": _build_accommodations,
    "distance": lambda: _build_distance(correct=True),
    "distance_generator": lambda: _build_distance(correct=False),
    "cities": _build_cities,
    "flights": _build_flights,
}


def preload(flights: bool = True) -> None:
    """Build every table up front so worker threads never pay the cost."""
    for k in ("attractions", "restaurants", "accommodations", "distance",
              "distance_generator", "cities"):
        _get(k)
    if flights:
        _get("flights")


# --------------------------------------------------------------------------- #
# Tools. Return shapes mirror the reference_information block Contents.        #
# --------------------------------------------------------------------------- #
def get_trip_dates(dates: list[str]) -> list[list[str]]:
    """Block: 'Trip dates and days-of-week' -> [[iso_date, weekday], ...].

    Dates are supplied BY THE AGENT, parsed from the query -- the harness does
    not pass the record's date list. No TravelPlanner tool hands the agent its
    dates either: `Flights.run(origin, destination, departure_date)` requires
    one, so the agent has always had to read them out of the query. Handing them
    over would be giving away a structured fact the query already states in
    prose, which is exactly the kind of shortcut the projection layer exists to
    prevent.

    Batched on purpose: one call for the whole trip rather than one per day,
    which on a 7-day trip would be six calls spent on something derivable in
    one. The tool-call budget counts it as 1.
    """
    return [[d, _date.fromisoformat(d).strftime("%A")] for d in dates]


def get_cities_in_state(state: str) -> list[str]:
    """Cities in a US state; [] for an unknown state.

    ``travelplanner-ports/tools/cities/apis.py:25`` *returns* (rather than
    raises) a ValueError object here, so we do not use it.
    """
    return list(_get("cities").get(state, []))


def search_attractions(city: str) -> list[dict]:
    """Block: 'Attractions in <city>'."""
    return list(_get("attractions").get(city, []))


def search_restaurants(city: str) -> list[dict]:
    """Block: 'Restaurants in <city>'."""
    return list(_get("restaurants").get(city, []))


def search_accommodations(city: str) -> list[dict]:
    """Block: 'Accommodations in <city>'."""
    return list(_get("accommodations").get(city, []))


def search_flights(origin: str, destination: str, date: str) -> list[dict]:
    """Block: 'Flight from <o> to <d> on <date>', in the flat-list form the HF
    export ships. [] when none."""
    df = _get("flights")
    try:
        hit = df.loc[[(origin, destination, date)]]
    except KeyError:
        return []
    out = []
    for r in hit.to_dict("records"):
        r = dict(r)
        for k in ("flight_number", "dep_time", "arr_time", "elapsed", "date",
                  "origin_city", "dest_city"):
            r[k] = str(r[k]).strip()
        r["price"] = float(r["price"])
        r["distance"] = None if pd.isna(r["distance"]) else float(r["distance"])
        out.append(r)
    return out


def get_ground_transport(origin: str, destination: str, mode: str) -> dict:
    """Blocks: 'Self-driving from <o> to <d>' / 'Taxi from <o> to <d>'.

    Over-a-day legs are reported unavailable, matching
    ``GoogleDistanceMatrix.run`` and ``commonsense_constraint``'s sandbox
    check. Residual divergence from the grader is rounding only: this returns
    ``round(km*rate, 2)`` where ``get_total_cost`` charges ``int(km*rate)``,
    so always <$1/leg -- pinned by ``verify_tool_evaluator_consistency.py``
    rather than "fixed", so the two definitions stay tied.
    """
    if mode not in ("self-driving", "taxi"):
        return {"available": False, "reason": f"unknown mode {mode!r}"}
    dist = _get("distance")
    row = dist.get((origin, destination)) or dist.get((destination, origin)) or {}
    km = row.get("distance_km")
    if km is None:
        return {"available": False, "reason": "no ground-transport distance data"}
    duration = row.get("duration")
    if duration and "day" in duration:
        return {"available": False,
                "reason": "trip exceeds one day; not a usable travel option"}
    rate = 0.05 if mode == "self-driving" else 1.00
    return {"available": True, "origin": origin, "dest": destination,
            "distance_km": km, "duration": duration, "cost": round(km * rate, 2)}


def ground_transport_raw(origin: str, destination: str, mode: str) -> dict:
    """The generator's exact behaviour, bugs included: buggy km parse,
    last-duplicate-wins, and over-a-day legs advertised as available. Used only
    by ``verify_pool_parity.py``; never exposed to an agent.
    """
    dist = _get("distance_generator")
    row = dist.get((origin, destination)) or dist.get((destination, origin)) or {}
    km = row.get("distance_km")
    if km is None:
        return {"available": False, "reason": "no ground-transport distance data"}
    rate = 0.05 if mode == "self-driving" else 1.00
    return {"available": True, "origin": origin, "dest": destination,
            "distance_km": km, "duration": row.get("duration"),
            "cost": round(km * rate, 2)}


# --------------------------------------------------------------------------- #
# Filtering and aggregation over a searched pool                              #
# --------------------------------------------------------------------------- #
# Search returns a city's whole pool; a query's hard constraints and preferences
# are then predicates over it ("cuisine in [French, Italian]", "cost <= 75",
# "rating >= 4.5", "room_type == Entire home/apt", "house_rules contains_all
# [...]"). Without these the agent has to eyeball 35 restaurant rows per city
# and filter in its head, which is where selection errors come from.
#
# Operators mirror the ones the preference bank actually uses, by frequency:
#   in 134, >= 95, == 61, <= 33, not_in 13, contains_all 3, != 1
# plus <, > and between for completeness.
#
# Stateless: a filter call names the same pool a search call would, so nothing
# has to remember "the last result". That keeps every call independently
# replayable from the trajectory.

_POOLS = {"attractions": search_attractions,
          "restaurants": search_restaurants,
          "accommodations": search_accommodations}

FILTERABLE = {
    "attractions": ("name", "city", "categories", "rating"),
    "restaurants": ("name", "city", "cuisines", "cost", "rating"),
    "accommodations": ("name", "city", "room_type", "cost", "rating",
                       "house_rules_list", "minimum_nights", "maximum_occupancy"),
    "flights": ("flight_number", "price", "dep_time", "arr_time", "date",
                "origin_city", "dest_city", "distance"),
}
OPERATORS = ("==", "!=", "<", "<=", ">", ">=", "in", "not_in", "contains_all", "between")
AGGREGATES = ("min", "max", "avg", "sum", "count")


def _as_list(v):
    return v if isinstance(v, (list, tuple, set)) else [v]


def _match(row: dict, f: dict) -> bool:
    """One predicate against one row.

    List-valued fields (categories, cuisines, house_rules_list) are treated as
    sets, which is what the preference bank means: `cuisine in [French, Italian]`
    asks whether the restaurant serves any of them, not whether its whole cuisine
    list equals one.
    """
    field, op, want = f.get("field"), f.get("op"), f.get("value")
    if field not in row:
        return False
    have = row[field]
    is_set = isinstance(have, (list, tuple, set))
    try:
        if op == "in":
            return bool(set(map(str, have)) & set(map(str, _as_list(want)))) if is_set                 else str(have) in {str(x) for x in _as_list(want)}
        if op == "not_in":
            return not (bool(set(map(str, have)) & set(map(str, _as_list(want)))) if is_set
                        else str(have) in {str(x) for x in _as_list(want)})
        if op == "contains_all":
            return set(map(str, _as_list(want))) <= set(map(str, have)) if is_set                 else str(have) in {str(x) for x in _as_list(want)}
        if op == "between":
            lo, hi = (list(want) + [None, None])[:2]
            return have is not None and float(lo) <= float(have) <= float(hi)
        if op == "==":
            return str(have) == str(want)
        if op == "!=":
            return str(have) != str(want)
        if have is None:
            return False
        a, b = float(have), float(want)
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
    except (TypeError, ValueError, KeyError):
        return False


def _pool(entity: str, city: str = "", origin: str = "", destination: str = "",
          date: str = "") -> list[dict]:
    if entity == "flights":
        return search_flights(origin, destination, date)
    fn = _POOLS.get(entity)
    return fn(city) if fn else []


def filter_items(entity: str, *, city: str = "", origin: str = "",
                 destination: str = "", date: str = "",
                 filters: list[dict] | None = None,
                 sort_by: str | None = None, desc: bool = False) -> dict:
    """Apply predicates to a pool. Returns matches plus how many were dropped."""
    if entity not in FILTERABLE:
        return {"error": "unknown_entity", "available": sorted(FILTERABLE)}
    rows = _pool(entity, city, origin, destination, date)
    total = len(rows)
    bad = [f.get("field") for f in (filters or [])
           if f.get("field") not in FILTERABLE[entity]]
    if bad:
        return {"error": "unknown_field", "fields": bad,
                "available": list(FILTERABLE[entity])}
    bad_op = [f.get("op") for f in (filters or []) if f.get("op") not in OPERATORS]
    if bad_op:
        return {"error": "unknown_operator", "operators": bad_op,
                "available": list(OPERATORS)}
    out = [r for r in rows if all(_match(r, f) for f in (filters or []))]
    if sort_by and out and sort_by in FILTERABLE[entity]:
        out.sort(key=lambda r: (r.get(sort_by) is None, r.get(sort_by)), reverse=desc)
    return {"entity": entity, "total": total, "matched": len(out),
            "dropped": total - len(out), "records": out}


def aggregate_items(entity: str, field: str, op: str, *, city: str = "",
                    origin: str = "", destination: str = "", date: str = "",
                    filters: list[dict] | None = None) -> dict:
    """min / max / avg / sum / count over a filtered pool.

    Numeric preferences are stated as optimisations ("maximise average rating",
    "minimise total cost"), so the agent needs the aggregate itself, not just
    the rows -- computing a mean over 35 rows in-context is exactly where
    arithmetic slips.
    """
    if op not in AGGREGATES:
        return {"error": "unknown_aggregate", "available": list(AGGREGATES)}
    res = filter_items(entity, city=city, origin=origin, destination=destination,
                       date=date, filters=filters)
    if "error" in res:
        return res
    vals = [r.get(field) for r in res["records"] if isinstance(r.get(field), (int, float))]
    if op == "count":
        return {"entity": entity, "field": field, "op": op,
                "value": len(res["records"]), "n": len(res["records"])}
    if not vals:
        return {"entity": entity, "field": field, "op": op, "value": None,
                "n": 0, "reason": "no numeric values after filtering"}
    v = {"min": min(vals), "max": max(vals), "sum": sum(vals),
         "avg": sum(vals) / len(vals)}[op]
    return {"entity": entity, "field": field, "op": op,
            "value": round(float(v), 2), "n": len(vals)}


# --------------------------------------------------------------------------- #
# Resolving a plan's named entities back to their attributes                  #
# --------------------------------------------------------------------------- #
# A plan line names an entity ("Kobe Hibachi & Sushi, Tampa") and nothing more.
# But preferences are almost entirely ABOUT attributes -- across the test split
# they test rating 210 times, cost 173, category 166, cuisine 24 -- so a check
# handed only names can verify essentially none of them.
#
# Annotating the plan line is not an option: the grader parses "Name, City" with
# a regex, and appending "; Cost: 321" makes the city parse as
# "Tampa; Cost: 321", which then fails the sandbox lookup. So the attributes are
# resolved here instead and handed to the checks alongside the plan.
#
# This is not extra information: it is exactly what the agent's own search tools
# already returned to it. What it removes is the need to carry it by hand.
# Attributes are given RAW -- the per-person and per-night arithmetic a
# preference may call for is left to the check, because the request states those
# units and reading them correctly is part of the task.

_NAME_CITY = re.compile(r"(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$")
_PAREN = re.compile(r"\s*\(.*?\)\s*$")


def _name_city(info: str) -> tuple[str, str]:
    """Split "Name, City" the way the grader does."""
    m = _NAME_CITY.search(info or "")
    if not m:
        return "", ""
    return m.group(1).strip(), _PAREN.sub("", m.group(2).strip()).strip()


def _lookup(rows: list[dict], name: str, city: str) -> dict | None:
    for r in rows:
        if r.get("name") == name and r.get("city") == city:
            return r
    for r in rows:                       # tolerate minor name drift
        if r.get("city") == city and name and name in str(r.get("name", "")):
            return r
    return None


_FROM_TO = re.compile(r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)")


def _leg_endpoints(text: str) -> tuple[str, str] | tuple[None, None]:
    m = _FROM_TO.search(text or "")
    if not m:
        return None, None
    return (_PAREN.sub("", m.group(1)).strip(), _PAREN.sub("", m.group(2)).strip())


def _flight_by_number(origin: str, dest: str, fn: str) -> dict | None:
    """Find a flight on an (origin, dest) pair by its number.

    The plan line has no date, so this scans the pair rather than the whole
    4.5M-row table -- the frame is indexed on (origin, dest, date), so a
    two-level `xs` is the cheap way in.
    """
    df = _get("flights")
    try:
        hit = df.xs((origin, dest), level=("origin_city", "dest_city"),
                    drop_level=False)
    except KeyError:
        return None
    m = hit[hit["flight_number"].astype(str).str.strip() == str(fn).strip()]
    if m.empty:
        return None
    r = m.iloc[0]
    return {"price": float(r["price"]), "elapsed": str(r["elapsed"]),
            "date": str(r["date"]),
            "distance": None if pd.isna(r["distance"]) else float(r["distance"])}


def resolve_plan(plan_days: list[dict]) -> list[dict]:
    """Attach DB attributes to every entity a plan names.

    `found: False` marks a name that does not resolve -- itself worth checking,
    since an unresolvable entity is one the grader will reject too.
    """
    out = []
    for unit in plan_days or []:
        # `current_city` is the one plan field convert_plan_text leaves as raw
        # text, and the environment parses it a second time inside its own
        # scorer. A check has no access to that parse, so every authored check
        # needing the day's cities had to re-derive them from the string --
        # observed failing on "from Washington to Tampa", where a hand-written
        # regex kept a stray "from" and rejected a valid plan for five rounds.
        # Parse it once here, with the same pattern the environment uses.
        cc = (unit.get("current_city") or "").strip()
        o_c, d_c = _leg_endpoints(cc)
        day = {"day": unit.get("days"), "current_city": unit.get("current_city"),
               # travel day -> origin/dest set and `city` is where the day ends,
               # which is where its meals and attractions are; stay day ->
               # origin/dest are None and `city` is the single city.
               "origin": o_c, "dest": d_c,
               "city": d_c if d_c else (_PAREN.sub("", cc).strip() or None),
               "is_travel_day": bool(o_c and d_c),
               "restaurants": [], "attractions": [], "accommodation": None,
               "transportation": []}
        for meal in ("breakfast", "lunch", "dinner"):
            v = (unit.get(meal) or "").strip()
            if not v or v == "-":
                continue
            n, c = _name_city(v)
            row = _lookup(search_restaurants(c), n, c) if c else None
            day["restaurants"].append(
                {"meal": meal, "name": n, "city": c, "found": row is not None,
                 **({k: row[k] for k in ("cost", "rating", "cuisines")} if row else {})})
        av = (unit.get("attraction") or "").strip()
        if av and av != "-":
            for part in (x for x in av.split(";") if x.strip() and x.strip() != "-"):
                n, c = _name_city(part.strip().rstrip("."))
                row = _lookup(search_attractions(c), n, c) if c else None
                day["attractions"].append(
                    {"name": n, "city": c, "found": row is not None,
                     **({k: row[k] for k in ("rating", "categories")} if row else {})})
        acc = (unit.get("accommodation") or "").strip()
        if acc and acc != "-":
            n, c = _name_city(acc)
            row = _lookup(search_accommodations(c), n, c) if c else None
            day["accommodation"] = {
                "name": n, "city": c, "found": row is not None,
                **({k: row[k] for k in ("cost", "rating", "room_type",
                                        "house_rules_list", "minimum_nights",
                                        "maximum_occupancy")} if row else {})}
        tr = (unit.get("transportation") or "").strip()
        if tr and tr != "-":
            low = tr.lower()
            mode = ("flight" if "flight number" in low else
                    "self-driving" if "self-driving" in low else
                    "taxi" if "taxi" in low else "unknown")
            # Times and flight number are parsed straight out of the plan line,
            # not interpreted -- a preference on departure_time should not have
            # to re-derive them from a string. Day-level notions (travel_phase,
            # week_group, city_index) are deliberately NOT supplied: they are
            # semantic readings of the request, and imposing our definition
            # would make a disagreement look like the agent's error.
            leg = {"mode": mode, "raw": tr}
            m = re.search(r"Flight Number:\s*([A-Za-z0-9]+)", tr)
            if m:
                leg["flight_number"] = m.group(1)
            for label, key in (("Departure Time", "departure_time"),
                               ("Arrival Time", "arrival_time")):
                m = re.search(rf"{label}:\s*([0-9]{{1,2}}:[0-9]{{2}})", tr)
                if m:
                    leg[key] = m.group(1)
            o, d = _leg_endpoints(tr)
            if not (o and d):
                o, d = o_c, d_c          # already parsed above
            if o and d:
                leg["origin"], leg["dest"] = o, d
                if mode in ("self-driving", "taxi"):
                    g = get_ground_transport(o, d, mode)
                    if g.get("available"):
                        leg.update({k: g[k] for k in ("distance_km", "duration", "cost")})
                elif mode == "flight" and leg.get("flight_number"):
                    # Same lookup the environment does: it resolves the price
                    # from the flight number via tools/flights/apis.py, which is
                    # the tool layer over the same CSV `search_flights` reads --
                    # not evaluation code. Resolving it here keeps flights
                    # consistent with every other entity, which all arrive with
                    # their attributes attached.
                    row = _flight_by_number(o, d, leg["flight_number"])
                    if row:
                        leg.update({k: row[k] for k in
                                    ("price", "elapsed", "date", "distance")
                                    if k in row})
            day["transportation"].append(leg)
        out.append(day)
    return out
