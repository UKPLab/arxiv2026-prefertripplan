"""Tool schemas for the agentic runtime.

Provider-neutral OpenAI-style function schemas, plus the serialiser that turns
a tool result into the text the model sees.

Two invariants, both load-bearing for the experiment:

  * **One tool surface.** Names, arguments and return shapes are fixed; the
    agent's information access is defined by what the tools return, never by
    reshaping the interface.
  * **One serialisation path.** Every tool result is rendered by
    ``render_records`` in the same ``- k=v  |  k=v`` layout the direct planner
    sees, so an agentic-vs-direct difference is an information difference and
    not a formatting one.
"""
from __future__ import annotations

import json
from typing import Any

# Result size is bounded by CHARACTERS, not row count. Measured on the real
# pools, a row-count cap binds the city tables pointlessly (none exceeds 65
# rows) while the character cap is what binds flights -- 50 flight rows is
# ~9,900 chars. One knob, tied to the context ceiling that matters.
# MAX_LIMIT remains only as a sanity bound on an agent-supplied `limit`.
MAX_LIMIT = 200
MAX_RESULT_CHARS = 8000


def _fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


_CITY = {"city": {"type": "string", "description": "Exact city name, e.g. 'Tampa'."}}
_PAGE = {
    "limit": {"type": "integer",
              "description": f"Optional cap on rows returned (max {MAX_LIMIT}). "
                             f"Results are size-limited regardless; a truncated "
                             f"result reports next_offset."},
    "offset": {"type": "integer", "description": "Row offset for paging (default 0)."},
}

SEARCH_TOOLS = [
    _fn("get_trip_dates",
        "Days of the week for the trip's dates. Read the dates out of the query "
        "and pass them in; several preferences depend on weekday vs weekend.",
        {"dates": {"type": "array", "items": {"type": "string"},
                   "description": "Trip dates as YYYY-MM-DD, in order."}},
        ["dates"]),
    _fn("get_cities_in_state",
        "List the cities available in a US state. Use when the destination is a "
        "state rather than a city and you must choose which cities to visit.",
        {"state": {"type": "string", "description": "US state name, e.g. 'Texas'."}},
        ["state"]),
    _fn("search_attractions",
        "Attractions in a city, with categories and rating.",
        {**_CITY, **_PAGE}, ["city"]),
    _fn("search_restaurants",
        "Restaurants in a city, with cuisines, average cost per person, and rating.",
        {**_CITY, **_PAGE}, ["city"]),
    _fn("search_accommodations",
        "Accommodations in a city, with room type, price, rating, house rules, "
        "minimum nights, and maximum occupancy.",
        {**_CITY, **_PAGE}, ["city"]),
    _fn("search_flights",
        "Flights between two cities on one date, with flight number, price, and times.",
        {"origin": {"type": "string"}, "destination": {"type": "string"},
         "date": {"type": "string", "description": "YYYY-MM-DD."}, **_PAGE},
        ["origin", "destination", "date"]),
    _fn("get_ground_transport",
        "Self-driving or taxi between two cities: distance, duration, and cost. "
        "Legs taking more than a day are reported unavailable and must not be used.",
        {"origin": {"type": "string"}, "destination": {"type": "string"},
         "mode": {"type": "string", "enum": ["self-driving", "taxi"]}},
        ["origin", "destination", "mode"]),
]


# Filters mirror the operators the preference bank actually uses. A filter call
# names the same pool a search call would, so it is stateless and independently
# replayable from the trajectory.
_FILTER_ARG = {
    "filters": {
        "type": "array",
        "description": "Predicates ANDed together, e.g. "
                       "[{\"field\": \"cost\", \"op\": \"<=\", \"value\": 75}, "
                       "{\"field\": \"cuisines\", \"op\": \"in\", "
                       "\"value\": [\"French\", \"Italian\"]}]. On list-valued "
                       "fields (categories, cuisines, house_rules_list), 'in' "
                       "means the row has any of the values and 'contains_all' "
                       "means it has all of them.",
        "items": {"type": "object", "properties": {
            "field": {"type": "string"},
            "op": {"type": "string",
                   "enum": ["==", "!=", "<", "<=", ">", ">=",
                            "in", "not_in", "contains_all", "between"]},
            "value": {}}},
    },
}
_WHERE = {
    "entity": {"type": "string",
               "enum": ["attractions", "restaurants", "accommodations", "flights"]},
    "city": {"type": "string", "description": "For attractions/restaurants/accommodations."},
    "origin": {"type": "string", "description": "For flights."},
    "destination": {"type": "string", "description": "For flights."},
    "date": {"type": "string", "description": "For flights, YYYY-MM-DD."},
}

FILTER_TOOLS = [
    _fn("filter_items",
        "Filter a pool by the query's hard constraints and preferences, instead "
        "of reading every row and filtering by eye. Reports how many rows were "
        "dropped, so an over-strict filter is visible.",
        {**_WHERE, **_FILTER_ARG,
         "sort_by": {"type": "string", "description": "Field to sort the matches by."},
         "desc": {"type": "boolean", "description": "Sort descending."}},
        ["entity"]),
    _fn("aggregate_items",
        "min / max / avg / sum / count over a filtered pool. Use for numeric "
        "preferences stated as optimisations (maximise average rating, minimise "
        "total cost) rather than computing the aggregate by hand.",
        {**_WHERE, **_FILTER_ARG,
         "field": {"type": "string", "description": "Numeric field to aggregate."},
         "op": {"type": "string", "enum": ["min", "max", "avg", "sum", "count"]}},
        ["entity", "field", "op"]),
]

NOTEBOOK_TOOLS = [
    _fn("notebook_write",
        "Save a short note for later, e.g. the shortlist for one city or leg. "
        "Use this instead of relying on earlier tool output staying in view.",
        {"label": {"type": "string", "description": "Short label, e.g. 'Dallas restaurants'."},
         "content": {"type": "string"}}, ["label", "content"]),
    _fn("notebook_list", "List the labels of everything saved so far.", {}, []),
    _fn("notebook_read", "Read back one saved note by its index.",
        {"index": {"type": "integer"}}, ["index"]),
]

REVISE_TOOL = _fn(
    "revise_checks",
    "Correct your own checks when one of them is wrong -- it crashed, it tests "
    "the wrong thing, or you have realised your phase-1 reading of the request "
    "was off. Only for fixing the CHECK; use it when the check is at fault, not "
    "when the plan is. A check that currently passes cannot be rewritten.",
    {"checks": {"type": "object",
                "description": "Map of constraint id -> new Python source, for "
                               "the ids you are changing only. New ids may be "
                               "added; existing passing ones may not be altered."},
     "reason": {"type": "string",
                "description": "Why the check was wrong, in one line."}},
    ["checks", "reason"])

SUBMIT_TOOL = _fn(
    "submit_plan",
    "Submit the finished plan. This ends the episode, so call it only when the "
    "plan is complete.",
    {"travel_plan": {"type": "string",
                     "description": "The full plan in the exact 'Day N:' line format "
                                    "given in the instructions."}},
    ["travel_plan"])

SEARCH_TOOL_NAMES = [t["function"]["name"] for t in SEARCH_TOOLS]
FILTER_TOOL_NAMES = [t["function"]["name"] for t in FILTER_TOOLS]
NOTEBOOK_TOOL_NAMES = [t["function"]["name"] for t in NOTEBOOK_TOOLS]


def toolset(*, search: bool, notebook: bool) -> list[dict]:
    """The mounted tool set. ``submit_plan`` is always included.

    ``search`` and ``notebook`` are switches for tests and future variants;
    the planner mounts both.
    """
    out: list[dict] = []
    if search:
        out += SEARCH_TOOLS + FILTER_TOOLS
    if notebook:
        out += NOTEBOOK_TOOLS
    out.append(SUBMIT_TOOL)
    return out


# --------------------------------------------------------------------------- #
# Result serialisation                                                        #
# --------------------------------------------------------------------------- #
def _row(d: dict) -> str:
    return "- " + "  |  ".join(f"{k}={v}" for k, v in d.items() if v not in (None, ""))


def render_records(records: list[dict], *, total: int, offset: int,
                   limit: int = MAX_LIMIT, header: str = "",
                   max_chars: int = MAX_RESULT_CHARS) -> str:
    """Render a page of rows in the direct planner's layout.

    Truncation drops whole rows from the tail and says so -- never a partial
    row, which would be a hallucination vector.
    """
    shown = records[offset:offset + limit]
    lines = [_row(r) for r in shown]
    body, used = [], 0
    for ln in lines:
        if used + len(ln) + 1 > max_chars:
            break
        body.append(ln); used += len(ln) + 1
    truncated = len(body) < len(shown)
    nxt = offset + len(body)
    head = header or f"{total} result(s)"
    parts = [f"{head}; showing {len(body)} (offset {offset})."]
    if nxt < total:
        parts.append(f"{total - nxt} more available -- call again with offset={nxt}.")
    if truncated:
        parts.append("Output truncated to fit the result size limit.")
    if not body:
        parts.append("(no rows)")
    return "\n".join(parts + body)


def render_scalar(obj: Any) -> str:
    if isinstance(obj, dict):
        return "  |  ".join(f"{k}: {v}" for k, v in obj.items())
    return json.dumps(obj, ensure_ascii=False)
