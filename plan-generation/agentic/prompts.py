"""Prompt templates for the agent's phases.

Built on a **shared trunk** taken verbatim from the direct planner's
`PLANNER_INSTRUCTION_PTP` (`plan-generation/generate_plans.py`): the task
framing, the profile-vs-query priority rule, the `Day N:` output contract, and
the Ithaca->Charlotte example. The agentic planner therefore receives identical
task instructions to the direct one, and differs only in having a loop, tools,
and a verifier -- so a direct-vs-agentic delta cannot be confounded with a
rewritten prompt.

TravelPlanner's own ReAct/Reflexion prompts are NOT vendored in this repo
(`tools/planner/apis.py` imports a missing `agents.prompts`), so the
phase-specific blocks below are ours. Each template records its provenance so
the paper can state exactly what was reused.

Nothing here may render a dataset field outside `record_view.AgentView` --
`diagnostics/verify_no_leakage.py` scans assembled prompts to enforce it.
"""
from __future__ import annotations

import _paths  # noqa: F401
import generate_plans as GP
from record_view import AgentView, HarnessView

PROVENANCE = ("trunk: PLANNER_INSTRUCTION_PTP (generate_plans.py:219-259), verbatim; "
              "phase blocks: original")


def _trunk() -> str:
    """The direct planner's instruction with its fill-slots removed.

    Everything up to `Given information:` -- i.e. the task statement, the
    priority rule, and the worked example -- is reused exactly.
    """
    t = GP.PLANNER_INSTRUCTION_PTP
    cut = t.find("Given information:")
    return (t[:cut] if cut > 0 else t).rstrip()


SYSTEM = """{trunk}

You are working as an agent: you may call tools, keep notes, check your own work,
and revise before submitting. Work in phases and do not skip ahead.

Rules that override anything else:
- Every entity in the plan (flight number, restaurant, accommodation, attraction)
  must come from tool results or the information given to you. Never invent one.
- Query constraints come first. The profile is motivating context and must yield
  to the query wherever they disagree.
- The plan you submit must use the exact `Day N:` layout shown above."""


# --------------------------------------------------------------------------- #
# Phase 1 -- SPEC. Sees query + profile only; no plan exists yet.              #
# --------------------------------------------------------------------------- #
SPEC_PSEUDOCODE = """PHASE 1a of 6 -- restate the trip facts and the constraints.

First read the trip's facts out of the request: how many days, the exact dates,
how many travellers, and the budget. Your checks will run against these values,
so if you misread them your checks will be wrong -- nothing else supplies them.

Then write down every requirement this trip must satisfy.
Cover three kinds:

  commonsense  unstated rules any sensible itinerary obeys (e.g. avoid or minimize 
               repetition; stay somewhere each night except the last; return to
               the origin; visit only cities on the itinerary etc.). These should
               be checks regarding completeness, consistency and validity of information, 
               reasonable considerations and scoping for within city and inter-city 
               activities, and the constraint values on the plan's entity side, if 
               any, e.g. any limiting or binding requirement.
  hard         requirements stated in the query that must hold (budget, cuisines,
               room type, house rules, transport restrictions)
  preference   softer wishes expressed in the query, including any conditional,
               ranked, or temporal phrasing etc. Capture the exact scope: does
               it apply to every item, or is one occurrence enough?

Every constraint must come from the trip request or from commonsense. The
traveller's profile is deliberately not shown here and is not a constraint
source: the request overrides it wherever the two disagree, so a check written
from the profile can demand the opposite of what is required.

For each, give a stable id and a one-line pseudo-code predicate. The pseudo-code
is the record of what you believe must hold -- write it so someone else could
check the plan by hand from it alone.

Scope is the part that is easiest to get wrong, so state it:

  universal    "every restaurant must be rated 3.5 or better" -- one bad meal fails it
  existential  "visit a museum at some point" -- one museum satisfies it
  count        "at most one expensive dinner"
  coverage     "we want to try Italian, Indian and Chinese" -- EVERY cuisine in
               the set must appear somewhere in the trip. This is not the same
               as the existential reading: one Italian meal does not satisfy it,
               and it is not the universal reading either, which would demand
               every single restaurant be one of the three. Coverage quantifies
               over the SET, and asks the plan to cover it.

Write which one you mean.

Costs split in two, and the split matters.

PER-ITEM costs are yours to check, and most preferences are about them -- "every
stay at $360 per person per night or less", "restaurants at $60 or more must be
rated 3.5+". Read the units the request gives: a listing price is per room per 
night, so "per person per night" needs dividing by how many the room sleeps, and 
your check has the numbers to do it. Similar estimates for road-based transportation 
e.g. self-driving or taxi with self-driving supporting 5 person in a vehicle while 
taxi supporting 4 person in a vehicle. The number of vehicle required would also 
then be required to be calculated for cost estimation.

The TOTAL trip cost against the budget is NOT yours. The environment computes it
every verification round and is authoritative, because that arithmetic is not a
lookup -- flights per traveller, ground transport per vehicle at its own
occupancy, accommodation per room for the nights used, meals per traveller per
sitting. A hand-rolled total will disagree with the real one and send you
chasing a violation that is not there. Keep a rough running estimate while
building so you do not overshoot, and let the environment settle the figure.

Return JSON only:
{{"facts": {{"days": 3, "dates": ["2025-11-02", "2025-11-03", "2025-11-04"],
           "people": 1, "budget": 2310}},
 "constraints": [
   {{"id": "C1", "kind": "commonsense", "source": "commonsense",
     "text": "no restaurant is visited more than once across the whole trip",
     "pseudocode": "FOR ALL meals m IN plan: COUNT(restaurant(m) IN plan) == 1"}},
   {{"id": "H1", "kind": "hard", "source": "query",
     "text": "expressed interest in a set of cuisines during the trip",
     "pseudocode": "FOR ALL c IN {{Italian, Indian, Chinese}}: EXISTS r IN restaurants(plan): c IN r.cuisines"}},
   {{"id": "P1", "kind": "preference", "source": "query",
     "text": "any restaurant costing 60 or more per person must be rated at least 3.5",
     "pseudocode": "FOR ALL r IN restaurants(plan): r.cost >= 60 IMPLIES r.rating >= 3.5"}},
   {{"id": "P2", "kind": "preference", "source": "query",
     "text": "visit at least one nature or parks attraction during the trip",
     "pseudocode": "EXISTS a IN attractions(plan): 'Nature & Parks' IN a.categories"}},
   {{"id": "P3", "kind": "preference", "source": "query",
     "text": "at most one dinner over 75 per person across the trip",
     "pseudocode": "COUNT(d IN dinners(plan) WHERE d.cost > 75) <= 1"}}
]}}

Those five are the shape, not the content -- take the actual constraints from
the request below. A trip usually has several commonsense entries, whatever hard
constraints the request states, and one or two preferences.

Trip request:
{query}"""

SPEC_PYTHON = """PHASE 1b of 6 -- implement the checks.

Write one Python function per constraint id from 1a. Same ids, no new ones.

Each must be exactly:

    def check(plan, ctx):
        # plan: list of day dicts with keys days, current_city, transportation,
        #       breakfast, attraction, lunch, dinner, accommodation.
        #       Values are strings; "-" means absent. Entity strings look like
        #       "Name, City" and transportation like
        #       "Flight Number: F123, from A to B, Departure Time: .., Arrival Time: .."
        # ctx:  the facts YOU stated in 1a, plus the resolved attributes of
        #       everything the plan names:
        #       {{"days": int, "people": int, "budget": number,
        #         "dates": [ISO...], "weekday": {{date: weekday_name}},
        #         "items": [ {{"day": 1, "current_city": str,
        #            # the day's cities, already parsed -- never re-parse
        #            # current_city yourself, a hand-written "from A to B"
        #            # regex is the single most common way a check goes wrong
        #            "origin": str|None, "dest": str|None,
        #            "city": str|None,          # where the day's activities are
        #            "is_travel_day": bool,     # True iff origin and dest are set
        #            "restaurants": [{{"meal","name","city","found",
        #                            "cost","rating","cuisines"}}],
        #            "attractions": [{{"name","city","found",
        #                            "rating","categories"}}],
        #            "accommodation": {{"name","city","found","cost","rating",
        #                              "room_type","house_rules_list",
        #                              "minimum_nights","maximum_occupancy"}},
        #            "transportation": [{{"mode","raw","origin","dest",
        #                            "flight_number","departure_time",
        #                            "arrival_time",          # flights
        #                            "price","elapsed","distance",
        #                            "distance_km","duration","cost"}}]  # ground
        #                            }} ... ] }}
        #       `found: False` means the name did not resolve.
        #
        #       Values are RAW, and the units differ per field. Getting these
        #       wrong is the likeliest way for a check to be confidently wrong:
        #         restaurant  `cost` is per person, per sitting
        #         accommodation `cost` is the LISTING price -- per room, per
        #                     night. A "per person per night" preference needs
        #                     dividing by how many the room sleeps
        #                     (`maximum_occupancy`), with proper rounding, and a 
        #                     whole-stay figure needs multiplying by the nights used
        #         ground leg  `cost` is PER VEHICLE for the leg, not per party
        #                     and not per person. A party larger than one
        #                     vehicle holds needs more than one vehicle, so a
        #                     per-party figure is the leg cost times the number
        #                     of vehicles required
        #         flight      `price` is per traveller for the leg, resolved
        #                     from the flight number -- multiply by the party
        #                     size for a per-party figure
        return (True_or_False, "short reason")

Constraints on your code, enforced automatically:
- allowed imports: json, re, math, datetime, collections, statistics, itertools,
  functools, operator. Nothing else.
- no file, network, or process access; no eval/exec/open/__import__.
- a function that raises is discarded, and the constraint counts as unverified --
  so prefer defensive parsing over clever parsing.

You cannot query the database from inside a check, but you do not need to:
ctx["items"] already carries the attributes of everything the plan names. Use
those for per-item comparisons. Skip the trip TOTAL -- the environment reports
it. Check only what is in the plan text and in ctx.

Before deciding a constraint is not checkable, look again: per-item cost, rating,
cuisine, category, room type, house rules, flight price, departure and arrival
time, and the day's origin/dest are all already in ctx["items"]. A constraint
over any of those IS checkable. Only if the data is genuinely absent, write the
check as `return (True, "not checkable from the plan alone")` -- that exact
reason, so it can be counted. A check that always returns True enforces nothing,
so use this as a last resort rather than a default.

Return JSON only:
{{"checks": {{"C1": "def check(plan, ctx):\\n    ...", "H1": "def check(plan, ctx):\\n    ..."}}}}

Your constraints from 1a:
{pseudocode}"""


# --------------------------------------------------------------------------- #
# Phases 2-3 -- TOOL USE and CONSTRUCT                                          #
# --------------------------------------------------------------------------- #
SEARCH = """PHASES 2-3 of 6 -- gather what you need, then write the plan.

You start with no information. Everything in the plan must come from a tool
result; never invent a flight number, restaurant, hotel or attraction.

Tools available now:

  get_trip_dates(dates)          days of the week for the dates YOU read out of
                                 the request. Pass them all at once.
  get_cities_in_state(state)     cities available in a US state
  search_attractions(city)       )
  search_restaurants(city)       ) the full pool for a city
  search_accommodations(city)    )
  search_flights(origin, destination, date)
  get_ground_transport(origin, destination, mode)   mode: self-driving | taxi

  filter_items(entity, ...)      apply your constraints to a pool instead of
                                 reading every row. Predicates are ANDed:
                                   {{"field": "cost", "op": "<=", "value": 75}}
                                   {{"field": "cuisines", "op": "in",
                                    "value": ["French", "Italian"]}}
                                 Operators: == != < <= > >= in not_in
                                 contains_all between. On list fields
                                 (categories, cuisines, house_rules_list) 'in'
                                 means the row has ANY of the values and
                                 'contains_all' means it has ALL of them.
                                 It reports how many rows were dropped, so you
                                 can tell an over-strict filter from an empty
                                 pool.
  aggregate_items(entity, field, op, ...)
                                 min / max / avg / sum / count over a filtered
                                 pool. Use it for preferences phrased as
                                 optimisations rather than averaging by hand.

  notebook_write(label, content) save a shortlist so you need not re-query
  notebook_list() / notebook_read(index)

A suggested order:
  1. get_trip_dates for the whole trip -- several constraints turn on weekday
     vs weekend.
  2. If the destination is a state rather than a city, get_cities_in_state and
     choose which to visit, and in what ORDER. The itinerary is a chain:

         origin -> city1 -> city2 -> ... -> origin

     one leg between consecutive stops, one leg home. What has to exist is each
     leg of the order you intend, on that leg's own date -- not each city
     measured against the origin. An intermediate city is never reached from
     the origin directly, so checking it against the origin tells you nothing.

     Order matters as much as the choice: the same cities in a different
     sequence are a different set of legs, and one arrangement can be
     travellable when another is not. If a leg has no flight on its date and no
     usable ground option, reorder the cities or swap one out rather than
     forcing it.
  3. Per chosen city: attractions, restaurants, accommodations. Filter them by
     the constraints you wrote in phase 1 rather than eyeballing the rows.
  4. Per leg: search_flights on that leg's date, and get_ground_transport for
     both self-driving and taxi.
  5. notebook_write your shortlist per city and per leg as you go.

A ground leg reported unavailable cannot be used, whatever its distance.

Watch the mapping from a constraint's wording to an operator. "Not a shared
room" is `room_type != "Shared room"`, not `room_type == "Not Shared room"` --
the latter matches nothing.

Then write the plan in the `Day N:` format. Your plan will be checked before it
counts as final, so submit it when it is complete rather than when it is
perfect.

Traveller profile:
{profile}

Trip request:
{query}"""


# --------------------------------------------------------------------------- #
# Phases 4-5 -- VERIFY and REPAIR                                             #
# --------------------------------------------------------------------------- #
VERIFY_FEEDBACK = """PHASES 4-5 of 6 -- your plan was checked. Round {round} of {max_rounds}.

Two independent reports. They cover different things, so read both.

Environment report -- authoritative on the total cost (priced with per-traveller
flights, per-vehicle ground transport, per-room accommodation and per-sitting
meals) and on whether an entity exists in the sandbox. It cannot see your
preferences:
{oracle}

Your own checks, from phase 1 -- the only thing covering the preferences:
{checks}

{unverified}{revise}
Produce the corrected FULL plan in the `Day N:` format, not a description of the
change. Rules:
  - if an entity is rejected as invalid, it is not in the sandbox; pick another
    from a tool result rather than rewording it
  - if a violation cannot be fixed with what you have, call more tools first
  - if no combination in the destination can satisfy the constraints, say so
    plainly instead of submitting a plan you know is invalid"""

REVISE_AVAILABLE = """
If one of your own checks is at fault rather than the plan -- it crashed, it
tests the wrong thing, or you can now see your phase-1 reading of the request
was off -- call revise_checks with just the ids you are changing and a one-line
reason. Verification then re-runs on the same plan against the corrected checks,
without spending a repair round.

Use it for the check, not for the plan. A check that currently passes cannot be
rewritten, so this cannot be used to make a failing plan pass.

"""

BUDGET_EXHAUSTED = """You have reached the step budget. Submit the best plan you have now
with submit_plan, in the exact `Day N:` layout. Do not call any other tool, and
do not explain -- an incomplete plan scores better than none.

Use `-` for anything you were unable to fill in."""

FALLBACK_COMPLETION = """{trunk}

Produce the plan now, as plain text in the `Day N:` layout. No tools are
available. Use only the information below -- do not invent entity names -- and
put `-` wherever you have nothing.

Traveller profile:
{profile}

Trip request:
{query}

Information gathered so far:
{notes}"""


def system_prompt() -> str:
    return SYSTEM.format(trunk=_trunk())


def spec_pseudocode(av: AgentView) -> str:
    """SPEC sees the request only. The profile is withheld deliberately -- see
    the note in the template and `verifier.SOURCES`."""
    return SPEC_PSEUDOCODE.format(query=av.query)


def spec_python(pseudocode_block: str) -> str:
    return SPEC_PYTHON.format(pseudocode=pseudocode_block)


def tool_use(av: AgentView) -> str:
    return SEARCH.format(profile=av.profile, query=av.query)


def assemble_all_for_audit(av: AgentView, hv: HarnessView) -> str:
    """Every prompt this record could produce, concatenated -- for the leakage
    scan in `diagnostics/verify_no_leakage.py`."""
    return "\n".join([
        system_prompt(), spec_pseudocode(av), spec_python("[C1] ... example"),
        tool_use(av),
        VERIFY_FEEDBACK.format(round=1, max_rounds=5, oracle="...", checks="...",
                               unverified="", revise=REVISE_AVAILABLE),
        BUDGET_EXHAUSTED,
        FALLBACK_COMPLETION.format(trunk=_trunk(), profile=av.profile,
                                   query=av.query, notes="..."),
    ])
