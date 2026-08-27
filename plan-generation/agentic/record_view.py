"""What the agent is allowed to see, enforced by projection.

The dataset row carries far more than a traveller would ever state: the
evaluator's own structured preference specs, drift provenance, feasibility
metadata, the oracle tour. Any of it in the prompt would be leakage. Rather
than trusting prompt-assembly code to avoid them, the runtime never handles a
raw row -- it projects each row into an `AgentView` (what may be shown) and a
`HarnessView` (what the runtime may use internally), and the projection is the
only path from one to the other.

  AGENT sees      query, profile
  HARNESS uses    id, date, days, people_number, org, dest, visiting_city_number

Every HARNESS field is restated in the NL `query`, so using it internally
reveals nothing the agent was not told -- and each use is deliberately one the
agent cannot observe:

  n_cities, dest   size the tool-call budget. Experiment configuration, not
                   information: the agent never learns the number.
  people           handed to the environment oracle for per-person costing.
                   Faithful to TravelPlanner, whose `ReactEnv` reads
                   `people_number` off the day dict -- the environment knows the
                   party size, the agent still has to read it from the query.
  dates, days,     REPORTING ONLY. The agent states its own reading of these in
  budget           SPEC, and its checks run against that; these are kept solely
                   to score fact-extraction accuracy afterwards. Nothing derived
                   from them reaches the agent or its checks.
  id               bookkeeping. `local_constraint` is deliberately excluded even
though it is only a restatement of the query: the direct planner does not
receive it either (`build_plan_prompt` passes profile/query/reference_information
only), and parity with the direct planner matters more than convenience.

Also unavailable by construction: web search. The agent's action space is
exactly `schemas.toolset()`, and the verifier sandbox allowlists imports with no
network module among them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Shown to the agent.
AGENT_PROMPT_FIELDS = ("query", "profile")
# Used by the runtime, never rendered into a prompt. Each is restated in `query`.
HARNESS_FIELDS = ("id", "date", "days", "people_number", "org", "dest",
                  "visiting_city_number", "budget")

# Everything else is forbidden. Listed explicitly so a new dataset column cannot
# silently become visible: the projection asserts the row's keys are covered.
FORBIDDEN_FIELDS = (
    "preferences_json",        # the evaluator's ground-truth preference specs
    "preferences_shorthand",   # ditto, compact form
    "preference_pair",         # pairing type/subtype
    "profile_json",            # structured mirror of the profile prose
    "drift_trace",             # which traits were drifted
    "profile_drift",           # drift mode
    "query_source",            # constraints-only half of the query
    "query_preferences",       # preferences-only half
    "local_constraint",        # structured constraints; the direct planner is not given them either
    "budget_update",           # escalation multiplier
    "trip_context",            # inferred trip framing
    "source_id", "level",      # split bookkeeping / difficulty label
    "feasibility_metadata",    # selected tour, candidate cities, transport caps
    "solution_information",    # feasibility-filtered pool
    "reference_information",   # the pre-retrieved pool: the agent must gather it
)


@dataclass
class AgentView:
    """Exactly what may reach the model."""
    query: str
    profile: str

    def as_prompt_fields(self) -> dict[str, str]:
        return {"query": self.query, "profile": self.profile}


@dataclass
class HarnessView:
    """Runtime-internal. Never rendered."""
    id: int
    dates: list[str]
    days: int
    people: int
    org: str
    dest: str
    n_cities: int
    budget: float | None = None     # fact-accuracy reporting, and the oracle's
                                    # budget threshold -- structured input, like
                                    # `people`, never rendered to the agent

    def tool_call_budget(self, extra_calls: int, dest_is_state: bool) -> int:
        """Tool calls allowed during the tool-use phase, derived from the instance.

        The minimum a perfect agent needs, measured across all 225 `test`
        records and matching exactly:

            1                get_trip_dates        (whole trip in one call)
            1                get_cities_in_state   (only if dest is a state)
            3 per city       attractions, restaurants, accommodations
            3 per leg        1 flight + 2 ground modes; legs = cities + 1
            -----------------------------------------------------------
            6C + 4 (+1 if the destination is a state)

        `extra_calls` is headroom on top of that: pagination when a result is
        truncated, probing candidate cities before committing to a tour, and
        notebook, filter and aggregate calls, which all draw on the same budget.
        Filtering is refinement rather than retrieval, so it is not in the
        minimum -- a plan can be built without it, just less accurately.
        A flat budget would be wrong in both directions -- it starves 3-city
        records while handing 1-city ones triple the headroom they need, which
        is a per-instance difficulty confound in a compute-controlled comparison.
        """
        return 6 * self.n_cities + 4 + (1 if dest_is_state else 0) + extra_calls


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class LeakageError(RuntimeError):
    pass


def project(row: dict) -> tuple[AgentView, HarnessView]:
    """The only sanctioned way to turn a dataset row into runtime inputs.

    Raises if the row carries an unrecognised column, so a future dataset
    revision cannot quietly widen what the agent can be shown.
    """
    known = set(AGENT_PROMPT_FIELDS) | set(HARNESS_FIELDS) | set(FORBIDDEN_FIELDS)
    unknown = set(row) - known
    if unknown:
        raise LeakageError(
            f"unclassified dataset column(s): {sorted(unknown)}. Add each to "
            f"HARNESS_FIELDS or FORBIDDEN_FIELDS in record_view.py before use.")

    agent = AgentView(query=row["query"], profile=row["profile"])
    harness = HarnessView(
        id=int(row["id"]), dates=list(row["date"]), days=int(row["days"]),
        people=max(int(row["people_number"]), 1), org=row["org"], dest=row["dest"],
        n_cities=max(int(row["visiting_city_number"]), 1),
        budget=_num(row.get("budget")))
    return agent, harness


def audit_text(text: str, row: dict) -> list[str]:
    """Scan an assembled prompt for content the agent should not have.

    Leakage means content present in the prompt but NOT already inside the
    agent-visible fields. That distinction matters: the dataset defines
    ``query == query_source + " " + query_preferences``, so both halves appear
    verbatim inside ``query`` by construction. Seeing them there is not leakage;
    receiving them as *separate fields* would be, because it hands the agent a
    clean split between constraints and preferences that the prose does not.

    Used by ``diagnostics/verify_no_leakage.py`` on real prompts.
    """
    visible = ((row.get("query") or "") + "\n" + (row.get("profile") or "")).lower()
    low = text.lower()

    hits: list[str] = []
    for fname in FORBIDDEN_FIELDS:
        v = row.get(fname)
        if not isinstance(v, str) or len(v) < 24:
            continue
        probe = v.strip()[:80].lower()
        if not probe or probe not in low:
            continue
        if probe in visible:
            continue          # already inside query/profile: not leakage
        hits.append(f"verbatim content of forbidden field {fname!r}")

    for marker in ("preferences_json", "preferences_shorthand", "local_constraint",
                   "feasibility_metadata", "selected_tour", "profile_drift",
                   "bank_id", "indicative_rationale", "stat_based_value"):
        if marker in low:
            hits.append(f"internal marker {marker!r} present")
    return sorted(set(hits))
