"""The agent loop: SPEC -> GATHER -> CONSTRUCT -> VERIFY -> REPAIR -> SUBMIT.

Budgets are per phase because the phases cost differently (DESIGN.md): SPEC is
one or two long turns, tool use many cheap ones, VERIFY/REPAIR a few expensive
ones. The tool-call budget is derived per record as
`6C + 4 (+1 state) + extra_tool_calls`, since a flat budget would hand a
1-city record triple the headroom a 3-city one gets.

Verification uses two signals together, and neither is optional:

  * TravelPlanner's `ReactReflectEnv` -- exact cost and entity validity,
    correct by construction, but silent on preferences;
  * the agent's own executable checks -- the only thing that can speak to
    preferences, but only as good as the agent wrote them.

Their disagreement (authored checks pass, environment reports invalid) is the
one signal that may re-open SPEC, and it exists only because both run.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field

import _paths  # noqa: F401
import db_tools as T
import prompts as P
import schemas as S
import verifier as V
from record_view import AgentView, HarnessView
from telemetry import StepRecord, Trajectory

# --------------------------------------------------------------------------- #
# Per-record scratchpad                                                       #
# --------------------------------------------------------------------------- #
# TravelPlanner ships a Notebook among its six agent tools
# (travelplanner-ports/tools/notebook/apis.py); this is the same idea with a
# string payload, since our tool results are already rendered text. In CoALA's
# terms (arXiv:2309.02427) it is working memory: somewhere to put a per-city
# shortlist so the model need not keep 30 tool results in view, which is what
# drives the context ceiling.
MAX_ENTRIES = 40
MAX_CHARS = 4000


@dataclass
class Notebook:
    entries: list[dict] = field(default_factory=list)

    def write(self, label: str, content: str) -> str:
        if len(self.entries) >= MAX_ENTRIES:
            return f"Notebook is full ({MAX_ENTRIES} entries). Reuse an index instead."
        self.entries.append({"label": str(label)[:120],
                             "content": str(content)[:MAX_CHARS]})
        return f"Saved at index {len(self.entries) - 1}."

    def list(self) -> str:
        if not self.entries:
            return "Notebook is empty."
        return "\n".join(f"[{i}] {e['label']}" for i, e in enumerate(self.entries))

    def read(self, index: int) -> str:
        try:
            e = self.entries[int(index)]
        except (ValueError, TypeError, IndexError):
            return f"No entry at index {index}. Use notebook_list to see what exists."
        return f"[{index}] {e['label']}\n{e['content']}"

    def dump(self) -> str:
        """Everything, for the plain-completion fallback."""
        return "\n\n".join(f"[{i}] {e['label']}\n{e['content']}"
                           for i, e in enumerate(self.entries)) or "(nothing gathered)"


# --------------------------------------------------------------------------- #
# Environment oracle -- TravelPlanner's ReactReflectEnv, used directly        #
# --------------------------------------------------------------------------- #
# A thin adapter, not a reimplementation: the cost model, entity lookups,
# message strings and retry bookkeeping are all the original
# travelplanner-ports/tools/planner/env.py (arXiv:2402.01622). Nothing here
# recomputes a price -- the whole value of this signal is that it is correct by
# construction and matches what the grader charges, and a parallel
# implementation would only add something that can drift.
#
# `ReactReflectEnv.run(day)` scores ONE day and tracks retry_step /
# is_terminated across calls (max_retry_step=3); the adapter aggregates per-day
# results into one plan-level message.
#
# IMPORT COST: env.py pulls evaluation.hard_constraint, which instantiates all
# five tool classes at module import -- ~8s and a second in-memory flights
# table alongside db_tools. Hence the lazy import, warmed once from the main
# thread via preload_oracle().
_LOCK = threading.Lock()
_ENV = None
_COST = re.compile(r"cost of your plan is ([\d.]+) dollars")
_REASONS = re.compile(r"\d+\.\s*(.*?)\s*(?=\d+\.|$)", re.S)


def _env():
    """Build the real ReactReflectEnv once, lazily."""
    global _ENV
    if _ENV is not None:
        return _ENV
    with _LOCK:
        if _ENV is None:
            import sys
            root = str(_paths.ROOT)                 # env.py needs `evaluation.*`
            if root not in sys.path:
                sys.path.insert(0, root)
            from tools.planner.env import ReactReflectEnv
            _ENV = ReactReflectEnv()
    return _ENV


def preload_oracle() -> None:
    """Warm the environment from the main thread. ~8s; do it before the pool."""
    _env()


@dataclass
class OracleVerdict:
    cost: float | None                      # None if any day was invalid
    problems: list[str] = field(default_factory=list)   # "day 2: ..." per issue
    per_day: list[str] = field(default_factory=list)    # raw ReactEnv messages
    terminated: bool = False                # env's own retry budget exhausted
    # The env prices a day and reports which named entities it could not find.
    # It is handed no budget and reads none -- `tested_data` has no such key --
    # so the comparison is made here and folded into the same verdict. Keeping
    # it separate bought nothing: the round gate ANDed them anyway, and
    # `problems` plus `cost` already tell the two failures apart afterwards.
    #
    # The threshold is the record's budget, arriving the same way people_number
    # does. An earlier version used the budget the agent had parsed, which made
    # the oracle depend on the thing it is meant to check independently -- a
    # misread budget would have moved its own goalposts. Whether the agent read
    # it correctly is a separate question, already answered by fact_accuracy.
    budget: float | None = None

    @property
    def over_by(self) -> float | None:
        if self.cost is None or not isinstance(self.budget, (int, float)):
            return None                     # a missing number is not a violation
        return round(self.cost - self.budget, 2) if self.cost > self.budget else None

    @property
    def ok(self) -> bool:
        return not self.problems and self.over_by is None

    def as_message(self) -> str:
        """What the agent sees. Plan-level, built from the env's own strings."""
        if self.problems:
            head = ("Sorry, the cost of your plan is not available because of the "
                    "following reasons:")
            return head + " " + " ".join(f"{i+1}. {p}"
                                         for i, p in enumerate(self.problems))
        msg = f"The cost of your plan is {self.cost} dollars."
        if self.over_by is not None:
            return (f"{msg} That is OVER your stated budget of {self.budget} by "
                    f"{self.over_by}. Bring the total down.")
        if isinstance(self.budget, (int, float)):
            return f"{msg} That is within your stated budget of {self.budget}."
        return msg


def evaluate_with_oracle(plan_days: list[dict], people: int, *,
                         budget: float | None = None,
                         reset: bool = True) -> OracleVerdict:
    """Score a whole plan by calling the original env once per day.

    `reset` clears `retry_step`/`is_terminated` so each record starts fresh;
    pass False within a record's repair loop to let the env's own 3-strike
    budget accumulate the way TravelPlanner intended.
    """
    env = _env()
    if reset:
        env.reset()

    total, problems, raw = 0.0, [], []
    for unit in plan_days:
        msg = env.run({**unit, "people_number": people})
        raw.append(msg)
        m = _COST.search(msg)
        if m:
            total += float(m.group(1))
        else:
            day = unit.get("days", "?")
            body = msg.split("reasons:", 1)[-1]
            for r in (x.strip().rstrip("\t").strip() for x in _REASONS.findall(body)):
                if r:
                    problems.append(f"day {day}: {r}")
    return OracleVerdict(cost=None if problems else round(total, 2),
                         problems=problems, per_day=raw, budget=budget,
                         terminated=bool(getattr(env, "is_terminated", False)))


@dataclass
class Budgets:
    """Every limit the loop obeys. Nothing is hardcoded below this line.

    Three scopes, and they fail differently:
      per phase   how much of each activity is allowed (spec turns, tool calls,
                  repair rounds, submit retries)
      per failure how many times a malfunction is tolerated before the loop
                  gives up on that activity (API retries, malformed tool args,
                  turns that call nothing)
      per episode hard ceilings so no record can run away (steps, tokens, wall)
    """
    # -- per phase --------------------------------------------------------
    spec_turns: int = 3             # 1 pseudo-code + 1 python + triggered revisions
    extra_tool_calls: int = 10      # adaptive headroom over the derived 6C + 4 (+1 state)
    max_tool_calls: int = 60        # absolute cap, whatever the derivation says
    verify_repair_rounds: int = 5   # CONSTRUCT -> VERIFY -> REPAIR cycles
    submit_retries: int = 1         # retries of a failed submit CALL, not a 2nd plan

    # -- per failure ------------------------------------------------------
    api_max_attempts: int = 3       # transient API errors, per single call
    request_timeout: float = 120.0  # HTTP timeout per attempt; also sizes the wall clock
    max_idle_turns: int = 3         # consecutive turns that call no tool
    # A malformed tool call already consumes the tool-call budget (see
    # _dispatch), so a model that cannot format arguments is stopped by that
    # budget without a separate counter. n_arg_errors is still recorded.

    # -- per episode ------------------------------------------------------
    # A RUNAWAY BACKSTOP, not a budget, and always derived. Every phase is
    # already bounded and those bounds sum to 36/43/49 turns for 1/2/3-city
    # records, so an absolute number never fires in normal operation -- and one
    # chosen without reference to the phase budgets fires for the WRONG reason
    # as soon as any of them is raised (--extra-search-calls 40 pushes a 3-city
    # record to 79 turns; a flat 60 would abort it and report a step overflow
    # rather than the search budget it actually is).
    #
    # Deriving it from the phase budgets means it can only ever catch a genuine
    # loop bug, which is the single thing a step ceiling is good for. There is
    # no margin: a margin would absorb a small overshoot, which is precisely the
    # bug worth catching, while doing nothing extra about an unbounded loop.
    # The ceiling is the exact sum, so any overshoot at all trips it.
    max_context_tokens: int = 100_000   # largest single prompt; see _over_budget

    # Wall clock, expressed as "how many worst-case calls is an episode allowed
    # to spend". A single call is already bounded by request_timeout x
    # api_max_attempts; the episode is not, and without a wall clock a stuck
    # record can hold a worker for hours.
    #
    # There is no absolute-seconds knob on purpose. A fixed number silently
    # decouples from the per-call bound it has to dominate -- the previous fixed
    # 900s was SHORTER than one worst-case call at 8 attempts, so it could not
    # do its job. Expressing it as a multiple of the per-call bound keeps the
    # two coherent by construction.
    max_agent_timeout_factor: float = 3.0

    def worst_case_call_seconds(self) -> float:
        """Upper bound on one LLM call: every attempt timing out, plus backoff."""
        backoff = sum(min(60.0, 2 ** i) for i in range(max(self.api_max_attempts - 1, 0)))
        return self.api_max_attempts * self.request_timeout + backoff

    def derived_wall_seconds(self) -> int:
        """Episode wall clock: `max_agent_timeout_factor` worst-case calls.

        Always derived; there is no absolute-seconds override, so it cannot
        drift below the per-call bound it exists to dominate. Checked BETWEEN
        turns, so it bounds when new work starts rather than interrupting a
        call in flight -- that is what request_timeout is for.
        """
        return int(self.max_agent_timeout_factor
                   * self.api_max_attempts * self.request_timeout)

    def derived_max_steps(self, tool_call_budget: int) -> int:
        """Exact sum of every phase's own bound. No margin, always derived.

        spec       spec_turns pseudo-code attempts + 1 python turn
        tool_use   the loop range in run_tool_use: one turn per iteration
        verify     one turn per verify/repair round, plus the spec revisions the
                   oracle-disagreement trigger can fire (capped at whatever SPEC
                   did not already spend)
        submit     submit_retries + 1 attempts, plus the plain-completion fallback
        """
        spec = self.spec_turns + 1
        tool_use = tool_call_budget + self.max_idle_turns
        verify = self.verify_repair_rounds + max(self.spec_turns - 2, 0)
        submit = self.submit_retries + 2
        return spec + tool_use + verify + submit

    # -- sizing -----------------------------------------------------------
    # One knob for result size, not two. A page-size cap and a character cap
    # bind in different places -- measured on real pools, page size bound the
    # city tables (pointlessly, since none exceeds 65 rows) while the character
    # cap bound flights (50 flight rows is ~9,900 chars). Keeping only the
    # character cap leaves the limit tied to the context ceiling that actually
    # matters; pagination still works because a truncated result reports
    # next_offset.
    tool_result_max_chars: int = S.MAX_RESULT_CHARS
    # Wall clock for executing the agent-authored verifier code. Named for what
    # it bounds: DB tool calls are also "sandboxed" in a loose sense, but they
    # are our code over local CSVs, whereas this runs a program the model wrote.
    verifier_exec_timeout: int = V.DEFAULT_TIMEOUT


class Runner:
    """One record, one episode. Not thread-safe; construct per worker task."""

    def __init__(self, chat, av: AgentView, hv: HarnessView, *,
                 budgets: Budgets, run_id: str, dest_is_state: bool,
                 max_tokens: int = 2048):
        self.chat, self.av, self.hv = chat, av, hv
        self.b = budgets
        self.max_tokens = max_tokens
        self.nb = Notebook()
        self.spec = V.Spec()
        # Adaptive budget, then the absolute cap. The derivation scales with
        # instance size; the cap bounds a pathological --extra-tool-calls.
        self.tool_call_budget = min(
            hv.tool_call_budget(budgets.extra_tool_calls, dest_is_state),
            budgets.max_tool_calls)
        self.step_ceiling = budgets.derived_max_steps(self.tool_call_budget)
        self.wall_ceiling = budgets.derived_wall_seconds()
        self.traj = Trajectory(id=hv.id, run_id=run_id, model=chat.model)
        self.messages: list[dict] = [{"role": "system", "content": P.system_prompt()}]
        self._t0 = time.time()
        self._plan: str | None = None
        # Best plan seen, not last plan seen. Scored (oracle_ok, -failing checks)
        # with ties going to whichever arrived first, so a later plan has to be
        # strictly better to displace an earlier one.
        # Candidates are recorded as they appear and scored ONCE at the end,
        # under the final check-set. Scoring online meant a plan from round 1
        # and a plan from round 3 could be compared under different checks if
        # the agent revised in between -- not a weakening problem specifically,
        # just two measurements with different rulers. Deferring makes them
        # comparable by construction, and makes a lenient check-set a constant
        # offset across all candidates, which cannot move an argmax.
        self._candidates: list[dict] = []
        self._seen_plans: set[str] = set()
        self._consec_idle = 0
        self._peak_context = 0
        self._convert = None          # set in run(); used to spot inline plans

    # -- budget bookkeeping ------------------------------------------------
    def _over_budget(self) -> str | None:
        """Token pressure is judged by CONTEXT SIZE, not cumulative spend.

        A cumulative prompt+completion total is tempting but wrong for this:
        each call resends the whole conversation, so the sum grows
        quadratically in turn count and says nothing about how large the
        context actually got. Capping it at 100K would abort around turn 13 --
        before a 3-city record has finished searching -- while its context was
        still only ~12K. That measures the budget, not the agent.

        `max_context_tokens` is the largest single prompt seen so far, which is
        what the ~96-112K ceiling in the literature refers to and the only
        quantity tied to degradation (Li et al., arXiv:2602.18998). Cumulative
        tokens and cost are still recorded per record for reporting; they just
        do not gate the loop.
        """
        if len(self.traj.steps) >= self.step_ceiling:
            return "max_total_steps"
        if self._peak_context >= self.b.max_context_tokens:
            return "max_context_tokens"
        if time.time() - self._t0 >= self.wall_ceiling:
            return "max_wall_seconds"
        return None

    def _turn(self, phase: str, user: str | None, tools, *,
              tool_choice: str = "auto") -> object:
        if user is not None:
            self.messages.append({"role": "user", "content": user})
        turn = self.chat.call(self.messages, tools, max_tokens=self.max_tokens,
                              tool_choice=tool_choice)
        self.traj.steps.append(StepRecord(
            i=len(self.traj.steps), phase=phase, text=turn.text,
            reasoning=turn.reasoning,
            tool_calls=[{"name": tc.name, "args": tc.args, "error": tc.error}
                        for tc in turn.tool_calls],
            prompt_tokens=turn.usage.prompt_tokens,
            completion_tokens=turn.usage.completion_tokens,
            reasoning_tokens=turn.usage.reasoning_tokens,
            cost_usd=turn.usage.cost_usd, latency_s=turn.latency_s,
            finish_reason=turn.finish_reason))
        self._peak_context = max(self._peak_context, turn.usage.prompt_tokens)
        # Mirror it onto the trajectory: the field was declared but never
        # assigned, so every record reported peak_context_tokens=0 at top level
        # while totals() carried the real figure. Same source as the budget
        # gate above, so the two can never disagree.
        self.traj.peak_context_tokens = self._peak_context
        return turn

    def _push_assistant(self, turn) -> None:
        m: dict = {"role": "assistant", "content": turn.text or None}
        if turn.tool_calls:
            m["tool_calls"] = [{"id": tc.id, "type": "function",
                                "function": {"name": tc.name, "arguments": tc.args_raw}}
                               for tc in turn.tool_calls]
        self.messages.append(m)

    # -- phase 1: SPEC -----------------------------------------------------
    def run_spec(self) -> None:
        """Pseudo-code first, then Python keyed to the same ids.

        The pseudo-code is canonical: it is what an LLM fallback would judge
        against, and what verifier recall/precision is scored on. Neither turn
        has seen a plan -- none exists yet.
        """
        turn = self._turn("spec_pseudocode", P.spec_pseudocode(self.av), None)
        self._push_assistant(turn)
        self.spec, problems = V.parse_pseudocode_turn(turn.text)
        used = 1
        while problems and used < self.b.spec_turns:
            turn = self._turn("spec_pseudocode",
                              f"That did not parse: {'; '.join(problems[:4])}. "
                              f"Return valid JSON in the required shape.", None)
            self._push_assistant(turn)
            self.spec, problems = V.parse_pseudocode_turn(turn.text)
            used += 1
        if not self.spec.constraints:
            return
        # Did the agent read the trip right? Pure telemetry -- computed once,
        # never fed back, never shown. The oracle thresholds on the record's
        # own numbers precisely so it does not depend on this; measuring the
        # agent's parse is the separate question, and this is where it is
        # answered. Declared long before it was wired, so it silently reported
        # {} on every record until now.
        self.traj.fact_accuracy = self.spec.fact_accuracy(
            {"days": self.hv.days, "people": self.hv.people,
             "budget": self.hv.budget, "dates": self.hv.dates})

        turn = self._turn("spec_python",
                          P.spec_python(self.spec.pseudocode_block()), None)
        self._push_assistant(turn)
        V.attach_python_turn(self.spec, turn.text)
        V.preflight(self.spec, self._ctx(), timeout=self.b.verifier_exec_timeout)
        self.traj.spec_json = self.spec.to_json()
        self.traj.verifier_mode = self.spec.mode()
        self.traj.verifier_coverage = self.spec.coverage()
        self.traj.n_inert_checks = sum(
            1 for c in self.spec.constraints if c.usable and c.inert)

    def _ctx(self) -> dict:
        """What a check sees: the facts the AGENT parsed, never the record's.

        A check reading a harness-supplied `days` would be correct even if the
        agent never understood the trip length -- which decouples verifier
        quality from comprehension, the thing verifier recall/precision exists
        to measure. Sourcing ctx from the spec means a misread query produces
        wrong checks, and that shows up.
        """
        ctx = self.spec.ctx()
        # Attributes of whatever the current draft names. Preferences are about
        # rating / cost / category / cuisine, none of which appear in the plan
        # text, so without this a check can verify almost nothing. It is the
        # same data the agent's own search tools returned; resolving it here
        # just saves carrying it by hand, and the plan line cannot carry it
        # anyway (an appended "; Cost: ..." breaks the grader's city parse).
        ctx["items"] = (T.resolve_plan(self._convert(self._plan))
                        if (self._plan and self._convert) else [])
        return ctx

    # -- phase 2-3: TOOL USE / CONSTRUCT -----------------------------------
    def run_tool_use(self) -> None:
        tools = S.toolset(search=True, notebook=True)
        user: str | None = P.tool_use(self.av)
        # +max_idle_turns so a run of narrating turns cannot itself
        # consume the tool budget before any tool is called.
        for _ in range(self.tool_call_budget + self.b.max_idle_turns):
            if (why := self._over_budget()):
                self.traj.stop_reason = why
                return
            turn = self._turn("tool_use", user, tools)
            user = None
            self._push_assistant(turn)
            if not turn.tool_calls:
                # The commonest reason a turn calls nothing is that the model
                # wrote the finished plan as prose instead of calling
                # submit_plan. Take it: discarding a complete plan and nudging
                # for a tool call wastes the budget and loses the work.
                if self._looks_like_plan(turn.text):
                    self._plan = turn.text
                    self.traj.stop_reason = "plan_emitted_inline"
                    return
                # Otherwise it is narration or a question. Tolerate a few, then
                # stop nudging and let SUBMIT force the call.
                self._consec_idle += 1
                if self._consec_idle >= self.b.max_idle_turns:
                    self.traj.stop_reason = "max_idle_turns"
                    return
                user = ("Continue: call a tool, or call submit_plan with the "
                        "finished plan.")
                continue
            self._consec_idle = 0
            for tc in turn.tool_calls:
                if tc.name == "submit_plan":
                    self._plan = (tc.args or {}).get("travel_plan", "")
                    self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                          "content": "Plan received."})
                    return
                self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                      "content": self._dispatch(tc)})
            if self.traj.n_tool_calls >= self.tool_call_budget:
                user = ("You have used your information-gathering budget. Write "
                        "the plan now and call submit_plan.")

    def _looks_like_plan(self, text: str) -> bool:
        """True when free text parses as at least one plan day.

        Uses the same converter the grader uses, so "this is a plan" means the
        same thing here as it will downstream -- not a regex guess.
        """
        return bool(text and text.strip() and self._convert
                    and len(self._convert(text)) >= 1)

    def _dispatch(self, tc) -> str:
        self.traj.n_tool_calls += 1
        if tc.error:
            # Counted, not capped: this call already consumed a slot of the
            # tool-call budget, so a model that cannot format arguments runs
            # out on its own.
            self.traj.n_arg_errors += 1
            return json.dumps({"error": "invalid_arguments", "detail": tc.error})
        a = tc.args or {}
        lim = min(int(a.get("limit", S.MAX_LIMIT) or S.MAX_LIMIT), S.MAX_LIMIT)
        off = max(int(a.get("offset", 0) or 0), 0)
        try:
            if tc.name == "get_trip_dates":
                # Agent-supplied, not harness-supplied: the dates are stated in
                # the query and no TravelPlanner tool hands them over either.
                return S.render_scalar(
                    {"dates": T.get_trip_dates(list(a.get("dates") or []))})
            if tc.name == "get_cities_in_state":
                cs = T.get_cities_in_state(a.get("state", ""))
                return S.render_scalar({"state": a.get("state"), "cities": cs}
                                       if cs else
                                       {"state": a.get("state"), "cities": [],
                                        "reason": "unknown_state"})
            if tc.name in ("search_attractions", "search_restaurants",
                           "search_accommodations"):
                fn = {"search_attractions": T.search_attractions,
                      "search_restaurants": T.search_restaurants,
                      "search_accommodations": T.search_accommodations}[tc.name]
                city = a.get("city", "")
                rows = fn(city)
                return S.render_records(rows, total=len(rows), offset=off, limit=lim,
                                        header=f"{tc.name.split('_')[1].title()} in {city}: {len(rows)}",
                                        max_chars=self.b.tool_result_max_chars)
            if tc.name == "search_flights":
                rows = T.search_flights(a.get("origin", ""), a.get("destination", ""),
                                        a.get("date", ""))
                return S.render_records(rows, total=len(rows), offset=off, limit=lim,
                                        header=f"Flights {a.get('origin')}->{a.get('destination')} on {a.get('date')}: {len(rows)}",
                                        max_chars=self.b.tool_result_max_chars)
            if tc.name == "get_ground_transport":
                return S.render_scalar(T.get_ground_transport(
                    a.get("origin", ""), a.get("destination", ""), a.get("mode", "")))
            if tc.name in ("filter_items", "aggregate_items"):
                kw = {k: a.get(k, "") for k in ("city", "origin", "destination", "date")}
                if tc.name == "filter_items":
                    r = T.filter_items(a.get("entity", ""), filters=a.get("filters"),
                                       sort_by=a.get("sort_by"),
                                       desc=bool(a.get("desc")), **kw)
                    if "error" in r:
                        return S.render_scalar(r)
                    return S.render_records(
                        r["records"], total=r["matched"], offset=off, limit=lim,
                        header=(f"{r['entity']}: {r['matched']} of {r['total']} match "
                                f"({r['dropped']} filtered out)"),
                        max_chars=self.b.tool_result_max_chars)
                return S.render_scalar(T.aggregate_items(
                    a.get("entity", ""), a.get("field", ""), a.get("op", ""),
                    filters=a.get("filters"), sort_by=a.get("sort_by"),
                    desc=bool(a.get("desc")), limit=a.get("limit"), **kw))
            if tc.name == "notebook_write":
                return self.nb.write(a.get("label", ""), a.get("content", ""))
            if tc.name == "notebook_list":
                return self.nb.list()
            if tc.name == "notebook_read":
                return self.nb.read(a.get("index", -1))
        except Exception as e:                       # a tool must never kill a run
            return json.dumps({"error": "tool_failed", "detail": f"{type(e).__name__}: {e}"[:200]})
        self.traj.n_unknown_tool += 1
        return json.dumps({"error": "unknown_tool", "available":
                           [t["function"]["name"] for t in
                            S.toolset(search=True, notebook=True)]})

    # -- phase 4-5: VERIFY / REPAIR ---------------------------------------
    def run_verify_repair(self, convert_plan) -> None:
        """Loop CONSTRUCT -> VERIFY -> REPAIR until clean or out of rounds.

        Two signals, kept separate in the feedback because they cover disjoint
        ground: the oracle is authoritative on cost and entity existence and
        blind to preferences; the authored checks are the only thing covering
        preferences and only as good as the agent wrote them.

        They are NOT cross-validating. An earlier version re-opened SPEC when
        the checks passed while the oracle failed, reading that as evidence the
        checks were wrong. It is not: preferences and entity validity are
        different concerns, so a plan with a bad restaurant and perfectly
        satisfied preferences produces exactly that pattern, and it is the
        normal case rather than a contradiction.

        Spec revision is instead agent-initiated, via `revise_checks`, because
        the agent is the only party that knows whether its own restatement was
        wrong. Soundness comes from the append-or-fix rule, not from the
        trigger: a check that currently passes cannot be rewritten.
        """
        if not self._plan:
            return

        spec_turns_left = max(self.b.spec_turns - 2, 0)
        for rnd in range(1, self.b.verify_repair_rounds + 1):
            if (why := self._over_budget()):
                self.traj.stop_reason = why
                return
            days = convert_plan(self._plan)
            if not days:
                oracle_msg, oracle_ok = "The plan could not be parsed into days.", False
                verdicts, checks_ok, cost = [], False, None
            else:
                v = evaluate_with_oracle(days, self.hv.people,
                                         budget=self.hv.budget, reset=(rnd == 1))
                oracle_msg, oracle_ok, cost = v.as_message(), v.ok, v.cost
                self.traj.oracle_rounds.append(
                    {"round": rnd, "ok": v.ok, "cost": v.cost,
                     "over_by": v.over_by, "problems": v.problems[:8],
                     "terminated": v.terminated})
                verdicts, checks_ok = self._run_checks(days, rnd)

            self.traj.verify_repair_rounds = rnd
            self._record_candidate(self._plan, rnd, oracle_ok, cost)
            if oracle_ok and checks_ok:
                self.traj.stop_reason = "verified"
                return

            # authored checks pass but the oracle disagrees -> the checks are wrong
            tools = [S.SUBMIT_TOOL] + ([S.REVISE_TOOL] if spec_turns_left > 0 else [])
            turn = self._turn("repair", P.VERIFY_FEEDBACK.format(
                round=rnd, max_rounds=self.b.verify_repair_rounds,
                oracle=oracle_msg, checks=self._render_checks(verdicts),
                unverified=self._unverified_note(),
                revise=(P.REVISE_AVAILABLE if spec_turns_left > 0 else "")), tools)
            self._push_assistant(turn)

            revised = False
            for tc in turn.tool_calls:
                if tc.name == "revise_checks" and tc.args:
                    if spec_turns_left > 0:
                        spec_turns_left -= 1
                        n = self._apply_revision(tc.args, rnd)
                        revised = True
                        self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                              "content": f"{n} check(s) updated. "
                                                         f"Re-running verification."})
                    else:
                        # Every agent that revised did so twice; the second call
                        # matched no branch, so no tool message was appended and
                        # the model was left waiting on a reply that never came.
                        # Refuse it out loud instead.
                        self.traj.n_revision_refused += 1
                        self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                              "content": "Revision budget spent -- "
                                                         "no further check edits this "
                                                         "episode. Submit a plan instead."})
                elif tc.name == "submit_plan" and tc.args:
                    self._plan = tc.args.get("travel_plan", self._plan)
                    self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                          "content": "Plan received."})
            if not turn.tool_calls and self._looks_like_plan(turn.text):
                self._plan = turn.text.strip()
            if revised:
                # A revision round re-verifies the SAME plan against corrected
                # checks; it is not a repair attempt, so it does not consume one.
                continue
            # Re-emitting a plan already tried means the agent is circling, not
            # converging; every further round costs tokens for a plan we have
            # already scored. Stop and fall back to the best one seen.
            if self._plan and self._plan.strip() in self._seen_plans:
                self.traj.repair_cycle = True
                self.traj.stop_reason = "repair_cycle"
                return
            self._record_candidate(self._plan, rnd + 1)
        self.traj.stop_reason = self.traj.stop_reason or "verify_repair_rounds_exhausted"
        self._record_candidate(self._plan, self.traj.verify_repair_rounds + 1)

    def _record_candidate(self, plan: str | None, rnd: int,
                          oracle_ok: bool | None = None,
                          cost: float | None = None) -> None:
        """Note a distinct plan. Scoring happens later, in `_select_plan`."""
        if not (plan or "").strip():
            return
        body = plan.strip()
        for c in self._candidates:
            if c["plan"] == body:                    # already a candidate
                if oracle_ok is not None and c["oracle_ok"] is None:
                    c["oracle_ok"] = oracle_ok       # first time it was verified
                    c["cost"] = cost
                return
        self._seen_plans.add(body)
        self._candidates.append({"round": rnd, "plan": body,
                                 "oracle_ok": oracle_ok, "cost": cost})

    def _select_plan(self) -> str | None:
        """Score every candidate under the FINAL check-set and return the best.

        Key is (oracle_ok, -failing checks), ties to the earliest round, so a
        later plan has to be strictly better to displace an earlier one. A
        candidate that never reached verification has oracle_ok=None, which
        sorts below a confirmed pass and above a confirmed failure -- unknown
        is not evidence either way.
        """
        if not self._candidates:
            return None
        runnable = {c.id: c.python for c in self.spec.constraints if c.usable}
        rank = {True: 2, None: 1, False: 0}
        scored = []
        for i, cand in enumerate(self._candidates):
            n_fail, note = 0, ""
            days = self._convert(cand["plan"]) if self._convert else None
            if not days:
                n_fail, note = 99, "unparseable"
            elif runnable:
                try:
                    vs = V.run(runnable, days, self._ctx(),
                               timeout=self.b.verifier_exec_timeout)
                    n_fail = sum(1 for v in vs if v.ok is False)
                except Exception as e:                # scoring must never sink a run
                    note = f"scoring failed: {type(e).__name__}"
                    n_fail = 99
            scored.append({"round": cand["round"], "oracle_ok": cand["oracle_ok"],
                           "cost": cand.get("cost"), "n_fail": n_fail, "note": note,
                           "key": (rank[cand["oracle_ok"]], -n_fail, -i)})
        best = max(range(len(scored)), key=lambda i: scored[i]["key"])
        for i, row in enumerate(scored):
            row["selected"] = (i == best)
            row.pop("key")
        self.traj.candidates = scored
        self.traj.best_plan_round = self._candidates[best]["round"]
        return self._candidates[best]["plan"]

    def _run_checks(self, days, rnd: int) -> tuple[list, bool]:
        runnable = {c.id: c.python for c in self.spec.constraints if c.usable}
        if not runnable:
            return [], True                      # nothing authored ran; oracle only
        verdicts = V.run(runnable, days, self._ctx(), timeout=self.b.verifier_exec_timeout)
        # A check can pass pre-flight and still die on a real plan: at SPEC time
        # ctx["items"] is empty, so anything iterating it passes vacuously.
        # Demote it on the first real error, which does two things -- it stops
        # counting toward verifier_coverage, and it unfreezes it so a revision
        # can repair it. Left usable it would be unblocking, unrepairable, and
        # inflating the coverage number all at once.
        by_id = {c.id: c for c in self.spec.constraints}
        for v in verdicts:
            if v.error and v.id in by_id and by_id[v.id].usable:
                by_id[v.id].smoke_ok = False
                by_id[v.id].error = f"failed on a real plan: {v.error}"[:200]
        self.traj.verifier_mode = self.spec.mode()
        self.traj.verifier_coverage = self.spec.coverage()
        self.traj.n_inert_checks = sum(
            1 for c in self.spec.constraints if c.usable and c.inert)
        self.traj.check_verdicts.append(
            {"round": rnd, "verdicts": [{"id": v.id, "ok": v.ok, "detail": v.detail,
                                         "error": v.error} for v in verdicts]})
        return verdicts, all(v.ok is not False for v in verdicts)

    def _render_checks(self, verdicts) -> str:
        if not verdicts:
            return "(none of your checks could be run)"
        by = {c.id: c for c in self.spec.constraints}
        out = []
        for v in verdicts:
            if v.ok is False:
                c = by.get(v.id)
                out.append(f"VIOLATED [{v.id}] {c.text if c else ''} -- {v.detail}")
        return "\n".join(out) or "All of your checks passed."

    def _unverified_note(self) -> str:
        bad = [c.id for c in self.spec.constraints if not c.usable]
        if not bad:
            return ""
        return (f"Note: {len(bad)} of your checks could not be run "
                f"({', '.join(bad[:6])}), so those constraints are unverified. "
                f"Satisfy them by reasoning rather than relying on the report.\n")

    def _apply_revision(self, args: dict, rnd: int) -> int:
        """Apply an agent-requested check revision. Append-or-fix only.

        This is where soundness lives, since the trigger no longer carries any:
        a check that currently passes pre-flight keeps its source, so a revision
        can add a check or repair a broken one but cannot weaken a working one
        into something the plan happens to satisfy.
        """
        raw = args.get("checks") or {}
        known = {c.id: c for c in self.spec.constraints}
        applied, shadowed, added = [], [], []
        for cid, src in raw.items():
            if not isinstance(src, str) or not src.strip():
                continue
            c = known.get(cid)
            if c is None:
                c = V.Constraint(id=str(cid), kind="preference",
                                 source="query", text="(added at revision)",
                                 pseudocode="(added at revision)", python=src)
                known[cid] = c
                self.spec.constraints.append(c)
                added.append(cid)
                continue
            # Revising a check that RUNS used to be refused outright, to stop an
            # agent softening a check its plan was failing. Measured, that rule
            # blocked 6 of 6 revisions across three records, every one of them a
            # correct diagnosis -- an over-strict city rule, a room type spelled
            # "Entire home/apt", a check counting absent "-" entries as
            # restaurants. Prohibition cannot tell a repair from a weakening, so
            # it is replaced by a shadow: the original keeps running, decides
            # which plan is delivered, and any disagreement is recorded.
            if c.usable and c.shadow_python is None:
                c.shadow_python = c.python
                shadowed.append(cid)
            c.python = src
            applied.append(cid)
        V.preflight(self.spec, self._ctx(), timeout=self.b.verifier_exec_timeout)
        n_applied = len(applied) + len(added)
        rev = {"round": rnd, "trigger": "agent_requested",
               "reason": str(args.get("reason", ""))[:200],
               "applied": n_applied, "applied_ids": applied, "added_ids": added,
               "shadowed_ids": shadowed}
        self.spec.revisions.append(rev)
        self.traj.spec_revisions.append(rev)
        self.traj.spec_json = self.spec.to_json()
        self.traj.verifier_mode = self.spec.mode()
        self.traj.verifier_coverage = self.spec.coverage()
        self.traj.n_inert_checks = sum(
            1 for c in self.spec.constraints if c.usable and c.inert)
        return n_applied

    # -- phase 6: SUBMIT ---------------------------------------------------
    def run_submit(self) -> str:
        """Final. Retries cover a failed call mechanism, never a second plan."""
        if self._plan:
            self.traj.submitted = True
            return self._plan
        for _ in range(self.b.submit_retries + 1):
            turn = self._turn("submit", P.BUDGET_EXHAUSTED, [S.SUBMIT_TOOL])
            self._push_assistant(turn)
            for tc in turn.tool_calls:
                if tc.name == "submit_plan" and tc.args:
                    self._plan = tc.args.get("travel_plan", "")
                    self.traj.submitted = True
                    self.traj.forced_submit = True
                    return self._plan
            self.messages.append({"role": "user",
                                  "content": "That was not a submit_plan call. Try again."})
        turn = self._turn("fallback", None, None)      # plain completion, no tools
        self.traj.fallback_used = True
        self.traj.submitted = bool(turn.text.strip())
        self._plan = turn.text
        return self._plan

    # -- driver ------------------------------------------------------------
    def run(self, convert_plan) -> tuple[str, Trajectory]:
        """SPEC -> TOOL USE -> [force a plan] -> VERIFY/REPAIR -> return.

        Submission is pulled BEFORE verification, not after. Previously
        `run_submit` came last, so a record whose agent never emitted a plan
        during tool use was forced to submit and then never checked -- observed
        on a 3-city record that exhausted its tool budget: rounds=0, no oracle
        call, no authored check ever run. Whatever produced the plan, it now
        goes through the same verification.
        """
        self._convert = convert_plan
        try:
            self.run_spec()
            self.run_tool_use()
            if not self._plan:
                self.run_submit()
            self.run_verify_repair(convert_plan)
            # The best candidate, not the last one written. Repair is not
            # monotone: a check that can never pass will keep the agent editing
            # a plan that was already valid, and the loop used to hand back
            # whatever that editing last produced.
            plan = self._select_plan() or self._plan or self.run_submit()
        except Exception as e:
            import traceback
            self.traj.stop_reason = f"error: {type(e).__name__}: {e}"[:200]
            # Keep the traceback: "'NoneType' object is not subscriptable" with
            # no frame cost an afternoon of bisecting.
            self.traj.error_traceback = traceback.format_exc()[-2000:]
            # A mid-episode crash is exactly when the best-seen plan matters:
            # the record still counts, and a verified earlier plan beats
            # whatever half-edited text the agent was holding.
            plan = self._select_plan() or self._plan or ""
        # `submitted` means "this record produced a plan", not "run_submit was
        # called". After the reorder, a plan emitted during tool use reaches
        # verification without run_submit ever running, and the flag was left
        # False on records that had in fact succeeded.
        self.traj.submitted = bool((plan or "").strip())
        self.traj.reverted_to_best = bool(
            plan and self._plan and plan.strip() != self._plan.strip())
        self.traj.wall_s = time.time() - self._t0
        self.traj.stop_reason = self.traj.stop_reason or "completed"
        self.traj.finished_at = time.strftime("%FT%TZ", time.gmtime())
        return plan, self.traj
