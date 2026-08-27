"""Per-record trajectory records.

Two files per run, deliberately separate:

  plan cache   one small line per record, keyed by dataset id -- what resume
               reads, and what `generate_plans.write_output_jsonl` consumes.
  trajectory   one large line per record -- every step, tool call, verdict and
               token count. Kept out of the cache so resume stays fast and the
               cache stays greppable.

The audit fields exist because a run can silently become a mixture: a record whose
authored checks failed pre-flight is verified only by the oracle. Reporting one
number over that mixture would be misleading, so `verifier_mode` and
`verifier_coverage` travel with every record and become analysis axes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class StepRecord:
    i: int
    phase: str                      # spec_pseudocode | spec_python | tool_use |
                                    # construct | verify | repair | submit | fallback
    text: str = ""
    reasoning: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    finish_reason: str | None = None


@dataclass
class Trajectory:
    id: int
    run_id: str
    model: str
    steps: list[StepRecord] = field(default_factory=list)

    # -- spec / verifier audit ------------------------------------------
    spec_json: str | None = None
    verifier_mode: str = "none"     # executable | partial | none
    verifier_coverage: float = 0.0
    spec_revisions: list[dict] = field(default_factory=list)
    fact_accuracy: dict = field(default_factory=dict)   # agent's parse vs truth
    check_verdicts: list[dict] = field(default_factory=list)   # per round, per id
    oracle_rounds: list[dict] = field(default_factory=list)

    # -- outcome ---------------------------------------------------------
    submitted: bool = False
    forced_submit: bool = False
    fallback_used: bool = False
    stop_reason: str = ""
    error_traceback: str | None = None
    finished_at: str = ""
    # True when this attempt's plan was written to the plan cache. A record that
    # produced no plan still gets its trajectory recorded -- the 502s and the
    # traceback are the evidence -- but it stays uncached and is re-run, so the
    # file accumulates one line PER ATTEMPT. Analysis should take the attempt
    # with cached=True, or failing that the last by `finished_at`.
    cached: bool = False
    verify_repair_rounds: int = 0        # rounds actually run, not the budget
    rounds_to_feasible: int | None = None      # first round the oracle passed
    # The repair loop can end on a plan worse than one it already had: an
    # observed record cycled between four plans chasing a check that could
    # never pass, and would have shipped its last rather than its best.
    best_plan_round: int | None = None   # round whose plan was returned (0 = pre-repair)
    reverted_to_best: bool = False       # the last plan was NOT the one returned
    repair_cycle: bool = False           # a plan repeated -> loop was thrashing
    n_inert_checks: int = 0              # usable checks that can never return False
    peak_context_tokens: int = 0
    n_tool_calls: int = 0
    n_arg_errors: int = 0
    n_unknown_tool: int = 0
    wall_s: float = 0.0

    def totals(self) -> dict:
        return {
            "prompt_tokens": sum(s.prompt_tokens for s in self.steps),
            "completion_tokens": sum(s.completion_tokens for s in self.steps),
            "reasoning_tokens": sum(s.reasoning_tokens for s in self.steps),
            "cost_usd": (sum(s.cost_usd for s in self.steps if s.cost_usd)
                         if any(s.cost_usd for s in self.steps) else None),
            "peak_context_tokens": max((s.prompt_tokens for s in self.steps),
                                       default=0),
            "n_steps": len(self.steps),
        }

    def cache_line(self, content: str, source_id: int | None) -> dict:
        """The small per-record line. `content` keeps the {"travel_plan": ...}
        envelope so `generate_plans.write_output_jsonl` and `convert_plans.py`
        consume it unchanged."""
        t = self.totals()
        return {"id": self.id, "source_id": source_id, "content": content,
                "reasoning": None,
                "verifier_mode": self.verifier_mode,
                "verifier_coverage": round(self.verifier_coverage, 3),
                "submitted": self.submitted, "forced_submit": self.forced_submit,
                "fallback_used": self.fallback_used, "stop_reason": self.stop_reason,
                "verify_repair_rounds": self.verify_repair_rounds,
                "rounds_to_feasible": self.rounds_to_feasible,
                "n_tool_calls": self.n_tool_calls, "wall_s": round(self.wall_s, 1),
                **t}

    def to_json(self) -> str:
        d = asdict(self)
        d["totals"] = self.totals()
        return json.dumps(d, ensure_ascii=False, default=str)
