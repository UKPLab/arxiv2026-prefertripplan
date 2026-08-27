#!/usr/bin/env python3
"""Agentic travel-plan generation -- the tool-using counterpart to generate_plans.py.

Same task, same output, same downstream. The direct planner is handed a
pre-retrieved pool in its prompt; this one is handed nothing and must gather the
information itself through tools, then verify and repair its own plan before
submitting.

The CLI deliberately mirrors ``plan-generation/generate_plans.py``: every shared
flag has the same name, type, default and meaning, and the default output paths
follow the same convention, so the two are interchangeable in a script. Flags
below the "agentic" group have no counterpart there.

    python3 plan-generation/generate_plans.py --backend openrouter --model M ...
    python3 plan-generation/agentic/run_agentic.py --backend openrouter --model M ...

Output is a ``plans_*.jsonl`` in exactly the shape ``generate_plans.py``
produces, so ``convert_plans.py`` -> ``eval.py`` -> ``analyze_performance.py``
consume it unmodified: the evaluator already grades against the full
``database/`` CSVs and ignores ``reference_information``.

Backends: the loop needs multi-turn tool calling, so ``vllm-offline`` is not
available -- it has no tool surface. Serve open-weight models through a vLLM
*server* and reach it with ``--backend openrouter --openrouter-base-url``, which
is how the published qwen results were produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

import _paths
import db_tools as T
import generate_plans as GP
import record_view as RV
import runtime as RT
from llm_backends import ToolChat

AGENTIC_DIR = _paths.ROOT / "plan-generation" / "agentic-runs"
BACKENDS = ("anthropic", "openai", "openrouter")     # tool-calling capable only

DEFAULT_MODELS = {"anthropic": "claude-haiku-4-5-20251001",
                  "openai": "gpt-5.4-mini",
                  "openrouter": "openai/gpt-5.4-mini"}


def _default_out_path(backend: str, model: str, split: str) -> Path:
    return AGENTIC_DIR / f"plans_{split}_{backend}_{GP._sanitize_model(model)}.jsonl"


def _default_cache_path(backend: str, model: str, split: str) -> Path:
    return AGENTIC_DIR / f"plan_{split}_cache_{backend}_{GP._sanitize_model(model)}.jsonl"


def _default_traj_path(backend: str, model: str, split: str) -> Path:
    return AGENTIC_DIR / f"traj_{split}_{backend}_{GP._sanitize_model(model)}.jsonl"


def _states() -> set[str]:
    out = set()
    with open(_paths.DB_DIR / "background" / "citySet_with_states.txt") as f:
        for line in f:
            if line.strip():
                out.add(line.rstrip("\n").split("\t")[1])
    return out


# --------------------------------------------------------------------------- #
# Resumable plan cache + trajectory sidecar                                   #
# --------------------------------------------------------------------------- #
# The plan cache reuses generate_plans' format and writer discipline -- one
# fsync'd JSON line per record keyed by dataset id -- so the output file it
# regenerates is consumed by convert_plans.py and eval.py with no modification.
# Extra agentic fields ride along on the same line.
#
# Trajectories go to a separate file: one large line per record, so the cache
# stays small and greppable and resume stays fast.
_LOCK = threading.Lock()


def _append_line(path: Path, text: str) -> None:
    """Append one line atomically.

    A single ``os.write`` to an ``O_APPEND`` descriptor is atomic for a regular
    file under POSIX, so a record can never be split or interleaved -- including
    by a SECOND PROCESS, which an in-process lock cannot help with. That is the
    case that actually bit: a cache line was found holding the tail of one
    record and the head of the next, losing both, because two runs overlapped.

    The lock is kept for ordering within this process; the atomic write is what
    makes the file safe.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (text.rstrip("\n") + "\n").encode()
    with _LOCK:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)


def append_plan(path: Path, line: dict) -> None:
    _append_line(path, json.dumps(line, ensure_ascii=False))


def append_trajectory(path: Path, traj_json: str) -> None:
    _append_line(path, traj_json)


def done_ids(path: Path) -> set[int]:
    """Ids already completed, for resume. Last line wins on duplicates."""
    out: set[int] = set()
    if not path.exists():
        return out
    with path.open() as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.add(int(json.loads(line)["id"]))
            except Exception:
                # Loud: a skipped line is a LOST record. Swallowing this is how
                # an interleaved write went unnoticed until the output was short.
                print(f"[warn] {path.name}: unparseable line {n} skipped "
                      f"({len(line)} chars). Those records are not cached and "
                      f"will be re-run.", file=sys.stderr)
    return out


def load_plans(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                out[int(e["id"])] = e
            except Exception:
                print(f"[warn] {path.name}: unparseable line {n} skipped "
                      f"({len(line)} chars)", file=sys.stderr)
    return out


def write_output(path: Path, cache: dict[int, dict]) -> int:
    """Regenerate the full `plans_*.jsonl` from the cache, sorted by id.

    Same shape `generate_plans.write_output_jsonl` produces, so downstream is
    unchanged: id, source_id, profile, query, reference_information,
    llm_travel_plan, llm_reasoning.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for id_ in sorted(cache):
            e = cache[id_]
            content = e.get("content") or ""
            plan = content
            try:
                obj = json.loads(content)
                if isinstance(obj, dict) and "travel_plan" in obj:
                    plan = obj["travel_plan"]
            except Exception:
                pass
            f.write(json.dumps({
                "id": id_, "source_id": e.get("source_id"),
                "profile": e.get("profile", ""), "query": e.get("query", ""),
                "reference_information": e.get("reference_information", ""),
                "llm_travel_plan": plan or None,
                "llm_reasoning": e.get("reasoning"),
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- shared with generate_plans.py: same names, types, defaults ----
    ap.add_argument("--backend", choices=list(BACKENDS),
                    default=os.environ.get("LLM_PLAN_BACKEND") or "anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--request-timeout", type=float, default=120.0)
    ap.add_argument("--dataset", default=GP.DEFAULT_DATASET)
    ap.add_argument("--split", default=GP.DEFAULT_SPLIT, choices=list(_paths.SPLITS))
    ap.add_argument("--config", default=None)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--sample-id", "--sample-idx", type=str, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--openrouter-base-url",
                    default=os.environ.get("OPENROUTER_BASE_URL")
                    or "https://openrouter.ai/api/v1")
    ap.add_argument("--openrouter-api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--extra-body", default=os.environ.get("LLM_PLAN_EXTRA_BODY"))
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                    help="reasoning tokens are billed as output and count against "
                         "--max-tokens, so the two move together: the submit turn "
                         "must fit a whole plan AFTER thinking. The published "
                         "direct runs used high with --max-tokens 16384")
    ap.add_argument("--verbosity", default=None, choices=["low", "medium", "high"],
                    help="response verbosity, for models that accept it")

    # ---- agentic-only ----
    g = ap.add_argument_group("agentic",
                              "no counterpart in generate_plans.py; phase budgets "
                              "and trajectory output (see agentic/DESIGN.md)")
    d = RT.Budgets()
    g.add_argument("--traj", default=None,
                   help="trajectory sidecar: one JSON line per record with every "
                        "step, tool call, verdict and token count "
                        "(default: alongside --cache)")
    g.add_argument("--dry-run", action="store_true",
                   help="resolve paths and budgets, print the work plan, make no API calls")

    ph = ap.add_argument_group("phase budgets", "how much of each activity is allowed")
    ph.add_argument("--spec-turns", type=int, default=d.spec_turns,
                    help="1 pseudo-code + 1 python + triggered revisions")
    ph.add_argument("--max-tool-calls", type=int, default=d.max_tool_calls,
                    help="absolute cap on tool calls per record, applied on top of "
                         "the adaptive budget below; the effective budget is the "
                         "smaller of the two")
    ph.add_argument("--extra-tool-calls", type=int, default=d.extra_tool_calls,
                    help="tool calls allowed ON TOP of the per-instance minimum "
                         "6C + 4 (+1 if the destination is a state), where C is the "
                         "number of cities to visit; covers pagination, probing "
                         "candidate cities, and notebook calls")
    ph.add_argument("--verify-repair-rounds", type=int, default=d.verify_repair_rounds,
                    help="CONSTRUCT -> VERIFY -> REPAIR cycles; each runs the "
                         "environment oracle and the agent's own checks, and the "
                         "loop exits early when both are clean")
    ph.add_argument("--submit-retries", type=int, default=d.submit_retries,
                    help="retries of a failed submit CALL; never a second plan")

    fa = ap.add_argument_group("failure budgets",
                               "how many times a malfunction is tolerated before "
                               "the loop gives up on that activity")
    fa.add_argument("--api-max-attempts", type=int, default=d.api_max_attempts,
                    help="attempts per single call on transient API errors; with "
                         "--request-timeout this bounds one call, and together they "
                         "size the episode wall clock")
    fa.add_argument("--max-idle-turns", type=int, default=d.max_idle_turns,
                    help="consecutive turns calling no tool before forcing submit")
    fa.add_argument("--verifier-exec-timeout", type=int, default=d.verifier_exec_timeout,
                    help="wall clock for executing the agent-authored verifier "
                         "code; named for what it bounds, since DB tool calls are "
                         "our own code and this runs a program the model wrote")

    ep = ap.add_argument_group("episode ceilings", "hard caps so no record runs away")
    ep.add_argument("--max-context-tokens", type=int, default=d.max_context_tokens,
                    help="largest single prompt allowed, i.e. how big the "
                         "conversation may get. This is the ~96-112K context "
                         "ceiling the literature reports and the quantity tied to "
                         "degradation. Cumulative tokens and cost are recorded per "
                         "record but do not gate the loop")
    ep.add_argument("--max-agent-timeout-factor", type=float,
                    default=d.max_agent_timeout_factor,
                    help="per-record wall clock, as a multiple of the per-call "
                         "bound: factor x --request-timeout x --api-max-attempts. "
                         "Checked between turns, so it bounds when new work starts "
                         "rather than interrupting a call in flight")
    ep.add_argument("--tool-result-max-chars", type=int, default=d.tool_result_max_chars,
                    help="a tool result is truncated to whole rows under this size")
    return ap


def _resolve_endpoint(a) -> tuple[str, str | None]:
    if a.backend == "openrouter":
        return a.openrouter_base_url, a.openrouter_api_key
    if a.backend == "openai":
        return a.openai_base_url, a.openai_api_key
    return None, os.environ.get("ANTHROPIC_API_KEY")


def main() -> int:
    a = build_parser().parse_args()
    a.model = a.model or DEFAULT_MODELS[a.backend]
    base_url, api_key = _resolve_endpoint(a)
    if a.backend == "anthropic":
        raise SystemExit("--backend anthropic is not wired for tool calling yet; "
                         "use openai or openrouter (a vLLM server counts as openrouter).")

    rows = _paths.load_split(a.split, dataset=a.dataset)
    if a.sample_id:
        raw = Path(a.sample_id).read_text() if Path(a.sample_id).exists() else a.sample_id
        want = {int(x) for x in re.split(r"[,\s]+", raw) if x.strip()}
        rows = [r for r in rows if int(r["id"]) in want]
    if a.sample:
        rows = rows[:a.sample]

    out_p = Path(a.out) if a.out else _default_out_path(a.backend, a.model, a.split)
    cache_p = Path(a.cache) if a.cache else _default_cache_path(a.backend, a.model, a.split)
    traj_p = Path(a.traj) if a.traj else _default_traj_path(a.backend, a.model, a.split)

    budgets = RT.Budgets(
        spec_turns=a.spec_turns, extra_tool_calls=a.extra_tool_calls,
        verify_repair_rounds=a.verify_repair_rounds, submit_retries=a.submit_retries,
        api_max_attempts=a.api_max_attempts,
        max_idle_turns=a.max_idle_turns, max_tool_calls=a.max_tool_calls,
        max_context_tokens=a.max_context_tokens,
        max_agent_timeout_factor=a.max_agent_timeout_factor,
        request_timeout=a.request_timeout,
        tool_result_max_chars=a.tool_result_max_chars,
        verifier_exec_timeout=a.verifier_exec_timeout)
    temperature = GP.TEMPERATURE if a.temperature is None else a.temperature
    cfg = {"model": a.model, "backend": a.backend, "base_url": base_url,
           "dataset": a.dataset, "split": a.split, "temperature": temperature,
           "reasoning_effort": a.reasoning_effort, "verbosity": a.verbosity,
           "max_tokens": a.max_tokens, "budgets": vars(budgets)}
    run_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    manifest = cache_p.parent / f"manifest_{cache_p.stem}.json"

    done = done_ids(cache_p)
    pending = [r for r in rows if int(r["id"]) not in done]
    states = _states()

    if a.reasoning_effort and a.reasoning_effort != "none":
        ratio = {"minimal": .1, "low": .2, "medium": .5, "high": .8, "xhigh": .95}[a.reasoning_effort]
        left = a.max_tokens - max(min(int(a.max_tokens * ratio), 128000), 1024)
        if left < 1500:
            print(f"[warn] --reasoning-effort {a.reasoning_effort} reserves ~"
                  f"{a.max_tokens - left} of {a.max_tokens} output tokens for "
                  f"thinking, leaving ~{left} for the answer. A 7-day plan needs "
                  f"~1200. Raise --max-tokens or lower the effort.", file=sys.stderr)
    print(f"[cfg] backend={a.backend}  model={a.model}  run_id={run_id}")
    print(f"[cfg] endpoint={base_url or '(sdk default)'}")
    print(f"[cfg] out=   {out_p}")
    print(f"[cfg] cache= {cache_p}")
    print(f"[cfg] traj=  {traj_p}")
    print(f"[in]  {len(rows)} rows from {a.dataset}:{a.split}")
    print(f"[run] cached={len(done)}  pending={len(pending)}  workers={a.workers}")
    if pending:
        b = sorted(min(RV.project(r)[1].tool_call_budget(a.extra_tool_calls,
                                                        r["dest"] in states),
                       a.max_tool_calls)
                   for r in pending)
        print(f"[run] tool-call budget: min {b[0]}  median {b[len(b)//2]}  max {b[-1]}")

    if a.dry_run:
        print("\n[dry-run] no API calls. Work plan:")
        for r in pending[:10]:
            _, hv = RV.project(r)
            print(f"   id={hv.id:4d} {hv.org}->{hv.dest} {hv.days}d {hv.n_cities}c "
                  f"people={hv.people} "
                  f"tools={min(hv.tool_call_budget(a.extra_tool_calls, hv.dest in states), a.max_tool_calls)}")
        if len(pending) > 10:
            print(f"   ... and {len(pending) - 10} more")
        return 0
    if not pending:
        n = write_output(out_p, load_plans(cache_p))
        print(f"[out] nothing pending; regenerated {n} records -> {out_p}")
        return 0

    print("[db] loading tool tables (flights CSV ~15s) ...", flush=True)
    T.preload()
    print("[db] loading TravelPlanner ReactReflectEnv (its own DB copies) ...", flush=True)
    RT.preload_oracle()

    sys.path.insert(0, str(_paths.ROOT / "evaluation"))
    from convert_plans import convert_plan_text                       # noqa: E402

    def convert(text: str):
        try:
            return convert_plan_text(text, day_base=1)
        except Exception:
            return []

    extra_body = json.loads(a.extra_body) if a.extra_body else None
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({**cfg, "run_id": run_id, "status": "running",
                                    "n_pending": len(pending),
                                    "started_at": time.strftime("%FT%TZ", time.gmtime())},
                                   indent=2))

    def work(row: dict):
        av, hv = RV.project(row)
        chat = ToolChat(a.model, base_url=base_url, api_key=api_key,
                        temperature=temperature, request_timeout=a.request_timeout,
                        max_attempts=budgets.api_max_attempts, extra_body=extra_body,
                        reasoning_effort=(None if a.reasoning_effort in (None, "none")
                                          else a.reasoning_effort),
                        verbosity=a.verbosity)
        runner = RT.Runner(chat, av, hv, budgets=budgets, run_id=run_id,
                           dest_is_state=(hv.dest in states), max_tokens=a.max_tokens)
        plan, traj = runner.run(convert)
        if not (plan or "").strip():
            # Do NOT cache a record with no plan. The direct planner guards the
            # same way (`if on_result is not None and comp.get("content")`), and
            # for the same reason: a cached empty record counts as done, so
            # resume skips it and the only way to retry is to edit the cache by
            # hand. Left uncached it simply stays pending.
            return None, traj
        line = traj.cache_line(json.dumps({"travel_plan": plan}, ensure_ascii=False),
                               row.get("source_id"))
        line.update({"profile": av.profile, "query": av.query})
        return line, traj

    # Label by backend and worker count, matching the direct path's
    # f"{self.name} ({self.workers}w)". The model is already on the [cfg]
    # line above, and a full model id eats most of the bar's width.
    prog = GP._ProgressReporter(len(pending),
                                label=f"agentic {a.backend} ({a.workers}w)")
    t0, n_ok, n_empty = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, r): int(r["id"]) for r in pending}
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                line, traj = fut.result()
            except Exception as e:
                print(f"  [err] id={rid}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            traj.cached = line is not None
            append_trajectory(traj_p, traj.to_json())   # keep the evidence either way
            if line is None:
                n_empty += 1
                print(f"  id={rid:4d} NO PLAN -- not cached, still pending. "
                      f"{traj.stop_reason[:90]}", flush=True)
                prog.step()
                continue
            append_plan(cache_p, line)
            n_ok += 1
            print(f"  id={rid:4d} {traj.stop_reason:22s} steps={len(traj.steps):3d} "
                  f"tools={traj.n_tool_calls:3d} "
                  f"verifier={traj.verifier_mode}({traj.verifier_coverage:.2f}) "
                  f"rounds={traj.verify_repair_rounds}", flush=True)
            prog.step()

    prog.close()
    n = write_output(out_p, load_plans(cache_p))
    manifest.write_text(json.dumps({**cfg, "run_id": run_id, "status": "complete",
                                    "n_records": n, "n_this_run": n_ok,
                                    "wall_s": round(time.time() - t0, 1),
                                    "finished_at": time.strftime("%FT%TZ", time.gmtime())},
                                   indent=2))
    if n_empty:
        print(f"\n[warn] {n_empty} record(s) produced no plan and were NOT cached. "
              f"Re-run the same command to retry only those.", file=sys.stderr)
    print(f"\n[out] wrote {n} records -> {out_p}")
    print(f"[out] trajectories    -> {traj_p}")
    print(f"[out] manifest        -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
