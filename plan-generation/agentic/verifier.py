"""The agent-authored verifier: its spec, and the sandbox that runs it.

One concern, so one module. SPEC is authored in two turns -- pseudo-code first,
then Python keyed to the same constraint ids -- and the sandbox scans, executes
and pre-flights that Python.

Why the pseudo-code is canonical rather than a comment on the code:

  * a **clean fallback** -- if the Python does not run, the same pseudo-code can
    be handed to an LLM judge without re-deriving anything, so the fallback
    judges the same spec rather than a different one;
  * **per-check granularity** -- ids link a pseudo-code entry, its
    implementation, its verdict and its score against ground truth, so coverage
    is a fraction rather than a boolean;
  * **auditability** -- the spec is written before any plan exists and is logged
    verbatim, so a reader can see exactly what the agent thought it had to satisfy.

Related representations: ATLAS's CSP-shaped constraint set (arXiv:2509.25586)
and ChinaTravel's DSL (arXiv:2412.13682), both of which keep a structured
constraint object separate from the planner.

Isolation is structural, not by instruction -- the agent must never reach
`evaluation/`, `preferences.py`, or anything else encoding benchmark semantics:

  1. AST scan   reject imports, attribute chains and string literals naming
                benchmark internals, plus dunder escapes. Runs before execution.
  2. subprocess separate interpreter, scrubbed environment, cwd in an empty temp
                dir, wall clock. Nothing it does can touch the parent.
  3. pre-flight every check is smoke-run against a SYNTHETIC plan before it is
                trusted -- not the agent's draft, which would let a check be
                tuned to the plan it will judge.

Per-check, not all-or-nothing: 6 of 8 checks running yields 6 verdicts and a
coverage of 0.75 rather than a discarded suite.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

KINDS = ("commonsense", "hard", "preference")
# The profile is deliberately NOT a constraint source. It is motivating
# context, not a requirement: the shared prompt trunk says the query wins
# wherever the two conflict, and on this dataset they conflict by design --
# 157/225 test records carry drift (78 inversion, 79 omission), so on inversion
# records the profile states the OPPOSITE of the query preference the evaluator
# scores. A profile-derived executable check would therefore be a hard predicate
# demanding the wrong thing, and the repair loop would drive the plan away from
# a correct answer while burning rounds on a constraint nobody grades.
#
# How the profile should influence a plan is left to the model. The shared
# prompt trunk already states the query/profile priority, verbatim from the
# direct planner; the harness adds nothing on top, since how well a planner
# resolves that tension is part of what the benchmark measures.
SOURCES = ("query", "commonsense")


@dataclass
class Constraint:
    id: str
    kind: str               # commonsense | hard | preference
    source: str             # query | profile | commonsense
    text: str               # one-line natural-language restatement
    pseudocode: str         # the canonical, backend-independent form
    python: str | None = None          # filled by SPEC turn 2
    compiles: bool | None = None       # set by pre-flight
    smoke_ok: bool | None = None       # ran on a synthetic plan without raising
    error: str | None = None
    inert: bool = False                # can never return False -> enforces nothing
    # The source this check had before its first revision, kept verbatim and
    # never executed at run time. Plan selection is already immune to a
    # weakened check -- every candidate is scored under the same final
    # check-set, so leniency is a constant offset -- which leaves one question
    # for afterwards: did a revision repair a broken check or soften a working
    # one? This is the record that lets a diagnostic answer it offline. Set
    # once, so it holds the blind-authored original rather than a chain of edits.
    shadow_python: str | None = None

    @property
    def usable(self) -> bool:
        return bool(self.compiles) and bool(self.smoke_ok)

    @property
    def enforcing(self) -> bool:
        """Runs AND can actually fail. `usable` alone counted checks that were
        hard-wired to pass, which inflated coverage on exactly the records where
        the agent had given up on a constraint."""
        return self.usable and not self.inert


@dataclass
class Spec:
    constraints: list[Constraint] = field(default_factory=list)
    revisions: list[dict] = field(default_factory=list)   # {turn, trigger, detail}
    # The agent's own reading of the trip facts, parsed from the query in SPEC
    # turn 1. This becomes the `ctx` its checks run against.
    #
    # The harness deliberately does NOT supply these. A check reading a
    # harness-supplied `days` is correct even when the agent never understood
    # the trip length, which decouples verifier quality from comprehension --
    # the very thing verifier recall/precision is meant to measure. Making the
    # agent state them means a misread query produces wrong checks, which is a
    # failure worth seeing.
    #
    # It also yields a free measurement: compare these against the record's
    # true values for fact-extraction accuracy, the "Constraint Extraction"
    # sub-capability isolated in arXiv:2605.03308.
    facts: dict = field(default_factory=dict)

    # -- coverage ---------------------------------------------------------
    @property
    def usable_ids(self) -> list[str]:
        return [c.id for c in self.constraints if c.usable]

    def coverage(self) -> float:
        """Fraction of authored checks that actually enforce something. Reported
        per record; results are stratified on it rather than averaged over a
        mixture."""
        if not self.constraints:
            return 0.0
        return sum(c.enforcing for c in self.constraints) / len(self.constraints)

    def mode(self) -> str:
        """executable | partial | none -- the per-record audit field."""
        if not self.constraints:
            return "none"
        cov = self.coverage()
        return "executable" if cov == 1.0 else ("none" if cov == 0.0 else "partial")

    def by_kind(self, kind: str) -> list[Constraint]:
        return [c for c in self.constraints if c.kind == kind]

    def to_json(self) -> str:
        return json.dumps({"facts": self.facts,
                           "constraints": [asdict(c) for c in self.constraints],
                           "revisions": self.revisions}, ensure_ascii=False)

    def ctx(self) -> dict:
        """What the checks see. Agent-supplied only -- nothing from the record.

        `weekday` is computed from the agent's own dates, so a misparsed date
        yields a wrong weekday and its weekend-scoped checks fail. That is the
        intended behaviour.
        """
        from datetime import date as _d
        dates = [str(x) for x in (self.facts.get("dates") or [])]
        weekday = {}
        for x in dates:
            try:
                weekday[x] = _d.fromisoformat(x).strftime("%A")
            except ValueError:
                weekday[x] = "invalid date"
        return {"days": self.facts.get("days"), "people": self.facts.get("people"),
                "budget": self.facts.get("budget"), "dates": dates,
                "weekday": weekday}

    def fact_accuracy(self, truth: dict) -> dict:
        """Per-field agreement with the record. Reporting only -- never fed back."""
        out = {}
        for k in ("days", "people", "budget", "dates"):
            got, want = self.facts.get(k), truth.get(k)
            if isinstance(want, list):
                out[k] = [str(x) for x in (got or [])] == [str(x) for x in want]
            else:
                try:
                    out[k] = got is not None and float(got) == float(want)
                except (TypeError, ValueError):
                    out[k] = str(got) == str(want)
        return out

    # -- the LLM-fallback view (same spec, no Python) ----------------------
    def pseudocode_block(self, only_unusable: bool = False) -> str:
        """Render for an LLM judge. Because this is the *same* artifact the
        Python was compiled from, an LLM verdict is comparable to an executed
        one rather than a separate opinion."""
        cs = [c for c in self.constraints if (not only_unusable or not c.usable)]
        return "\n".join(f"[{c.id}] ({c.kind}) {c.text}\n    {c.pseudocode}" for c in cs)


_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")


def parse_pseudocode_turn(raw: str) -> tuple[Spec, list[str]]:
    """Parse SPEC turn 1. Returns (spec, problems); problems drive the retry."""
    problems: list[str] = []
    try:
        obj = json.loads(_strip_fence(raw))
    except Exception as e:
        return Spec(), [f"not valid JSON: {e}"]
    items = obj.get("constraints") if isinstance(obj, dict) else obj
    if not isinstance(items, list) or not items:
        return Spec(), ["expected a non-empty 'constraints' list"]

    spec, seen = Spec(), set()
    f = obj.get("facts") if isinstance(obj, dict) else None
    if isinstance(f, dict):
        spec.facts = {"days": f.get("days"), "people": f.get("people"),
                      "budget": f.get("budget"),
                      "dates": list(f.get("dates") or [])}
    else:
        problems.append("missing 'facts' object (days, dates, people, budget)")
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            problems.append(f"constraint {i} is not an object"); continue
        cid = str(it.get("id", "")).strip()
        if not _ID.match(cid):
            problems.append(f"constraint {i}: bad or missing id {cid!r}"); continue
        if cid in seen:
            problems.append(f"duplicate id {cid!r}"); continue
        kind = str(it.get("kind", "")).lower()
        src = str(it.get("source", "")).lower()
        if kind not in KINDS:
            problems.append(f"{cid}: kind must be one of {KINDS}, got {kind!r}"); continue
        if src == "profile":
            # Rejected outright rather than coerced. The query overrides the
            # profile wherever they disagree, and on this dataset they disagree
            # by design (157/225 test records drift, 78 by inversion), so a
            # profile-derived check can demand exactly the opposite of what is
            # scored -- and the repair loop would then drive a correct plan away
            # from the right answer.
            problems.append(
                f"{cid}: the profile is not a constraint source. The request "
                f"overrides it wherever they conflict, so a check written from "
                f"the profile can demand the wrong thing.")
            continue
        if src not in SOURCES:
            src = "query"
        pc = str(it.get("pseudocode", "")).strip()
        if not pc:
            problems.append(f"{cid}: empty pseudocode"); continue
        seen.add(cid)
        spec.constraints.append(Constraint(id=cid, kind=kind, source=src,
                                           text=str(it.get("text", "")).strip(),
                                           pseudocode=pc))
    if not spec.constraints:
        problems.append("no usable constraints parsed")
    return spec, problems


def attach_python_turn(spec: Spec, raw: str) -> list[str]:
    """Parse SPEC turn 2 and attach implementations by id. In place."""
    problems: list[str] = []
    try:
        obj = json.loads(_strip_fence(raw))
    except Exception as e:
        return [f"not valid JSON: {e}"]
    impls = obj.get("checks") if isinstance(obj, dict) else obj
    if not isinstance(impls, dict):
        return ["expected an object mapping constraint id -> python source"]
    known = {c.id: c for c in spec.constraints}
    for cid, src in impls.items():
        c = known.get(str(cid))
        if c is None:
            problems.append(f"unknown constraint id {cid!r}"); continue
        if not isinstance(src, str) or not src.strip():
            problems.append(f"{cid}: empty implementation"); continue
        c.python = src
    missing = [c.id for c in spec.constraints if not c.python]
    if missing:
        problems.append(f"no implementation for: {', '.join(missing)}")
    return problems


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    return s.strip()


# --------------------------------------------------------------------------- #
# Sandbox: scan, execute, pre-flight                                        #
# --------------------------------------------------------------------------- #
# Modules the checks may use. Deliberately small: enough to express predicates
# over a plan, not enough to reach the filesystem, the network, or the repo.
ALLOWED_IMPORTS = {"json", "re", "math", "datetime", "collections", "statistics",
                   "itertools", "functools", "operator"}

# Names that would reach benchmark semantics. Matched against imports, attribute
# chains and string literals, so `__import__("preferences")` is caught too.
BANNED_SUBSTRINGS = ("evaluation", "commonsense_constraint", "hard_constraint",
                     "evaluate_preferences", "preferences", "augment_preferences",
                     "reference_information", "prefertripplan", "eval.py")
BANNED_NAMES = {"__import__", "eval", "exec", "compile", "open", "input",
                "globals", "locals", "vars", "getattr", "setattr", "delattr",
                "__builtins__", "__subclasses__", "__bases__", "__mro__"}

DEFAULT_TIMEOUT = 10


@dataclass
class CheckVerdict:
    id: str
    ok: bool | None        # True pass, False violation, None could not run
    detail: str = ""
    error: str | None = None


def scan(src: str) -> list[str]:
    """Static rejection. Returns a list of problems; empty means acceptable."""
    problems: list[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod not in ALLOWED_IMPORTS:
                problems.append(f"import not allowed: from {node.module}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
            problems.append(f"attribute not allowed: .{node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            low = node.value.lower()
            for b in BANNED_SUBSTRINGS:
                if b in low:
                    problems.append(f"string references benchmark internals: {b!r}")
                    break
    return sorted(set(problems))


# The harness the checks run inside. Each check is a `check(plan, ctx)` returning
# (bool, detail). Exceptions are caught per check so one bad function cannot take
# the suite down.
_RUNNER = '''
import json, sys
payload = json.loads(sys.stdin.read())
plan, ctx, checks = payload["plan"], payload["ctx"], payload["checks"]
out = []
for cid, src in checks.items():
    ns = {}
    try:
        exec(src, ns, ns)
        fn = ns.get("check")
        if not callable(fn):
            out.append({"id": cid, "ok": None, "error": "no callable named 'check'"}); continue
        r = fn(plan, ctx)
        if isinstance(r, tuple):
            ok, detail = (list(r) + [""])[:2]
        else:
            ok, detail = r, ""
        out.append({"id": cid, "ok": bool(ok), "detail": str(detail)[:400]})
    except Exception as e:
        out.append({"id": cid, "ok": None, "error": f"{type(e).__name__}: {e}"[:300]})
print(json.dumps(out))
'''


def run(checks: dict[str, str], plan: Any, ctx: dict, *,
        timeout: int = DEFAULT_TIMEOUT) -> list[CheckVerdict]:
    """Execute a set of checks in a subprocess. Never raises."""
    if not checks:
        return []
    with tempfile.TemporaryDirectory() as td:
        runner = Path(td) / "_runner.py"
        runner.write_text(_RUNNER)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": td,
               "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1"}
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-S", str(runner)],
                input=json.dumps({"plan": plan, "ctx": ctx, "checks": checks}),
                capture_output=True, text=True, timeout=timeout, cwd=td, env=env)
        except subprocess.TimeoutExpired:
            return [CheckVerdict(cid, None, error=f"suite timed out after {timeout}s")
                    for cid in checks]
        if p.returncode != 0:
            return [CheckVerdict(cid, None, error=f"runner failed: {p.stderr[-200:]}")
                    for cid in checks]
        try:
            rows = json.loads(p.stdout)
        except Exception:
            return [CheckVerdict(cid, None, error="runner produced no JSON") for cid in checks]
    return [CheckVerdict(r["id"], r.get("ok"), r.get("detail", ""), r.get("error")) for r in rows]


def synthetic_plan(days: int, cities: list[str]) -> list[dict]:
    """A well-formed but arbitrary plan, used only for pre-flight.

    Deliberately NOT the agent's draft: smoke-running on the draft would let a
    check be written -- or silently tuned -- to the plan it will judge.
    """
    out = []
    for d in range(1, days + 1):
        city = cities[min(d - 1, len(cities) - 1)] if cities else "Somewhere"
        out.append({"days": d, "current_city": city,
                    "transportation": "-", "breakfast": f"Cafe {d}, {city}",
                    "attraction": f"Museum {d}, {city}", "lunch": f"Diner {d}, {city}",
                    "dinner": f"Bistro {d}, {city}",
                    "accommodation": f"Hotel {d}, {city}"})
    return out


def is_inert(src: str) -> bool:
    """True when every return path hands back a literal True.

    Observed shape: `return (True, "not checkable from the plan alone")` as the
    function's only statement. Such a check compiles, survives pre-flight and is
    marked usable, so it counted toward coverage while enforcing nothing -- on
    one record the sole preference check was of this form and coverage still
    read 1.00. Structural rather than behavioural, so it cannot mistake a check
    that merely happens to pass for one that can never fail.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    if not returns:
        return False
    for r in returns:
        v = r.value
        if isinstance(v, ast.Tuple) and v.elts:
            v = v.elts[0]
        if not (isinstance(v, ast.Constant) and v.value is True):
            return False
    return True


def preflight(spec, ctx: dict, *, timeout: int = DEFAULT_TIMEOUT) -> None:
    """Scan, then smoke-run every check. Sets compiles/smoke_ok/error in place."""
    runnable: dict[str, str] = {}
    for c in spec.constraints:
        if not c.python:
            c.compiles, c.smoke_ok, c.error = False, False, "no implementation"
            continue
        problems = scan(c.python)
        if problems:
            c.compiles, c.smoke_ok = False, False
            c.error = "; ".join(problems[:3])
            continue
        c.compiles = True
        c.inert = is_inert(c.python)
        runnable[c.id] = c.python

    if not runnable:
        return
    plan = synthetic_plan(int(ctx.get("days") or 3), list(ctx.get("cities") or []))
    verdicts = {v.id: v for v in run(runnable, plan, ctx, timeout=timeout)}
    for c in spec.constraints:
        if c.id in verdicts:
            v = verdicts[c.id]
            c.smoke_ok = v.error is None
            if v.error:
                c.error = v.error
