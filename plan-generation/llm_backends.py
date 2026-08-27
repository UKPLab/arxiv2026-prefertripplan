#!/usr/bin/env python3
"""LLM backend layer -- provider clients, retry ladder, batch generation.

Extracted verbatim from ``generate_plans.py`` so that both the direct planner
and the agentic track (``plan-generation/agentic/``) drive models through one
implementation. The retry ladders here encode provider-specific failure modes
this project actually hit -- OpenRouter proxies emitting bodies that break the
SDK's own JSON parser, GPT-5-family empty content when reasoning consumes the
token budget, Anthropic ``retry-after`` headers -- so they are shared rather
than reimplemented.

Nothing in this file was rewritten during the extraction; the classes are the
code that produced the published results, moved. ``generate_plans.py`` imports
them back and behaves identically.

Note on ``--backend openrouter``: it is a plain OpenAI-compatible client, so it
also drives a **locally hosted vLLM server** when ``--openrouter-base-url`` (or
``OPENROUTER_BASE_URL``) points at it. That is how ``qwen3.8-27b`` is served in
this project. ``VLLMOfflineBackend`` is the separate in-process path and has no
tool-calling surface.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

# Sampling default. Kept here because every backend reads it; re-exported by
# generate_plans so existing references keep working.
TEMPERATURE = 0.2


# --------------------------------------------------------------------------- #
# Progress reporter                                                           #
# --------------------------------------------------------------------------- #
class _ProgressReporter:
    def __init__(self, total: int, label: str = "generating"):
        self.total = total
        self.done  = 0
        self.label = label
        self.start = time.time()
        self._last_line_len = 0
        self._tqdm = None
        try:
            from tqdm import tqdm as _tqdm
            self._tqdm = _tqdm(total=total, desc=label, unit="req",
                               dynamic_ncols=True, mininterval=0.3, leave=True)
        except Exception:
            self._fallback_write(f"[progress] {label}: 0/{total} (starting...)")

    def _fmt_dt(self, s: float) -> str:
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
        elapsed = time.time() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remain = (self.total - self.done) / rate if rate > 0 else 0.0
        pct = 100.0 * self.done / self.total if self.total else 100.0
        self._fallback_write(
            f"[progress] {self.label}: {self.done}/{self.total} "
            f"({pct:5.1f}%)  elapsed={self._fmt_dt(elapsed)}  "
            f"eta={self._fmt_dt(remain)}  rate={rate:.2f} req/s")

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close(); self._tqdm = None
        elif self._last_line_len:
            sys.stderr.write("\n"); sys.stderr.flush()
            self._last_line_len = 0


# --------------------------------------------------------------------------- #
# Backends                                                                    #
# --------------------------------------------------------------------------- #
def _completion(content: str, reasoning: str | None = None) -> dict[str, Any]:
    """One backend result.  Every backend returns a list of these.

    Fields:
      ``content``   : the model's final response text (the plan itself,
                      after any reasoning phase).  Always a string --
                      empty ``""`` if the call failed or the model
                      produced no visible content.
      ``reasoning`` : the reasoning trace when the model / backend
                      exposes one, else ``None``.
                        - OpenRouter    : ``message.reasoning`` (when
                                          ``reasoning_effort`` is set or
                                          ``include_reasoning=True`` is
                                          passed via ``extra_body``).
                        - vLLM offline  : ``<think>...</think>`` blocks
                                          parsed out of the raw text.
                        - Anthropic     : ``thinking`` content blocks
                                          (only when thinking mode is on
                                          -- currently disabled so this
                                          is always ``None``).
                        - OpenAI (GPT-5)/o1/o3): opaque; reasoning
                                          tokens are billed but not
                                          returned in chat/completions,
                                          so this is ``None``.
    """
    return {"content": content, "reasoning": reasoning}


# Regex for pulling reasoning traces out of raw model text -- both
# ``<think>...</think>`` (DeepSeek-R1 style) and ``<thinking>...</thinking>``
# (Claude-style, though never emitted directly by that backend).
_THINK_RE = re.compile(r"<(think|thinking)>(.*?)</\1>",
                       re.IGNORECASE | re.DOTALL)


def _split_think(raw: str) -> tuple[str, str | None]:
    """Extract every ``<think>``/``<thinking>`` block from ``raw`` and
    return ``(content, reasoning)`` where content has those blocks
    stripped out and reasoning is the concatenation of every block's
    interior.  When no reasoning tags are present, returns
    ``(raw, None)`` unchanged -- callers can just spread the tuple
    straight into ``_completion``."""
    if not raw:
        return raw, None
    parts = _THINK_RE.findall(raw)
    if not parts:
        return raw, None
    reasoning = "\n\n".join(inner.strip() for _, inner in parts if inner.strip())
    content   = _THINK_RE.sub("", raw).strip()
    return content, (reasoning or None)


def _extract_reasoning_from_openai_message(message: Any) -> str | None:
    """Pull the reasoning trace out of a Chat-Completions ``message``
    object.  Different providers put it in different places -- try each
    known channel, take the first non-empty string:

      * ``message.reasoning``           -- OpenRouter, some providers'
                                          Chat-Completions extension.
      * ``message.reasoning_content``   -- vLLM's parsed-reasoning
                                          extension, some HuggingFace
                                          inference endpoints, DeepSeek
                                          direct API.
      * ``message.model_dump()['reasoning']`` -- when the SDK didn't
                                          declare the field as a class
                                          attribute but pydantic still
                                          kept it in ``model_extra``.
    """
    for attr in ("reasoning", "reasoning_content"):
        v = getattr(message, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # pydantic extra-fields fallback
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        try:
            d = dump()
            for k in ("reasoning", "reasoning_content"):
                v = d.get(k) if isinstance(d, dict) else None
                if isinstance(v, str) and v.strip():
                    return v.strip()
        except Exception:
            pass
    return None


class Backend:
    """Abstract base. ``generate_batch`` returns one completion dict per
    prompt (see ``_completion`` for the schema).

    ``on_result`` is an optional per-completion callback: as soon as a
    single prompt finishes, backends invoke it with ``(index, completion)``
    where ``completion`` is the same dict shape.  The caller persists
    the result to disk immediately so Ctrl-C interruptions leave every
    completed plan safely on disk.
    """
    name: str = "abstract"
    def generate_batch(self, prompts: list[str], *, max_tokens: int,
                       on_result=None) -> list[dict[str, Any]]:
        raise NotImplementedError


class AnthropicBackend(Backend):
    name = "anthropic"
    def __init__(self, model: str, workers: int = 8,
                 thinking_budget: int = 0,
                 temperature: float | None = None,
                 request_timeout: float = 120.0):
        import anthropic
        # Disable SDK's silent internal retries so ``_one`` handles 429s
        # visibly; per-request timeout cap keeps stuck sockets from
        # freezing a worker.
        self.client = anthropic.Anthropic(max_retries=0,
                                          timeout=request_timeout)
        # None -> module default, so historical runs reproduce exactly.
        # Ignored in thinking mode, which mandates the default.
        self.temperature = TEMPERATURE if temperature is None else temperature
        self.model = model
        self.workers = workers
        # Extended thinking: when > 0 we pass
        # ``thinking={"type":"enabled", "budget_tokens": <this>}`` to
        # ``messages.create``, which makes Claude emit ``thinking``
        # content blocks alongside the final ``text`` blocks.  Anthropic
        # requires the caller's ``max_tokens > budget_tokens`` and
        # forces ``temperature`` back to the default (1) when thinking
        # is on -- both are handled in ``_one``.  Set to 0 to disable.
        self.thinking_budget = int(thinking_budget or 0)

    def _one(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        """Single Anthropic call with EXPLICIT rate-limit handling.

        The default Anthropic SDK retries 429s internally with silent
        exponential backoff -- which looks like the process 'stalled'
        for tens of seconds with no output.  We surface the wait
        directly by catching RateLimitError, parsing the ``retry-after``
        header when available, logging to stderr so the user knows why
        the progress bar isn't moving, and retrying.

        Returns a completion dict.  ``reasoning`` carries the
        concatenated ``thinking`` content blocks if extended-thinking
        mode is on; ``None`` otherwise.
        """
        import re as _re, time as _time, sys as _sys, random as _random
        from anthropic import RateLimitError, APITimeoutError, APIConnectionError

        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                # Extended thinking, when enabled, requires the default
                # temperature (1) and ``max_tokens > budget_tokens``.
                # We size ``max_tokens`` to ``max(max_tokens, budget+1)``
                # defensively so a small caller max doesn't underflow
                # the budget.
                call_kwargs: dict = {
                    "model":    self.model,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if self.thinking_budget > 0:
                    call_kwargs["max_tokens"] = max(
                        max_tokens, self.thinking_budget + 1024)
                    call_kwargs["thinking"] = {
                        "type":          "enabled",
                        "budget_tokens": self.thinking_budget,
                    }
                    # Thinking mode: default temperature only.
                else:
                    call_kwargs["max_tokens"]  = max_tokens
                    call_kwargs["temperature"] = self.temperature
                msg = self.client.messages.create(**call_kwargs)
                content   = "".join(b.text for b in msg.content
                                    if getattr(b, "type", None) == "text").strip()
                thinking  = "\n\n".join(
                    getattr(b, "thinking", "") for b in msg.content
                    if getattr(b, "type", None) == "thinking"
                ).strip()
                return _completion(content, thinking or None)
            except RateLimitError as e:
                retry_after = None
                resp = getattr(e, "response", None)
                if resp is not None and hasattr(resp, "headers"):
                    try:
                        retry_after = float(resp.headers.get("retry-after") or 0)
                    except (TypeError, ValueError):
                        retry_after = None
                wait = retry_after if retry_after else min(60.0, 2 ** attempt)
                wait = max(wait, 0.5) + _random.uniform(0.0, 0.5)
                print(f"\n[rate-limit] anthropic 429   sleeping {wait:.2f}s "
                      f"(attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except (APITimeoutError, APIConnectionError) as e:
                wait = min(30.0, 2 ** attempt)
                print(f"\n[api-transient] {type(e).__name__}: "
                      f"retrying in {wait:.1f}s (attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
        print(f"\n[error] Anthropic call failed after {max_attempts} attempts; skipping.",
              file=_sys.stderr, flush=True)
        return _completion("", None)

    def generate_batch(self, prompts, *, max_tokens, on_result=None):
        results: list[dict[str, Any] | None] = [None] * len(prompts)
        reporter = _ProgressReporter(len(prompts),
                                     label=f"{self.name} ({self.workers}w)")
        import sys as _sys
        _first_error_logged = False
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._one, p, max_tokens): i
                        for i, p in enumerate(prompts)}
                for f in as_completed(futs):
                    idx = futs[f]
                    try:
                        comp = f.result()
                    except Exception as e:
                        if not _first_error_logged:
                            print(f"\n[error] worker {idx} raised {type(e).__name__}: {e}",
                                  file=_sys.stderr, flush=True)
                            _first_error_logged = True
                        comp = _completion("", None)
                    results[idx] = comp
                    if on_result is not None and comp.get("content"):
                        try:
                            on_result(idx, comp)
                        except Exception:
                            pass
                    reporter.step()
        finally:
            reporter.close()
        return results


# --------------------------------------------------------------------------- #
# Model-family sampling defaults                                              #
# --------------------------------------------------------------------------- #
# GPT-5-family / o1 / o3 reasoning models require the DEFAULT temperature
# (1) and expose the ``reasoning_effort`` + ``verbosity`` knobs.  Non-
# reasoning models (Claude, Llama, Gemini, GPT-4o family, ...) accept a
# custom temperature but reject the reasoning knobs.
#
# When routing through OpenRouter, model IDs carry a provider prefix
# (``openai/gpt-5.4-mini``, ``anthropic/claude-3.5-sonnet``, ...) -- the
# regex accepts either the bare or provider-prefixed form.

_REASONING_MODEL_RE = re.compile(
    r"^(?:[a-z0-9._-]+/)?(?:o1|o3|o4|gpt-5)", re.IGNORECASE)


def _is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_MODEL_RE.match(model or ""))


def _sampling_kwargs(model: str, temperature: float | None = None) -> dict:
    """Sampling / response-format kwargs to pass to
    ``chat.completions.create`` for a given model.  Reasoning models
    (GPT-5 family / o-series) omit ``temperature`` and add the
    reasoning-specific knobs; everything else uses the standard
    ``temperature`` path and skips the reasoning knobs.

    We ALSO request ``response_format=json_object`` on both branches so
    the planner's ``{"travel_plan": "..."}`` envelope is enforced
    consistently -- ``PLANNER_INSTRUCTION_PTP`` already mentions "JSON"
    in its final paragraph, which satisfies OpenAI's/OpenRouter's
    prompt-must-mention-JSON requirement.

    ``response_format=json_object`` is intentionally OMITTED on both
    branches: OpenRouter can proxy to providers (DeepSeek, some Llama
    endpoints, ...) that don't cleanly implement the flag and hand back
    malformed responses that surface as JSONDecodeError inside the SDK
    body parser.  ``_extract_plan_text`` gracefully handles both a JSON-
    wrapped ``{"travel_plan":"..."}`` response AND plain-text plans, so
    the JSON envelope keeps working without the strict-mode flag."""
    if _is_reasoning_model(model):
        return {
            "reasoning_effort":  "medium",
            "verbosity":         "medium",
        }
    return {
        # ``temperature`` None means "caller didn't ask" -> fall back to
        # the module default so every historical run reproduces exactly.
        "temperature": TEMPERATURE if temperature is None else temperature,
    }


_OPENAI_ENDPOINTS = ("chat_completions", "responses")


class OpenAIBackend(Backend):
    name = "openai"
    def __init__(self, model: str, workers: int = 8,
                 base_url: str | None = None, api_key: str | None = None,
                 reasoning_effort: str = "none",
                 verbosity: str = "medium",
                 endpoint: str = "responses",
                 reasoning_summary: str = "auto",
                 extra_body: dict | None = None,
                 temperature: float | None = None,
                 request_timeout: float = 120.0):
        from openai import OpenAI
        kwargs: dict = {}
        if base_url is not None: kwargs["base_url"] = base_url
        if api_key  is not None: kwargs["api_key"]  = api_key
        # Disable the SDK's silent internal retry so ``_one`` handles
        # 429s visibly.  Cap per-request timeout so stuck connections
        # can't freeze a worker forever.
        kwargs["max_retries"] = 0
        # Per-request wall-clock cap.  120s suits hosted APIs; a
        # self-hosted vLLM server generating a long reasoning trace plus
        # a 7-day plan under a deep request queue can exceed it, and
        # because ``max_retries=0`` every timeout regenerates from
        # scratch.  Raise via ``--request-timeout`` for local serving.
        kwargs["timeout"] = request_timeout
        self.client = OpenAI(**kwargs)
        self.model = model
        self.workers = workers
        # Reasoning-family knobs are per-instance so ``--reasoning-effort``
        # / ``--verbosity`` from the CLI can override the constructor
        # defaults without touching the source.  Defaults preserve the
        # previously-hardcoded values so callers relying on the old
        # behavior see no change.
        self.reasoning_effort = reasoning_effort
        self.verbosity        = verbosity
        if endpoint not in _OPENAI_ENDPOINTS:
            raise ValueError(
                f"OpenAIBackend endpoint must be one of {_OPENAI_ENDPOINTS}; "
                f"got {endpoint!r}")
        # ``chat_completions`` (default) preserves the exact prior
        # behavior -- content only, no reasoning trace returned.
        # ``responses`` routes through ``client.responses.create`` (the
        # Responses API), which surfaces reasoning summaries as items
        # of type="reasoning" on ``response.output``.  Summary opt-in
        # is controlled by ``reasoning_summary`` ("auto" / "concise" /
        # "detailed"); the Responses API returns nothing under
        # ``summary`` unless one of these is set.
        self.endpoint          = endpoint
        self.reasoning_summary = reasoning_summary
        # Free-form passthrough merged into the request body on every
        # call.  Exists for self-hosted OpenAI-compatible servers whose
        # request schema is a SUPERSET of OpenAI's -- notably vLLM,
        # which accepts `chat_template_kwargs` (e.g.
        # {"enable_thinking": true} for Qwen3), `top_k`, `min_p`,
        # `repetition_penalty` and `skip_special_tokens`, none of which
        # the OpenAI SDK models as named parameters.  Empty by default,
        # so hosted-API behavior is byte-identical to before.
        self.extra_body = dict(extra_body or {})
        # ``temperature`` stays None unless the caller asked for one.
        # The chat-completions path below then OMITS the field entirely,
        # which is REQUIRED for the GPT-5 family (they accept only the
        # default 1 and raise unsupported_value otherwise).  Setting it
        # is how you pin sampling on an OpenAI-compatible local server.
        self.temperature     = temperature
        self.request_timeout = request_timeout

    def _one(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        """Dispatch a single OpenAI call to the configured endpoint."""
        if self.endpoint == "responses":
            return self._one_responses(prompt, max_tokens)
        return self._one_chat_completions(prompt, max_tokens)

    def _one_chat_completions(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        """Single OpenAI call with EXPLICIT rate-limit handling.

        See AnthropicBackend._one for the rationale -- OpenAI's SDK
        also retries 429s silently under the hood, which mimics a
        hang from the caller's point of view.  We surface it.
        """
        import re as _re, time as _time, sys as _sys, random as _random
        from openai import RateLimitError, APITimeoutError, APIConnectionError

        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                # GPT-5-family reasoning models: `temperature` MUST be
                # the default (1); passing anything else raises
                # unsupported_value.  Omit it entirely so the SDK uses
                # the default.  Reasoning knobs + json_object response
                # format are supported.  See PLANNER_INSTRUCTION_PTP
                # which instructs the model to return a JSON object.
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                    reasoning_effort=self.reasoning_effort,
                    verbosity=self.verbosity,
                    # response_format={"type": "json_object"},
                    **({"temperature": self.temperature}
                       if self.temperature is not None else {}),
                    **({"extra_body": self.extra_body} if self.extra_body else {}),
                )
                choice = resp.choices[0]
                content = (choice.message.content or "").strip()
                # Reasoning content on GPT-5-family via chat/completions
                # is opaque -- only summarised via the Responses API,
                # not returned here.  Extract if the SDK ever surfaces
                # it as a plain attribute; otherwise None.
                reasoning = _extract_reasoning_from_openai_message(choice.message)
                # Empty content on GPT-5-family means the reasoning
                # tokens ate the budget (finish_reason = "length") or
                # the API refused (finish_reason = "content_filter" /
                # other).  Log the diagnostic ONCE per retry so
                # silent-empty-writes don't look like a hang.
                if not content:
                    fr = getattr(choice, "finish_reason", None)
                    usage = getattr(resp, "usage", None)
                    reasoning_tok = None
                    if usage is not None:
                        details = getattr(usage, "completion_tokens_details", None)
                        if details is not None:
                            reasoning_tok = getattr(details, "reasoning_tokens", None)
                    print(f"\n[empty] finish_reason={fr!r}  "
                          f"completion_tokens={getattr(usage,'completion_tokens',None)}  "
                          f"reasoning_tokens={reasoning_tok}  "
                          f"max_completion_tokens={max_tokens}"
                          f"   -> raise max_completion_tokens or drop reasoning_effort to 'low'",
                          file=_sys.stderr, flush=True)
                return _completion(content, reasoning)
            except RateLimitError as e:
                msg = str(getattr(e, "message", e))
                m = _re.search(r"try again in\s+([\d.]+)\s*(ms|s|m)", msg)
                if m:
                    v = float(m.group(1)); unit = m.group(2)
                    wait = v/1000.0 if unit == "ms" else (v*60.0 if unit == "m" else v)
                else:
                    wait = min(60.0, 2 ** attempt)
                wait = max(wait, 0.5) + _random.uniform(0.0, 0.5)
                # Trim the (typically-long) SDK message down to essentials.
                lim = _re.search(r"Limit\s+(\d+)", msg)
                usd = _re.search(r"Used\s+(\d+)", msg)
                req = _re.search(r"Requested\s+(\d+)", msg)
                brief = (f"limit={lim.group(1) if lim else '?'} "
                         f"used={usd.group(1) if usd else '?'} "
                         f"req={req.group(1) if req else '?'}"
                         if (lim or usd or req) else msg[:80])
                print(f"\n[rate-limit] {brief}   sleeping {wait:.2f}s "
                      f"(attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except (APITimeoutError, APIConnectionError) as e:
                wait = min(30.0, 2 ** attempt)
                print(f"\n[api-transient] {type(e).__name__}: "
                      f"retrying in {wait:.1f}s (attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except json.JSONDecodeError as e:
                # OpenAI SDK's own response-body JSON parser failed --
                # typically a truncated response (proxy timeout, TLS
                # reset mid-stream) that made it past the low-level
                # network layer but arrived as invalid JSON at the
                # decoder.  Treated as a transient error worth retrying.
                wait = min(30.0, 2 ** attempt)
                print(f"\n[api-transient] JSONDecodeError from SDK "
                      f"(truncated response?):  {e}   "
                      f"retrying in {wait:.1f}s (attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except Exception as e:
                # Backstop: any OTHER exception (unexpected SDK error,
                # httpx transport blip, etc.) becomes retry-able rather
                # than a silent-empty write.  BadRequestError on
                # unsupported params etc. STILL re-raises on the last
                # attempt so the outer generate_batch logs it.
                if attempt < max_attempts - 1:
                    wait = min(30.0, 2 ** attempt)
                    print(f"\n[api-unknown] {type(e).__name__}: {e}   "
                          f"retrying in {wait:.1f}s "
                          f"(attempt {attempt+1}/{max_attempts})",
                          file=_sys.stderr, flush=True)
                    _time.sleep(wait)
                    continue
                # Last attempt failed with an unknown error -- re-raise
                # so generate_batch's [error] log surfaces it.
                raise
        print(f"\n[error] OpenAI call failed after {max_attempts} attempts; skipping.",
              file=_sys.stderr, flush=True)
        return _completion("", None)

    def _one_responses(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        """Same retry semantics as the chat-completions path, but
        routes through ``client.responses.create`` (the Responses API).

        The Responses API surfaces reasoning summaries as items of
        type="reasoning" on ``response.output``, each carrying a
        ``.summary`` list of ``{type: "summary_text", text: str}``
        entries.  We concatenate every summary_text as the ``reasoning``
        field.  Final text comes from ``response.output_text`` -- the
        SDK's convenience accessor that pulls together every
        ``output_text`` chunk from message items.

        Reasoning summary is opt-in on the Responses API; nothing is
        returned unless ``reasoning={"summary": ...}`` is set (we use
        ``self.reasoning_summary`` -- "auto" by default)."""
        import re as _re, time as _time, sys as _sys, random as _random
        from openai import RateLimitError, APITimeoutError, APIConnectionError

        # Build the reasoning config: effort + summary opt-in.
        reasoning_cfg: dict = {}
        if self.reasoning_effort and self.reasoning_effort != "none":
            reasoning_cfg["effort"] = self.reasoning_effort
        if self.reasoning_summary:
            reasoning_cfg["summary"] = self.reasoning_summary

        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                resp = self.client.responses.create(
                    model             = self.model,
                    input             = prompt,
                    max_output_tokens = max_tokens,
                    reasoning         = reasoning_cfg or None,
                )
                # Content: prefer the SDK's flattened convenience.
                content = (getattr(resp, "output_text", "") or "").strip()
                # Reasoning: walk response.output for reasoning items.
                reasoning_bits: list[str] = []
                for item in (getattr(resp, "output", None) or []):
                    if getattr(item, "type", None) != "reasoning":
                        continue
                    for s in (getattr(item, "summary", None) or []):
                        s_type = getattr(s, "type", None)
                        s_text = getattr(s, "text", None)
                        if s_type in ("summary_text", "text") and s_text:
                            reasoning_bits.append(s_text.strip())
                reasoning = "\n\n".join(reasoning_bits).strip() or None

                if not content:
                    fr    = getattr(resp, "status", None)
                    usage = getattr(resp, "usage", None)
                    reasoning_tok = None
                    if usage is not None:
                        details = getattr(usage, "output_tokens_details", None)
                        if details is not None:
                            reasoning_tok = getattr(details, "reasoning_tokens", None)
                    print(f"\n[empty] responses status={fr!r}  "
                          f"output_tokens={getattr(usage,'output_tokens',None)}  "
                          f"reasoning_tokens={reasoning_tok}  "
                          f"max_output_tokens={max_tokens}   "
                          f"-> raise max_output_tokens or drop reasoning_effort",
                          file=_sys.stderr, flush=True)
                return _completion(content, reasoning)
            except RateLimitError as e:
                msg = str(getattr(e, "message", e))
                m = _re.search(r"try again in\s+([\d.]+)\s*(ms|s|m)", msg)
                if m:
                    v = float(m.group(1)); unit = m.group(2)
                    wait = v/1000.0 if unit == "ms" else (v*60.0 if unit == "m" else v)
                else:
                    wait = min(60.0, 2 ** attempt)
                wait = max(wait, 0.5) + _random.uniform(0.0, 0.5)
                print(f"\n[rate-limit] responses 429   sleeping {wait:.2f}s "
                      f"(attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except (APITimeoutError, APIConnectionError) as e:
                wait = min(30.0, 2 ** attempt)
                print(f"\n[api-transient] responses {type(e).__name__}: "
                      f"retrying in {wait:.1f}s (attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except json.JSONDecodeError as e:
                wait = min(30.0, 2 ** attempt)
                print(f"\n[api-transient] responses JSONDecodeError "
                      f"(truncated response?):  {e}   "
                      f"retrying in {wait:.1f}s (attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except Exception as e:
                if attempt < max_attempts - 1:
                    wait = min(30.0, 2 ** attempt)
                    print(f"\n[api-unknown] responses {type(e).__name__}: {e}   "
                          f"retrying in {wait:.1f}s "
                          f"(attempt {attempt+1}/{max_attempts})",
                          file=_sys.stderr, flush=True)
                    _time.sleep(wait)
                    continue
                raise
        print(f"\n[error] Responses call failed after {max_attempts} attempts; skipping.",
              file=_sys.stderr, flush=True)
        return _completion("", None)

    def generate_batch(self, prompts, *, max_tokens, on_result=None):
        results: list[dict[str, Any] | None] = [None] * len(prompts)
        reporter = _ProgressReporter(len(prompts),
                                     label=f"{self.name} ({self.workers}w)")
        import sys as _sys
        _first_error_logged = False
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._one, p, max_tokens): i
                        for i, p in enumerate(prompts)}
                for f in as_completed(futs):
                    idx = futs[f]
                    try:
                        comp = f.result()
                    except Exception as e:
                        # Surface the FIRST unexpected exception so silent
                        # cache-writes-not-happening doesn't look like a hang.
                        # (RateLimit / Timeout / ConnectionError are handled
                        # inside _one with visible logs; anything reaching
                        # here is an unexpected error worth calling out --
                        # e.g., BadRequestError on unsupported params.)
                        if not _first_error_logged:
                            print(f"\n[error] worker {idx} raised {type(e).__name__}: {e}",
                                  file=_sys.stderr, flush=True)
                            _first_error_logged = True
                        comp = _completion("", None)
                    results[idx] = comp
                    if on_result is not None and comp.get("content"):
                        try:
                            on_result(idx, comp)
                        except Exception:
                            pass
                    reporter.step()
        finally:
            reporter.close()
        return results


class OpenRouterBackend(OpenAIBackend):
    """OpenRouter uses the OpenAI-compatible chat/completions protocol,
    so we inherit the entire retry / callback / thread-pool plumbing
    from ``OpenAIBackend`` and only override:

      * ``__init__`` -- point the OpenAI SDK at OpenRouter's base URL
        and use ``OPENROUTER_API_KEY`` (falls back to ``OPENAI_API_KEY``
        if you've overloaded that env var, or accept an explicit key).
      * ``_one``     -- pick sampling params based on the routed model
        (see ``_sampling_kwargs``): a reasoning-family model routed via
        ``openai/gpt-5.4-mini`` gets ``reasoning_effort`` + ``verbosity``
        with no ``temperature``; a non-reasoning model like
        ``anthropic/claude-3.5-sonnet`` gets ``temperature`` and skips
        the reasoning knobs.  Both request ``response_format=json_object``
        so ``PLANNER_INSTRUCTION_PTP``'s JSON envelope is enforced.

    Model IDs use OpenRouter's ``provider/model`` scheme -- e.g.
    ``openai/gpt-5.4-mini``, ``anthropic/claude-3.5-sonnet``,
    ``meta-llama/llama-3.3-70b-instruct``, ``google/gemini-2.0-flash-exp:free``.
    """
    name = "openrouter"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str, workers: int = 8,
                 base_url: str | None = None, api_key: str | None = None,
                 reasoning_effort: str | None = None,
                 verbosity: str | None = None,
                 extra_body: dict | None = None,
                 temperature: float | None = None,
                 request_timeout: float = 120.0):
        # OPENROUTER_API_KEY is the canonical env var; the parent
        # OpenAI SDK client would otherwise look at OPENAI_API_KEY.
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        # OpenRouter proxies to Chat Completions ONLY; the Responses
        # API doesn't exist upstream.  Pin the parent endpoint back to
        # chat_completions so the recently-flipped OpenAIBackend default
        # doesn't accidentally route OpenRouter through .responses.
        super().__init__(
            model    = model,
            workers  = workers,
            base_url   = base_url or self.DEFAULT_BASE_URL,
            api_key    = api_key,
            endpoint   = "chat_completions",
            extra_body = extra_body,
            request_timeout = request_timeout,
        )
        # NOTE: deliberately NOT forwarded to the parent as
        # ``temperature``.  This backend builds its request through
        # ``_sampling_kwargs`` (see ``_one``), which is where the value
        # is applied -- and which correctly SKIPS it for a reasoning
        # model.  Forwarding it as well would send the field twice.
        self.temperature = temperature
        # CLI-driven overrides.  When ``reasoning_effort`` / ``verbosity``
        # is None, ``_sampling_kwargs`` picks the model-family default
        # for a reasoning model and omits the knob entirely for a
        # non-reasoning one.  When set, we override / inject it for THIS
        # backend so the user can dial the effort per run.
        self.reasoning_effort = reasoning_effort
        self.verbosity        = verbosity

    def _one(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        """OpenRouter version of the retry loop.  Same shape as
        ``OpenAIBackend._one`` but routes sampling params through
        ``_sampling_kwargs`` so mixed-family models work.

        Returns a completion dict.  ``reasoning`` is populated from
        ``message.reasoning`` when the upstream reasoning model returns
        it -- OpenRouter forwards the trace as an extension field on
        the Chat-Completions envelope when either ``reasoning_effort``
        is set or ``include_reasoning=True`` is passed via
        ``extra_body``.  For safety we do BOTH: the model-family knob
        AND the explicit toggle."""
        import re as _re, time as _time, sys as _sys, random as _random
        from openai import RateLimitError, APITimeoutError, APIConnectionError

        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                # Model-family default sampling params (see ``_sampling_kwargs``)
                # with CLI overrides applied on top.  ``reasoning_effort`` and
                # ``verbosity`` are ADDED even if the model wasn't classified
                # as a reasoning model -- the user knows what they routed to.
                sk = _sampling_kwargs(self.model, self.temperature)
                if self.reasoning_effort is not None:
                    sk["reasoning_effort"] = self.reasoning_effort
                    # When the user explicitly turns reasoning on for a
                    # non-classified reasoning model (e.g. DeepSeek-V4-
                    # Flash on OpenRouter which supports high / xhigh),
                    # drop `temperature` -- reasoning-capable providers
                    # generally require the default temperature and
                    # reject a custom value alongside reasoning_effort.
                    sk.pop("temperature", None)
                if self.verbosity is not None:
                    sk["verbosity"] = self.verbosity
                resp = self.client.chat.completions.create(
                    model                 = self.model,
                    messages              = [{"role": "user", "content": prompt}],
                    # messages              = [{"role": "user", "content": prompt + "\nInclude your full, step-by-step thinking process inside <think>...</think> tags before giving your final structured plan under <plan>...</plan>."}],
                    max_completion_tokens = max_tokens,
                    # OpenRouter-specific: force the API to return the
                    # reasoning trace on the message.  Harmless when
                    # the upstream model doesn't have reasoning.
                    # User-supplied `--extra-body` keys are merged on
                    # top, so a vLLM run can add e.g.
                    # {"chat_template_kwargs": {"enable_thinking": true}}
                    # and may override include_reasoning if it wants.
                    extra_body            = {"include_reasoning": True,
                                             **self.extra_body},
                    **sk,
                )
                # import pdb; pdb.set_trace()
                # Shared parser: ChatTurn is a superset of what this path
                # returns, so both sides cannot drift on extraction.
                turn      = parse_chat_response(resp, self.model)
                choice    = resp.choices[0]
                content, reasoning = turn.text, turn.reasoning
                if not content:
                    fr = getattr(choice, "finish_reason", None)
                    usage = getattr(resp, "usage", None)
                    reasoning_tok = None
                    if usage is not None:
                        details = getattr(usage, "completion_tokens_details", None)
                        if details is not None:
                            reasoning_tok = getattr(details, "reasoning_tokens", None)
                    print(f"\n[empty] openrouter model={self.model!r}  "
                          f"finish_reason={fr!r}  "
                          f"completion_tokens={getattr(usage,'completion_tokens',None)}  "
                          f"reasoning_tokens={reasoning_tok}  "
                          f"max_completion_tokens={max_tokens}",
                          file=_sys.stderr, flush=True)
                return _completion(content, reasoning)
            except RateLimitError as e:
                # OpenRouter forwards upstream provider 429s.  Parse the
                # server's suggested wait when present, fall back to
                # exponential backoff otherwise.
                msg = str(getattr(e, "message", e))
                m = _re.search(r"try again in\s+([\d.]+)\s*(ms|s|m)", msg)
                if m:
                    v = float(m.group(1)); unit = m.group(2)
                    wait = v/1000.0 if unit == "ms" else (v*60.0 if unit == "m" else v)
                else:
                    wait = min(60.0, 2 ** attempt)
                wait = max(wait, 0.5) + _random.uniform(0.0, 0.5)
                print(f"\n[rate-limit] openrouter 429   sleeping {wait:.2f}s "
                      f"(attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except (APITimeoutError, APIConnectionError) as e:
                wait = min(30.0, 2 ** attempt)
                print(f"\n[api-transient] openrouter {type(e).__name__}: "
                      f"retrying in {wait:.1f}s (attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except json.JSONDecodeError as e:
                # Truncated / malformed response body from the SDK's
                # own JSON parser (proxy timeout, TLS reset, upstream
                # provider blip forwarded by OpenRouter).  Retriable.
                wait = min(30.0, 2 ** attempt)
                print(f"\n[api-transient] openrouter JSONDecodeError "
                      f"(truncated response?):  {e}   "
                      f"retrying in {wait:.1f}s (attempt {attempt+1}/{max_attempts})",
                      file=_sys.stderr, flush=True)
                _time.sleep(wait)
                continue
            except Exception as e:
                # Unknown SDK / transport error -- treat as transient
                # for all but the final attempt.  On the last attempt
                # we re-raise so generate_batch's [error] log fires.
                if attempt < max_attempts - 1:
                    wait = min(30.0, 2 ** attempt)
                    print(f"\n[api-unknown] openrouter {type(e).__name__}: {e}   "
                          f"retrying in {wait:.1f}s "
                          f"(attempt {attempt+1}/{max_attempts})",
                          file=_sys.stderr, flush=True)
                    _time.sleep(wait)
                    continue
                raise
        print(f"\n[error] OpenRouter call failed after {max_attempts} attempts; skipping.",
              file=_sys.stderr, flush=True)
        return _completion("", None)


class VLLMOfflineBackend(Backend):
    name = "vllm-offline"
    def __init__(self, model: str, *,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: int | None = None,
                 dtype: str = "auto",
                 tensor_parallel_size: int = 1,
                 trust_remote_code: bool = True,
                 temperature: float | None = None,
                 **extra):
        from vllm import LLM
        self.LLM_class = LLM
        self.model = model
        # None -> module default, so historical runs reproduce exactly.
        self.temperature = TEMPERATURE if temperature is None else temperature
        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            **extra,
        )

    def generate_batch(self, prompts, *, max_tokens, on_result=None):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=self.temperature, max_tokens=max_tokens)
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self.llm.chat(conversations, sampling_params=sp, use_tqdm=True)

        # Extract reasoning per completion.  Two paths:
        #   1) vLLM's reasoning parser (``--enable-reasoning
        #      --reasoning-parser deepseek-r1``) surfaces the reasoning
        #      under ``o.outputs[0].reasoning_content`` and keeps the
        #      final response in ``.text`` clean of think tags.
        #   2) Without the parser, reasoning models emit
        #      ``<think>...</think>`` inline in ``.text``; we split
        #      those blocks out here via ``_split_think``.
        completions: list[dict[str, Any]] = []
        for o in outputs:
            out0 = o.outputs[0]
            reasoning = getattr(out0, "reasoning_content", None)
            raw_text  = (out0.text or "").strip()
            if reasoning and isinstance(reasoning, str) and reasoning.strip():
                content   = raw_text
                reasoning = reasoning.strip()
            else:
                content, reasoning = _split_think(raw_text)
            completions.append(_completion(content, reasoning))

        # vLLM's internal batching is opaque; the best we can do here
        # is fire on_result after the whole batch completes so every
        # completed plan lands in the cache before the run's end (and
        # merge / write_output_jsonl) is invoked.
        if on_result is not None:
            for i, comp in enumerate(completions):
                if comp.get("content"):
                    try:
                        on_result(i, comp)
                    except Exception:
                        pass
        return completions


def make_backend(name: str, model: str, *,
                 workers: int = 8,
                 openai_base_url: str | None = None,
                 openai_api_key: str | None = None,
                 openai_endpoint: str = "responses",
                 openai_reasoning_summary: str = "auto",
                 openrouter_base_url: str | None = None,
                 openrouter_api_key: str | None = None,
                 anthropic_thinking_budget: int = 0,
                 reasoning_effort: str | None = None,
                 verbosity: str | None = None,
                 vllm_gpu_memory: float = 0.9,
                 vllm_max_model_len: int | None = None,
                 vllm_dtype: str = "auto",
                 vllm_tensor_parallel: int = 1,
                 extra_body: dict | None = None,
                 temperature: float | None = None,
                 request_timeout: float = 120.0) -> Backend:
    if name == "anthropic":
        return AnthropicBackend(model=model, workers=workers,
                                thinking_budget=anthropic_thinking_budget,
                                temperature=temperature,
                                request_timeout=request_timeout)
    if name == "openai":
        # Pass through CLI-set knobs; None means "use constructor
        # default" (which preserves the previously-hardcoded values).
        kw: dict = {"endpoint": openai_endpoint,
                    "reasoning_summary": openai_reasoning_summary,
                    "extra_body": extra_body,
                    "temperature": temperature,
                    "request_timeout": request_timeout}
        if reasoning_effort is not None: kw["reasoning_effort"] = reasoning_effort
        if verbosity        is not None: kw["verbosity"]        = verbosity
        return OpenAIBackend(model=model, workers=workers,
                             base_url=openai_base_url, api_key=openai_api_key,
                             **kw)
    if name == "openrouter":
        return OpenRouterBackend(model=model, workers=workers,
                                 base_url=openrouter_base_url,
                                 api_key=openrouter_api_key,
                                 reasoning_effort=reasoning_effort,
                                 verbosity=verbosity,
                                 extra_body=extra_body,
                                 temperature=temperature,
                                 request_timeout=request_timeout)
    if name == "vllm-offline":
        return VLLMOfflineBackend(
            model=model,
            gpu_memory_utilization=vllm_gpu_memory,
            max_model_len=vllm_max_model_len,
            dtype=vllm_dtype,
            tensor_parallel_size=vllm_tensor_parallel,
            temperature=temperature,
        )
    raise SystemExit(f"unknown backend: {name}")

# --------------------------------------------------------------------------- #
# Shared client construction + retry ladder                                   #
# --------------------------------------------------------------------------- #
# Extracted so the agentic runtime drives models through the same code path as
# the direct planner. The four Backend classes above keep their own inlined
# copies of this ladder deliberately: they produced the published results and
# are left byte-identical. New callers use these.
#
# The ladder encodes failure modes this project actually hit -- OpenRouter
# proxies returning bodies that break the SDK's JSON parser, GPT-5-family empty
# content when reasoning eats the token budget, Anthropic retry-after headers --
# so it is worth having exactly one of.

def make_openai_client(base_url: str | None = None, api_key: str | None = None,
                       request_timeout: float = 120.0):
    """An OpenAI-compatible client. Also drives a self-hosted vLLM server when
    ``base_url`` points at one; such servers ignore the key, so a placeholder is
    sent rather than failing on a missing one.

    ``max_retries=0`` disables the SDK's silent internal retries so that
    ``retry_call`` owns backoff and the waits are visible in logs.
    """
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key or "local",
                  max_retries=0, timeout=request_timeout)


def parse_retry_after(exc: Exception) -> float | None:
    """Seconds requested by the provider, if it said. Handles the header and the
    ``try again in 350ms`` phrasing both OpenAI and OpenRouter use."""
    import re as _re
    hdrs = getattr(getattr(exc, "response", None), "headers", None)
    if hdrs:
        v = hdrs.get("retry-after") or hdrs.get("Retry-After")
        if v:
            try:
                return float(v)
            except ValueError:
                pass
    m = _re.search(r"try again in ([\d.]+)\s*(ms|s|m)", str(exc))
    if m:
        n, unit = float(m.group(1)), m.group(2)
        return n / 1000 if unit == "ms" else (n * 60 if unit == "m" else n)
    return None


def build_chat_request(model: str, messages: list[dict], max_tokens: int, *,
                       temperature: float | None = None,
                       reasoning_effort: str | None = None,
                       verbosity: str | None = None,
                       extra_body: dict | None = None,
                       tools: list[dict] | None = None,
                       tool_choice: str | dict | None = None) -> dict:
    """Assemble a chat-completions request the one way this project does it.

    Extracted so the batch planner and the tool-calling loop cannot drift on
    plumbing. They already had: the batch path sends ``max_completion_tokens``
    while the tool path was sending ``max_tokens`` -- a different API parameter,
    deprecated for newer models and rejected outright by some.

    Sampling resolution is `_sampling_kwargs`' model-family default with CLI
    overrides on top, and ``temperature`` dropped whenever ``reasoning_effort``
    is in play, since reasoning-capable providers generally reject both.

    ``include_reasoning`` is requested so the trace comes back on the message;
    user ``extra_body`` keys are merged over it and may turn it off.
    """
    sk = _sampling_kwargs(model, temperature)
    if reasoning_effort is not None:
        sk["reasoning_effort"] = reasoning_effort
        sk.pop("temperature", None)
    if verbosity is not None:
        sk["verbosity"] = verbosity
    kw: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "extra_body": {"include_reasoning": True, **(extra_body or {})},
        **sk,
    }
    if tools:
        kw["tools"] = tools
        kw["tool_choice"] = tool_choice or "auto"
        # One observation per step keeps step accounting exact.
        kw["parallel_tool_calls"] = False
    return kw


_TEXT_TOOLCALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[\w.-]+)>(?P<body>.*?)</function>\s*</tool_call>",
    re.S)
_TEXT_PARAM_RE = re.compile(
    r"<parameter=(?P<key>[\w.-]+)>(?P<val>.*?)</parameter>", re.S)


def _parse_text_tool_calls(text: str) -> list["ToolCall"]:
    """Recover tool calls a model emitted as markup rather than natively.

    Hermes-style XML:

        <tool_call><function=submit_plan>
          <parameter=travel_plan>Day 1: ...</parameter>
        </function></tool_call>

    vLLM parses this server-side given --tool-call-parser hermes; hosted
    endpoints do not always, and the raw markup then ends up inside whatever
    the caller does with the text -- for submit_plan that means the plan itself
    carries `</parameter></function></tool_call>`, which the plan parser cannot
    read. Recovering it here keeps one code path downstream.
    """
    out: list[ToolCall] = []
    for i, m in enumerate(_TEXT_TOOLCALL_RE.finditer(text)):
        args = {k: v.strip() for k, v in _TEXT_PARAM_RE.findall(m.group("body"))}
        out.append(ToolCall(id=f"text_{i}", name=m.group("name"),
                            args_raw=json.dumps(args), args=args, error=None))
    return out


def parse_chat_response(resp, model: str = "", *, latency_s: float = 0.0) -> "ChatTurn":
    """Parse one chat-completions response, the one way this project does it.

    `ChatTurn` is a strict superset of the batch path's `{content, reasoning}`,
    so both callers use this and the batch one projects: `_completion(turn.text,
    turn.reasoning)`. Keeping two parsers is how `max_completion_tokens` vs
    `max_tokens` and a missing `reasoning_effort` went unnoticed -- invisible
    until a run fails.

    Verified a no-op on the published runs: across all 1,125 cached rows, none
    carried `<think>` tags in content and none needed stripping, so the
    `_split_think` fallback and `.strip()` change nothing there. On a response
    that DOES carry think tags the batch path previously left them in the plan
    text; now they are extracted, which is a fix.
    """
    choices = getattr(resp, "choices", None)
    if not choices:
        # OpenRouter returns a body with `choices: null` when an upstream
        # provider errors. Previously this surfaced as
        # "TypeError: 'NoneType' object is not subscriptable" from choices[0],
        # outside the retry, killing the whole record.
        raise ValueError(f"response carried no choices: {getattr(resp, 'error', resp)!r}")
    choice = choices[0]
    msg = choice.message
    turn = ChatTurn(text=(msg.content or "").strip(), latency_s=latency_s,
                    raw_message=msg,
                    finish_reason=getattr(choice, "finish_reason", None))
    turn.reasoning = _extract_reasoning_from_openai_message(msg)
    if not turn.reasoning and turn.text:
        # Providers that inline the trace rather than returning it separately.
        turn.text, turn.reasoning = _split_think(turn.text)

    for tc in (getattr(msg, "tool_calls", None) or []):
        raw = tc.function.arguments or "{}"
        try:
            args, err = json.loads(raw), None
            if not isinstance(args, dict):
                args, err = None, (f"arguments must be a JSON object, got "
                                   f"{type(args).__name__}")
        except Exception as e:
            args, err = None, f"arguments are not valid JSON: {e}"
        turn.tool_calls.append(ToolCall(id=tc.id, name=tc.function.name,
                                        args_raw=raw, args=args, error=err))

    # Some endpoints emit a tool call as Hermes-style XML in the content
    # instead of a native tool_call -- observed on hosted nemotron for ~4% of
    # turns. Recover it: otherwise the markup lands in the plan text, and a
    # submit_plan issued this way is lost entirely.
    if not turn.tool_calls and turn.text and "<function=" in turn.text:
        recovered = _parse_text_tool_calls(turn.text)
        if recovered:
            turn.tool_calls = recovered
            turn.text = _TEXT_TOOLCALL_RE.sub("", turn.text).strip()

    u = getattr(resp, "usage", None)
    if u:
        det = getattr(u, "completion_tokens_details", None)
        turn.usage = Usage(
            prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
            reasoning_tokens=int(getattr(det, "reasoning_tokens", 0) or 0) if det else 0)
        turn.usage.cost_usd = _cost(model or getattr(resp, "model", ""), turn.usage)
    return turn


def retry_call(fn, *, max_attempts: int = 8, label: str = "call"):
    """Run ``fn`` with the project's backoff ladder. Retries the SINGLE call --
    never an agent loop, which would repeat tool side effects."""
    import random as _random
    import openai as _openai
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt == max_attempts - 1:
                break
            if isinstance(e, _openai.RateLimitError):
                wait = parse_retry_after(e) or min(60.0, 2 ** attempt)
                kind = "rate-limit"
            elif isinstance(e, (_openai.APITimeoutError, _openai.APIConnectionError)):
                wait, kind = min(30.0, 2 ** attempt), "api-transient"
            elif isinstance(e, json.JSONDecodeError):
                wait, kind = min(30.0, 2 ** attempt), "api-badbody"
            elif isinstance(e, ValueError) and "no choices" in str(e):
                # An upstream provider 5xx forwarded as a 200 with
                # `choices: null` -- e.g. "Upstream error from Nvidia: Service
                # temporarily overloaded". Transient, and worth its own label
                # so it is not mistaken for a bug in our parsing.
                wait, kind = min(60.0, 2 ** attempt), "api-upstream"
            else:
                wait, kind = min(30.0, 2 ** attempt), "api-unknown"
            wait += _random.uniform(0, 0.5)
            print(f"[{kind}] {label}: {type(e).__name__}; retrying in {wait:.1f}s "
                  f"(attempt {attempt + 1}/{max_attempts})", file=sys.stderr)
            time.sleep(wait)
    raise last  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Multi-turn tool calling                                                     #
# --------------------------------------------------------------------------- #
# The Backend classes above expose ``generate_batch(prompts) -> completions``:
# one shot, no conversation. A tool-using agent needs the other shape -- a turn
# at a time, with ``tools`` on the way out and ``role: "tool"`` messages coming
# back. That is what ToolChat adds. It lives here rather than in the agentic
# package because it is provider-level machinery, not agent logic: the loop that
# decides WHICH tools to call belongs to the caller, the mechanics of calling
# them belong with the other clients.
#
# Both shapes share this module's client construction, retry ladder and
# reasoning extraction, so there is one place where "talking to a model" is
# implemented.
#
# Forced submission (probed 2026-08-22, see agentic/DESIGN.md):
#   named tool_choice={"function": ...}   works on hosted, HTTP 500 on vLLM
#   tool_choice="required"                fails on both
#   lone tool + tool_choice="auto"        works on both   <- what force_submit uses

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    args_raw: str
    args: dict | None      # None when args_raw is not valid JSON
    error: str | None = None


@dataclass
class ChatTurn:
    text: str = ""
    reasoning: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    latency_s: float = 0.0
    raw_message: Any = None


def _cost(model: str, u: "Usage") -> float | None:
    """No price map, so no dollar figure. Always None.

    This project used to reach for litellm's offline price table here. It was
    dropped: it returned None for both models actually being run -- one is a
    free tier, the other is self-hosted -- while adding a dependency, an import
    cost, and a provider-list warning on every turn.

    Nothing is lost. The fairness control is iso-TOKEN, and token counts come
    from the API's own `usage` object, recorded per step regardless. A run
    against a self-hosted server has no dollar cost to report anyway, and None
    says that honestly where 0.0 would not.

    If a hosted model is ever run and dollars are wanted, put a small explicit
    price dict here: six lines, auditable, and it cannot shift between runs the
    way a shipped price map can.
    """
    return None


class ToolChat:
    """One model endpoint, driven with tools.

    ``max_attempts`` mirrors ``llm_backends``' ladder: rate limits, timeouts,
    connection drops and SDK body-parse failures are all transient. The retry
    wraps a SINGLE call, never the agent loop -- retrying the loop would repeat
    tool side effects and corrupt step accounting.
    """

    def __init__(self, model: str, *, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 request_timeout: float = 180.0, max_attempts: int = 8,
                 extra_body: dict | None = None,
                 reasoning_effort: str | None = None,
                 verbosity: str | None = None,
                 include_reasoning: bool = True):
        self.model = model
        self.temperature = temperature
        self.max_attempts = max_attempts
        # include_reasoning is set by build_chat_request; passing False here
        # opts out via extra_body, the same escape hatch the batch path has.
        self.extra_body = dict(extra_body or {})
        if not include_reasoning:
            self.extra_body["include_reasoning"] = False
        # CLI overrides. When both are None, `_sampling_kwargs` picks the
        # model-family default, exactly as OpenRouterBackend does.
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.client = make_openai_client(base_url, api_key, request_timeout)

    # -- one turn ------------------------------------------------------------
    def call(self, messages: list[dict], tools: list[dict] | None, *,
             max_tokens: int = 2048, tool_choice: str = "auto") -> ChatTurn:
        kw = build_chat_request(
            self.model, messages, max_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort, verbosity=self.verbosity,
            extra_body=self.extra_body,
            tools=tools, tool_choice=(tool_choice if tools else None))

        t0 = time.time()
        resp = retry_call(lambda: self.client.chat.completions.create(**kw),
                             max_attempts=self.max_attempts,
                             label=f"{self.model} chat")
        dt = time.time() - t0

        turn = parse_chat_response(resp, self.model, latency_s=dt)
        return turn

    def force_submit(self, messages: list[dict], submit_tool: dict, *,
                     max_tokens: int = 2048) -> ChatTurn:
        """Last turn: mount only ``submit_plan`` and let ``auto`` do the work.

        The one forcing mechanism that works on both endpoints. Deliberately
        used even where named ``tool_choice`` is available, so the two models
        run mechanically identical loops.
        """
        return self.call(messages, [submit_tool], max_tokens=max_tokens,
                         tool_choice="auto")
