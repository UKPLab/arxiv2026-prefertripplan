"""LLM trip-plan generator for the PreferTripPlan HF dataset.

Loads records from the HuggingFace dataset ``UKPLab/PreferTripPlan``
(private today, public later; the same code path serves both — pass an
HF token via ``--hf-token`` / ``$HF_TOKEN`` for the private phase, and
omit it once the repo is public).

For every record it asks a planner LLM to produce a full trip plan.

Prompt: ``PLANNER_INSTRUCTION`` from the original TravelPlanner project's
``agents/prompts.py`` (sole-planning setup) is reused VERBATIM.  The
only change we make is a single added line in the template's tail so
the traveler's profile is surfaced alongside the query:

    ...
    Given information: {text}
    Traveler profile: {profile}     <-- one line added for PreferTripPlan
    Query: {query}
    Travel Plan:

Source of ``PLANNER_INSTRUCTION``:
    https://github.com/OSU-NLP-Group/TravelPlanner/blob/main/agents/prompts.py
This preserves TravelPlanner's plaintext output shape (Day N: Current
City / Transportation / Breakfast / Attraction / Lunch / Dinner /
Accommodation) so downstream evaluators built for TravelPlanner consume
PreferTripPlan plans without adaptation.

Three fields per HF record are consumed:
    profile                -- fluent first-person profile introduction
    query                  -- fluent trip-request message
    reference_information  -- JSON-serialised list of candidate-pool blocks

Supports three backends (pick via ``--backend`` or env ``LLM_PLAN_BACKEND``):
    anthropic    -- Anthropic API.       Needs ANTHROPIC_API_KEY.
    openai       -- OpenAI API.          Needs OPENAI_API_KEY (and
                    optionally OPENAI_BASE_URL for compatible endpoints).
    vllm-offline -- In-process vLLM.     Batched inference.

Sidecar cache at ``plan-generation/plan_cache_<backend>_<model>.jsonl``
keyed by dataset row index so runs are resumable across the private→
public transition.  The cache file name is model-specific (mirroring
``--out``) so multiple simultaneous runs with different backends /
models don't clobber each other's caches.

Output: JSONL at ``--out`` (default:
``plan-generation/plans_<backend>_<model>.jsonl``) with, per row,
``{"idx", "profile", "query", "reference_information", "llm_travel_plan"}``.
The input HF dataset is treated as read-only.

Usage:
  python3 plan-generation/generate_plans.py                              # anthropic default
  python3 plan-generation/generate_plans.py --backend openai --model gpt-4o-mini
  python3 plan-generation/generate_plans.py --backend vllm-offline --model meta-llama/Llama-3.1-8B-Instruct
  python3 plan-generation/generate_plans.py --sample 5                   # smoke test
  python3 plan-generation/generate_plans.py --sample-idx 0,10,17         # by dataset row index
  python3 plan-generation/generate_plans.py --hf-token hf_xxx            # private repo phase
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN_DIR = ROOT / "plan-generation"

DEFAULT_DATASET = "UKPLab/PreferTripPlan"
DEFAULT_SPLIT   = "test"

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
# PLANNER_INSTRUCTION -- verbatim from OSU-NLP-Group/TravelPlanner            #
# agents/prompts.py  (only the tail line is extended to carry `profile`).     #
# --------------------------------------------------------------------------- #
# _PLANNER_INSTRUCTION_TP = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).

# ***** Example *****
# Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 10th, 2025, with a budget of $30,200?
# Travel Plan:
# Day 1:
# Current City: from Ithaca to Charlotte
# Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
# Breakfast: Nagaland's Kitchen, Charlotte
# Attraction: The Charlotte Museum of History, Charlotte
# Lunch: Cafe Maple Street, Charlotte
# Dinner: Bombay Vada Pav, Charlotte
# Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

# Day 2:
# Current City: Charlotte
# Transportation: -
# Breakfast: Olive Tree Cafe, Charlotte
# Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
# Lunch: Birbal Ji Dhaba, Charlotte
# Dinner: Pind Balluchi, Charlotte
# Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

# Day 3:
# Current City: from Charlotte to Ithaca
# Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
# Breakfast: Subway, Charlotte
# Attraction: Books Monument, Charlotte.
# Lunch: Olive Tree Cafe, Charlotte
# Dinner: Kylin Skybar, Charlotte
# Accommodation: -

# ***** Example Ends *****

# Given information: {text}
# Query: {query}
# Travel Plan:"""

# # PreferTripPlan modification: one added line for the traveler profile,
# # inserted immediately before the "Query: ..." line.  Everything above
# # (the instructions + Ithaca/Charlotte example) is TravelPlanner-verbatim.
# _TP_TAIL   = "Given information: {text}\nQuery: {query}\nTravel Plan:"
# _PTP_TAIL  = ("Given information: {text}\n"
#               "Traveler profile: {profile}\n"
#               "Query: {query}\n"
#               "Travel Plan:")
# PLANNER_INSTRUCTION_PTP = _PLANNER_INSTRUCTION_TP.replace(_TP_TAIL, _PTP_TAIL)
# assert _PTP_TAIL in PLANNER_INSTRUCTION_PTP, "prompt-tail surgery failed"

# Adapted from TravelPlanner+ (Table 9)
PLANNER_INSTRUCTION_PTP = """You are a proficient planner with a keen understanding of personal preferences and styles. Based on the provided information, user profile, and query, please give me a detailed and personalized plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation . Note that all the information in your plan should be derived from the provided data and aligned with the profile details. You must adhere to the format given in the example. Additionally, all details should align with common sense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B). Always prioritize the query constraints first, especially when they conflict with user profiles. Incorporate personal preferences based on user profiles as secondary considerations.

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 10th, 2025, with a budget of $30,200?
Travel Plan:
Day 1:
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -

***** Example Ends *****

Given information: {text}

User profile: {profile}

Query: {query}

Return your response as a JSON object with a single key `travel_plan` whose value is the plan formatted EXACTLY as shown in the example above (the same "Day N:" / "Current City:" / "Transportation:" / "Breakfast:" / "Attraction:" / "Lunch:" / "Dinner:" / "Accommodation:" line-based layout). Use `\n` as newlines inside the string; do NOT wrap the string in markdown code fences; do NOT add any other top-level keys."""


# --------------------------------------------------------------------------- #
# reference_information renderer                                              #
# --------------------------------------------------------------------------- #
def _render_content_table(items: list) -> str:
    lines: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            lines.append(f"- {it}")
            continue
        parts = [f"{k}={v}" for k, v in it.items() if v not in (None, "")]
        lines.append("- " + "  |  ".join(parts))
    return "\n".join(lines) if lines else "(no items)"


def render_reference_information(ref) -> str:
    """Turn the reference_information list into a flat LLM-friendly text
    blob, preserving block order.  Accepts either the already-parsed
    list-of-dicts OR the JSON-string form emitted by our HF export."""
    if isinstance(ref, str):
        try:
            ref = json.loads(ref)
        except Exception:
            return ref
    if not isinstance(ref, list):
        return str(ref)
    out: list[str] = []
    for b in ref:
        if not isinstance(b, dict):
            continue
        desc = b.get("Description", "")
        content = b.get("Content")
        out.append(f"## {desc}")
        if isinstance(content, list):
            out.append(_render_content_table(content))
        elif isinstance(content, dict):
            if content.get("records") is not None:
                # Flight block.
                cnt = content.get("count", len(content.get("records", []) or []))
                stats = content.get("price_stats") or {}
                out.append(f"count: {cnt}")
                if stats:
                    out.append(f"price_stats: {stats}")
                recs = content.get("records") or []
                if recs:
                    out.append(_render_content_table(recs[:10]))
                    if len(recs) > 10:
                        out.append(f"...and {len(recs) - 10} more flight records")
            else:
                # Ground-transport / other single-dict block.
                out.append("  |  ".join(f"{k}: {v}" for k, v in content.items()))
        else:
            out.append(str(content))
        out.append("")
    return "\n".join(out).strip()


def build_plan_prompt(record) -> str:
    """Fill our extended PLANNER_INSTRUCTION with the record's profile,
    query, and rendered reference_information.  `record` can be either
    a dict OR a HF-dataset row (both indexable by key)."""
    profile = record["profile"] or ""
    query   = record["query"]   or ""
    text    = render_reference_information(record["reference_information"])
    return PLANNER_INSTRUCTION_PTP.format(text=text, profile=profile, query=query)


# --------------------------------------------------------------------------- #
# Backends                                                                    #
# --------------------------------------------------------------------------- #
class Backend:
    """Abstract base. ``generate_batch`` returns one text per prompt.

    ``on_result`` is an optional per-completion callback: as soon as a
    single prompt finishes, backends invoke it with ``(index, text)``
    so the caller can persist the result to disk immediately, which
    makes Ctrl-C interruptions non-destructive.
    """
    name: str = "abstract"
    def generate_batch(self, prompts: list[str], *, max_tokens: int,
                       on_result=None) -> list[str]:
        raise NotImplementedError


class AnthropicBackend(Backend):
    name = "anthropic"
    def __init__(self, model: str, workers: int = 8):
        import anthropic
        # Disable SDK's silent internal retries so ``_one`` handles 429s
        # visibly; per-request timeout cap keeps stuck sockets from
        # freezing a worker.
        self.client = anthropic.Anthropic(max_retries=0, timeout=120.0)
        self.model = model
        self.workers = workers

    def _one(self, prompt: str, max_tokens: int) -> str:
        """Single Anthropic call with EXPLICIT rate-limit handling.

        The default Anthropic SDK retries 429s internally with silent
        exponential backoff -- which looks like the process 'stalled'
        for tens of seconds with no output.  We surface the wait
        directly by catching RateLimitError, parsing the ``retry-after``
        header when available, logging to stderr so the user knows why
        the progress bar isn't moving, and retrying.
        """
        import re as _re, time as _time, sys as _sys, random as _random
        from anthropic import RateLimitError, APITimeoutError, APIConnectionError

        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content
                               if getattr(b, "type", None) == "text").strip()
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
        return ""

    def generate_batch(self, prompts, *, max_tokens, on_result=None):
        results = [None] * len(prompts)
        reporter = _ProgressReporter(len(prompts),
                                     label=f"anthropic ({self.workers}w)")
        import sys as _sys
        _first_error_logged = False
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._one, p, max_tokens): i
                        for i, p in enumerate(prompts)}
                for f in as_completed(futs):
                    idx = futs[f]
                    try:
                        text = f.result()
                    except Exception as e:
                        if not _first_error_logged:
                            print(f"\n[error] worker {idx} raised {type(e).__name__}: {e}",
                                  file=_sys.stderr, flush=True)
                            _first_error_logged = True
                        text = ""
                    results[idx] = text
                    if on_result is not None and text:
                        try:
                            on_result(idx, text)
                        except Exception:
                            pass
                    reporter.step()
        finally:
            reporter.close()
        return results


class OpenAIBackend(Backend):
    name = "openai"
    def __init__(self, model: str, workers: int = 8,
                 base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI
        kwargs: dict = {}
        if base_url is not None: kwargs["base_url"] = base_url
        if api_key  is not None: kwargs["api_key"]  = api_key
        # Disable the SDK's silent internal retry so ``_one`` handles
        # 429s visibly.  Cap per-request timeout so stuck connections
        # can't freeze a worker forever.
        kwargs["max_retries"] = 0
        kwargs["timeout"] = 120.0
        self.client = OpenAI(**kwargs)
        self.model = model
        self.workers = workers

    def _one(self, prompt: str, max_tokens: int) -> str:
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
                    reasoning_effort="none", #"medium",
                    verbosity="medium",
                    # response_format={"type": "json_object"},
                )
                choice = resp.choices[0]
                content = (choice.message.content or "").strip()
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
                return content
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
        print(f"\n[error] OpenAI call failed after {max_attempts} attempts; skipping.",
              file=_sys.stderr, flush=True)
        return ""

    def generate_batch(self, prompts, *, max_tokens, on_result=None):
        results = [None] * len(prompts)
        reporter = _ProgressReporter(len(prompts),
                                     label=f"openai ({self.workers}w)")
        import sys as _sys
        _first_error_logged = False
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._one, p, max_tokens): i
                        for i, p in enumerate(prompts)}
                for f in as_completed(futs):
                    idx = futs[f]
                    try:
                        text = f.result()
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
                        text = ""
                    results[idx] = text
                    if on_result is not None and text:
                        try:
                            on_result(idx, text)
                        except Exception:
                            pass
                    reporter.step()
        finally:
            reporter.close()
        return results


class VLLMOfflineBackend(Backend):
    name = "vllm-offline"
    def __init__(self, model: str, *,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: int | None = None,
                 dtype: str = "auto",
                 tensor_parallel_size: int = 1,
                 trust_remote_code: bool = True,
                 **extra):
        from vllm import LLM
        self.LLM_class = LLM
        self.model = model
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
        sp = SamplingParams(temperature=TEMPERATURE, max_tokens=max_tokens)
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self.llm.chat(conversations, sampling_params=sp, use_tqdm=True)
        texts = [o.outputs[0].text.strip() for o in outputs]
        # vLLM's internal batching is opaque; the best we can do here
        # is fire on_result after the whole batch completes so every
        # completed plan lands in the cache before the run's end (and
        # merge / write_output_jsonl) is invoked.
        if on_result is not None:
            for i, t in enumerate(texts):
                if t:
                    try:
                        on_result(i, t)
                    except Exception:
                        pass
        return texts


def make_backend(name: str, model: str, *,
                 workers: int = 8,
                 openai_base_url: str | None = None,
                 openai_api_key: str | None = None,
                 vllm_gpu_memory: float = 0.9,
                 vllm_max_model_len: int | None = None,
                 vllm_dtype: str = "auto",
                 vllm_tensor_parallel: int = 1) -> Backend:
    if name == "anthropic":
        return AnthropicBackend(model=model, workers=workers)
    if name == "openai":
        return OpenAIBackend(model=model, workers=workers,
                             base_url=openai_base_url, api_key=openai_api_key)
    if name == "vllm-offline":
        return VLLMOfflineBackend(
            model=model,
            gpu_memory_utilization=vllm_gpu_memory,
            max_model_len=vllm_max_model_len,
            dtype=vllm_dtype,
            tensor_parallel_size=vllm_tensor_parallel,
        )
    raise SystemExit(f"unknown backend: {name}")


# --------------------------------------------------------------------------- #
# HF dataset loader                                                           #
# --------------------------------------------------------------------------- #
def load_hf_dataset(dataset: str, split: str, config: str | None,
                    hf_token: str | None):
    """Load a HuggingFace dataset with token-aware auth.  Same call
    works for private repos (token required) and public repos (token
    is either an empty string or None).  Newer `datasets` releases
    also honour ``$HF_TOKEN`` and ``$HUGGING_FACE_HUB_TOKEN`` even
    when we don't pass one explicitly."""
    from datasets import load_dataset
    kwargs: dict = {"split": split}
    if config:
        kwargs["name"] = config
    if hf_token:
        # `token=` is the newer kwarg; older `datasets` used `use_auth_token=`.
        # `load_dataset` accepts both on any recent-ish version; prefer `token`.
        kwargs["token"] = hf_token
    return load_dataset(dataset, **kwargs)


# --------------------------------------------------------------------------- #
# Cache + orchestration                                                       #
# --------------------------------------------------------------------------- #
def _load_cache(cache_path: Path) -> dict[int, str]:
    cache: dict[int, str] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                cache[int(e["idx"])] = e["text"]
            except Exception:
                continue
    return cache


def _append_cache(cache_path: Path, idx: int, text: str) -> None:
    # Append + flush + fsync so a Ctrl-C or a killed request leaves every
    # already-completed plan safely on disk instead of stuck in Python's
    # default buffer.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps({"idx": idx, "text": text}) + "\n")
        f.flush()
        try:
            import os as _os
            _os.fsync(f.fileno())
        except OSError:
            pass


def _parse_index_list(spec: str) -> set[int]:
    """Parse a comma / newline list of row indices, or a path to a file
    containing the same."""
    p = Path(spec)
    text = p.read_text() if p.exists() else spec
    out: set[int] = set()
    for tok in re.split(r"[,\s]+", text):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            continue
    return out


def _sanitize_model(model: str) -> str:
    return re.sub(r"[^\w.-]+", "_", model).strip("_")


def _default_out_path(backend_name: str, model: str) -> Path:
    return PLAN_DIR / f"plans_{backend_name}_{_sanitize_model(model)}.jsonl"


def _default_cache_path(backend_name: str, model: str) -> Path:
    """Model-specific cache path so parallel runs against different
    backends / models don't collide on the same file."""
    return PLAN_DIR / f"plan_cache_{backend_name}_{_sanitize_model(model)}.jsonl"


def generate_all(rows: list[tuple[int, dict]], backend: Backend, *,
                 cache_path: Path, max_tokens: int = 4096,
                 verbose: bool = True) -> None:
    cache = _load_cache(cache_path)
    if verbose:
        print(f"[cache] file: {cache_path}")
        print(f"[cache] plans cached: {len(cache)}")

    pending: list[tuple[int, str]] = []
    for idx, r in rows:
        if idx in cache:
            continue
        pending.append((idx, build_plan_prompt(r)))

    if verbose:
        print(f"[run] backend={backend.name}  pending={len(pending)}  "
              f"already_cached={len(rows) - len(pending)}")
    if not pending:
        return

    prompts = [p for (_, p) in pending]

    # Serialize cache writes across worker threads so appends can't
    # interleave partial JSON lines on disk.
    import threading
    _cache_lock = threading.Lock()

    def _persist(local_index: int, text: str) -> None:
        """Called from the backend as soon as prompt ``local_index``
        finishes.  Look up the pending row's real row-index and append
        immediately -- so any Ctrl-C partway through leaves every
        completed plan on disk instead of only after the batch ends.
        """
        row_idx, _ = pending[local_index]
        with _cache_lock:
            _append_cache(cache_path, row_idx, text)

    t0 = time.time()
    if verbose:
        print(f"[run] dispatching {len(prompts)} prompts (max_tokens={max_tokens})...")
    try:
        texts = backend.generate_batch(prompts, max_tokens=max_tokens,
                                       on_result=_persist)
    except KeyboardInterrupt:
        import sys as _sys
        print(f"\n[run] interrupted; cache holds every plan completed so far",
              file=_sys.stderr, flush=True)
        raise
    dt = time.time() - t0
    if verbose:
        print(f"[run] generated {len(texts)} plans in {dt:.1f}s "
              f"({len(texts)/max(dt, 1e-6):.2f} plans/sec)")


def write_output_jsonl(rows: list[tuple[int, dict]], out_path: Path, *,
                       cache_path: Path) -> None:
    """Emit a JSONL with `{idx, profile, query, reference_information,
    llm_travel_plan}` for every row whose plan is now in the cache.

    Under json_object response mode (default for the OpenAI backend), the
    cached text is a JSON envelope like ``{"travel_plan": "..."}``; here
    we extract the inner string so ``llm_travel_plan`` stays the same
    line-based text the downstream TravelPlanner evaluator expects.
    Non-JSON cached plans (e.g. Anthropic / vLLM backends that don't
    use response_format) pass through unchanged."""
    cache = _load_cache(cache_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for idx, r in rows:
            plan = _extract_plan_text(cache.get(idx))
            rec = {
                "idx":                   idx,
                "profile":               r["profile"],
                "query":                 r["query"],
                "reference_information": r["reference_information"],
                "llm_travel_plan":       plan,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if plan:
                n += 1
    print(f"[out] wrote {n} records with plans (of {len(rows)} total) → {out_path}")


def _extract_plan_text(raw: str | None) -> str | None:
    """Strip a ``{"travel_plan": "..."}`` JSON envelope if present.
    Returns the original string when the raw text isn't a JSON object
    with a ``travel_plan`` key (Anthropic / vLLM backends emit plain
    text; older cached entries pre-json_object also emit plain text).
    """
    if not raw:
        return raw
    stripped = raw.strip()
    if not stripped.startswith("{"):
        return raw
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return raw
    if isinstance(obj, dict) and "travel_plan" in obj:
        v = obj["travel_plan"]
        return v if isinstance(v, str) else raw
    return raw


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--backend", choices=["anthropic", "openai", "vllm-offline"],
                    default=os.environ.get("LLM_PLAN_BACKEND", "anthropic"))
    ap.add_argument("--model", default=None,
                    help="Model name.  Defaults: anthropic→claude-haiku-4-5-20251001, "
                         "openai→gpt-5.4-mini, vllm-offline→meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--workers", type=int, default=4)
    # `max_completion_tokens` for reasoning models includes hidden
    # chain-of-thought tokens.  A 3-7 day plan needs ~500-1500 output
    # tokens; `reasoning_effort='medium'` can add 2000-6000 reasoning
    # tokens.  Default 16384 leaves comfortable headroom.  Drop this
    # (e.g. 8192) if you switch to `reasoning_effort='low'` / 'minimal'
    # in the backend for cost / speed.
    ap.add_argument("--max-tokens", type=int, default=4096)

    # HF dataset selection.
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="HuggingFace dataset repo (default: UKPLab/PreferTripPlan).")
    ap.add_argument("--split", default=DEFAULT_SPLIT,
                    help="Dataset split (default: test).")
    ap.add_argument("--config", default=None,
                    help="Dataset config name (default: unset -> uses the default config).")
    ap.add_argument("--hf-token",
                    default=os.environ.get("HF_TOKEN")
                            or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                    help="HF token for private repos.  Env fallbacks: "
                         "HF_TOKEN, HUGGING_FACE_HUB_TOKEN.  Omit for public repos.")

    # Row selection.
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, only generate for the first N rows (smoke test).")
    ap.add_argument("--sample-idx", type=str, default=None,
                    help="Comma-list of dataset row indices, or path to a file "
                         "with one index per line.")

    # Output + cache.
    ap.add_argument("--out", default=None,
                    help="Output JSONL.  Default: plan-generation/"
                         "plans_<backend>_<model-sanitized>.jsonl.")
    ap.add_argument("--cache", default=None,
                    help="Sidecar cache JSONL (append-only, keyed by dataset "
                         "row index).  Default: plan-generation/"
                         "plan_cache_<backend>_<model-sanitized>.jsonl.  "
                         "Different backends / models get separate caches "
                         "so parallel runs never collide.")

    # OpenAI-compat + vLLM extras.
    ap.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--openai-api-key",  default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--vllm-gpu-memory", type=float, default=0.9)
    ap.add_argument("--vllm-max-model-len", type=int, default=None)
    ap.add_argument("--vllm-dtype", default="auto")
    ap.add_argument("--vllm-tensor-parallel", type=int, default=1)

    args = ap.parse_args()

    default_model = {
        "anthropic":    "claude-haiku-4-5-20251001",
        "openai":       "gpt-5.4-mini",
        "vllm-offline": "meta-llama/Llama-3.1-8B-Instruct",
    }
    model = args.model or default_model[args.backend]
    out_path   = Path(args.out)   if args.out   else _default_out_path(args.backend, model)
    cache_path = Path(args.cache) if args.cache else _default_cache_path(args.backend, model)

    print(f"[cfg] backend={args.backend}  model={model}  workers={args.workers}")
    print(f"[cfg] dataset={args.dataset}  split={args.split}"
          + (f"  config={args.config}" if args.config else "")
          + f"  private-token={'yes' if args.hf_token else 'no'}")
    print(f"[cfg] cache={cache_path}")
    print(f"[cfg] out=  {out_path}")

    # Load HF dataset.
    ds = load_hf_dataset(args.dataset, args.split, args.config, args.hf_token)
    n_total = len(ds)
    print(f"[in]  {n_total} rows loaded from HF: {args.dataset}:{args.split}")
    print(f"[in]  columns: {ds.column_names}")

    required = {"profile", "query", "reference_information"}
    missing = required - set(ds.column_names)
    if missing:
        sys.exit(f"dataset is missing required columns: {sorted(missing)}")

    # Build (idx, row) list.
    rows: list[tuple[int, dict]] = [(i, ds[i]) for i in range(n_total)]

    # Row filter.
    if args.sample_idx:
        target = _parse_index_list(args.sample_idx)
        rows = [(i, r) for (i, r) in rows if i in target]
        print(f"[in]  --sample-idx filter: {len(rows)} rows match")
    elif args.sample > 0:
        rows = rows[: args.sample]
        print(f"[in]  --sample={args.sample}: {len(rows)} rows")

    # Backend + generate.
    backend = make_backend(
        args.backend, model,
        workers=args.workers,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        vllm_gpu_memory=args.vllm_gpu_memory,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_dtype=args.vllm_dtype,
        vllm_tensor_parallel=args.vllm_tensor_parallel,
    )
    generate_all(rows, backend, cache_path=cache_path,
                 max_tokens=args.max_tokens)

    # Write output JSONL.
    write_output_jsonl(rows, out_path, cache_path=cache_path)


if __name__ == "__main__":
    main()
