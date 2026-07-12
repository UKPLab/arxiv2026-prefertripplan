"""LLM trip-plan generator for the PreferTripPlan HF dataset.

Loads records from the HuggingFace dataset ``UKPLab/PreferTripPlan``
(private today, public later; the same code path serves both — pass an
HF token via ``--hf-token`` / ``$HF_TOKEN`` for the private phase, and
omit it once the repo is public).

For every record it asks a planner LLM to produce a full trip plan.

Prompt: ``PLANNER_INSTRUCTION`` from the original TravelPlanner project's
``agents/prompts.py`` (sole-planning setup) is reused VERBATIM.  The
only change we make is a single added line in the template's tail so
the traveler's persona is surfaced alongside the query:

    ...
    Given information: {text}
    Traveler persona: {persona}     <-- one line added for PreferTripPlan
    Query: {query}
    Travel Plan:

Source of ``PLANNER_INSTRUCTION``:
    https://github.com/OSU-NLP-Group/TravelPlanner/blob/main/agents/prompts.py
This preserves TravelPlanner's plaintext output shape (Day N: Current
City / Transportation / Breakfast / Attraction / Lunch / Dinner /
Accommodation) so downstream evaluators built for TravelPlanner consume
PreferTripPlan plans without adaptation.

Three fields per HF record are consumed:
    persona                -- fluent first-person persona introduction
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
``{"idx", "persona", "query", "reference_information", "llm_travel_plan"}``.
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
# agents/prompts.py  (only the tail line is extended to carry `persona`).     #
# --------------------------------------------------------------------------- #
_PLANNER_INSTRUCTION_TP = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B). For any day with multiple attractions, the order in which they appear in the "Attraction" line reflects the order they are visited.

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
Query: {query}
Travel Plan:"""

# PreferTripPlan modification: one added line for the traveler persona,
# inserted immediately before the "Query: ..." line.  Everything above
# (the instructions + Ithaca/Charlotte example) is TravelPlanner-verbatim.
_TP_TAIL   = "Given information: {text}\nQuery: {query}\nTravel Plan:"
_PTP_TAIL  = ("Given information: {text}\n"
              "Traveler persona: {persona}\n"
              "Query: {query}\n"
              "Travel Plan:")
PLANNER_INSTRUCTION_PTP = _PLANNER_INSTRUCTION_TP.replace(_TP_TAIL, _PTP_TAIL)
assert _PTP_TAIL in PLANNER_INSTRUCTION_PTP, "prompt-tail surgery failed"


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
    """Fill our extended PLANNER_INSTRUCTION with the record's persona,
    query, and rendered reference_information.  `record` can be either
    a dict OR a HF-dataset row (both indexable by key)."""
    persona = record["persona"] or ""
    query   = record["query"]   or ""
    text    = render_reference_information(record["reference_information"])
    return PLANNER_INSTRUCTION_PTP.format(text=text, persona=persona, query=query)


# --------------------------------------------------------------------------- #
# Backends                                                                    #
# --------------------------------------------------------------------------- #
class Backend:
    name: str = "abstract"
    def generate_batch(self, prompts: list[str], *, max_tokens: int) -> list[str]:
        raise NotImplementedError


class AnthropicBackend(Backend):
    name = "anthropic"
    def __init__(self, model: str, workers: int = 8):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.workers = workers

    def _one(self, prompt: str, max_tokens: int) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content
                       if getattr(b, "type", None) == "text").strip()

    def generate_batch(self, prompts, *, max_tokens):
        results = [None] * len(prompts)
        reporter = _ProgressReporter(len(prompts),
                                     label=f"anthropic ({self.workers}w)")
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._one, p, max_tokens): i
                        for i, p in enumerate(prompts)}
                for f in as_completed(futs):
                    results[futs[f]] = f.result()
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
        self.client = OpenAI(**kwargs)
        self.model = model
        self.workers = workers

    def _one(self, prompt: str, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_completion_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def generate_batch(self, prompts, *, max_tokens):
        results = [None] * len(prompts)
        reporter = _ProgressReporter(len(prompts),
                                     label=f"openai ({self.workers}w)")
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._one, p, max_tokens): i
                        for i, p in enumerate(prompts)}
                for f in as_completed(futs):
                    results[futs[f]] = f.result()
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

    def generate_batch(self, prompts, *, max_tokens):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=TEMPERATURE, max_tokens=max_tokens)
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self.llm.chat(conversations, sampling_params=sp, use_tqdm=True)
        return [o.outputs[0].text.strip() for o in outputs]


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
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps({"idx": idx, "text": text}) + "\n")


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
    t0 = time.time()
    if verbose:
        print(f"[run] dispatching {len(prompts)} prompts (max_tokens={max_tokens})...")
    texts = backend.generate_batch(prompts, max_tokens=max_tokens)
    dt = time.time() - t0
    if verbose:
        print(f"[run] generated {len(texts)} plans in {dt:.1f}s "
              f"({len(texts)/max(dt, 1e-6):.2f} plans/sec)")

    for (idx, _), text in zip(pending, texts):
        if text:
            _append_cache(cache_path, idx, text)


def write_output_jsonl(rows: list[tuple[int, dict]], out_path: Path, *,
                       cache_path: Path) -> None:
    """Emit a JSONL with `{idx, persona, query, reference_information,
    llm_travel_plan}` for every row whose plan is now in the cache."""
    cache = _load_cache(cache_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for idx, r in rows:
            plan = cache.get(idx)
            rec = {
                "idx":                   idx,
                "persona":               r["persona"],
                "query":                 r["query"],
                "reference_information": r["reference_information"],
                "llm_travel_plan":       plan,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if plan:
                n += 1
    print(f"[out] wrote {n} records with plans (of {len(rows)} total) → {out_path}")


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

    required = {"persona", "query", "reference_information"}
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
