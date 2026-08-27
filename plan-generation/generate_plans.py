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
keyed by the HF dataset's ``id`` (1-indexed contiguous integer, present
on every row of both ``test`` and ``test_large``).  Each cache entry
also stores ``source_id`` -- an integer on ``test`` (pointer into
``test_large``) and ``null`` on ``test_large`` -- so a cache from one
split can be promoted to the other without any re-generation.  Cache
file names are model-specific (mirroring ``--out``) so simultaneous
runs against different backends / models never collide.

Output: JSONL at ``--out`` (default:
``plan-generation/plans_<backend>_<model>.jsonl``) with, per row,
``{"id", "source_id", "profile", "query", "reference_information",
"llm_travel_plan", "llm_reasoning"}``.  ``llm_reasoning`` carries the
model's reasoning trace when the backend surfaces one (OpenAI Responses
API reasoning items, OpenRouter's ``message.reasoning``, vLLM's
``reasoning_content`` or extracted ``<think>...</think>`` blocks,
Anthropic's ``thinking`` content blocks); ``None`` otherwise.

The output file MIRRORS THE CACHE (cumulative across partial runs):
it emits exactly the rows whose plans are in the cache, sorted by
dataset ``id`` -- so ``--sample-id 1,10,17`` produces a 3-line file
whose ``id`` values are 1/10/17, and a subsequent run APPENDS new
plans without clobbering earlier ones.  Rows never dispatched are
simply absent from the file; there is no ``None``-padding.
The input HF dataset is treated as read-only.

Cross-split cache promotion (``--seed-cache``):
  Passing ``--seed-cache PATH`` before generation merges a cache from
  the OTHER split into the primary cache, avoiding re-generation of
  overlapping rows.  Direction is inferred from the target ``--split``:
    * target=test_large, seed=test cache   -> each seed entry with
        source_id=N is written as a test_large cache entry keyed by
        id=N (source_id=null).
    * target=test, seed=test_large cache   -> for each test row whose
        source_id equals a seed entry's id, that seed entry is written
        as a test cache entry keyed by the matching test id (with
        source_id set to the seed id).
  Primary-cache entries win on any id collision.  The seed file is
  unchanged; promoted entries are appended to ``--cache``.

Legacy files: existing plan_cache_* / plans_* files that were keyed
by 0-indexed ``idx`` predate this schema.  Convert them once with the
one-off script ``plan-generation/migrate_legacy_cache.py`` (uses the
HF split to map every legacy ``idx`` to its ``id`` / ``source_id``).

Usage:
  python3 plan-generation/generate_plans.py                              # anthropic default
  python3 plan-generation/generate_plans.py --backend openai --model gpt-4o-mini
  python3 plan-generation/generate_plans.py --backend vllm-offline --model meta-llama/Llama-3.1-8B-Instruct
  python3 plan-generation/generate_plans.py --sample 5                   # smoke test
  python3 plan-generation/generate_plans.py --sample-id 1,10,17          # by dataset id (1-indexed)
  python3 plan-generation/generate_plans.py --hf-token hf_xxx            # private repo phase
  python3 plan-generation/generate_plans.py --split test_large \\
      --seed-cache plan_cache_.../test.jsonl                             # promote test cache -> test_large
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
# Backends, progress reporting, and sampling defaults now live in            #
# ``llm_backends`` so the agentic track can drive models through the same    #
# clients and the same retry ladder. Imported back under their original      #
# names: every reference below this point is unchanged.                      #
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_backends import (            # noqa: E402
    TEMPERATURE,
    _ProgressReporter,
    _completion,
    _split_think,
    _extract_reasoning_from_openai_message,
    _sampling_kwargs,
    _OPENAI_ENDPOINTS,
    Backend,
    AnthropicBackend,
    OpenAIBackend,
    OpenRouterBackend,
    VLLMOfflineBackend,
    make_backend,
)



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
PLANNER_INSTRUCTION_PTP = """You are a proficient planner with a keen understanding of personal preferences and styles. Based on the provided information, user profile, and query, please give me a detailed and personalized plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data and aligned with the profile details. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B). Always prioritize the query constraints first, especially when they conflict with user profiles. Incorporate personal preferences based on user profiles as secondary considerations.

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 10th, 2025, with a budget of $30,200? We would ideally like to visit a nature and parks attraction at least once during the trip. 
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
def _load_cache(cache_path: Path) -> dict[int, dict[str, Any]]:
    """Read the per-model plan cache, keyed by the HF dataset ``id``
    (1-indexed).  Each entry value is
    ``{content, reasoning, source_id}``.  ``source_id`` is an integer
    on ``test``-split caches (pointer into ``test_large``) and ``None``
    on ``test_large``-split caches.

    Legacy files keyed by ``idx`` (0-indexed positional) are REJECTED
    with a clear error pointing to the one-off migration script.
    """
    cache: dict[int, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    saw_legacy = False
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if "id" not in e and "idx" in e:
                saw_legacy = True
                continue
            try:
                id_ = int(e["id"])
            except (KeyError, TypeError, ValueError):
                continue
            content   = e.get("content")
            if content is None:
                content = e.get("text", "")
            reasoning = e.get("reasoning")
            sid = e.get("source_id")
            cache[id_] = {"content": content, "reasoning": reasoning,
                          "source_id": sid}
    if saw_legacy:
        raise SystemExit(
            f"[cache] {cache_path} contains legacy 'idx'-keyed entries.\n"
            f"        Run `python3 plan-generation/migrate_legacy_cache.py "
            f"{cache_path}` first (uses the HF split to map every legacy "
            f"idx to its id / source_id)."
        )
    return cache


def _append_cache(cache_path: Path, id_: int, source_id: int | None,
                  content: str, reasoning: str | None = None) -> None:
    """Append one completion to the cache with fsync.

    Entry shape: ``{"id": <int>, "source_id": <int|null>,
    "content": <str>, "reasoning": <str|null>}`` -- ``id`` is the
    HF dataset ``id`` (1-indexed); ``source_id`` is the ``test`` row's
    pointer into ``test_large`` (``null`` on ``test_large`` caches).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps({
            "id":        id_,
            "source_id": source_id,
            "content":   content,
            "reasoning": reasoning,
        }) + "\n")
        f.flush()
        try:
            import os as _os
            _os.fsync(f.fileno())
        except OSError:
            pass


def promote_seed_into_cache(seed_path: Path, cache_path: Path,
                            target_rows: list[tuple[int, int | None, dict]],
                            *, verbose: bool = True) -> int:
    """Merge a seed cache from the OTHER split into ``cache_path``.

    Direction is inferred from ``target_rows`` (i.e. from ``--split``):

      * If the target split carries ``source_id`` (values are integers)
        -- this is ``test`` -- and the seed was produced against
        ``test_large`` (its entries have ``source_id`` null), then for
        each target row whose ``source_id`` matches a seed entry's
        ``id``, the seed content is appended to ``cache_path`` under
        the target row's ``id`` (with ``source_id`` set to the seed
        entry's id, i.e. the ``test_large`` id).

      * If the target split has ``source_id`` null on every row -- this
        is ``test_large`` -- and the seed was produced against ``test``
        (its entries carry a non-null ``source_id``), then each seed
        entry's ``source_id`` becomes the new ``id`` in the appended
        entry (with ``source_id`` set to null).  Seed entries with a
        ``source_id`` that isn't a valid ``id`` in the target split
        are skipped.

    Primary-cache entries win on any id collision -- the seed is only
    used to fill gaps.  The seed file is left untouched; promoted
    entries are appended to ``cache_path`` and become part of the
    normal cache going forward.
    """
    if not seed_path.exists():
        raise SystemExit(f"[seed-cache] {seed_path} does not exist")
    seed = _load_cache(seed_path)
    if verbose:
        print(f"[seed-cache] seed:    {seed_path}  ({len(seed)} entries)")
        print(f"[seed-cache] target:  {cache_path}")

    primary = _load_cache(cache_path)

    target_source_ids = {sid for (_i, sid, _r) in target_rows}
    target_source_ids.discard(None)
    target_expects_sid = bool(target_source_ids)

    seed_has_sid = any(v.get("source_id") is not None for v in seed.values())

    if target_expects_sid and seed_has_sid:
        raise SystemExit(
            "[seed-cache] both target split and seed carry non-null "
            "source_id; nothing to promote.  Point --seed-cache at a "
            "cache from the OTHER split."
        )
    if not target_expects_sid and not seed_has_sid:
        raise SystemExit(
            "[seed-cache] both target split and seed have null "
            "source_id; nothing to promote.  Point --seed-cache at a "
            "cache from the OTHER split."
        )

    n_promoted = n_skipped = 0
    if target_expects_sid:
        # target = test, seed = test_large.  For each target row whose
        # source_id equals a seed entry's id, promote that seed entry
        # to the target id (test id).
        seed_ids = set(seed.keys())
        for tid, tsid, _r in target_rows:
            if tid in primary:
                continue
            if tsid is None or tsid not in seed_ids:
                continue
            entry = seed[tsid]
            _append_cache(cache_path, tid, tsid,
                          entry.get("content", ""),
                          reasoning=entry.get("reasoning"))
            n_promoted += 1
    else:
        # target = test_large, seed = test.  For each seed entry, its
        # source_id (test_large id) becomes the new id.
        valid_target_ids = {i for (i, _s, _r) in target_rows}
        for seed_id, entry in seed.items():
            new_id = entry.get("source_id")
            if new_id is None:
                n_skipped += 1
                continue
            if new_id in primary:
                continue
            if new_id not in valid_target_ids:
                n_skipped += 1
                continue
            _append_cache(cache_path, new_id, None,
                          entry.get("content", ""),
                          reasoning=entry.get("reasoning"))
            n_promoted += 1
    if verbose:
        extra = f", skipped={n_skipped}" if n_skipped else ""
        print(f"[seed-cache] promoted {n_promoted} entries into {cache_path}"
              f"{extra}")
    return n_promoted


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


def generate_all(rows: list[tuple[int, int | None, dict]], backend: Backend, *,
                 cache_path: Path, max_tokens: int = 4096,
                 verbose: bool = True) -> None:
    cache = _load_cache(cache_path)
    if verbose:
        print(f"[cache] file: {cache_path}")
        print(f"[cache] plans cached: {len(cache)}")

    # Each pending entry carries (id, source_id, prompt) so ``_persist``
    # can write both id and source_id back into the cache line.
    pending: list[tuple[int, int | None, str]] = []
    for id_, sid, r in rows:
        if id_ in cache:
            continue
        pending.append((id_, sid, build_plan_prompt(r)))

    if verbose:
        print(f"[run] backend={backend.name}  pending={len(pending)}  "
              f"already_cached={len(rows) - len(pending)}")
    if not pending:
        return

    prompts = [p for (_, _, p) in pending]

    # Serialize cache writes across worker threads so appends can't
    # interleave partial JSON lines on disk.
    import threading
    _cache_lock = threading.Lock()

    def _persist(local_index: int, comp: dict[str, Any]) -> None:
        """Called from the backend as soon as prompt ``local_index``
        finishes.  Look up the pending row's (id, source_id) and append
        immediately -- so any Ctrl-C partway through leaves every
        completed plan (and its reasoning trace, when the model
        exposed one) on disk instead of only after the batch ends.
        """
        row_id, row_sid, _ = pending[local_index]
        with _cache_lock:
            _append_cache(cache_path, row_id, row_sid,
                           comp.get("content", ""),
                           reasoning=comp.get("reasoning"))

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


def write_output_jsonl(rows: list[tuple[int, int | None, dict]], out_path: Path, *,
                       cache_path: Path) -> None:
    """Emit a JSONL with ``{id, source_id, profile, query,
    reference_information, llm_travel_plan, llm_reasoning}`` for every
    row whose plan is present in the cache.

    The file mirrors the cache: rows without a cached plan are NOT
    emitted (they would show up as ``llm_travel_plan=None`` padding and
    make id-based evaluation confusing).  ``id`` is the HF dataset id
    (1-indexed); ``source_id`` is an integer on ``test``-split rows
    (pointer into ``test_large``) and ``null`` on ``test_large`` rows,
    so evaluators can match records back to either split directly.

    Emission order is ascending ``id`` -- independent of dispatch
    order, so re-runs are byte-stable modulo new cache entries.

    Under json_object response mode (default for the OpenAI backend), the
    cached content is a JSON envelope like ``{"travel_plan": "..."}``;
    here we extract the inner string so ``llm_travel_plan`` stays the
    same line-based text the downstream TravelPlanner evaluator expects.
    Non-JSON cached plans (e.g. Anthropic / vLLM backends that don't
    use response_format) pass through unchanged.

    ``llm_reasoning`` carries the model's reasoning trace when the
    backend surfaced one (OpenRouter's ``message.reasoning``, vLLM's
    ``reasoning_content`` or extracted ``<think>...</think>`` blocks,
    Anthropic's ``thinking`` content blocks) -- ``None`` otherwise."""
    cache = _load_cache(cache_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ascending id keeps the file stable across partial re-runs.
    cached_rows = sorted(
        ((id_, sid, r) for (id_, sid, r) in rows if id_ in cache),
        key=lambda t: t[0],
    )
    n = 0
    with out_path.open("w") as f:
        for id_, sid, r in cached_rows:
            entry     = cache[id_]
            plan      = _extract_plan_text(entry.get("content"))
            reasoning = entry.get("reasoning")
            rec = {
                "id":                    id_,
                "source_id":             sid,
                "profile":               r["profile"],
                "query":                 r["query"],
                "reference_information": r["reference_information"],
                "llm_travel_plan":       plan,
                "llm_reasoning":         reasoning,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if plan:
                n += 1
    total = len(cached_rows)
    print(f"[out] wrote {total} records ({n} with plans, {total - n} cached-but-empty) → {out_path}")


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
    ap.add_argument("--backend",
                    choices=["anthropic", "openai", "openrouter", "vllm-offline"],
                    default=os.environ.get("LLM_PLAN_BACKEND", "anthropic"))
    ap.add_argument("--model", default=None,
                    help="Model name.  Defaults: "
                         "anthropic→claude-haiku-4-5-20251001, "
                         "openai→gpt-5.4-mini, "
                         "openrouter→openai/gpt-5.4-mini "
                         "(use OpenRouter's provider/model form, e.g. "
                         "anthropic/claude-3.5-sonnet, meta-llama/llama-3.3-70b-instruct), "
                         "vllm-offline→meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--workers", type=int, default=4)
    # `max_completion_tokens` for reasoning models includes hidden
    # chain-of-thought tokens.  A 3-7 day plan needs ~500-1500 output
    # tokens; `reasoning_effort='medium'` can add 2000-6000 reasoning
    # tokens.  Default 16384 leaves comfortable headroom.  Drop this
    # (e.g. 8192) if you switch to `reasoning_effort='low'` / 'minimal'
    # in the backend for cost / speed.
    ap.add_argument("--max-tokens", type=int, default=4096)
    # Sampling temperature.  Default None means "use the module-level
    # TEMPERATURE constant" (0.2) -- the value every result in the
    # paper was generated at -- so omitting the flag reproduces prior
    # runs exactly.  Applied by anthropic (non-thinking mode only),
    # openrouter (via _sampling_kwargs, skipped for reasoning models
    # which mandate the default) and vllm-offline.  On the `openai`
    # backend the field is omitted unless you pass this flag, because
    # the GPT-5 family rejects any non-default temperature; set it only
    # when pointing that backend at an OpenAI-compatible local server.
    ap.add_argument("--temperature", type=float, default=None,
                    help=f"Sampling temperature (default: {TEMPERATURE}).  "
                         "Ignored for reasoning models, which require "
                         "the provider default.")
    # Per-request wall-clock cap handed to the SDK client.  Both the
    # anthropic and openai/openrouter clients run with max_retries=0,
    # so a timeout means the whole generation is thrown away and redone
    # -- expensive on a local server where a long trace plus a 7-day
    # plan can legitimately exceed 120s under a deep request queue.
    ap.add_argument("--request-timeout", type=float, default=120.0,
                    help="Per-request timeout in seconds for the "
                         "anthropic / openai / openrouter HTTP clients "
                         "(default: 120).  Raise it when serving a "
                         "large model locally.  No effect on "
                         "vllm-offline, which makes no HTTP calls.")

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
    ap.add_argument("--sample-id", "--sample-idx", dest="sample_id",
                    type=str, default=None,
                    help="Comma-list of dataset ids (1-indexed), or path to a "
                         "file with one id per line.  --sample-idx is accepted "
                         "as a legacy alias and refers to the same 1-indexed "
                         "dataset id field.")

    # Output + cache.
    ap.add_argument("--out", default=None,
                    help="Output JSONL.  Default: plan-generation/"
                         "plans_<backend>_<model-sanitized>.jsonl.")
    ap.add_argument("--cache", default=None,
                    help="Sidecar cache JSONL (append-only, keyed by dataset "
                         "id).  Default: plan-generation/"
                         "plan_cache_<backend>_<model-sanitized>.jsonl.  "
                         "Different backends / models get separate caches "
                         "so parallel runs never collide.")
    ap.add_argument("--seed-cache", default=None,
                    help="Optional path to a cache from the OTHER split.  "
                         "Promoted entries are appended to --cache before "
                         "generation begins (direction inferred from --split): "
                         "test <-> test_large.  Skips re-generation of "
                         "overlapping rows.")

    # OpenAI-compat + OpenRouter + vLLM extras.
    ap.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--openai-api-key",  default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--openrouter-base-url",
                    default=os.environ.get("OPENROUTER_BASE_URL"),
                    help="OpenRouter API base URL (default: "
                         "https://openrouter.ai/api/v1).")
    ap.add_argument("--openrouter-api-key",
                    default=os.environ.get("OPENROUTER_API_KEY"),
                    help="OpenRouter API key (defaults to $OPENROUTER_API_KEY).")
    # Free-form request-body passthrough for OpenAI-compatible servers
    # whose schema is a SUPERSET of OpenAI's.  The motivating case is a
    # self-hosted vLLM online server, which accepts `chat_template_kwargs`
    # (the only way to toggle Qwen3-style thinking per request) plus
    # sampling knobs the OpenAI SDK has no named parameter for -- top_k,
    # min_p, repetition_penalty, skip_special_tokens.  Ignored-with-a-
    # warning by servers that don't declare the keys; NOT safe to point
    # at api.openai.com, which rejects unknown body fields outright.
    ap.add_argument("--extra-body", default=os.environ.get("LLM_PLAN_EXTRA_BODY"),
                    help="JSON object merged into the request body on "
                         "every call (openai / openrouter backends).  "
                         "For a vLLM server, e.g. "
                         "'{\"chat_template_kwargs\": {\"enable_thinking\": true}}' "
                         "or '{\"top_k\": 20, \"min_p\": 0.0}'.  Keys "
                         "override the backend's own defaults, so "
                         "'{\"include_reasoning\": false}' turns the "
                         "reasoning trace off.")
    # Reasoning-model knobs.  Applied by the OPENAI and OPENROUTER
    # backends.  Useful when routing e.g. `openai/gpt-5.4-mini` at
    # reduced effort for cheaper runs, or forcing `high` on a hard
    # subset.  Ignored by anthropic / vllm-offline backends.
    #
    # OpenAI backend: unset -> constructor default (see OpenAIBackend
    # ``__init__``); set -> passed directly to the API on every call.
    # OpenRouter backend: unset -> per-model default from
    # ``_sampling_kwargs`` (medium for gpt-5 / o-series, unset for
    # non-reasoning models); set -> added / overridden on every call.
    ap.add_argument("--reasoning-effort",
                    choices=["none", "minimal", "low", "medium",
                             "high", "xhigh"],
                    default=None,
                    help="Reasoning effort knob for the openai and "
                         "openrouter backends.  Default None uses the "
                         "backend's own default (see OpenAIBackend "
                         "constructor / _sampling_kwargs).")
    ap.add_argument("--verbosity",
                    choices=["low", "medium", "high"],
                    default=None,
                    help="Verbosity knob for the openai and openrouter "
                         "backends.  Default None uses the backend's "
                         "own default.")
    # Endpoint switch for the OpenAI backend.  ``responses`` (default)
    # routes through the Responses API, which is the endpoint OpenAI
    # documents as the primary path for reasoning models and which
    # returns reasoning-summary items on every call (captured on the
    # ``llm_reasoning`` output field).  Pass ``chat_completions`` to
    # opt back into the older Chat Completions path -- content only,
    # no reasoning trace surfaced.  Ignored by every other backend.
    ap.add_argument("--openai-endpoint",
                    choices=list(_OPENAI_ENDPOINTS),
                    default="responses",
                    help="OpenAI endpoint to route through.  Default "
                         "`responses` uses the Responses API and "
                         "captures reasoning summaries.  Pass "
                         "`chat_completions` to force the older "
                         "endpoint (content only, no reasoning trace).")
    ap.add_argument("--openai-reasoning-summary",
                    choices=["auto", "concise", "detailed"],
                    default="auto",
                    help="Reasoning-summary verbosity for the Responses "
                         "API (see the OpenAI reasoning guide -- "
                         "https://platform.openai.com/docs/guides/reasoning "
                         "-- for details).  `auto` lets the model pick; "
                         "`concise` returns a brief summary; `detailed` "
                         "returns a longer one.  Ignored under "
                         "--openai-endpoint chat_completions.  Default `auto`.")
    # Anthropic extended-thinking budget.  0 (default) keeps thinking
    # disabled -- Anthropic behaves as before, content only.  A positive
    # value enables thinking mode with that many tokens of budget; the
    # backend passes ``thinking={"type":"enabled","budget_tokens":N}``
    # and content-block-level ``thinking`` blocks are captured on the
    # ``llm_reasoning`` field.  Recommended range: 1024-16384.
    ap.add_argument("--anthropic-thinking-budget",
                    type=int, default=0,
                    help="Budget tokens for Anthropic extended thinking.  "
                         "0 (default) disables thinking mode.  A positive "
                         "value enables it; reasoning trace lands on the "
                         "`llm_reasoning` output field.  When enabled, "
                         "the effective max_tokens is bumped to "
                         "budget + 1024 as required by the API.")
    ap.add_argument("--vllm-gpu-memory", type=float, default=0.9)
    ap.add_argument("--vllm-max-model-len", type=int, default=None)
    ap.add_argument("--vllm-dtype", default="auto")
    ap.add_argument("--vllm-tensor-parallel", type=int, default=1)

    args = ap.parse_args()

    # Validate --extra-body FIRST: a typo in the JSON should abort
    # before the dataset download, not surface as a confusing 400
    # midway through a run that has already burned tokens.
    extra_body: dict | None = None
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--extra-body is not valid JSON: {e}")
        if not isinstance(extra_body, dict):
            raise SystemExit(
                f"--extra-body must be a JSON object, got "
                f"{type(extra_body).__name__}")

    default_model = {
        "anthropic":    "claude-haiku-4-5-20251001",
        "openai":       "gpt-5.4-mini",
        "openrouter":   "openai/gpt-5.4-mini",
        "vllm-offline": "meta-llama/Llama-3.1-8B-Instruct",
    }
    model = args.model or default_model[args.backend]
    out_path   = Path(args.out)   if args.out   else _default_out_path(args.backend, model)
    cache_path = Path(args.cache) if args.cache else _default_cache_path(args.backend, model)

    print(f"[cfg] backend={args.backend}  model={model}  workers={args.workers}")
    # Echo the sampling knobs so a run's provenance is recoverable from
    # its log alone -- these change results and are easy to forget.
    print(f"[cfg] temperature="
          f"{TEMPERATURE if args.temperature is None else args.temperature}"
          f"{' (default)' if args.temperature is None else ''}"
          f"  max-tokens={args.max_tokens}"
          f"  request-timeout={args.request_timeout}s"
          + (f"  extra-body={json.dumps(extra_body)}" if extra_body else ""))
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

    required = {"id", "profile", "query", "reference_information"}
    missing = required - set(ds.column_names)
    if missing:
        sys.exit(f"dataset is missing required columns: {sorted(missing)}")

    # Build (id, source_id, row) triples.  ``all_rows`` covers the
    # ENTIRE dataset and feeds ``write_output_jsonl`` at the end so
    # the output file stays cumulative across partial re-runs (matching
    # the cache, which is already append-only + keyed by dataset id).
    # ``rows`` below is the invocation-scoped filter that drives ONLY
    # ``generate_all``'s pending-list -- so a run with ``--sample-id
    # 50,...,149`` dispatches just those 100 prompts while the emitted
    # plan file still contains previously completed rows.
    all_rows: list[tuple[int, int | None, dict]] = [
        (int(ds[i]["id"]),
         (int(ds[i]["source_id"]) if ds[i].get("source_id") is not None else None),
         ds[i])
        for i in range(n_total)
    ]
    rows: list[tuple[int, int | None, dict]] = list(all_rows)

    # Row filter (dispatch-only; does not narrow the output).
    if args.sample_id:
        target = _parse_index_list(args.sample_id)
        rows = [(i, s, r) for (i, s, r) in all_rows if i in target]
        print(f"[in]  --sample-id filter: {len(rows)} rows match")
    elif args.sample > 0:
        rows = all_rows[: args.sample]
        print(f"[in]  --sample={args.sample}: {len(rows)} rows")

    # Cross-split cache promotion (optional).  Promote BEFORE generation
    # so the pending-list already accounts for merged entries.
    if args.seed_cache:
        promote_seed_into_cache(Path(args.seed_cache), cache_path, all_rows)

    # Backend + generate.
    backend = make_backend(
        args.backend, model,
        workers=args.workers,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        openai_endpoint=args.openai_endpoint,
        openai_reasoning_summary=args.openai_reasoning_summary,
        openrouter_base_url=args.openrouter_base_url,
        openrouter_api_key=args.openrouter_api_key,
        anthropic_thinking_budget=args.anthropic_thinking_budget,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        vllm_gpu_memory=args.vllm_gpu_memory,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_dtype=args.vllm_dtype,
        vllm_tensor_parallel=args.vllm_tensor_parallel,
        extra_body=extra_body,
        temperature=args.temperature,
        request_timeout=args.request_timeout,
    )
    generate_all(rows, backend, cache_path=cache_path,
                 max_tokens=args.max_tokens)

    # Write output JSONL over the FULL dataset (not just the filtered
    # slice) so partial re-runs accumulate plans in the output file
    # rather than clobbering earlier runs' entries.  Uncached rows are
    # still emitted with ``llm_travel_plan=None`` / ``llm_reasoning=None``
    # so downstream can tell "pending" apart from "generated".
    write_output_jsonl(all_rows, out_path, cache_path=cache_path)


if __name__ == "__main__":
    main()
