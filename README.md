<!-- <p align="center">
  <img src='logo.png' width='200'>
</p> -->

# PreferTripPlan
[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow?style=flat-square)](https://huggingface.co/datasets/UKPLab/PreferTripPlan)
<!-- [![License](https://img.shields.io/github/license/UKPLab/arxiv2026-prefertripplan)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/) -->

**PreferTripPlan** is a benchmark for evaluating language-model travel planners under structured, multi-paradigm user preferences and persona-driven drift. It extends the [TravelPlanner](https://osu-nlp-group.github.io/TravelPlanner/) test split with 1000 preference-augmented queries covering **8 preference paradigms** (Atomic, Composite, Numeric, Conditional, Lexicographic, Compensatory, Temporal, Scoped) and a controlled **persona drift** protocol (aligned / omission / inversion) that decouples the traveler's stated persona from their in-query preferences.

> **Abstract:** Existing agentic travel-planning benchmarks focus on hard constraints (budget, cuisine, room type) and reward planners that satisfy the constraints regardless of *how* the user actually reasons about tradeoffs. Real travelers rank, compensate, condition, and time-bound their preferences — and their stated persona is often a partial or noisy signal for the plan they actually want. **PreferTripPlan** targets that gap. Every query carries (i) a resolved preference in one of 8 well-typed paradigms with structured predicates and quantifier scopes, (ii) a natural-language persona rendered from a curated trait bank, and (iii) a persona drift mode that either aligns, omits, or inverts a persona trait relative to the trip's preference. Feasibility is gated end-to-end — tour selection, pool floors, temporal-scope buckets, transport-budget, min-nights lodging, and vacuous-satisfaction guards — so every admitted record poses a genuine trade-off decision. The result is a benchmark that tests whether a planner can (a) parse structured preferences beyond simple filters and (b) navigate a mismatch between what the user's persona suggests and what the query actually asks for.

Contact person: [Md Imbesat Hassan Rizvi](mailto:imbesat.rizvi@tu-darmstadt.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue if something is broken or if you have further questions.


## Dataset at a glance

| Property | Value |
|---|---|
| HF splits | `test` (225 curated balanced rows) · `test_large` (1000 full rows) |
| Records | 1000 (984 augmented + 16 non-augmentable pass-through) |
| Difficulty split | easy 348 · medium 333 · hard 319 |
| Persona drift per level | aligned 30% · omission 35% · inversion 35% |
| Pairing (medium/hard) | independent 60% · overlapping 40% |
| Overlap subtypes | competing 25% · non-competing 75% |
| Preference paradigms | 8 (Atomic, Composite, Numeric, Conditional, Lexicographic, Compensatory, Temporal, Scoped) |
| Temporal sub-operators | 9 (always, sometime, within, atmost_once, sometime_before, sometime_after, always_within, hold_during, hold_after) |
| Preference bank size | ~150 curated entries with `example_values` + rationales |


## Getting Started

Clone the repo and set up a Python 3.10 virtual environment:

```bash
git clone https://github.com/UKPLab/arxiv2026-prefertripplan.git
cd arxiv2026-prefertripplan
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The TravelPlanner CSV / JSONL sources live under `database/` (accommodations, restaurants, attractions, flights, distance matrix, city↔state mapping). The base test queries are `database/travelplanner_test.jsonl` and are used verbatim — the augmenter only ADDS fields, never mutates the original query facts.


## Usage

The benchmark is generated in three sequential stages. Each stage reads the previous stage's output and writes to `prefertripplan.jsonl` in place; the sidecar files (`llm_nl_cache.jsonl`, `analysis/`) are optional.

### Stage 1 — Preference augmentation

Attaches one (easy) or two (medium/hard) resolved preferences to every query, drawn from `preference_bank.json` and balanced across paradigms, sub-paradigms, bank ids, pairing types, and overlap subtypes. Every admitted record passes an end-to-end feasibility gate: candidate cities have viable pools, the ordered tour has a within-budget transport plan, every temporal-scope group has ≥1 matching item, no ScopedPreference names a city outside the query's state, and the reference `solution_information` block draws from a feasibility-filtered pool.

```bash
python3 augment_preferences.py \
    --bank preference_bank.json \
    --queries database/travelplanner_test.jsonl \
    --db database \
    --out prefertripplan.jsonl \
    --seed 20260601
```

Targeted regeneration of specific queries (preserving the rest) is supported via `--regen '712,713:overlapping:competing'`.

### Stage 2 — Templated persona + query generation

Builds the traceable persona from the trait bank, applies per-query drift (aligned / omission / inversion), and emits a templated natural-language `nl_persona` + `nl_query`. This stage is deterministic and offline (no LLM calls).

```bash
python3 generate_personas.py
```

### Stage 3 — LLM natural-language rendering (optional, resumable)

Rewrites the templated persona and query into fluent traveler prose while preserving every structured predicate value, scope tag, and drift semantic. Supports three backends and a sidecar `(qid, kind)`-keyed cache so runs are resumable and can be scoped to a sample of qids:

```bash
# Anthropic default
python3 render_llm_nl.py --backend anthropic --model claude-haiku-4-5-20251001

# OpenAI (or OpenAI-compatible endpoint via --openai-base-url)
python3 render_llm_nl.py --backend openai --model gpt-5.4-mini --workers 8

# vLLM in-process, single-pass batched inference
python3 render_llm_nl.py --backend vllm-offline --model meta-llama/Llama-3.1-8B-Instruct

# Smoke test on a curated qid list
python3 render_llm_nl.py --sample-qids data-augmentation/representative_qids.txt
```

Progress + ETA are shown live via tqdm (falls back to a plain `\r`-updated line if tqdm isn't installed).


### Stage 4 — HuggingFace export: `test` + `test_large` splits

Once Stage 3 has written `llm_nl_query` / `llm_nl_profile` into the source jsonl, the two HF-ready splits are produced from it:

- **`test_large`** — the full **1000-record** set, one row per source record. Serves as the wide-coverage split for statistical power and per-slice analysis.
- **`test`** — a curated, balanced **225-record** subset drawn from `test_large`. Balanced across level (75 easy / 75 medium / 75 hard), pairing (single / independent / overlapping × competing / non_competing), profile-drift mode (~30/35/35 aligned/omission/inversion), all 8 paradigms (≈9–10 each in easy/single), and all 9 Temporal sub-ops. This is the recommended default for headline scoring.

Both splits share the same schema; every row carries an `id` field (contiguous 1..N, unique within the split) as its first key. Rows in `test` additionally carry `source_id` — the `id` of the same record in `test_large` — so the two splits can be joined without an external map.

Both files land under `data-generation/HF-data/` alongside a README with YAML frontmatter that HuggingFace's Hub loader parses to expose the two splits under one `default` config:

```
data-generation/HF-data/
├─ prefertripplan_test.jsonl          # 225 rows, split="test"
├─ prefertripplan_test_large.jsonl    # 1000 rows, split="test_large"
└─ README.md
```

Two-step pipeline (run in order):

```bash
# 1. Select the balanced 225-record subset by query_id.
#    Writes: data-generation/prefertripplan.subset.ids.txt
#            (+ prefertripplan.subset.jsonl and .summary.txt for auditing)
python3 data-generation/build_balanced_subset.py --target 225

# 2. Emit the two HF splits from the LLM-rendered source.
#    Reads:  data-generation/prefertripplan.<model-slug>.jsonl,
#            data-generation/prefertripplan.subset.ids.txt
#    Writes: data-generation/HF-data/prefertripplan_test.jsonl,
#            data-generation/HF-data/prefertripplan_test_large.jsonl
python3 data-generation/build_hf_dataset.py \
    --in data-generation/prefertripplan.<model-slug>.jsonl
```

The emitted splits are published on the HuggingFace Hub at [UKPLab/PreferTripPlan](https://huggingface.co/datasets/UKPLab/PreferTripPlan).

Downstream consumption:

```python
from datasets import load_dataset
ds_test       = load_dataset("UKPLab/PreferTripPlan", split="test")        # 225
ds_test_large = load_dataset("UKPLab/PreferTripPlan", split="test_large")  # 1000
```


### Expected results

After Stage 2 you have `prefertripplan.jsonl` with 1000 records. Each augmented record carries:

| Field | Type | Description |
|---|---|---|
| `query_id`, `org`, `dest`, `days`, `date`, `budget`, `people_number`, `local_constraint` | — | Original TravelPlanner query facts (unchanged) |
| `level` | `easy` / `medium` / `hard` | Difficulty; drives paradigm count and pairing |
| `preferences` | list of dicts | Resolved preferences with structured predicate template, `bank_id`, `trace`, and `rationale` |
| `preference_traces` | list of strings | Full trace per pref: `<paradigm>[.<subtype>]:<bank_id>` |
| `pairing_type` | `single` / `independent` / `overlapping` | Relationship between the two prefs (medium/hard) |
| `pairing_subtype` | `null` / `competing` / `non_competing` | Overlap sub-tag |
| `budget_original`, `budget_multiplier`, `budget` | numbers | Original budget and any escalation applied to keep the record feasible (cap 1.5×) |
| `persona` | dict | Trait-populated persona with Travel Style / Food / Hobbies / Lifestyle / Preferred Destinations / Dislikes |
| `persona_drift_mode` | `aligned` / `omission` / `inversion` | Persona-vs-preference relationship |
| `persona_trace` | list | Full source provenance for the persona traits |
| `nl_persona`, `nl_query` | strings | Templated natural-language rendering |
| `feasibility_metadata` | dict | Full audit trail: candidate cities, selected tour, per-city pool counts, per-check breakdown, transport cost, day↔city↔phase↔week-group tie-ins, solution + reference information |

Stage 3 adds:

| Field | Description |
|---|---|
| `llm_nl_persona` | LLM-rewritten fluent persona introduction |
| `llm_nl_query` | LLM-rewritten fluent trip-request message, drift-aware |

Optional distribution report:

```bash
python3 analyze_distributions.py
# → analysis/distribution_analysis.txt  +  analysis/plots/*.png
```


### Key CLI parameters

**`augment_preferences.py`**
- `--bank`: preference-bank JSON path (default: `preference_bank.json`)
- `--queries`: base TravelPlanner test JSONL (default: `database/travelplanner_test.jsonl`)
- `--db`: database directory holding accommodations/restaurants/attractions/flights CSVs
- `--out`: output JSONL (default: `prefertripplan.jsonl`)
- `--seed`: RNG seed (default: 20260601)
- `--regen`: comma-separated `id[:pairing[:subtype]]` targets for surgical regeneration

**`render_llm_nl.py`**
- `--backend`: `anthropic` / `openai` / `vllm-offline`
- `--model`: model name (backend-specific default)
- `--workers`: concurrent in-flight requests for API backends (default 4)
- `--sample-qids`: restrict to a comma-list or a text file of query_ids
- `--sample N`: restrict to the first N records (smoke test)


## Repository layout

```
Preference-Augmentation/
├─ augment_preferences.py     Stage-1 augmenter: bank → resolved preferences + feasibility gate
├─ generate_personas.py       Stage-2 templated persona + query renderer
├─ render_llm_nl.py           Stage-3 LLM NL renderer (Anthropic / OpenAI / vLLM), resumable
├─ analyze_distributions.py   Distribution report over the augmented JSONL
├─ preferences.py             8-paradigm preference type hierarchy + evaluators
├─ persona_traits.py          Curated trait tables (aligned + inversion variants per field)
├─ preference_bank.json       Human-curated bank of preference templates with example values
├─ data-generation/
│   ├─ build_balanced_subset.py    Stage-4a: pick a balanced 225-record subset by query_id
│   ├─ build_hf_dataset.py         Stage-4b: emit prefertripplan_test{,_large}.jsonl
│   └─ HF-data/                    Emitted HF-ready splits + README (uploaded to the Hub)
├─ database/                  TravelPlanner sources (CSVs + base test queries)
├─ analysis/                  Distribution reports and plots
└─ data-augmentation/         Auxiliary scripts (flight DB build, query redate, curated qid list)
```


## Design highlights

- **Feasibility as a hard invariant.** Every admitted record has (a) a coherent ordered tour where every leg has a viable transport mode under `local_constraint`, (b) transport cost ≤60% of the applied budget, (c) accommodations with `minimum_nights ≤ 3` in every stay city, (d) per-city pool floors cleared for each preference's scope group, and (e) no vacuous-satisfaction case anywhere (every scope group of every temporal operator has ≥1 matching item; `atmost_once` and ordering ops included).
- **Temporal scope routing.** Nine temporal operators × five scope classes (`per_city`, `per_day`, `travel_phase`, `week_group`, `global`) each have a dedicated bucket enforcement — per-city individual, per-phase union, per-week-group union, per-hold-window per-city. Rationale explanations don't override the resolved scope.
- **Cross-state city guard.** Any `Day.city` reference (in ScopedPreference scope_filters or ConditionalPreference condition/then_pref) that names a city outside the query's state is rejected upstream.
- **Compensatory sibling distinctness.** When a paired anchor forces re-sampling of primary_ap and margin_ap on the same categorical attribute, the two slots are guaranteed to land on distinct values compatible with the paired predicate.
- **Persona drift ≠ preference drift.** Persona traits drift independently of the query's preferences (30/35/35 aligned/omission/inversion per level) — the LLM must reconcile the two without inferring one from the other.
- **Provenance everywhere.** `preference_traces`, `persona_trace`, `feasibility_metadata`, `solution_information`, and `reference_information` together form a self-verifying audit trail per record.


## Development

### End-to-end regeneration (sequential)

Every stage reads what the previous stage wrote and mutates `prefertripplan.jsonl` in place. Run the scripts in this exact order — no other invocation is required, no manual editing between stages.

```bash
# Stage 1 — Preference augmentation
#   Reads:  preference_bank.json, database/travelplanner_test.jsonl, database/*
#   Writes: prefertripplan.jsonl (preferences, preference_traces, pairing_type,
#           pairing_subtype, budget_multiplier, feasibility_metadata,
#           solution_information, reference_information)
python3 augment_preferences.py

# Stage 2 — Templated persona + query rendering (deterministic, offline)
#   Reads:  prefertripplan.jsonl, persona_traits.py trait tables
#   Writes: prefertripplan.jsonl (persona, persona_drift_mode, persona_trace,
#           nl_persona, nl_query)
python3 generate_personas.py

# Stage 3 (optional) — LLM NL rendering, resumable via llm_nl_cache.jsonl
#   Reads:  prefertripplan.jsonl, llm_nl_cache.jsonl (if present)
#   Writes: prefertripplan.jsonl (llm_nl_persona, llm_nl_query),
#           llm_nl_cache.jsonl (append-only)
python3 render_llm_nl.py --backend anthropic

# Stage 4a — Pick balanced 225-record subset by query_id
#   Reads:  data-generation/prefertripplan.jsonl
#   Writes: data-generation/prefertripplan.subset.ids.txt
#           data-generation/prefertripplan.subset.jsonl (+ .summary.txt)
python3 data-generation/build_balanced_subset.py --target 225

# Stage 4b — Emit the two HF splits from the LLM-rendered source
#   Reads:  data-generation/prefertripplan.<model-slug>.jsonl,
#           data-generation/prefertripplan.subset.ids.txt
#   Writes: data-generation/HF-data/prefertripplan_test.jsonl        (225 rows,  split="test")
#           data-generation/HF-data/prefertripplan_test_large.jsonl  (1000 rows, split="test_large")
python3 data-generation/build_hf_dataset.py \
    --in data-generation/prefertripplan.<model-slug>.jsonl

# Optional — Distribution / balance audit
#   Reads:  prefertripplan.jsonl
#   Writes: analysis/distribution_analysis.txt, analysis/plots/*.png
python3 analyze_distributions.py
```

### Targeted / partial re-runs

- **Regenerate a subset of queries** without touching the rest (Stage 1):
  ```bash
  python3 augment_preferences.py --regen '712,713:overlapping:competing,996'
  ```
  The listed queries are re-augmented (optionally with forced pairing / subtype); every other record is read back from `--out` verbatim.
- **After** any Stage-1 regeneration, always re-run Stage 2 so `persona`, `persona_drift_mode`, `nl_persona`, and `nl_query` are re-derived from the fresh preferences.
- **Stage 3 cache invalidation.** The `llm_nl_cache.jsonl` sidecar is keyed by `(qid, kind)` only — not content-hashed. If Stage 1 changed the resolved preferences for a `qid`, its old LLM NL is stale. Either delete the whole cache before the next Stage-3 run (`rm llm_nl_cache.jsonl`) or prune the affected qid lines. Persona-only entries usually survive across regens; query entries almost always need to be regenerated.
- **Smoke test Stage 3 on a curated qid list** before a full run:
  ```bash
  python3 render_llm_nl.py --sample-qids data-augmentation/representative_qids.txt
  ```

### Balance / health checks

- `python3 analyze_distributions.py` writes level / drift / pairing / paradigm / bank-id / entity-attribute distributions to `analysis/distribution_analysis.txt` and per-axis plots to `analysis/plots/`.
- Every admitted record's `feasibility_metadata` block is self-verifying: `selected_tour`, `per_city_check_breakdown`, `prehoc_transport_cost` vs `transport_budget_cap`, `day_to_city` / `day_to_phase` / `day_to_week_group`, and `solution_information` are all inspectable per record without re-running the augmenter.


## Cite

Please use the following citation:

```
@misc{rizvi2026prefertripplan,
      title={A Multi-Paradigm Preference-Fidelity Benchmark for Long Horizon Planning}, 
      author={Rizvi, Md Imbesat Hassan and Dutta, Subhabrata and Zhu, Xiaodan and Gurevych, Iryna},
      month=jul,
      year={2026},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={}, 
}
```


## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
