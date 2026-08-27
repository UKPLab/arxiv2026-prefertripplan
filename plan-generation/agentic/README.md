# `agentic/` — agentic evaluation track for PreferTripPlan

Design and experimental plan: **[DESIGN.md](DESIGN.md)**.

## Self-containment

This package imports **nothing** from `data-generation/` or `preferences.py`.
`db_tools.py` reads `database/*.csv` directly with its own parsers. The only
permitted external code is the TravelPlanner-ported tool APIs under
`travelplanner-ports/tools`, and only the verification scripts use them — to
check our tools against the grader's view of the world.

`_paths.HF_DIR` points at `data-generation/HF-data/*.jsonl`, but that is the
shipped **dataset**, read as data. It is not a code dependency.

This is why the parity result below is strong: `db_tools.py` is an
**independent reimplementation** of the record shapes in
`reference_information`, not a caller of the code that produced them. Matching
byte-for-byte across 19,852 blocks is therefore evidence about the data, and
any future drift on either side will break the test.

Nothing outside this directory is modified. The existing pipeline
(`generate_plans.py`, `evaluation/*`) is used read-only or invoked as a CLI —
`eval.py` already grades against the full `database/` CSVs and ignores
`reference_information`, so a tool-using planner needs no evaluator change.

## Stage 0 — zero-inference gates

Run these before any model is invoked.

| script | asserts |
|---|---|
| `verify_pool_parity.py` | DB tool output == `reference_information`, block for block |
| `verify_tool_evaluator_consistency.py` | the agent's ground-transport view matches the grader's |
| `audit_dataset.py` | emits per-row `has_baited_ground_leg` / `dead_leg` covariates |

```bash
python3 verify_pool_parity.py --split test       --strict
python3 verify_pool_parity.py --split test_large --strict
python3 verify_tool_evaluator_consistency.py     --strict
python3 audit_dataset.py --split test
python3 audit_dataset.py --split test_large
```

### Results

**Pool parity — 100%, both splits.** This is the gate the I2-vs-I4 contrast
rests on: both conditions expose the same items, so the delta measures
retrieval rather than pool differences.

| split | blocks compared | identical |
|---|---|---|
| `test` | 3,654 | **3,654 (100.00%)** |
| `test_large` | 16,198 | **16,198 (100.00%)** |

Both parsers were written independently — `db_tools.py` shares no code with the
generator — so this is a genuine cross-check rather than a tautology.

**Tool/grader consistency.** Refusal semantics agree perfectly — 0 mismatches
across all 32,312 (pair × mode) combinations between `GoogleDistanceMatrix.run`,
`run_for_evaluation`, and `db_tools.get_ground_transport`. An agent can never
pick a ground leg the sandbox check will then reject. Residual cost divergence
is rounding only: **max $0.99**, the exact bound of `round(km*rate,2)` vs the
grader's `int(km*rate)`. Pinned by the test rather than "fixed", so the two
definitions stay tied.

The test also **found two dataset errata on its first two runs** (DESIGN.md
facts 4 and 5), both cases of `reference_information` disagreeing with the
grader on ground-transport cost:

| erratum | cause | `test` rows | worst error |
|---|---|---|---|
| decimal distances mis-parsed | `parse_km` regex cannot span `.`, so `"94.0 km"` → `4.0`; 143/17602 CSV rows | 47/225 (20.9%) | $94 |
| duplicate rows resolved differently | 1,382 duplicate pairs, 25 conflicting; grader takes first row, `load_distance` takes last | 7/225 (3.1%) | $19 |

Outcome impact is minor — 2 `valid_cost` failures for gpt-5.6-terra, 0 for the
other four models — but the agent-facing tool must not inherit either bug, so
`_load_distance_true` parses correctly and resolves duplicates first-wins.
`ground_transport_raw` keeps the buggy behaviour so the parity test can still
reproduce shipped blocks byte-for-byte.

**Dataset covariates.**

| tag | `test` | `test_large` |
|---|---|---|
| `has_baited_ground_leg` | 18/225 (8.0%) | 92/1000 (9.2%) |
| `dead_leg` | 5/225 (2.2%) — ids 27, 33, 52, 144, 214 | 33/1000 (3.3%) |

`dead_leg` rows have **no usable transport on some leg of the prescribed tour**
(zero flights, both ground modes unusable), so no plan can pass
`is_valid_information_in_sandbox` there under that tour. See DESIGN.md fact 3.

## Layout

Pipeline code in the package root, one module per concern; everything else is a
diagnostic.

| file | role |
|---|---|
| `_paths.py` | repo paths, `sys.path` shim, dataset loading |
| `db_tools.py` | DB-backed tool layer over `database/*.csv` |
| `schemas.py` | tool JSON schemas + result serialisation |
| `record_view.py` | projection: what the agent may see, and the tool-call budget |
| `prompts.py` | the six phase templates, on the direct planner's trunk |
| `verifier.py` | the agent-authored spec (pseudo-code + Python) and its sandbox |
| `runtime.py` | the loop, budgets, notebook, and the `ReactReflectEnv` adapter |
| `telemetry.py` | per-record trajectory records |
| `run_agentic.py` | the CLI, plus the resumable cache |
| `diagnostics/` | validation and audit scripts, plus their outputs |

Model calls go through `plan-generation/llm_backends.py` -- the same clients,
retry ladder and reasoning extraction the direct planner uses. `ToolChat` lives
there too, since multi-turn tool calling is provider machinery rather than agent
logic; only the loop that decides *which* tools to call is here.

```
python3 plan-generation/agentic/diagnostics/verify_pool_parity.py --split test --strict
python3 plan-generation/agentic/diagnostics/verify_no_leakage.py  --split test --strict
python3 plan-generation/agentic/diagnostics/verify_tool_evaluator_consistency.py --strict
python3 plan-generation/agentic/diagnostics/audit_dataset.py --split test_large
```

## Dataset access

Rows come from `UKPLab/PreferTripPlan` via `--dataset`, the same source
`generate_plans.py` and `eval.py` use, so the agentic track is scored against
identical rows. `--dataset` also accepts a local HF-compatible directory:

```bash
--dataset ../../data-generation/HF-data     # verified byte-identical to the Hub
```

One normalisation is applied on load. **Any** load through `datasets` — Hub or
local directory — returns `date` as `"2025-11-02 00:00:00"` while the source
JSONL holds `"2025-11-02"`: Arrow infers `timestamp[s]` from the ISO strings,
the dataset README declares the column `string`, and the cast stringifies the
datetime. `date.fromisoformat` rejects the result, so `_paths._normalise`
strips the time component. See DESIGN.md fact 6.

## Not yet built

Stage A onward: `chat.py`, `runtime.py`, `prompts.py`, `scaffolds/react.py`,
`selfcheck.py`, `telemetry.py`, `run_agentic.py`, and
`evaluation/agent_metrics.py`. See DESIGN.md § Implementation.
