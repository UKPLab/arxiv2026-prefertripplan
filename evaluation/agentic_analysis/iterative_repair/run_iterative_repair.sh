#!/usr/bin/env bash
# Score every plan an agentic run produced, not only the one it returned.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=qwen3.8-27b
SPLIT=test
AGENTIC_RUNS=../../../plan-generation/agentic-runs

python3 -u iterative_repair.py \
  --traj   "$AGENTIC_RUNS/traj_${SPLIT}_openrouter_${MODEL}.jsonl" \
  --cache  "$AGENTIC_RUNS/plans_${SPLIT}_cache_openrouter_${MODEL}.jsonl" \
  --direct "../../${MODEL}_${SPLIT}/eval_${MODEL}.jsonl" \
  --split  "$SPLIT" \
  --out     "runs/iter_${MODEL}_${SPLIT}.jsonl" \
  --summary "runs/iter_${MODEL}_${SPLIT}.txt" \
  --chunk 40 2>&1 | tee iterative_repair_pipeline.log
