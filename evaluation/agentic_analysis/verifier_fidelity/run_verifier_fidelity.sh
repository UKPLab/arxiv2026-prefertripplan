#!/usr/bin/env bash
# Score the agent's self-authored verifier against the benchmark evaluator.
# Requires iterative_repair/runs/iter_*.jsonl, which supplies the ground truth
# per plan; run that pipeline first.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=qwen3.8-27b
SPLIT=test
AGENTIC_RUNS=../../../plan-generation/agentic-runs

python3 -u verifier_fidelity.py \
  --traj "$AGENTIC_RUNS/traj_${SPLIT}_openrouter_${MODEL}.jsonl" \
  --iter "../iterative_repair/runs/iter_${MODEL}_${SPLIT}.jsonl" \
  --split "$SPLIT" \
  --out     "runs/fidelity_${MODEL}_${SPLIT}.jsonl" \
  --summary "runs/fidelity_${MODEL}_${SPLIT}.txt" \
  --chunk 40 2>&1 | tee verifier_fidelity_pipeline.log
