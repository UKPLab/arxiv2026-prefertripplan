#!/bin/bash
# F4 tier-completion pipeline: per-model runs -> two-level summary.
#   LEVEL 1  within pairing   t1 - t2
#   LEVEL 2  within scope     (as tier 1) - (as tier 2)
# Both graded and binary, levels always printed with the differences.
# Agentic runs excluded by design (tool-based DB access means
# reference_information is not their choice set).
cd "$(dirname "$0")" || exit 1
mkdir -p runs
for split in test test_large; do
  for m in gpt-5.6-terra nemotron deepseek qwen3.8-27b gemma-4-26b-a4b; do
    ef=$(ls ../"${m}_${split}"/eval_*.jsonl 2>/dev/null | head -1)
    [ -n "$ef" ] || { echo "MISSING eval for $m/$split"; continue; }
    python3 tier_completion.py --split "$split" --model "$m" --eval-file "$ef" \
      --out "runs/tier_${m}_${split}.txt" \
      --json-out "runs/tier_${m}_${split}.json" >/dev/null 2>&1 \
      && echo "ok   $m/$split" || echo "FAIL $m/$split"
  done
done
python3 tier_completion_summary.py --runs-dir runs \
  --out TIER_COMPLETION_approach_and_results.txt
echo "TIER COMPLETION PIPELINE DONE"
