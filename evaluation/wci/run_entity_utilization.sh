#!/bin/bash
# Entity-utilization pipeline: per-model runs -> combined summary.
#   ENTITY UTILIZATION:  REALIZED = RELATIVE x EXPECTED
# Agentic runs excluded by design (tool-based DB access means
# reference_information is not their choice set).
cd "$(dirname "$0")" || exit 1
declare -A PLAN=(
 [gpt-5.6-terra]="gpt-5.6-terra/structured_plans_SPLIT_openai_gpt-5.6-terra.jsonl"
 [nemotron]="nemotron/structured_plans_SPLIT_openrouter_nvidia_nemotron-3-ultra-550b-a55b_free.jsonl"
 [deepseek]="deepseek/structured_plans_SPLIT_openrouter_deepseek_deepseek-v4-flash.jsonl"
 [qwen3.8-27b]="qwen3.8-27b/structured_plans_SPLIT_openrouter_qwen3.8-27b.jsonl"
 [gemma-4-26b-a4b]="gemma-4-26b-a4b/structured_plans_SPLIT_openrouter_google_gemma-4-26b-a4b-it_free.jsonl"
)
mkdir -p runs
for split in test test_large; do
  for m in gpt-5.6-terra nemotron deepseek qwen3.8-27b gemma-4-26b-a4b; do
    pf="../../plan-generation/${PLAN[$m]/SPLIT/$split}"
    [ -f "$pf" ] || { echo "MISSING $pf"; continue; }
    python3 entity_utilization.py --split "$split" --model "$m" --plan-file "$pf" \
      --out "runs/entity_util_${m}_${split}.txt" \
      --json-out "runs/entity_util_${m}_${split}.json" >/dev/null 2>&1 \
      && echo "ok   $m/$split" || echo "FAIL $m/$split"
  done
done
python3 entity_utilization_summary.py --runs-dir runs --out ENTITY_UTILIZATION_approach_and_results.txt
echo "ENTITY UTILIZATION PIPELINE DONE"
