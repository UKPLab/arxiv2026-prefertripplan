#!/bin/bash
# PFI pipeline: per-model runs -> per-model summary.
# PER MODEL ALWAYS -- the same drifted leaves are scored for every model,
# so pooling would stack correlated copies and inflate n.
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
    ev=$(ls ../"${m}_${split}"/eval_*.jsonl 2>/dev/null | head -1)
    pf="../../plan-generation/${PLAN[$m]/SPLIT/$split}"
    [ -n "$ev" ] || { echo "MISSING eval  $m/$split"; continue; }
    [ -f "$pf" ] || { echo "MISSING plan  $pf"; continue; }
    python3 pfi.py --split "$split" --model "$m" \
      --detailed "$ev" --plan-file "$pf" \
      --out "runs/pfi_${m}_${split}.txt" \
      --json-out "runs/pfi_${m}_${split}.json" >/dev/null 2>&1 \
      && echo "ok   $m/$split" || echo "FAIL $m/$split"
  done
done
python3 pfi_summary.py --runs-dir runs --out PFI_approach_and_results.txt
echo "PFI PIPELINE DONE"
