#!/bin/bash
# Batch profile vision encoder on diverse image samples.
# Usage: bash scripts/vulcan/batch_profile_samples.sh

set -euo pipefail

MODEL="saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa"
JSONL="datasets/vqa_rad/train.jsonl"
ROOT="datasets/vqa_rad"
OUTDIR="results/batch_profile"
WARMUP=10
ITERS=30

# sample_index  image_size  description
SAMPLES=(
  "0  566x555   medium-square"
  "2  766x978   medium-tall"
  "3  555x693   medium-tall-sm"
  "8  1024x1286 large-tall"
  "80 337x451   small"
  "429 768x1020 medium-tall-lg"
  "432 513x513  small-square"
  "463 1024x1309 large-tall-lg"
  "538 296x336  tiny"
  "547 589x623  medium-square-sm"
)

mkdir -p "$OUTDIR"

for entry in "${SAMPLES[@]}"; do
  read -r idx size label <<< "$entry"
  outfile="$OUTDIR/sample${idx}_${label}.json"
  echo "=== Profiling sample $idx ($size, $label) ==="
  WANDB_DISABLED=true python3 scripts/vulcan/profile_vision_encoder.py \
    --model_name_or_path "$MODEL" \
    --dataset_jsonl "$JSONL" \
    --dataset_root "$ROOT" \
    --sample_index "$idx" \
    --template qwen3_5_nothink \
    --trust_remote_code \
    --warmup "$WARMUP" \
    --iterations "$ITERS" \
    --max_new_tokens 1 \
    --output_path "$outfile" \
    2>&1 | grep -E '(prompt_tokens|image_sizes|total_ms|vision_encoder_est|vision_encoder_ratio|Saved|Error)'
  echo ""
done

echo "=== All done. Results in $OUTDIR/ ==="
