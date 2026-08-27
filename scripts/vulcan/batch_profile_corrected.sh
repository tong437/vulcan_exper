#!/bin/bash
# Corrected batch profile: force_max_tokens + diverse samples.
set -euo pipefail

MODEL="saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa"
JSONL="datasets/vqa_rad/train.jsonl"
ROOT="datasets/vqa_rad"
OUTDIR="results/batch_profile_corrected"
WARMUP=5
ITERS=20

# sample_index  label
SAMPLES=(
  "0  medium-square"
  "80 small"
  "432 small-square"
  "538 tiny"
  "8  large-tall"
  "463 large-tall-lg"
)

mkdir -p "$OUTDIR"

for entry in "${SAMPLES[@]}"; do
  read -r idx label <<< "$entry"
  for tokens in 1 64; do
    outfile="$OUTDIR/sample${idx}_${label}_t${tokens}.json"
    echo "=== sample $idx ($label) max_new_tokens=$tokens force_max ==="
    WANDB_DISABLED=true python3 scripts/vulcan/profile_vision_encoder.py \
      --model_name_or_path "$MODEL" \
      --dataset_jsonl "$JSONL" \
      --dataset_root "$ROOT" \
      --sample_index "$idx" \
      --template qwen3_5_nothink \
      --trust_remote_code \
      --warmup "$WARMUP" \
      --iterations "$ITERS" \
      --max_new_tokens "$tokens" \
      --force_max_tokens \
      --output_path "$outfile" \
      2>&1 | grep -E '(prompt_tokens|actual_gen|image_grid|effective_pix|total_ms|vision_encoder_est|vision_encoder_ratio|Saved)'
    echo ""
  done
done

echo "=== All done ==="
