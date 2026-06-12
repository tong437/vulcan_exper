<!--
Copyright 2026 the LlamaFactory team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Vulcan VQA-RAD Setup

The VQA-RAD dataset on Hugging Face uses `image`, `question`, and `answer` columns with `train` and `test` splits. Convert it to LlamaFactory's multimodal sharegpt format before training.

## Convert Dataset

If the server has network access:

```bash
python scripts/vulcan/convert_vqa_rad.py \
  --dataset_name_or_path flaviagiammarino/vqa-rad \
  --output_dir datasets/vqa_rad \
  --overwrite
```

If the dataset has already been downloaded to the server, pass the local snapshot directory, saved-dataset directory, parquet directory, or parquet file:

```bash
python scripts/vulcan/convert_vqa_rad.py \
  --dataset_name_or_path /path/to/vqa-rad \
  --output_dir datasets/vqa_rad \
  --overwrite
```

The converter writes:

```text
datasets/vqa_rad/
  dataset_info.json
  train.jsonl
  test.jsonl
  images/
```

## Debug Train

Edit `model_name_or_path` in `examples/vulcan/qwen35_08b_vqa_rad_full_sft_debug.yaml` if the Qwen model is stored at a local server path, then run:

```bash
WANDB_DISABLED=true llamafactory-cli train examples/vulcan/qwen35_08b_vqa_rad_full_sft_debug.yaml
```

## Full Baseline Train

After the debug run passes, edit `examples/vulcan/qwen35_08b_vqa_rad_full_sft.yaml` for the server GPU memory budget and run:

```bash
WANDB_DISABLED=true llamafactory-cli train examples/vulcan/qwen35_08b_vqa_rad_full_sft.yaml
```

## Build Cluster Index

Collect down-projection input activations and build greedy-match clusters. Start with `--max_batches` for a quick smoke test, then remove it or set a larger value for the real run.

```bash
python scripts/vulcan/collect_cluster_idx.py \
  --config examples/vulcan/qwen35_08b_vqa_rad_full_sft.yaml \
  --output_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50.json \
  --keep_ratio 0.5 \
  --max_batches 100 \
  model_name_or_path=/path/to/baseline/model_or_checkpoint \
  dataset_dir=/path/to/datasets/vqa_rad
```

The extra `key=value` arguments override fields from the YAML, matching the `python src/train.py <yaml> key=value` style.

For a front/middle/back layer split, pass group keep ratios. For example, keep the first third unpruned,
keep 75% in the middle third, and keep 50% in the last third:

```bash
python scripts/vulcan/collect_cluster_idx.py \
  --config examples/vulcan/qwen35_08b_vqa_rad_full_sft.yaml \
  --output_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_layerwise_1_00_0_75_0_50.json \
  --first_keep_ratio 1.0 \
  --middle_keep_ratio 0.75 \
  --last_keep_ratio 0.5 \
  --max_batches 100 \
  model_name_or_path=/path/to/baseline/model_or_checkpoint \
  dataset_dir=/path/to/datasets/vqa_rad
```

Layerwise `cluster_idx` files are supported by both collapse SFT and structural pruning. For non-uniform target widths,
`save_pruned_model.py` stores `vulcan_intermediate_sizes` and installs a small Qwen3.5 remote-code loader in the output
checkpoint. Reload such checkpoints with `trust_remote_code: true`.

## VQA-Med Multimodal Clustering

Regenerate the VQA-Med converted dataset once to create category-specific training splits, then collect image,
question, and causal prediction-position activations separately:

```bash
python scripts/vulcan/convert_vqa_med.py \
  --dataset_dir /path/to/raw/vqa_med \
  --output_dir datasets/vqa_med \
  --overwrite

python scripts/vulcan/collect_cluster_idx.py \
  --config examples/vulcan/qwen35_08b_vqa_med_cls_full_sft.yaml \
  --output_path saves/qwen35-0_8b-vqa-med-cls/vulcan/cluster_idx_multimodal_0_50_ckpt1200.json \
  --activation_mode multimodal \
  --category_datasets vqa_train_modality,vqa_train_plane,vqa_train_organ \
  --image_activation_weight 0.4 \
  --question_activation_weight 0.4 \
  --prediction_activation_weight 0.2 \
  --activation_distance_weight 0.25 \
  --keep_ratio 0.5 \
  --max_batches 200 \
  --shuffle \
  --seed 42 \
  model_name_or_path=/path/to/checkpoint-1200 \
  dataset_dir=datasets/vqa_med \
  deepspeed=null
```

In multimodal mode, `--max_batches` is applied to each category, so the example above collects at most 600 batches
in total. For arbitrary adaptive layer budgets, pass a JSON/YAML list with `--keep_ratios_path`; nonuniform Qwen3.5
checkpoints saved by Vulcan are reloadable with `trust_remote_code=True`.

The multimodal mode averages tokens within each sample and samples within each category before macro-averaging the
three categories. Anchor scores use `0.4 image + 0.4 question + 0.2 prediction` contributions, optionally weighted by
the corresponding `down_proj` column norm. Clustering combines normalized up/gate weight directions with standardized
modality signatures. Legacy collection remains the default.

For normalized collapse with a delayed ramp, use:

```yaml
collapse_reduction: normalized
collapse_warmup_steps: 200
collapse_ramp_steps: 800
collapse_loss_scale: 1.0
```

See `qwen35_08b_vqa_med_cls_multimodal_vulcan_sft.yaml` for a complete configuration.

## Collapse SFT

After `cluster_idx_greedy_match_0_50.json` exists, run the Vulcan regularized SFT:

```bash
WANDB_DISABLED=true python src/train.py \
  examples/vulcan/qwen35_08b_vqa_rad_vulcan_sft.yaml \
  model_name_or_path=/path/to/baseline/model_or_checkpoint \
  dataset_dir=/path/to/datasets/vqa_rad \
  collapse_cluster_idx_path=/path/to/cluster_idx_greedy_match_0_50.json \
  output_dir=/path/to/output/vulcan-sft
```

Recommended collapse settings for the current VQA-RAD yes/no experiment are:

```yaml
use_collapse_loss: true
collapse_lambda1: 0.0
collapse_lambda2: 0.0
collapse_learnable_lambda: true
collapse_lambda_lr: -1.0
collapse_use_weight_proxy: false
learning_rate: 1.0e-4
gradient_accumulation_steps: 4
num_train_epochs: 6.0
```

The best 0.50 run so far uses plain SGD gradient ascent for the lambda parameters. `collapse_use_weight_proxy: true`
was useful for the original DeepSpeed compatibility design, but it failed in the current server experiment because the
collapse gradients did not produce the desired MLP redundancy; keep it disabled for the main VQA-RAD run.

## Activation Alignment

Activation alignment is the multimodal extension path for Vulcan. It aligns the soft top-k MLP activation masks between
visual tokens and task-relevant text tokens. Start with the align-only config before combining it with collapse loss:

```yaml
use_activation_align: true
align_lambda: 10.0
align_temperature: 0.05
align_quantile: 0.8
align_pool_type: mean
align_loss_type: soft_iou
align_text_mode: answer
```

`align_text_mode: answer` pools only assistant response tokens (`labels != IGNORE_INDEX`). Use `question` and `qa` for
ablation runs after the answer-token baseline is stable.

You can compare intra-cluster redundancy before and after collapse SFT:

```bash
python scripts/vulcan/inspect_model_redundancy.py \
  --model_name_or_path saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa \
  --cluster_idx_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50_baseline_full_vqa.json \
  --template qwen3_5_nothink \
  --trust_remote_code \
  --output_path saves/qwen35-0_8b-vqa-rad/vulcan/redundancy_baseline_full_vqa_0_50.json

python scripts/vulcan/inspect_model_redundancy.py \
  --model_name_or_path saves/qwen35-0_8b-vqa-rad/full/vulcan-from-baseline-full-vqa-lr1e5-lam01 \
  --cluster_idx_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50_baseline_full_vqa.json \
  --template qwen3_5_nothink \
  --trust_remote_code \
  --output_path saves/qwen35-0_8b-vqa-rad/vulcan/redundancy_vulcan_from_baseline_full_vqa_lr1e5_lam01_0_50.json
```

During Vulcan training, do not use total loss alone as the stopping signal. Track `sft_loss`, `collapse_loss`, and
`collapse_loss_ratio` from the trainer logs, then run generated yes/no accuracy and redundancy inspection on saved
checkpoints. A useful checkpoint should keep unpruned yes/no accuracy close to `baseline-full-vqa` while clearly
reducing intra-cluster distance.

## Prune Model

Prune the baseline or Vulcan checkpoint with the same cluster file:

```bash
python scripts/vulcan/save_pruned_model.py \
  --model_name_or_path saves/qwen35-0_8b-vqa-rad/full/vulcan-from-baseline-full-vqa-lr1e5-lam01 \
  --cluster_idx_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50_baseline_full_vqa.json \
  --output_dir saves/qwen35-0_8b-vqa-rad/full/vulcan-from-baseline-full-vqa-lr1e5-lam01-pruned-0_50 \
  --template qwen3_5_nothink \
  --trust_remote_code
```

## Evaluate Predictions

After running LlamaFactory prediction and producing `generated_predictions.jsonl`, compute exact match and token F1:

```bash
python scripts/vulcan/eval_vqa_predictions.py \
  --prediction_file saves/qwen35-0_8b-vqa-rad/full/vulcan-pruned/generated_predictions.jsonl
```

## Yes/No Evaluation

VQA-RAD contains many closed-ended questions whose labels are exactly `yes` or `no`. Build a yes/no-only eval split after dataset conversion:

```bash
python scripts/vulcan/filter_vqa_rad_yesno.py \
  --dataset_dir datasets/vqa_rad
```

Run prediction on only that split. Override paths from the command line so the checked-in YAML can stay unchanged:

```bash
WANDB_DISABLED=true python src/train.py \
  examples/vulcan/qwen35_08b_vqa_rad_yesno_predict.yaml \
  model_name_or_path=/path/to/trained-or-pruned-model \
  dataset_dir=/path/to/datasets/vqa_rad \
  output_dir=/path/to/output/yesno-predict
```

Then compute VQA metrics and closed-ended accuracy:

```bash
python scripts/vulcan/eval_vqa_predictions.py \
  --prediction_file /path/to/output/yesno-predict/generated_predictions.jsonl
```
