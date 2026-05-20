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

You can compare intra-cluster redundancy before and after collapse SFT:

```bash
python scripts/vulcan/inspect_model_redundancy.py \
  --model_name_or_path saves/qwen35-0_8b-vqa-rad/full/vulcan-sft \
  --cluster_idx_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50.json \
  --template qwen3_5_nothink \
  --trust_remote_code \
  --output_path saves/qwen35-0_8b-vqa-rad/vulcan/redundancy_vulcan_sft.json
```

## Prune Model

Prune the baseline or Vulcan checkpoint with the same cluster file:

```bash
python scripts/vulcan/save_pruned_model.py \
  --model_name_or_path saves/qwen35-0_8b-vqa-rad/full/vulcan-sft \
  --cluster_idx_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50.json \
  --output_dir saves/qwen35-0_8b-vqa-rad/full/vulcan-pruned \
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
