# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.vulcan import (
    build_layerwise_cluster_idx,
    build_third_keep_ratios,
    build_uniform_cluster_idx,
    collect_mlp_activations,
    find_mlp_layers,
    save_cluster_idx,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Vulcan greedy-match cluster_idx from an SFT YAML config.")
    parser.add_argument("--config", required=True, help="LlamaFactory SFT YAML config used to load model and dataset.")
    parser.add_argument("--output_path", required=True, help="Path to save cluster_idx JSON.")
    parser.add_argument("--keep_ratio", type=float, default=0.5, help="Target MLP intermediate keep ratio.")
    parser.add_argument("--first_keep_ratio", type=float, default=None, help="Keep ratio for the first layer group.")
    parser.add_argument("--middle_keep_ratio", type=float, default=None, help="Keep ratio for the middle layer group.")
    parser.add_argument("--last_keep_ratio", type=float, default=None, help="Keep ratio for the last layer group.")
    parser.add_argument(
        "--first_layer_ratio", type=float, default=1.0 / 3.0, help="Fraction of layers in the first group."
    )
    parser.add_argument(
        "--last_layer_ratio", type=float, default=1.0 / 3.0, help="Fraction of layers in the last group."
    )
    parser.add_argument("--max_batches", type=int, default=None, help="Limit activation collection batches.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override dataloader batch size.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader num_workers.")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def parse_config_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise ValueError(f"Config overrides must use key=value syntax, got: {override}")

    key, value = override.split("=", maxsplit=1)
    key = key.strip()
    if not key:
        raise ValueError(f"Config override key cannot be empty: {override}")

    return key, yaml.safe_load(value)


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    train_config = load_config(args.config)
    for override in args.overrides:
        key, value = parse_config_override(override)
        train_config[key] = value

    train_config["do_train"] = False
    train_config["do_eval"] = False
    train_config["do_predict"] = False
    train_config.setdefault("output_dir", "saves/vulcan/cluster_idx_tmp")

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(train_config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    dataloader = DataLoader(
        dataset_module["train_dataset"],
        batch_size=args.batch_size or training_args.per_device_train_batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers if args.num_workers is not None else training_args.dataloader_num_workers,
    )
    activations = collect_mlp_activations(model, dataloader, max_batches=args.max_batches)
    if any(ratio is not None for ratio in (args.first_keep_ratio, args.middle_keep_ratio, args.last_keep_ratio)):
        num_layers = len(find_mlp_layers(model))
        keep_ratios = build_third_keep_ratios(
            num_layers=num_layers,
            first_keep_ratio=args.first_keep_ratio if args.first_keep_ratio is not None else args.keep_ratio,
            middle_keep_ratio=args.middle_keep_ratio if args.middle_keep_ratio is not None else args.keep_ratio,
            last_keep_ratio=args.last_keep_ratio if args.last_keep_ratio is not None else args.keep_ratio,
            first_layer_ratio=args.first_layer_ratio,
            last_layer_ratio=args.last_layer_ratio,
        )
        print(f"Using layerwise keep ratios: {keep_ratios}", flush=True)
        cluster_idx = build_layerwise_cluster_idx(model, activations, keep_ratios=keep_ratios)
    else:
        cluster_idx = build_uniform_cluster_idx(model, activations, keep_ratio=args.keep_ratio)

    save_cluster_idx(cluster_idx, args.output_path)
    print(f"Saved Vulcan cluster_idx to {args.output_path}")


if __name__ == "__main__":
    main()
