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
import copy
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

from llamafactory.data import (  # noqa: E402
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402
from llamafactory.train.vulcan import (  # noqa: E402
    build_layerwise_cluster_idx,
    build_multimodal_cluster_idx,
    build_third_keep_ratios,
    build_uniform_cluster_idx,
    collect_mlp_activations,
    collect_multimodal_mlp_activations,
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
        "--keep_ratios_path",
        type=str,
        default=None,
        help="JSON/YAML file containing a per-layer ratio list, or a mapping with a keep_ratios field.",
    )
    parser.add_argument(
        "--first_layer_ratio", type=float, default=1.0 / 3.0, help="Fraction of layers in the first group."
    )
    parser.add_argument(
        "--last_layer_ratio", type=float, default=1.0 / 3.0, help="Fraction of layers in the last group."
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Limit batches globally in legacy mode or per category dataset in multimodal mode.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override dataloader batch size.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader num_workers.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the training split before collecting activations. Recommended for category-grouped datasets.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --shuffle is enabled.")
    parser.add_argument(
        "--activation_mode",
        choices=["legacy", "multimodal"],
        default="legacy",
        help="Use legacy all-token activation means or VQA modality-aware sample/category macro means.",
    )
    parser.add_argument(
        "--category_datasets",
        default="vqa_train_modality,vqa_train_plane,vqa_train_organ",
        help="Comma-separated training datasets macro-averaged in multimodal mode.",
    )
    parser.add_argument("--image_activation_weight", type=float, default=0.4)
    parser.add_argument("--question_activation_weight", type=float, default=0.4)
    parser.add_argument("--prediction_activation_weight", type=float, default=0.2)
    parser.add_argument(
        "--disable_down_proj_weighting",
        action="store_true",
        help="Do not multiply activation signatures by the corresponding down_proj column norm.",
    )
    parser.add_argument(
        "--disable_weight_row_normalization",
        action="store_true",
        help="Use raw up/gate rows instead of direction-normalized rows for multimodal clustering.",
    )
    parser.add_argument(
        "--activation_distance_weight",
        type=float,
        default=0.25,
        help="Weight of standardized image/question/prediction signatures in multimodal clustering distance.",
    )
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


def build_dataloader(
    train_config: dict[str, Any],
    dataset_name: str,
    model,
    tokenizer_module,
    template,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
):
    category_config = copy.deepcopy(train_config)
    category_config["dataset"] = dataset_name
    model_args, data_args, training_args, _, _ = get_train_args(category_config)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=None,
        label_pad_token_id=(
            IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer_module["tokenizer"].pad_token_id
        ),
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset_module["train_dataset"],
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        shuffle=shuffle,
        generator=generator,
    )


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
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_size = args.batch_size or training_args.per_device_train_batch_size
    num_workers = args.num_workers if args.num_workers is not None else training_args.dataloader_num_workers
    dataloader = build_dataloader(
        train_config,
        train_config["dataset"],
        model,
        tokenizer_module,
        template,
        batch_size,
        num_workers,
        args.shuffle,
        args.seed,
    )
    if args.keep_ratios_path is not None:
        with open(args.keep_ratios_path, encoding="utf-8") as f:
            keep_ratios_data = yaml.safe_load(f)
        if isinstance(keep_ratios_data, dict):
            keep_ratios_data = keep_ratios_data.get("keep_ratios")
        if not isinstance(keep_ratios_data, list):
            raise ValueError("--keep_ratios_path must contain a list or a mapping with a keep_ratios list.")
        keep_ratios = [float(ratio) for ratio in keep_ratios_data]
        print(f"Using keep ratios from {args.keep_ratios_path}: {keep_ratios}", flush=True)
    elif any(ratio is not None for ratio in (args.first_keep_ratio, args.middle_keep_ratio, args.last_keep_ratio)):
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
    else:
        keep_ratios = [args.keep_ratio] * len(find_mlp_layers(model))

    if args.activation_mode == "legacy":
        activations = collect_mlp_activations(model, dataloader, max_batches=args.max_batches)
        if len(set(keep_ratios)) == 1:
            cluster_idx = build_uniform_cluster_idx(model, activations, keep_ratio=keep_ratios[0])
        else:
            cluster_idx = build_layerwise_cluster_idx(model, activations, keep_ratios=keep_ratios)
    else:
        category_names = [name.strip() for name in args.category_datasets.split(",") if name.strip()]
        if not category_names:
            raise ValueError("--category_datasets must contain at least one dataset in multimodal mode.")

        dataloaders = {
            category_name: build_dataloader(
                train_config,
                category_name,
                model,
                tokenizer_module,
                template,
                batch_size,
                num_workers,
                args.shuffle,
                args.seed,
            )
            for category_name in category_names
        }
        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None and tokenizer_module.get("processor") is not None:
            image_token_id = getattr(tokenizer_module["processor"], "image_token_id", None)
        if image_token_id is None:
            raise ValueError("Cannot determine image_token_id for multimodal activation collection.")

        stats = collect_multimodal_mlp_activations(
            model,
            dataloaders,
            image_token_id=int(image_token_id),
            special_token_ids=set(tokenizer.all_special_ids),
            max_batches=args.max_batches,
            image_weight=args.image_activation_weight,
            question_weight=args.question_activation_weight,
            prediction_weight=args.prediction_activation_weight,
            weight_by_down_proj=not args.disable_down_proj_weighting,
        )
        cluster_idx = build_multimodal_cluster_idx(
            model,
            stats,
            keep_ratios=keep_ratios,
            normalize_weight_rows=not args.disable_weight_row_normalization,
            activation_distance_weight=args.activation_distance_weight,
        )

    save_cluster_idx(cluster_idx, args.output_path)
    print(f"Saved Vulcan cluster_idx to {args.output_path}")


if __name__ == "__main__":
    main()
