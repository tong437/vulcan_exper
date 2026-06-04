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


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.vulcan import load_cluster_idx
from llamafactory.train.vulcan.pruning import pruning_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune a trained model with a Vulcan cluster_idx JSON file.")
    parser.add_argument("--model_name_or_path", required=True, help="Trained checkpoint or model directory.")
    parser.add_argument("--cluster_idx_path", required=True, help="Path to cluster_idx JSON.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the pruned model.")
    parser.add_argument("--template", default="qwen3_5_nothink", help="LlamaFactory prompt template.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote model code when loading.")
    parser.add_argument("--infer_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max_shard_size", default="5GB", help="Max shard size passed to save_pretrained.")
    parser.add_argument("--config", default=None, help="Optional YAML config to reuse model/template arguments from.")
    return parser.parse_args()


def load_yaml(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}

    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    infer_config = load_yaml(args.config)
    infer_config.update(
        {
            "model_name_or_path": args.model_name_or_path,
            "template": args.template or infer_config.get("template", "qwen3_5_nothink"),
            "trust_remote_code": args.trust_remote_code or infer_config.get("trust_remote_code", False),
            "infer_dtype": args.infer_dtype,
        }
    )
    model_args, data_args, finetuning_args, _ = get_infer_args(infer_config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)

    cluster_idx = load_cluster_idx(args.cluster_idx_path)
    summary = pruning_mlp(model, cluster_idx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_args.infer_dtype == "auto":
        output_dtype = getattr(model.config, "torch_dtype", torch.float32)
    else:
        output_dtype = getattr(torch, model_args.infer_dtype)

    if output_dtype != torch.float32:
        model = model.to(output_dtype)

    save_kwargs = {"save_directory": output_dir, "max_shard_size": args.max_shard_size}
    if not is_transformers_version_greater_than("5.0.0"):
        save_kwargs["safe_serialization"] = True

    model.save_pretrained(**save_kwargs)
    tokenizer.save_pretrained(output_dir)
    if processor is not None:
        processor.save_pretrained(output_dir)

    print(
        "Pruned "
        f"{summary.num_layers} MLP layers from intermediate_size={summary.original_intermediate_size} "
        f"to {summary.pruned_intermediate_size}. Saved to {output_dir}."
    )


if __name__ == "__main__":
    main()
