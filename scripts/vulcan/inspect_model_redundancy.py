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
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.vulcan import find_mlp_layers, load_cluster_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect intra-cluster MLP weight redundancy.")
    parser.add_argument("--model_name_or_path", required=True, help="Model or checkpoint path.")
    parser.add_argument("--cluster_idx_path", required=True, help="Path to cluster_idx JSON.")
    parser.add_argument("--template", default="qwen3_5_nothink", help="LlamaFactory prompt template.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote model code when loading.")
    parser.add_argument("--output_path", default=None, help="Optional JSON output path.")
    return parser.parse_args()


@torch.no_grad()
def compute_layer_stats(mlp: torch.nn.Module, layer_clusters) -> dict[str, float]:
    vectors = torch.cat([mlp.up_proj.weight.float(), mlp.gate_proj.weight.float()], dim=1)
    l1_sum = 0.0
    l2_sum = 0.0
    cosine_sum = 0.0
    count = 0
    for cluster in layer_clusters:
        anchor = vectors[int(cluster["anchor"])]
        members = vectors[torch.tensor(cluster["neuron"], device=vectors.device, dtype=torch.long)]
        diff = members - anchor.unsqueeze(0)
        l1_sum += diff.abs().mean(dim=1).sum().item()
        l2_sum += diff.pow(2).mean(dim=1).sqrt().sum().item()
        cosine_sum += torch.nn.functional.cosine_similarity(members, anchor.unsqueeze(0), dim=1).sum().item()
        count += len(cluster["neuron"])

    return {
        "mean_l1": l1_sum / max(count, 1),
        "mean_l2": l2_sum / max(count, 1),
        "mean_cosine": cosine_sum / max(count, 1),
        "num_clustered_neurons": count,
    }


def main() -> None:
    args = parse_args()
    model_args, _, finetuning_args, _ = get_infer_args(
        {
            "model_name_or_path": args.model_name_or_path,
            "template": args.template,
            "trust_remote_code": args.trust_remote_code,
        }
    )
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    cluster_idx = load_cluster_idx(args.cluster_idx_path)
    mlp_layers = find_mlp_layers(model)
    if len(cluster_idx) != len(mlp_layers):
        raise ValueError(f"cluster_idx has {len(cluster_idx)} layers, but model has {len(mlp_layers)} MLP layers.")

    layer_stats = []
    for layer_ref, layer_clusters in zip(mlp_layers, cluster_idx):
        if not layer_clusters:
            continue

        stats = compute_layer_stats(layer_ref.mlp, layer_clusters)
        stats["layer"] = layer_ref.index
        layer_stats.append(stats)

    summary = {
        "num_layers": len(layer_stats),
        "mean_l1": sum(stats["mean_l1"] for stats in layer_stats) / max(len(layer_stats), 1),
        "mean_l2": sum(stats["mean_l2"] for stats in layer_stats) / max(len(layer_stats), 1),
        "mean_cosine": sum(stats["mean_cosine"] for stats in layer_stats) / max(len(layer_stats), 1),
        "layers": layer_stats,
    }
    print(json.dumps(summary, indent=2))
    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
