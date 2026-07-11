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

"""Phase 2: Causal Ablation Experiments.

Ablate specific neuron types and evaluate impact on downstream tasks.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: Causal Ablation Experiments.")
    parser.add_argument("--config", required=True, help="LlamaFactory SFT YAML config.")
    parser.add_argument("--neuron_scores", required=True, help="Path to neuron_type_scores.parquet.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--ablation_ratios", default="0.001,0.003,0.005,0.01,0.02,0.03,0.05",
                        help="Comma-separated ablation ratios.")
    parser.add_argument("--high_conf_threshold", type=float, default=0.7, help="High-confidence threshold.")
    parser.add_argument("--eval_tasks", default="caption,vqa,text",
                        help="Comma-separated evaluation tasks.")
    parser.add_argument("--max_eval_samples", type=int, default=1000, help="Max samples per eval task.")
    return parser.parse_args()


FA_LAYERS = {3, 7, 11, 15, 19, 23}


def load_neuron_scores(path: str) -> pd.DataFrame:
    """Load neuron type scores from parquet."""
    return pd.read_parquet(path)


def select_neurons(
    df: pd.DataFrame,
    group: str,
    n_neurons: int,
    threshold: float,
    rng: np.random.RandomState,
) -> list[tuple[int, int]]:
    """Select neurons for ablation based on group type.

    Returns:
        list of (layer_idx, neuron_idx) tuples
    """
    if group == "visual":
        candidates = df[df["p_visual"] >= threshold].nlargest(n_neurons, "p_visual")
    elif group == "text":
        candidates = df[df["p_text"] >= threshold].nlargest(n_neurons, "p_text")
    elif group == "multimodal":
        candidates = df[df["p_multimodal"] >= threshold].nlargest(n_neurons, "p_multimodal")
    elif group == "unknown":
        candidates = df[df["p_unknown"] >= threshold].nlargest(n_neurons, "p_unknown")
    elif group == "random":
        candidates = df.sample(n=min(n_neurons, len(df)), random_state=rng)
    elif group == "layer_type_matched_random":
        fa_neurons = df[df["attention_type"] == "FA"]
        gdn_neurons = df[df["attention_type"] == "GDN"]
        fa_ratio = len(fa_neurons) / len(df)
        n_fa = int(n_neurons * fa_ratio)
        n_gdn = n_neurons - n_fa
        fa_sample = fa_neurons.sample(n=min(n_fa, len(fa_neurons)), random_state=rng)
        gdn_sample = gdn_neurons.sample(n=min(n_gdn, len(gdn_neurons)), random_state=rng)
        candidates = pd.concat([fa_sample, gdn_sample])
    elif group == "low_magnitude":
        candidates = df.nsmallest(n_neurons, "p_visual")
    else:
        raise ValueError(f"Unknown group: {group}")

    return list(zip(candidates["layer"].tolist(), candidates["neuron_idx"].tolist()))


def make_ablation_hook(neuron_indices: list[int]):
    """Create a forward-pre-hook that zeros out specified neurons."""
    def hook(module, args):
        intermediate = args[0]
        if len(neuron_indices) > 0:
            intermediate[..., neuron_indices] = 0
        return (intermediate,)
    return hook


def register_ablation_hooks(model, ablation_plan: dict[int, list[int]]):
    """Register ablation hooks for each layer.

    Args:
        model: The model
        ablation_plan: dict mapping layer_idx to list of neuron indices to ablate
    """
    from llamafactory.train.vulcan import find_mlp_layers

    mlp_layers = find_mlp_layers(model)
    hooks = []

    for layer_ref in mlp_layers:
        if layer_ref.index in ablation_plan:
            neuron_indices = ablation_plan[layer_ref.index]
            if neuron_indices:
                h = layer_ref.mlp.down_proj.register_forward_pre_hook(
                    make_ablation_hook(neuron_indices)
                )
                hooks.append(h)

    return hooks


def build_ablation_plan(
    neuron_list: list[tuple[int, int]],
    num_layers: int,
) -> dict[int, list[int]]:
    """Convert neuron list to per-layer ablation plan."""
    plan = {}
    for layer_idx, neuron_idx in neuron_list:
        if layer_idx not in plan:
            plan[layer_idx] = []
        plan[layer_idx].append(neuron_idx)
    return plan


def evaluate_caption(model, processor, dataset, max_samples: int, device: str) -> dict:
    """Evaluate on COCO Caption task."""
    # Placeholder - implement actual evaluation
    return {"task": "caption", "metrics": {"bleu4": 0.0, "cider": 0.0}}


def evaluate_vqa(model, processor, dataset, max_samples: int, device: str) -> dict:
    """Evaluate on VQA task."""
    # Placeholder - implement actual evaluation
    return {"task": "vqa", "metrics": {"accuracy": 0.0}}


def evaluate_text(model, tokenizer, dataset, max_samples: int, device: str) -> dict:
    """Evaluate on text-only task."""
    # Placeholder - implement actual evaluation
    return {"task": "text", "metrics": {"perplexity": 0.0}}


def run_ablation_experiment(
    model,
    processor,
    df: pd.DataFrame,
    group: str,
    ratio: float,
    total_neurons: int,
    threshold: float,
    eval_tasks: list[str],
    max_eval_samples: int,
    device: str,
    rng: np.random.RandomState,
) -> dict:
    """Run a single ablation experiment."""
    n_neurons = int(total_neurons * ratio)
    neurons = select_neurons(df, group, n_neurons, threshold, rng)
    ablation_plan = build_ablation_plan(neurons, df["layer"].max() + 1)

    hooks = register_ablation_hooks(model, ablation_plan)

    results = {
        "group": group,
        "ratio": ratio,
        "n_neurons": len(neurons),
        "n_layers_affected": len(ablation_plan),
        "evaluations": {},
    }

    try:
        if "caption" in eval_tasks:
            results["evaluations"]["caption"] = evaluate_caption(
                model, processor, None, max_eval_samples, device
            )
        if "vqa" in eval_tasks:
            results["evaluations"]["vqa"] = evaluate_vqa(
                model, processor, None, max_eval_samples, device
            )
        if "text" in eval_tasks:
            results["evaluations"]["text"] = evaluate_text(
                model, processor.tokenizer, None, max_eval_samples, device
            )
    finally:
        for h in hooks:
            h.remove()

    return results


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading neuron scores from {args.neuron_scores}")
    df = load_neuron_scores(args.neuron_scores)

    ratios = [float(r) for r in args.ablation_ratios.split(",")]
    eval_tasks = [t.strip() for t in args.eval_tasks.split(",")]

    groups = ["visual", "text", "multimodal", "unknown", "random", "layer_type_matched_random"]

    total_neurons = len(df)
    print(f"Total neurons: {total_neurons}")
    print(f"Ablation ratios: {ratios}")
    print(f"Eval tasks: {eval_tasks}")
    print(f"Ablation groups: {groups}")

    # Placeholder for model loading and evaluation
    print("\nPhase 2 ablation framework ready.")
    print("Implement model loading and evaluation tasks to run experiments.")

    config = {
        "neuron_scores": args.neuron_scores,
        "ablation_ratios": ratios,
        "eval_tasks": eval_tasks,
        "high_conf_threshold": args.high_conf_threshold,
        "max_eval_samples": args.max_eval_samples,
        "total_neurons": total_neurons,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig saved to {output_dir / 'config.json'}")


if __name__ == "__main__":
    main()
