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

"""Compute pruning scores based on neuron type analysis.

Prune score = p_unknown - λ_m * p_multimodal - λ_v * p_visual - λ_t * p_text
Higher score = safer to prune.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute pruning scores.")
    parser.add_argument("--neuron_scores", required=True, help="Path to neuron_type_scores.parquet.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--lambda_multimodal", type=float, default=1.0, help="Multimodal protection weight.")
    parser.add_argument("--lambda_visual", type=float, default=0.3, help="Visual protection weight.")
    parser.add_argument("--lambda_text", type=float, default=0.3, help="Text protection weight.")
    parser.add_argument("--simplified", action="store_true", help="Use simplified formula: p_unknown - p_multimodal.")
    return parser.parse_args()


def compute_pruning_scores(
    df: pd.DataFrame,
    lambda_multimodal: float = 1.0,
    lambda_visual: float = 0.3,
    lambda_text: float = 0.3,
    simplified: bool = False,
) -> pd.DataFrame:
    """Compute pruning scores for each neuron.

    Args:
        df: DataFrame with neuron type scores
        lambda_multimodal: Weight for multimodal protection
        lambda_visual: Weight for visual protection
        lambda_text: Weight for text protection
        simplified: Use simplified formula

    Returns:
        DataFrame with pruning scores added
    """
    result = df.copy()

    if simplified:
        result["prune_score"] = result["p_unknown"] - result["p_multimodal"]
    else:
        result["prune_score"] = (
            result["p_unknown"]
            - lambda_multimodal * result["p_multimodal"]
            - lambda_visual * result["p_visual"]
            - lambda_text * result["p_text"]
        )

    return result


def generate_pruning_mask(
    df: pd.DataFrame,
    ratio: float,
    per_layer: bool = True,
) -> dict[int, list[int]]:
    """Generate pruning mask based on prune scores.

    Args:
        df: DataFrame with prune_score column
        ratio: Pruning ratio (0.0 to 1.0)
        per_layer: If True, prune top-r% within each layer

    Returns:
        dict mapping layer_idx to list of neuron indices to prune
    """
    mask = {}

    if per_layer:
        for layer in sorted(df["layer"].unique()):
            layer_df = df[df["layer"] == layer]
            n_prune = max(1, int(len(layer_df) * ratio))
            prunable = layer_df.nlargest(n_prune, "prune_score")
            mask[int(layer)] = prunable["neuron_idx"].tolist()
    else:
        n_prune = max(1, int(len(df) * ratio))
        prunable = df.nlargest(n_prune, "prune_score")
        for _, row in prunable.iterrows():
            layer = int(row["layer"])
            if layer not in mask:
                mask[layer] = []
            mask[layer].append(int(row["neuron_idx"]))

    return mask


def generate_baseline_masks(
    df: pd.DataFrame,
    ratio: float,
    num_layers: int,
    seed: int = 42,
) -> dict[str, dict[int, list[int]]]:
    """Generate baseline pruning masks for comparison.

    Returns:
        dict mapping method name to pruning mask
    """
    rng = np.random.RandomState(seed)
    masks = {}

    n_prune = max(1, int(len(df) * ratio))

    random_neurons = df.sample(n=n_prune, random_state=rng)
    masks["random"] = {}
    for _, row in random_neurons.iterrows():
        layer = int(row["layer"])
        if layer not in masks["random"]:
            masks["random"][layer] = []
        masks["random"][layer].append(int(row["neuron_idx"]))

    fa_df = df[df["attention_type"] == "FA"]
    gdn_df = df[df["attention_type"] == "GDN"]
    fa_ratio = len(fa_df) / len(df)
    n_fa = int(n_prune * fa_ratio)
    n_gdn = n_prune - n_fa

    fa_sample = fa_df.sample(n=min(n_fa, len(fa_df)), random_state=rng)
    gdn_sample = gdn_df.sample(n=min(n_gdn, len(gdn_df)), random_state=rng)
    matched = pd.concat([fa_sample, gdn_sample])
    masks["layer_type_matched_random"] = {}
    for _, row in matched.iterrows():
        layer = int(row["layer"])
        if layer not in masks["layer_type_matched_random"]:
            masks["layer_type_matched_random"][layer] = []
        masks["layer_type_matched_random"][layer].append(int(row["neuron_idx"]))

    masks["magnitude"] = {}
    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer]
        n_layer_prune = max(1, int(len(layer_df) * ratio))
        prunable = layer_df.nsmallest(n_layer_prune, "p_visual")
        masks["magnitude"][int(layer)] = prunable["neuron_idx"].tolist()

    masks["unknown_score"] = {}
    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer]
        n_layer_prune = max(1, int(len(layer_df) * ratio))
        prunable = layer_df.nlargest(n_layer_prune, "p_unknown")
        masks["unknown_score"][int(layer)] = prunable["neuron_idx"].tolist()

    return masks


def save_results(
    output_dir: str,
    df: pd.DataFrame,
    masks: dict[str, dict[int, list[int]]],
    config: dict,
):
    """Save pruning scores and masks."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scores_path = output_path / "pruning_scores.parquet"
    df.to_parquet(scores_path, index=False)
    print(f"Saved pruning_scores.parquet ({len(df)} neurons)")

    masks_dir = output_path / "masks"
    masks_dir.mkdir(exist_ok=True)
    for method, mask in masks.items():
        mask_path = masks_dir / f"{method}.json"
        with open(mask_path, "w") as f:
            json.dump({str(k): v for k, v in mask.items()}, f, indent=2)
        print(f"Saved mask: {method}")

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("Pruning Score Summary")
    print(f"{'='*60}")

    print(f"\nTotal neurons: {len(df)}")
    print(f"Prune score stats:")
    print(f"  Mean:   {df['prune_score'].mean():.4f}")
    print(f"  Std:    {df['prune_score'].std():.4f}")
    print(f"  Min:    {df['prune_score'].min():.4f}")
    print(f"  Max:    {df['prune_score'].max():.4f}")

    print(f"\nTop 10 safest to prune (highest score):")
    top10 = df.nlargest(10, "prune_score")
    for _, row in top10.iterrows():
        print(f"  Layer {int(row['layer']):2d}, Neuron {int(row['neuron_idx']):4d}: "
              f"score={row['prune_score']:.4f} (unknown={row['p_unknown']:.3f}, multimodal={row['p_multimodal']:.3f})")

    print(f"\nTop 10 most important to keep (lowest score):")
    bottom10 = df.nsmallest(10, "prune_score")
    for _, row in bottom10.iterrows():
        print(f"  Layer {int(row['layer']):2d}, Neuron {int(row['neuron_idx']):4d}: "
              f"score={row['prune_score']:.4f} (unknown={row['p_unknown']:.3f}, multimodal={row['p_multimodal']:.3f})")


def main():
    args = parse_args()

    print(f"Loading neuron scores from {args.neuron_scores}")
    df = pd.read_parquet(args.neuron_scores)

    print(f"Computing pruning scores (simplified={args.simplified})...")
    df = compute_pruning_scores(
        df,
        lambda_multimodal=args.lambda_multimodal,
        lambda_visual=args.lambda_visual,
        lambda_text=args.lambda_text,
        simplified=args.simplified,
    )

    num_layers = df["layer"].max() + 1
    ratios = [0.05, 0.10, 0.15, 0.20, 0.30]

    all_masks = {}
    for ratio in ratios:
        print(f"\nGenerating masks for ratio={ratio}...")
        type_aware_mask = generate_pruning_mask(df, ratio, per_layer=True)
        all_masks[f"type_aware_{ratio}"] = type_aware_mask

        baseline_masks = generate_baseline_masks(df, ratio, num_layers)
        for method, mask in baseline_masks.items():
            all_masks[f"{method}_{ratio}"] = mask

    config = {
        "neuron_scores": args.neuron_scores,
        "lambda_multimodal": args.lambda_multimodal,
        "lambda_visual": args.lambda_visual,
        "lambda_text": args.lambda_text,
        "simplified": args.simplified,
        "ratios": ratios,
        "num_layers": num_layers,
    }

    save_results(args.output_dir, df, all_masks, config)
    print_summary(df)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
