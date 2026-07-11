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

"""Compute neuron type scores from activation collection results.

Reads neuron_scores.json and generates:
- neuron_type_scores.parquet: per-neuron type scores
- Summary statistics and visualizations
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute neuron type scores.")
    parser.add_argument("--input_dir", required=True, help="Directory with activation collection results.")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: input_dir/scores).")
    parser.add_argument("--high_conf_threshold", type=float, default=0.7, help="High-confidence threshold.")
    parser.add_argument("--attention_types", default=None, help="JSON file mapping layer_idx to FA/GDN type.")
    return parser.parse_args()


FA_LAYERS = {3, 7, 11, 15, 19, 23}


def load_neuron_scores(input_dir: str) -> dict:
    scores_path = Path(input_dir) / "neuron_scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"neuron_scores.json not found in {input_dir}")

    with open(scores_path) as f:
        return json.load(f)


def compute_type_scores(scores: dict) -> pd.DataFrame:
    """Convert neuron scores to a DataFrame with type classifications."""
    rows = []
    for layer_key, layer_data in scores.items():
        layer_idx = int(layer_key.split("_")[1])
        p_visual = np.array(layer_data["p_visual"])
        p_text = np.array(layer_data["p_text"])
        p_multimodal = np.array(layer_data["p_multimodal"])
        p_unknown = np.array(layer_data["p_unknown"])
        dead_mask = layer_data.get("dead_mask", [False] * len(p_visual))

        for neuron_idx in range(len(p_visual)):
            is_dead = dead_mask[neuron_idx] if neuron_idx < len(dead_mask) else False
            if is_dead:
                dominant_type = "dead"
            else:
                dominant_type = max(
                    [("visual", p_visual[neuron_idx]),
                     ("text", p_text[neuron_idx]),
                     ("multimodal", p_multimodal[neuron_idx]),
                     ("unknown", p_unknown[neuron_idx])],
                    key=lambda x: x[1]
                )[0]

            rows.append({
                "layer": layer_idx,
                "neuron_idx": neuron_idx,
                "p_visual": p_visual[neuron_idx],
                "p_text": p_text[neuron_idx],
                "p_multimodal": p_multimodal[neuron_idx],
                "p_unknown": p_unknown[neuron_idx],
                "dominant_type": dominant_type,
                "is_dead": is_dead,
                "attention_type": "FA" if layer_idx in FA_LAYERS else "GDN",
            })

    return pd.DataFrame(rows)


def compute_layer_statistics(df: pd.DataFrame, threshold: float) -> dict:
    """Compute per-layer neuron type statistics."""
    stats = {}
    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer]
        high_conf = layer_df[
            (layer_df["p_visual"] >= threshold) |
            (layer_df["p_text"] >= threshold) |
            (layer_df["p_multimodal"] >= threshold) |
            (layer_df["p_unknown"] >= threshold)
        ]

        dead_df = layer_df[layer_df["is_dead"]]
        alive_df = layer_df[~layer_df["is_dead"]]

        stats[int(layer)] = {
            "total_neurons": len(layer_df),
            "dead_neurons": int(layer_df["is_dead"].sum()),
            "attention_type": layer_df["attention_type"].iloc[0],
            "type_counts": {
                "visual": int((alive_df["dominant_type"] == "visual").sum()),
                "text": int((alive_df["dominant_type"] == "text").sum()),
                "multimodal": int((alive_df["dominant_type"] == "multimodal").sum()),
                "unknown": int((alive_df["dominant_type"] == "unknown").sum()),
                "dead": int(layer_df["is_dead"].sum()),
            },
            "high_conf_counts": {
                "visual": int((high_conf["p_visual"] >= threshold).sum()),
                "text": int((high_conf["p_text"] >= threshold).sum()),
                "multimodal": int((high_conf["p_multimodal"] >= threshold).sum()),
                "unknown": int((high_conf["p_unknown"] >= threshold).sum()),
            },
            "mean_scores": {
                "p_visual": float(layer_df["p_visual"].mean()),
                "p_text": float(layer_df["p_text"].mean()),
                "p_multimodal": float(layer_df["p_multimodal"].mean()),
                "p_unknown": float(layer_df["p_unknown"].mean()),
            },
        }

    return stats


def compute_fa_vs_gdn_statistics(df: pd.DataFrame, threshold: float) -> dict:
    """Compare neuron type distributions between FA and GDN layers."""
    fa_df = df[df["attention_type"] == "FA"]
    gdn_df = df[df["attention_type"] == "GDN"]

    stats = {}
    for neuron_type in ["visual", "text", "multimodal", "unknown"]:
        p_col = f"p_{neuron_type}"
        fa_high_conf = (fa_df[p_col] >= threshold).sum()
        gdn_high_conf = (gdn_df[p_col] >= threshold).sum()

        stats[neuron_type] = {
            "fa_count": int(fa_high_conf),
            "gdn_count": int(gdn_high_conf),
            "fa_ratio": float(fa_high_conf / len(fa_df)) if len(fa_df) > 0 else 0,
            "gdn_ratio": float(gdn_high_conf / len(gdn_df)) if len(gdn_df) > 0 else 0,
            "fa_mean": float(fa_df[p_col].mean()),
            "gdn_mean": float(gdn_df[p_col].mean()),
        }

    return stats


def save_results(
    output_dir: str,
    df: pd.DataFrame,
    layer_stats: dict,
    fa_gdn_stats: dict,
    config: dict,
):
    """Save neuron type scores and statistics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_path = output_path / "neuron_type_scores.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved neuron_type_scores.parquet ({len(df)} neurons)")

    layer_stats_path = output_path / "layer_statistics.json"
    with open(layer_stats_path, "w") as f:
        json.dump(layer_stats, f, indent=2)
    print(f"Saved layer_statistics.json")

    fa_gdn_path = output_path / "fa_vs_gdn_statistics.json"
    with open(fa_gdn_path, "w") as f:
        json.dump(fa_gdn_stats, f, indent=2)
    print(f"Saved fa_vs_gdn_statistics.json")

    config_path = output_path / "scoring_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def print_summary(df: pd.DataFrame, layer_stats: dict, fa_gdn_stats: dict):
    """Print a summary of neuron type scores."""
    print(f"\n{'='*60}")
    print("Neuron Type Score Summary")
    print(f"{'='*60}")

    total = len(df)
    n_dead = df["is_dead"].sum()
    print(f"\nTotal neurons: {total}")
    print(f"Dead neurons (global_max <= 1e-6): {n_dead} ({100*n_dead/total:.1f}%)")
    print(f"Type distribution (dominant, alive neurons only):")
    alive_df = df[~df["is_dead"]]
    for ntype in ["visual", "text", "multimodal", "unknown"]:
        count = (alive_df["dominant_type"] == ntype).sum()
        print(f"  {ntype:12s}: {count:6d} ({100*count/total:.1f}%)")

    print(f"\nFA vs GDN comparison:")
    for ntype, stats in fa_gdn_stats.items():
        print(f"  {ntype:12s}:")
        print(f"    FA  ratio: {stats['fa_ratio']:.3f}  (mean p: {stats['fa_mean']:.3f})")
        print(f"    GDN ratio: {stats['gdn_ratio']:.3f}  (mean p: {stats['gdn_mean']:.3f})")
        print(f"    Difference: {stats['fa_ratio'] - stats['gdn_ratio']:+.3f}")


def main():
    args = parse_args()
    output_dir = args.output_dir or str(Path(args.input_dir) / "scores")

    print(f"Loading neuron scores from {args.input_dir}")
    scores = load_neuron_scores(args.input_dir)

    print(f"Computing type scores...")
    df = compute_type_scores(scores)

    print(f"Computing layer statistics...")
    layer_stats = compute_layer_statistics(df, args.high_conf_threshold)

    print(f"Computing FA vs GDN statistics...")
    fa_gdn_stats = compute_fa_vs_gdn_statistics(df, args.high_conf_threshold)

    config = {
        "input_dir": args.input_dir,
        "high_conf_threshold": args.high_conf_threshold,
        "num_layers": len(layer_stats),
        "num_neurons": len(df),
    }

    save_results(output_dir, df, layer_stats, fa_gdn_stats, config)
    print_summary(df, layer_stats, fa_gdn_stats)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
