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


def compute_type_scores(scores: dict, high_conf_threshold: float = 0.7) -> pd.DataFrame:
    """Convert neuron scores to a DataFrame with type classifications."""
    rows = []
    for layer_key, layer_data in scores.items():
        layer_idx = int(layer_key.split("_")[1])
        q_visual = np.array(layer_data["q_visual"], dtype=float)
        q_text = np.array(layer_data["q_text"], dtype=float)
        q_multimodal = np.array(layer_data["q_multimodal"], dtype=float)
        q_unknown = np.array(layer_data["q_unknown"], dtype=float)
        r_visual = np.array(layer_data["r_visual"], dtype=float)
        r_text = np.array(layer_data["r_text"], dtype=float)
        r_multimodal = np.array(layer_data["r_multimodal"], dtype=float)
        r_unknown = np.array(layer_data["r_unknown"], dtype=float)
        dead_mask = np.asarray(layer_data.get("dead_mask", [False] * len(q_visual)), dtype=bool)

        for neuron_idx in range(len(q_visual)):
            is_dead = dead_mask[neuron_idx] if neuron_idx < len(dead_mask) else False
            dominant_tie = False
            if is_dead:
                dominant_type = "dead"
            else:
                type_names = np.array(["visual", "text", "multimodal", "unknown"])
                q_array = np.array([
                    q_visual[neuron_idx], q_text[neuron_idx],
                    q_multimodal[neuron_idx], q_unknown[neuron_idx],
                ])
                r_array = np.array([
                    r_visual[neuron_idx], r_text[neuron_idx],
                    r_multimodal[neuron_idx], r_unknown[neuron_idx],
                ])
                finite = np.isfinite(q_array)
                if finite.any():
                    max_value = np.nanmax(q_array)
                    candidates = np.flatnonzero(finite & np.isclose(q_array, max_value, rtol=0.0, atol=1e-12))
                    dominant_tie = len(candidates) > 1
                    # Resolve q ties by full-dataset frequency, then stable type order.
                    candidate_r = np.nan_to_num(r_array[candidates], nan=-np.inf)
                    winner = candidates[int(np.argmax(candidate_r))]
                    dominant_type = str(type_names[winner])
                else:
                    dominant_type = "unknown"

            # Compute top-two q margin
            q_values = [q_visual[neuron_idx], q_text[neuron_idx], q_multimodal[neuron_idx], q_unknown[neuron_idx]]
            # Handle None values (from JSON null) and NaN
            valid_q_values = []
            for v in q_values:
                if v is None:
                    continue
                try:
                    if not np.isnan(v):
                        valid_q_values.append(v)
                except (TypeError, ValueError):
                    continue
            if len(valid_q_values) >= 2:
                sorted_q = sorted(valid_q_values, reverse=True)
                q_1 = sorted_q[0]
                q_2 = sorted_q[1]
                q_margin = q_1 - q_2
                exact_tie = (q_margin == 0)
                low_margin_05 = (q_margin < 0.05)
                low_margin_10 = (q_margin < 0.10)
            else:
                q_1 = valid_q_values[0] if valid_q_values else np.nan
                q_2 = np.nan
                q_margin = np.nan
                exact_tie = False
                low_margin_05 = False
                low_margin_10 = False

            # Confidence category
            max_q = max(valid_q_values) if valid_q_values else np.nan
            if is_dead:
                confidence_category = "dead"
            elif max_q >= high_conf_threshold:
                confidence_category = "high_confidence"
            else:
                confidence_category = "mixed_low_confidence"

            rows.append({
                "layer": layer_idx,
                "neuron_idx": neuron_idx,
                "q_visual": q_visual[neuron_idx],
                "q_text": q_text[neuron_idx],
                "q_multimodal": q_multimodal[neuron_idx],
                "q_unknown": q_unknown[neuron_idx],
                "r_visual": r_visual[neuron_idx],
                "r_text": r_text[neuron_idx],
                "r_multimodal": r_multimodal[neuron_idx],
                "r_unknown": r_unknown[neuron_idx],
                "dominant_type": dominant_type,
                "dominant_tie": dominant_tie,
                "is_dead": is_dead,
                "attention_type": "FA" if layer_idx in FA_LAYERS else "GDN",
                "q_1": q_1,
                "q_2": q_2,
                "q_margin": q_margin,
                "exact_tie": exact_tie,
                "low_margin_05": low_margin_05,
                "low_margin_10": low_margin_10,
                "max_q": max_q,
                "confidence_category": confidence_category,
            })

    return pd.DataFrame(rows)


def compute_layer_statistics(df: pd.DataFrame, threshold: float) -> dict:
    """Compute per-layer neuron type statistics."""
    stats = {}
    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer]
        # Use q_* for high-confidence thresholding
        high_conf = layer_df[
            (layer_df["q_visual"] >= threshold) |
            (layer_df["q_text"] >= threshold) |
            (layer_df["q_multimodal"] >= threshold) |
            (layer_df["q_unknown"] >= threshold)
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
                "visual": int((high_conf["q_visual"] >= threshold).sum()),
                "text": int((high_conf["q_text"] >= threshold).sum()),
                "multimodal": int((high_conf["q_multimodal"] >= threshold).sum()),
                "unknown": int((high_conf["q_unknown"] >= threshold).sum()),
            },
            "mean_scores": {
                "q_visual": float(layer_df["q_visual"].mean()),
                "q_text": float(layer_df["q_text"].mean()),
                "q_multimodal": float(layer_df["q_multimodal"].mean()),
                "q_unknown": float(layer_df["q_unknown"].mean()),
                "r_visual": float(layer_df["r_visual"].mean()),
                "r_text": float(layer_df["r_text"].mean()),
                "r_multimodal": float(layer_df["r_multimodal"].mean()),
                "r_unknown": float(layer_df["r_unknown"].mean()),
            },
        }

    return stats


def compute_fa_vs_gdn_statistics(df: pd.DataFrame, threshold: float) -> dict:
    """Compare neuron type distributions between FA and GDN layers (alive neurons only)."""
    # Exclude dead neurons from FA/GDN comparison
    fa_df = df[(df["attention_type"] == "FA") & (~df["is_dead"])]
    gdn_df = df[(df["attention_type"] == "GDN") & (~df["is_dead"])]

    stats = {}
    for neuron_type in ["visual", "text", "multimodal", "unknown"]:
        q_col = f"q_{neuron_type}"
        fa_high_conf = (fa_df[q_col] >= threshold).sum()
        gdn_high_conf = (gdn_df[q_col] >= threshold).sum()

        stats[neuron_type] = {
            "fa_count": int(fa_high_conf),
            "gdn_count": int(gdn_high_conf),
            "fa_ratio": float(fa_high_conf / len(fa_df)) if len(fa_df) > 0 else 0,
            "gdn_ratio": float(gdn_high_conf / len(gdn_df)) if len(gdn_df) > 0 else 0,
            "fa_mean": float(fa_df[q_col].mean()),
            "gdn_mean": float(gdn_df[q_col].mean()),
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
    print("Saved layer_statistics.json")

    fa_gdn_path = output_path / "fa_vs_gdn_statistics.json"
    with open(fa_gdn_path, "w") as f:
        json.dump(fa_gdn_stats, f, indent=2)
    print("Saved fa_vs_gdn_statistics.json")

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
    print("Type distribution (dominant, alive neurons only):")
    alive_df = df[~df["is_dead"]]
    for ntype in ["visual", "text", "multimodal", "unknown"]:
        count = (alive_df["dominant_type"] == ntype).sum()
        print(f"  {ntype:12s}: {count:6d} ({100*count/total:.1f}%)")

    print("\nFA vs GDN comparison:")
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

    print("Computing type scores...")
    df = compute_type_scores(scores, args.high_conf_threshold)

    print("Computing layer statistics...")
    layer_stats = compute_layer_statistics(df, args.high_conf_threshold)

    print("Computing FA vs GDN statistics...")
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
