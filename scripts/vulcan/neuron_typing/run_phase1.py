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

"""End-to-end Phase 1 pipeline: activation collection → scoring → statistics → visualization."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 neuron typing pipeline.")
    parser.add_argument("--config", required=True, help="LlamaFactory SFT YAML config.")
    parser.add_argument("--output_dir", required=True, help="Output directory for all results.")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--t_visual", type=float, default=2.0, help="Visual threshold.")
    parser.add_argument("--t_text", type=float, default=3.0, help="Text threshold.")
    parser.add_argument("--n_visual", type=int, default=4, help="Visual token count threshold.")
    parser.add_argument("--n_text", type=int, default=2, help="Text token count threshold.")
    parser.add_argument("--high_conf_threshold", type=float, default=0.7, help="High-confidence threshold.")
    parser.add_argument("--skip_collection", action="store_true", help="Skip activation collection.")
    parser.add_argument("--skip_stats", action="store_true", help="Skip statistical tests.")
    parser.add_argument("--skip_plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


FA_LAYERS = {3, 7, 11, 15, 19, 23}


def run_command(cmd: list[str], desc: str):
    """Run a subprocess command."""
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with return code {result.returncode}")
        sys.exit(1)


def plot_layer_distribution(df: pd.DataFrame, output_path: Path, threshold: float):
    """Plot per-layer count of 4 neuron types (high-confidence)."""
    layers = sorted(df["layer"].unique())
    counts = {"visual": [], "text": [], "multimodal": [], "unknown": []}

    for layer in layers:
        layer_df = df[df["layer"] == layer]
        counts["visual"].append((layer_df["p_visual"] >= threshold).sum())
        counts["text"].append((layer_df["p_text"] >= threshold).sum())
        counts["multimodal"].append((layer_df["p_multimodal"] >= threshold).sum())
        counts["unknown"].append((layer_df["p_unknown"] >= threshold).sum())

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.2
    x = np.arange(len(layers))

    colors = {"visual": "#2196F3", "text": "#4CAF50", "multimodal": "#FF9800", "unknown": "#9E9E9E"}

    for i, (ntype, color) in enumerate(colors.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, counts[ntype], width, label=ntype, color=color, alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Neuron Count")
    ax.set_title(f"Per-Layer High-Confidence Neuron Counts (threshold={threshold})")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()

    for layer_idx in FA_LAYERS:
        if layer_idx in layers:
            idx = layers.index(layer_idx)
            ax.axvspan(idx - 0.4, idx + 0.4, alpha=0.1, color="red")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path.name}")


def plot_layer_ratio(df: pd.DataFrame, output_path: Path, threshold: float):
    """Plot per-layer ratio of 4 neuron types."""
    layers = sorted(df["layer"].unique())
    ratios = {"visual": [], "text": [], "multimodal": [], "unknown": []}

    for layer in layers:
        layer_df = df[df["layer"] == layer]
        total = len(layer_df)
        ratios["visual"].append((layer_df["p_visual"] >= threshold).sum() / total)
        ratios["text"].append((layer_df["p_text"] >= threshold).sum() / total)
        ratios["multimodal"].append((layer_df["p_multimodal"] >= threshold).sum() / total)
        ratios["unknown"].append((layer_df["p_unknown"] >= threshold).sum() / total)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {"visual": "#2196F3", "text": "#4CAF50", "multimodal": "#FF9800", "unknown": "#9E9E9E"}

    for ntype, color in colors.items():
        ax.plot(layers, ratios[ntype], marker="o", label=ntype, color=color, linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Ratio")
    ax.set_title(f"Per-Layer Neuron Type Ratios (threshold={threshold})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for layer_idx in FA_LAYERS:
        if layer_idx in layers:
            ax.axvline(x=layer_idx, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path.name}")


def plot_scatter(df: pd.DataFrame, output_path: Path):
    """Scatter plot: x=p_visual, y=p_text, color=p_multimodal."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        df["p_visual"],
        df["p_text"],
        c=df["p_multimodal"],
        cmap="YlOrRd",
        alpha=0.3,
        s=10,
    )

    ax.set_xlabel("P(visual)")
    ax.set_ylabel("P(text)")
    ax.set_title("Neuron Type Scores Scatter")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.colorbar(scatter, label="P(multimodal)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path.name}")


def plot_fa_vs_gdn(df: pd.DataFrame, output_path: Path, threshold: float):
    """Bar chart: mean type ratios in FA vs GDN layers with CI error bars."""
    fa_df = df[df["attention_type"] == "FA"]
    gdn_df = df[df["attention_type"] == "GDN"]

    types = ["visual", "text", "multimodal", "unknown"]
    fa_means = []
    gdn_means = []
    fa_stds = []
    gdn_stds = []

    for ntype in types:
        p_col = f"p_{ntype}"
        fa_means.append((fa_df[p_col] >= threshold).mean())
        gdn_means.append((gdn_df[p_col] >= threshold).mean())
        fa_stds.append(fa_df[p_col].std())
        gdn_stds.append(gdn_df[p_col].std())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(types))
    width = 0.35

    ax.bar(x - width/2, fa_means, width, label="FA", color="#2196F3", alpha=0.8, yerr=fa_stds, capsize=5)
    ax.bar(x + width/2, gdn_means, width, label="GDN", color="#4CAF50", alpha=0.8, yerr=gdn_stds, capsize=5)

    ax.set_xlabel("Neuron Type")
    ax.set_ylabel("Ratio")
    ax.set_title(f"FA vs GDN Neuron Type Ratios (threshold={threshold})")
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path.name}")


def plot_threshold_sensitivity(input_dir: Path, output_path: Path):
    """Plot how neuron counts change across threshold sweep."""
    scores_path = input_dir / "neuron_scores.json"
    with open(scores_path) as f:
        scores = json.load(f)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    counts = {"visual": [], "text": [], "multimodal": [], "unknown": []}

    for t in thresholds:
        total_v, total_t, total_m, total_u = 0, 0, 0, 0
        total_neurons = 0
        for layer_key, layer_data in scores.items():
            p_visual = np.array(layer_data["p_visual"])
            p_text = np.array(layer_data["p_text"])
            p_multimodal = np.array(layer_data["p_multimodal"])
            p_unknown = np.array(layer_data["p_unknown"])

            total_neurons += len(p_visual)
            total_v += (p_visual >= t).sum()
            total_t += (p_text >= t).sum()
            total_m += (p_multimodal >= t).sum()
            total_u += (p_unknown >= t).sum()

        counts["visual"].append(total_v)
        counts["text"].append(total_t)
        counts["multimodal"].append(total_m)
        counts["unknown"].append(total_u)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"visual": "#2196F3", "text": "#4CAF50", "multimodal": "#FF9800", "unknown": "#9E9E9E"}

    for ntype, color in colors.items():
        ax.plot(thresholds, counts[ntype], marker="o", label=ntype, color=color, linewidth=2)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Neuron Count")
    ax.set_title("Threshold Sensitivity: High-Confidence Neuron Counts")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path.name}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent

    if not args.skip_collection:
        cmd = [
            sys.executable,
            str(script_dir / "collect_ffn_activations.py"),
            "--config", args.config,
            "--output_dir", str(output_dir / "activations"),
            "--max_samples", str(args.max_samples),
            "--batch_size", str(args.batch_size),
            "--t_visual", str(args.t_visual),
            "--t_text", str(args.t_text),
            "--n_visual", str(args.n_visual),
            "--n_text", str(args.n_text),
        ]
        run_command(cmd, "Activation Collection")

    if not args.skip_collection:
        cmd = [
            sys.executable,
            str(script_dir / "score_neuron_types.py"),
            "--input_dir", str(output_dir / "activations"),
            "--output_dir", str(output_dir / "scores"),
            "--high_conf_threshold", str(args.high_conf_threshold),
        ]
        run_command(cmd, "Neuron Type Scoring")

    if not args.skip_stats:
        cmd = [
            sys.executable,
            str(script_dir / "statistical_tests.py"),
            "--input_dir", str(output_dir / "scores"),
            "--output_dir", str(output_dir / "stats"),
        ]
        run_command(cmd, "Statistical Tests")

    if not args.skip_plots:
        print(f"\n{'='*60}")
        print("Generating Plots")
        print(f"{'='*60}")

        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        parquet_path = output_dir / "scores" / "neuron_type_scores.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)

            plot_layer_distribution(df, plots_dir / "fig_layer_distribution.png", args.high_conf_threshold)
            plot_layer_ratio(df, plots_dir / "fig_layer_ratio.png", args.high_conf_threshold)
            plot_scatter(df, plots_dir / "fig_scatter.png")
            plot_fa_vs_gdn(df, plots_dir / "fig_fa_vs_gdn.png", args.high_conf_threshold)
            plot_threshold_sensitivity(output_dir / "activations", plots_dir / "fig_threshold_sensitivity.png")

    print(f"\n{'='*60}")
    print("Phase 1 Complete!")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nOutputs:")
    print(f"  - activations/global_max.pt")
    print(f"  - activations/neuron_scores.json")
    print(f"  - scores/neuron_type_scores.parquet")
    print(f"  - scores/layer_statistics.json")
    print(f"  - scores/fa_vs_gdn_statistics.json")
    print(f"  - stats/perm_test_results.json")
    print(f"  - plots/fig_*.png")


if __name__ == "__main__":
    main()
