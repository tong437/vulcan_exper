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

"""End-to-end Phase 1 pipeline with calibration + typing.

Pipeline:
1. Pilot: Collect global_max (Pass 1 only)
2. Calibration: Estimate per-neuron quantile thresholds
3. Typing: Classify neurons using calibrated thresholds
4. Scoring: Compute type scores and statistics
5. Visualization: Generate diagnostic plots
"""

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
    parser.add_argument("--calibration_samples", type=int, default=500,
                        help="Samples for calibration (threshold estimation).")
    parser.add_argument("--typing_samples", type=int, default=5000,
                        help="Samples for typing (neuron classification).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--threshold_mode", choices=["fixed", "quantile"], default="quantile",
                        help="Threshold mode.")
    parser.add_argument("--quantiles", default="0.95,0.97,0.99",
                        help="Quantiles to compute for calibration.")
    parser.add_argument("--quantile_idx_visual", type=int, default=1,
                        help="Quantile index for visual threshold (0=q95, 1=q97, 2=q99). Default: 1 (q97).")
    parser.add_argument("--quantile_idx_text", type=int, default=0,
                        help="Quantile index for text threshold (0=q95, 1=q97, 2=q99). Default: 0 (q95).")
    parser.add_argument("--quantile_idx", type=int, default=None,
                        help="Fallback: set both visual and text quantile index (0=q95, 1=q97, 2=q99).")
    parser.add_argument("--visual_ratio", type=float, default=0.005,
                        help="Min ratio of visual tokens that must exceed threshold.")
    parser.add_argument("--visual_min_count", type=int, default=4,
                        help="Min absolute visual token count.")
    parser.add_argument("--text_ratio", type=float, default=0.10,
                        help="Min ratio of text tokens that must exceed threshold.")
    parser.add_argument("--text_min_count", type=int, default=2,
                        help="Min absolute text token count.")
    parser.add_argument("--high_conf_threshold", type=float, default=0.7,
                        help="High-confidence threshold for type assignment.")
    parser.add_argument("--skip_calibration", action="store_true",
                        help="Skip calibration (use existing neuron_quantiles.pt).")
    parser.add_argument("--skip_typing", action="store_true",
                        help="Skip typing (use existing neuron_scores.json).")
    parser.add_argument("--skip_stats", action="store_true", help="Skip statistical tests.")
    parser.add_argument("--skip_plots", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()
    # Handle --quantile_idx fallback: only override if explicitly provided
    if args.quantile_idx is not None:
        args.quantile_idx_visual = args.quantile_idx
        args.quantile_idx_text = args.quantile_idx
    return args


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
        counts["visual"].append((layer_df["q_visual"] >= threshold).sum())
        counts["text"].append((layer_df["q_text"] >= threshold).sum())
        counts["multimodal"].append((layer_df["q_multimodal"] >= threshold).sum())
        counts["unknown"].append((layer_df["q_unknown"] >= threshold).sum())

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.2
    x = np.arange(len(layers))

    colors = {"visual": "#2196F3", "text": "#4CAF50", "multimodal": "#FF9800", "unknown": "#9E9E9E"}

    for i, (ntype, color) in enumerate(colors.items()):
        offset = (i - 1.5) * width
        ax.bar(x + offset, counts[ntype], width, label=ntype, color=color, alpha=0.8)

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
        # Exclude dead neurons from denominator
        layer_df = df[(df["layer"] == layer) & (~df["is_dead"])]
        total = len(layer_df)
        if total == 0:
            ratios["visual"].append(0.0)
            ratios["text"].append(0.0)
            ratios["multimodal"].append(0.0)
            ratios["unknown"].append(0.0)
            continue
        ratios["visual"].append((layer_df["q_visual"] >= threshold).sum() / total)
        ratios["text"].append((layer_df["q_text"] >= threshold).sum() / total)
        ratios["multimodal"].append((layer_df["q_multimodal"] >= threshold).sum() / total)
        ratios["unknown"].append((layer_df["q_unknown"] >= threshold).sum() / total)

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
    """Scatter plot: x=q_visual, y=q_text, color=q_multimodal."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        df["q_visual"],
        df["q_text"],
        c=df["q_multimodal"],
        cmap="YlOrRd",
        alpha=0.3,
        s=10,
    )

    ax.set_xlabel("q(visual)")
    ax.set_ylabel("q(text)")
    ax.set_title("Neuron Type Purity Scores Scatter")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.colorbar(scatter, label="q(multimodal)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path.name}")


def plot_fa_vs_gdn(df: pd.DataFrame, output_path: Path, threshold: float):
    """Bar chart: mean type ratios in FA vs GDN layers with CI error bars (alive neurons only)."""
    # Exclude dead neurons from FA/GDN comparison
    fa_df = df[(df["attention_type"] == "FA") & (~df["is_dead"])]
    gdn_df = df[(df["attention_type"] == "GDN") & (~df["is_dead"])]

    types = ["visual", "text", "multimodal", "unknown"]
    fa_means = []
    gdn_means = []
    fa_stds = []
    gdn_stds = []

    for ntype in types:
        q_col = f"q_{ntype}"
        fa_means.append((fa_df[q_col] >= threshold).mean())
        gdn_means.append((gdn_df[q_col] >= threshold).mean())
        fa_stds.append(fa_df[q_col].std())
        gdn_stds.append(gdn_df[q_col].std())

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
            # Use dtype=float to handle null values (converted from NaN)
            q_visual = np.asarray(layer_data["q_visual"], dtype=float)
            q_text = np.asarray(layer_data["q_text"], dtype=float)
            q_multimodal = np.asarray(layer_data["q_multimodal"], dtype=float)
            q_unknown = np.asarray(layer_data["q_unknown"], dtype=float)

            total_neurons += len(q_visual)
            total_v += (q_visual >= t).sum()
            total_t += (q_text >= t).sum()
            total_m += (q_multimodal >= t).sum()
            total_u += (q_unknown >= t).sum()

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


def plot_quantile_sensitivity(scores_dir: Path, output_path: Path):
    """Plot type distribution across different quantile thresholds.

    Expects multiple neuron_scores files from different quantile runs.
    """
    quantile_files = sorted(scores_dir.glob("neuron_scores_q*.json"))
    if not quantile_files:
        print("No quantile-specific scores found, skipping quantile sensitivity plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    quantile_labels = []
    type_counts = {"visual": [], "text": [], "multimodal": [], "unknown": []}

    for qf in quantile_files:
        with open(qf) as f:
            scores = json.load(f)

        q_label = qf.stem.split("_")[-1]
        quantile_labels.append(q_label)

        total_v, total_t, total_m, total_u = 0, 0, 0, 0
        total_neurons = 0
        for layer_key, layer_data in scores.items():
            # Use dtype=float to handle null values (converted from NaN)
            q_visual = np.asarray(layer_data["q_visual"], dtype=float)
            q_text = np.asarray(layer_data["q_text"], dtype=float)
            q_multimodal = np.asarray(layer_data["q_multimodal"], dtype=float)
            q_unknown = np.asarray(layer_data["q_unknown"], dtype=float)

            total_neurons += len(q_visual)
            total_v += (q_visual >= 0.7).sum()
            total_t += (q_text >= 0.7).sum()
            total_m += (q_multimodal >= 0.7).sum()
            total_u += (q_unknown >= 0.7).sum()

        type_counts["visual"].append(total_v)
        type_counts["text"].append(total_t)
        type_counts["multimodal"].append(total_m)
        type_counts["unknown"].append(total_u)

    x = np.arange(len(quantile_labels))
    width = 0.2
    colors = {"visual": "#2196F3", "text": "#4CAF50", "multimodal": "#FF9800", "unknown": "#9E9E9E"}

    for i, (ntype, color) in enumerate(colors.items()):
        offset = (i - 1.5) * width
        ax.bar(x + offset, type_counts[ntype], width, label=ntype, color=color, alpha=0.8)

    ax.set_xlabel("Quantile")
    ax.set_ylabel("Neuron Count")
    ax.set_title("Quantile Sensitivity: High-Confidence Neuron Counts (p>=0.7)")
    ax.set_xticks(x)
    ax.set_xticklabels(quantile_labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path.name}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent

    print(f"\n{'='*60}")
    print("Phase 1 Pipeline: Neuron Typing")
    print(f"{'='*60}")
    print(f"Mode: {args.threshold_mode}")
    print(f"Calibration samples: {args.calibration_samples}")
    print(f"Typing samples: {args.typing_samples}")

    calibration_dir = output_dir / "calibration"
    activations_dir = output_dir / "activations"

    if not args.skip_calibration:
        print(f"\n{'='*60}")
        print("Step 1: Pilot (Global Max Collection)")
        print(f"{'='*60}")
        cmd = [
            sys.executable,
            str(script_dir / "collect_ffn_activations.py"),
            "--config", args.config,
            "--output_dir", str(activations_dir),
            "--max_samples", str(args.calibration_samples),
            "--batch_size", str(args.batch_size),
            "--pilot",
        ]
        run_command(cmd, "Pilot - Global Max Collection")

        print(f"\n{'='*60}")
        print("Step 2: Calibration (Quantile Thresholds)")
        print(f"{'='*60}")
        cmd = [
            sys.executable,
            str(script_dir / "calibrate_thresholds.py"),
            "--config", args.config,
            "--output_dir", str(calibration_dir),
            "--max_samples", str(args.calibration_samples),
            "--batch_size", str(args.batch_size),
            "--quantiles", args.quantiles,
        ]
        run_command(cmd, "Calibration - Quantile Thresholds")

    if not args.skip_typing:
        print(f"\n{'='*60}")
        print("Step 3: Typing (Neuron Classification)")
        print(f"{'='*60}")

        quantile_path = str(calibration_dir / "neuron_quantiles.pt")

        cmd = [
            sys.executable,
            str(script_dir / "collect_ffn_activations.py"),
            "--config", args.config,
            "--output_dir", str(activations_dir),
            "--max_samples", str(args.typing_samples),
            "--batch_size", str(args.batch_size),
            "--threshold_mode", args.threshold_mode,
            "--quantile_path", quantile_path,
            "--quantile_idx_visual", str(args.quantile_idx_visual),
            "--quantile_idx_text", str(args.quantile_idx_text),
            "--visual_ratio", str(args.visual_ratio),
            "--visual_min_count", str(args.visual_min_count),
            "--text_ratio", str(args.text_ratio),
            "--text_min_count", str(args.text_min_count),
        ]
        run_command(cmd, "Typing - Neuron Classification")

    print(f"\n{'='*60}")
    print("Step 4: Scoring")
    print(f"{'='*60}")
    cmd = [
        sys.executable,
        str(script_dir / "score_neuron_types.py"),
        "--input_dir", str(activations_dir),
        "--output_dir", str(output_dir / "scores"),
        "--high_conf_threshold", str(args.high_conf_threshold),
    ]
    run_command(cmd, "Neuron Type Scoring")

    if not args.skip_stats:
        print(f"\n{'='*60}")
        print("Step 5: Statistical Tests")
        print(f"{'='*60}")
        cmd = [
            sys.executable,
            str(script_dir / "statistical_tests.py"),
            "--input_dir", str(output_dir / "scores"),
            "--output_dir", str(output_dir / "stats"),
        ]
        run_command(cmd, "Statistical Tests")

    if not args.skip_plots:
        print(f"\n{'='*60}")
        print("Step 6: Visualization")
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
            plot_threshold_sensitivity(activations_dir, plots_dir / "fig_threshold_sensitivity.png")
            plot_quantile_sensitivity(activations_dir, plots_dir / "fig_quantile_sensitivity.png")

    print(f"\n{'='*60}")
    print("Phase 1 Complete!")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print("\nOutputs:")
    print("  - calibration/global_max.pt")
    print("  - calibration/neuron_quantiles.pt")
    print("  - activations/global_max.pt")
    print("  - activations/neuron_scores.json")
    print("  - scores/neuron_type_scores.parquet")
    print("  - scores/layer_statistics.json")
    print("  - scores/fa_vs_gdn_statistics.json")
    print("  - stats/perm_test_results.json")
    print("  - plots/fig_*.png")


if __name__ == "__main__":
    main()
