#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate and report the measured A2 epsilon sweep over all 16 frozen runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/vulcan_epsilon_report_mpl")

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_2b_instruct": "Qwen3-VL-2B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "gemma_4_e2b_it": "Gemma-4-E2B-it",
}
DATASETS = {"mme": "MME", "mmbench": "MMBench", "textvqa": "TextVQA", "chartqa": "ChartQA"}
METRICS = {"mme": "Accuracy", "mmbench": "Accuracy", "textvqa": "Soft accuracy", "chartqa": "Relaxed accuracy"}
EPSILONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _read_run(directory: Path, model: str, dataset: str, epsilon: float) -> tuple[dict, pd.DataFrame]:
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (config["model"], config["dataset"]) != (model, dataset) or not np.isclose(config["epsilon"], epsilon):
        raise ValueError(f"Incorrect model, dataset, or epsilon in {directory}")
    if summary["expected_samples"] != 1000 or not np.isclose(summary["epsilon"], epsilon):
        raise ValueError(f"Incomplete summary: {directory}")
    expected_conditions = {"baseline", "A1", "A2"} if epsilon == 0.05 else {"baseline", "A2"}
    if epsilon != 0.05 and config.get("methods") != ["A2"]:
        raise ValueError(f"New sweep must contain A2 only: {directory}")
    frame = pd.read_json(directory / "predictions.jsonl", lines=True)
    if len(frame) != 1000 * len(expected_conditions) or set(frame["condition"]) != expected_conditions:
        raise ValueError(f"Incorrect prediction matrix: {directory}")
    if frame.duplicated(["sample_index", "condition"]).any():
        raise ValueError(f"Duplicate sample-condition rows: {directory}")
    for condition in expected_conditions:
        subset = frame.loc[frame["condition"] == condition].sort_values("sample_index")
        if not np.array_equal(subset["sample_index"].to_numpy(), np.arange(1000)):
            raise ValueError(f"Missing sample indices for {condition}: {directory}")
    if not frame["score"].between(0, 1).all() or not frame["actual_ratio"].between(0, 1).all():
        raise ValueError(f"Invalid score or actual pruning ratio: {directory}")
    if not frame["actual_score_share"].between(0, 1).all():
        raise ValueError(f"Invalid cumulative response-score share: {directory}")
    if {row["condition"] for row in summary["conditions"]} != expected_conditions:
        raise ValueError(f"Summary condition set differs from predictions: {directory}")
    for row in summary["conditions"]:
        subset = frame.loc[frame["condition"] == row["condition"]]
        for column, summary_key in (
            ("score", "score"),
            ("actual_ratio", "mean_actual_ratio"),
            ("actual_score_share", "mean_actual_score_share"),
        ):
            if not np.isclose(subset[column].mean(), row[summary_key], rtol=0, atol=1e-9):
                raise ValueError(f"Summary {summary_key} differs from predictions: {directory}")
    ablated = frame.loc[frame["condition"] == "A2"]
    if (ablated["actual_score_share"] > epsilon + 1e-10).any():
        raise ValueError(f"A2 exceeds its response-score budget: {directory}")
    if not np.allclose(ablated["actual_ratio"], ablated["selected_neurons"] / summary["total_neurons"]):
        raise ValueError(f"Actual pruning rate disagrees with neuron counts: {directory}")
    return config, frame


def _paired_group_ci(drops_pp: np.ndarray, groups: np.ndarray, seed: int, repeats: int) -> tuple[float, float]:
    _, group_index = np.unique(groups.astype(str), return_inverse=True)
    group_count = group_index.max() + 1
    if group_count < 2:
        raise ValueError("Paired bootstrap requires at least two independent groups.")
    sizes = np.bincount(group_index, minlength=group_count)
    sums = np.bincount(group_index, weights=drops_pp, minlength=group_count)
    random = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        selected = random.integers(0, group_count, size=group_count)
        draws[repeat] = sums[selected].sum() / sizes[selected].sum()
    low, high = np.percentile(draws, (2.5, 97.5))
    return float(low), float(high)


def _plot_one(frame: pd.DataFrame, model: str, dataset: str, output: Path) -> None:
    epsilons = frame["epsilon"].to_numpy()
    scores = frame["a2_score_percent"].to_numpy()
    baseline = float(frame["baseline_score_percent"].iloc[0])
    actual = frame["mean_actual_pruning_percent"].to_numpy()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(9.0, 6.4))
    grid = fig.add_gridspec(2, 1, height_ratios=(4.0, 0.9), hspace=0.60)
    ax = fig.add_subplot(grid[0])
    table_ax = fig.add_subplot(grid[1])
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.14, top=0.73)
    fig.suptitle("Task score vs response-score budget ε", fontsize=19, fontweight="bold", y=0.97)
    fig.text(0.545, 0.89, f"{MODELS[model]} | {DATASETS[dataset]} | N = 1000", ha="center", color="#475569")
    fig.text(
        0.545,
        0.835,
        f"Measured results | Baseline {METRICS[dataset].lower()}: {baseline:.1f}%",
        ha="center",
        color="#64748B",
        fontsize=10.5,
    )
    ax.axhline(baseline, color="#8B99AB", ls="--", lw=1.8, label="Unpruned baseline")
    ax.plot(epsilons, scores, color="#397DB5", marker="o", ms=7, lw=2.8, label="A2 task score")
    for epsilon, score in zip(epsilons, scores):
        ax.annotate(
            f"{score:.1f}",
            (epsilon, score),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            color="#334155",
            fontsize=9,
        )
    values = np.r_[scores, baseline]
    spread = max(float(values.max() - values.min()), 2.0)
    ax.set_xlim(0.035, 0.315)
    ax.set_ylim(float(values.min() - 0.18 * spread), float(values.max() + 0.17 * spread))
    ax.set_xticks(epsilons, [f"{value:.2f}" for value in epsilons])
    ax.set_xlabel("Response-score budget ε", labelpad=9)
    ax.set_ylabel(f"{METRICS[dataset]} (%)", labelpad=8)
    ax.grid(color="#E4EAF0", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(length=0, pad=6)
    ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white", edgecolor="#E2E8F0")
    table_ax.axis("off")
    table_ax.text(0.0, 1.12, "Mean actual pruning rate", transform=table_ax.transAxes, color="#475569", fontsize=9)
    table = table_ax.table(
        cellText=[[f"{value:.1f}%" for value in actual]],
        colLabels=[f"ε={value:.2f}" for value in epsilons],
        cellLoc="center",
        loc="center",
        bbox=(0.0, 0.0, 1.0, 0.90),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#DDE5EC")
        cell.set_linewidth(0.8)
    fig.text(
        0.13,
        0.055,
        "Measured task scores; y-axis is zoomed. Confidence intervals remain available in summary.csv.",
        color="#64748B",
        fontsize=8.5,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=240, facecolor="white")
    plt.close(fig)


def _plot_overview(frame: pd.DataFrame, output: Path) -> None:
    colors = {
        "llava_1_5_7b": "#397DB5",
        "qwen3_vl_2b_instruct": "#D98A49",
        "qwen3_5_2b": "#5B9D8C",
        "gemma_4_e2b_it": "#8F75A1",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.13, top=0.84, hspace=0.28, wspace=0.22)
    fig.suptitle("Task-score loss across response-score budgets", fontsize=19, fontweight="bold", y=0.98)
    fig.text(
        0.525, 0.925, "Measured A2 results | 4 models × 4 datasets | N = 1000 per run", ha="center", color="#64748B"
    )
    for axis, dataset in zip(axes.flat, DATASETS):
        for model in MODELS:
            subset = frame.loc[(frame["model"] == model) & (frame["dataset"] == dataset)].sort_values("epsilon")
            axis.plot(
                subset["epsilon"],
                subset["score_drop_pp"],
                marker="o",
                ms=4.5,
                lw=2,
                color=colors[model],
                label=MODELS[model],
            )
        axis.axhline(0, color="#94A3B8", lw=1.1, ls="--")
        axis.set_title(DATASETS[dataset], fontsize=12)
        axis.set_xticks(EPSILONS)
        axis.set_xlim(0.04, 0.31)
        axis.grid(color="#E4EAF0", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#CBD5E1")
        axis.tick_params(length=0, pad=5)
    for axis in axes[-1]:
        axis.set_xlabel("ε (response-score budget)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Baseline − A2 score (pp)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.525, 0.9), ncol=4, frameon=False, fontsize=9)
    fig.text(
        0.08,
        0.035,
        "Positive = worse. Panel y-axis ranges differ; effective pruning rates also differ by model and ε (summary.csv).",
        fontsize=9,
        color="#64748B",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior-root",
        type=Path,
        default=Path("artifacts/neuron_typing_generalization/adaptive_pruning_a1_a2_2026_10"),
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("artifacts/neuron_typing_generalization/adaptive_pruning_epsilon_sweep_2026_10"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/neuron_typing_generalization/reports/advisor_2026_09/figures/epsilon_sweep"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_repeats < 2:
        parser.error("At least two bootstrap repeats are required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            reference_baseline = None
            config_reference = None
            previous_ratios = None
            previous_shares = None
            for index, epsilon in enumerate(EPSILONS):
                directory = (
                    args.prior_root / f"{model}__{dataset}"
                    if index == 0
                    else args.sweep_root / f"{model}__{dataset}__eps_{epsilon:.2f}".replace(".", "p")
                )
                config, data = _read_run(directory, model, dataset, epsilon)
                comparable_keys = (
                    "study",
                    "model",
                    "dataset",
                    "statistics_root",
                    "baseline_source",
                    "source_run_id",
                    "input_policy",
                    "max_new_tokens",
                )
                comparable = {key: config[key] for key in comparable_keys}
                if config_reference is None:
                    config_reference = comparable
                elif comparable != config_reference:
                    raise ValueError(f"Input, source, or generation configuration differs: {directory}")
                baseline = data.loc[data["condition"] == "baseline"].sort_values("sample_index").reset_index(drop=True)
                ablated = data.loc[data["condition"] == "A2"].sort_values("sample_index").reset_index(drop=True)
                paired_keys = ("sample_id", "group_id")
                for key in paired_keys:
                    if not baseline[key].equals(ablated[key]):
                        raise ValueError(f"Unpaired {key}: {directory}")
                if reference_baseline is None:
                    reference_baseline = baseline
                else:
                    for key in ("sample_id", "group_id", "prediction", "score"):
                        if not reference_baseline[key].equals(baseline[key]):
                            raise ValueError(f"Baseline {key} differs across epsilon: {directory}")
                ratios = ablated["actual_ratio"].to_numpy(dtype=float)
                shares = ablated["actual_score_share"].to_numpy(dtype=float)
                if previous_ratios is not None and (
                    np.any(ratios < previous_ratios - 1e-12) or np.any(shares < previous_shares - 1e-12)
                ):
                    raise ValueError(f"Selection is not monotone in epsilon: {directory}")
                previous_ratios, previous_shares = ratios, shares
                baseline_scores = baseline["score"].to_numpy(dtype=float)
                a2_scores = ablated["score"].to_numpy(dtype=float)
                drop = (baseline_scores - a2_scores) * 100
                low, high = _paired_group_ci(
                    drop,
                    baseline["group_id"].to_numpy(dtype=str),
                    seed=20261010 + index,
                    repeats=args.bootstrap_repeats,
                )
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "epsilon": epsilon,
                        "n_samples": 1000,
                        "n_groups": baseline["group_id"].nunique(),
                        "baseline_score_percent": baseline_scores.mean() * 100,
                        "a2_score_percent": a2_scores.mean() * 100,
                        "score_drop_pp": drop.mean(),
                        "drop_ci_low_pp": low,
                        "drop_ci_high_pp": high,
                        "mean_actual_pruning_percent": ratios.mean() * 100,
                        "p10_actual_pruning_percent": np.percentile(ratios, 10) * 100,
                        "p90_actual_pruning_percent": np.percentile(ratios, 90) * 100,
                        "mean_actual_score_share_percent": shares.mean() * 100,
                        "source": str(directory / "predictions.jsonl"),
                    }
                )
            print(f"Verified {model}__{dataset}: 6 epsilon levels × 1000 paired samples", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "summary.csv", index=False)
    for model in MODELS:
        for dataset in DATASETS:
            subset = frame.loc[(frame["model"] == model) & (frame["dataset"] == dataset)].sort_values("epsilon")
            _plot_one(subset, model, dataset, args.output_dir / f"{model}__{dataset}")
    _plot_overview(frame, args.output_dir / "epsilon_sweep_overview")
    print(f"Wrote {len(frame)} measured conditions and 17 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
