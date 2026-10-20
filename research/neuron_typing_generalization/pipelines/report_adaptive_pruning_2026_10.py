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

"""Report measured adaptive-pruning ratios and paired task-score drops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


MODELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_2b_instruct": "Qwen3-VL-2B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "gemma_4_e2b_it": "Gemma-4-E2B-it",
}
DATASETS = {"mme": "MME", "mmbench": "MMBench", "textvqa": "TextVQA", "chartqa": "ChartQA"}
METRICS = {"mme": "accuracy", "mmbench": "accuracy", "textvqa": "soft accuracy", "chartqa": "relaxed accuracy"}


def paired_group_bootstrap(
    differences_pp: np.ndarray, groups: np.ndarray, *, repeats: int = 2000, seed: int = 20261009
) -> tuple[np.ndarray, np.ndarray]:
    """Paired percentile CIs, resampling image/chart groups with replacement."""
    if differences_pp.ndim != 2 or differences_pp.shape[1] != 2 or len(groups) != len(differences_pp):
        raise ValueError("Need paired A1/A2 per-example differences and one group per example.")
    _, group_index = np.unique(groups.astype(str), return_inverse=True)
    group_count = group_index.max() + 1
    if group_count < 2 or repeats < 2:
        raise ValueError("At least two groups and two bootstrap repeats are required.")
    sizes = np.bincount(group_index, minlength=group_count)
    sums = np.column_stack(
        [np.bincount(group_index, weights=differences_pp[:, method], minlength=group_count) for method in range(2)]
    )
    random = np.random.default_rng(seed)
    draws = np.empty((repeats, 2), dtype=np.float64)
    for repeat in range(repeats):
        selected = random.integers(0, group_count, size=group_count)
        draws[repeat] = sums[selected].sum(axis=0) / sizes[selected].sum()
    low, high = np.percentile(draws, (2.5, 97.5), axis=0)
    return low, high


def load_one(
    input_root: Path, model: str, dataset: str
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directory = input_root / f"{model}__{dataset}"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["expected_samples"] != 1000 or summary["model"] != model or summary["dataset"] != dataset:
        raise ValueError(f"Adaptive-pruning summary is incomplete: {directory}")
    frame = pd.read_json(directory / "predictions.jsonl", lines=True)
    if len(frame) != 3000 or frame.duplicated(["sample_index", "condition"]).any():
        raise ValueError(f"Expected three unique conditions for each of 1,000 samples: {directory}")
    if set(frame["condition"]) != {"baseline", "A1", "A2"}:
        raise ValueError(f"Unexpected condition set: {directory}")
    for condition in ("baseline", "A1", "A2"):
        subset = frame.loc[frame["condition"] == condition].sort_values("sample_index")
        if len(subset) != 1000 or not np.array_equal(subset["sample_index"], np.arange(1000)):
            raise ValueError(f"Incomplete sample order for {condition}: {directory}")
        if not subset["score"].between(0, 1).all() or not subset["actual_ratio"].between(0, 1).all():
            raise ValueError(f"Invalid score or actual ratio for {condition}: {directory}")
    ordered = {
        condition: frame.loc[frame["condition"] == condition].sort_values("sample_index")
        for condition in ("baseline", "A1", "A2")
    }
    for condition in ("A1", "A2"):
        if not np.array_equal(ordered["baseline"]["sample_id"], ordered[condition]["sample_id"]):
            raise ValueError(f"Paired sample IDs disagree for {condition}: {directory}")
        if not np.array_equal(ordered["baseline"]["group_id"], ordered[condition]["group_id"]):
            raise ValueError(f"Paired group IDs disagree for {condition}: {directory}")
    scores = np.column_stack(
        [ordered[condition]["score"].to_numpy(dtype=float) for condition in ("baseline", "A1", "A2")]
    )
    ratios = np.column_stack([ordered[condition]["actual_ratio"].to_numpy(dtype=float) for condition in ("A1", "A2")])
    shares = np.column_stack(
        [ordered[condition]["actual_score_share"].to_numpy(dtype=float) for condition in ("A1", "A2")]
    )
    if np.any(shares[:, 1] > summary["epsilon"] + 1e-10):
        raise ValueError(f"A2 score budget was exceeded: {directory}")
    return summary, scores, ratios, shares, ordered["baseline"]["group_id"].to_numpy(dtype=str)


def plot_one(output: Path, model: str, dataset: str, epsilon: float, scores, ratios, drops, low, high) -> None:
    colors = ("#397DB5", "#D98A49")
    means = ratios.mean(axis=0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.3, 5.4), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.22, top=0.72, wspace=0.35)
    fig.suptitle("Sample-adaptive pruning: A1 vs A2", fontsize=19, fontweight="bold", y=0.975)
    fig.text(
        0.52,
        0.90,
        f"{MODELS[model]} | {DATASETS[dataset]} | N = 1000 | Uniform ratio across layers per sample",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(
        0.52,
        0.85,
        f"Baseline {METRICS[dataset]}: {scores[:, 0].mean():.1%} | Measured results",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    bins = np.linspace(0, 1, 21)
    for method, color, label in ((0, colors[0], "A1: knee"), (1, colors[1], f"A2: budget (ε = {epsilon:g})")):
        left.hist(
            ratios[:, method],
            bins=bins,
            weights=np.full(len(ratios), 1 / len(ratios)),
            color=color,
            alpha=0.42,
            edgecolor=color,
            linewidth=0.7,
            label=label,
            zorder=3,
        )
        left.axvline(means[method], color=color, ls="--", lw=1.7, zorder=4)
    left.set_title("Actual pruning-rate distribution", fontsize=12, pad=13)
    left.set_xlim(0, 1)
    left.set_xticks(np.linspace(0, 1, 6))
    left.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.set_xlabel("Actual pruning rate per sample", labelpad=10)
    left.set_ylabel("Share of samples per 5% bin", labelpad=8)
    left.legend(loc="upper left", fontsize=9, frameon=False)
    left.text(
        0.97,
        0.97,
        f"Mean A1: {means[0]:.1%}\nMean A2: {means[1]:.1%}",
        transform=left.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color="#475569",
    )
    right.bar(np.arange(2), drops, color=colors, width=0.52, zorder=3)
    for method in range(2):
        right.vlines(method, low[method], high[method], color="#334155", lw=1.3, zorder=4)
        right.hlines((low[method], high[method]), method - 0.06, method + 0.06, color="#334155", lw=1.3, zorder=4)
    span = max(float(high.max() - min(0, low.min())), 1.0)
    for method in range(2):
        right.text(
            method,
            max(drops[method], high[method]) + span * 0.055,
            f"{drops[method]:.2f} pp\nMean pruning: {means[method]:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#334155",
        )
    right.axhline(0, color="#64748B", lw=1.1)
    right.set_ylim(min(0, low.min(), drops.min()) - span * 0.12, max(0, high.max(), drops.max()) + span * 0.34)
    right.set_xticks((0, 1), ("A1\nKnee", "A2\nBudget"))
    right.set_title("Task score drop", fontsize=12, pad=13)
    right.set_ylabel("Baseline - pruned score (pp)", labelpad=8)
    right.set_xlabel("Pruning-rate selection method", labelpad=10)
    for axis in (left, right):
        axis.grid(axis="y", color="#E4EAF0", lw=0.7, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#CBD5E1")
        axis.tick_params(length=0, pad=6)
    fig.text(
        0.075,
        0.09,
        "Dashed lines = mean pruning rates. Intervals = paired image/chart-group bootstrap 95% CIs.",
        fontsize=9,
        color="#64748B",
    )
    fig.text(
        0.075,
        0.046,
        "Different removal amounts: compare quality and dose together. Positive drop = worse.",
        fontsize=9,
        color="#64748B",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=250, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("artifacts/neuron_typing_generalization/adaptive_pruning_a1_a2_2026_10"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/neuron_typing_generalization/reports/advisor_2026_09/figures/adaptive_pruning"),
    )
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model in MODELS:
        if args.model and model != args.model:
            continue
        for dataset in DATASETS:
            if args.dataset and dataset != args.dataset:
                continue
            summary, scores, ratios, shares, groups = load_one(args.input_root, model, dataset)
            differences = (scores[:, [0]] - scores[:, 1:]) * 100
            low, high = paired_group_bootstrap(differences, groups, repeats=args.bootstrap_repeats)
            drops = differences.mean(axis=0)
            stem = args.output_dir / f"{model}__{dataset}"
            plot_one(stem, model, dataset, summary["epsilon"], scores, ratios, drops, low, high)
            for method in range(2):
                results.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "method": ("A1", "A2")[method],
                        "n_samples": 1000,
                        "n_groups": len(np.unique(groups)),
                        "epsilon": summary["epsilon"],
                        "baseline_score_percent": scores[:, 0].mean() * 100,
                        "pruned_score_percent": scores[:, method + 1].mean() * 100,
                        "score_drop_pp": drops[method],
                        "ci_low_pp": low[method],
                        "ci_high_pp": high[method],
                        "mean_actual_pruning_percent": ratios[:, method].mean() * 100,
                        "mean_actual_score_share_percent": shares[:, method].mean() * 100,
                    }
                )
            print(f"Reported {model}__{dataset}", flush=True)
    pd.DataFrame(results).to_csv(args.output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
