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

"""Validate A1/A2 pruning runs and draw the advisor's 16 comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from neuron_typing_generalization.weak_pruning import TARGET_PERCENTS


MODELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_2b_instruct": "Qwen3-VL-2B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "gemma_4_e2b_it": "Gemma-4-E2B-it",
}
DATASETS = {"mme": "MME", "mmbench": "MMBench", "textvqa": "TextVQA", "chartqa": "ChartQA"}
METRICS = {"mme": "accuracy", "mmbench": "accuracy", "textvqa": "soft accuracy", "chartqa": "relaxed accuracy"}


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    return plt


def _load_one(input_root: Path, model: str, dataset: str) -> pd.DataFrame:
    directory = input_root / f"{model}__{dataset}"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["expected_samples"] != 1000 or summary["target_percents"] != list(TARGET_PERCENTS):
        raise ValueError(f"Wrong sample count or targets: {directory}")
    rows = pd.read_json(directory / "predictions.jsonl", lines=True)
    if len(rows) != 11000 or rows.duplicated(["sample_index", "condition", "repeat"]).any():
        raise ValueError(f"Incomplete or duplicate predictions: {directory}")
    if rows["sample_index"].nunique() != 1000 or not rows["score"].between(0, 1).all():
        raise ValueError(f"Invalid sample coverage or score range: {directory}")
    by_condition = rows.groupby("condition", sort=False)
    if len(by_condition) != 11 or not (by_condition.size() == 1000).all():
        raise ValueError(f"Incomplete conditions: {directory}")
    saved = {condition["condition"]: condition for condition in summary["conditions"]}
    table = []
    for condition, group in by_condition:
        if condition not in saved:
            raise ValueError(f"Unexpected condition {condition}: {directory}")
        score = float(group["score"].mean())
        actual = (
            float(group["selected_neurons"].mean() / group["target_neurons"].max()) if condition != "baseline" else 0.0
        )
        if not np.isclose(score, saved[condition]["score"], atol=1e-10):
            raise ValueError(f"Score disagrees with summary for {condition}: {directory}")
        if not np.isclose(
            float((group["selected_neurons"] / summary["total_neurons"] * 100).mean()),
            saved[condition]["actual_removal_percent"],
            atol=1e-10,
        ):
            raise ValueError(f"Realized dose disagrees with summary for {condition}: {directory}")
        table.append(
            {
                "model": model,
                "dataset": dataset,
                "condition": condition,
                "method": saved[condition]["method"],
                "target_percent": saved[condition]["removal_percent"],
                "score_percent": score * 100,
                "actual_percent": saved[condition]["actual_removal_percent"],
                "target_attainment_fraction": actual,
                "shortfall_prediction_fraction": saved[condition]["shortfall_prediction_count"] / 1000,
            }
        )
    return pd.DataFrame(table)


def _plot_one(frame: pd.DataFrame, model: str, dataset: str, figure_dir: Path) -> None:
    plt = _load_pyplot()
    values = frame.set_index("condition")
    baseline = float(values.loc["baseline", "score_percent"])
    x = np.array(TARGET_PERCENTS, dtype=float)
    y1 = np.array([values.loc[f"A1_{percent:02d}pct", "score_percent"] for percent in TARGET_PERCENTS])
    y2 = np.array([values.loc[f"A2_{percent:02d}pct", "score_percent"] for percent in TARGET_PERCENTS])
    actual_a1 = [values.loc[f"A1_{percent:02d}pct", "actual_percent"] for percent in TARGET_PERCENTS]
    actual_a2 = [values.loc[f"A2_{percent:02d}pct", "actual_percent"] for percent in TARGET_PERCENTS]
    all_targets_met = not frame["shortfall_prediction_fraction"].gt(0).any()
    figure, axis = plt.subplots(figsize=(8.4, 5.8), facecolor="white")
    figure.subplots_adjust(left=0.12, right=0.965, bottom=0.35, top=0.75)
    figure.suptitle("Weak-neuron removal: A1 vs A2", fontsize=18, fontweight="bold", y=0.98)
    figure.text(
        0.54, 0.908, f"{MODELS[model]} | {DATASETS[dataset]} | All layers", ha="center", fontsize=11, color="#475569"
    )
    figure.text(0.54, 0.863, "Measured ablation results", ha="center", fontsize=10, color="#64748B")
    xx = np.r_[0, x]
    axis.plot(xx, np.r_[baseline, y1], "-o", color="#397DB5", lw=2.2, ms=6, label="A1: separate V / T / C")
    axis.plot(xx, np.r_[baseline, y2], "-s", color="#D78A49", lw=2.2, ms=6, label="A2: pooled input tokens")
    axis.axhline(baseline, color="#788493", ls="--", lw=1.3, label=f"Baseline: {baseline:.2f}%")
    low = min(y1.min(), y2.min(), baseline)
    high = max(y1.max(), y2.max(), baseline)
    span = max(high - low, 2)
    axis.set_ylim(max(0, low - 0.16 * span), min(100, high + 0.25 * span))
    axis.set_xlim(-0.5, x[-1] + 0.7)
    axis.set_xticks(xx, [f"{value:g}%" for value in xx])
    axis.set_xlabel("Target removal per layer (% of all neurons)", labelpad=10)
    axis.set_ylabel(f"{METRICS[dataset].capitalize()} (%)", labelpad=10)
    axis.legend(loc="lower left", fontsize=9, framealpha=0.96, edgecolor="#E2E8F0")
    axis.grid(color="#E4EAF0", lw=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#CBD5E1")
    axis.tick_params(length=0, pad=6)
    table = axis.table(
        cellText=[[f"{value:.3f}%" for value in series] for series in (actual_a1, actual_a2)],
        rowLabels=["A1 actual", "A2 actual"],
        colLabels=[f"Target {value:g}%" for value in x],
        cellLoc="center",
        rowLoc="center",
        bbox=[0.13, -0.66, 0.87, 0.31],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#E2E8F0")
        cell.set_facecolor("#F1F5F9" if row == 0 else "white")
    figure.text(
        0.12,
        0.045,
        "Actual = mean selected / total FFN neurons across 1,000 samples.",
        fontsize=8.5,
        color="#64748B",
    )
    figure.text(
        0.12,
        0.023,
        (
            "All targets met; small gaps reflect integer rounding per layer. Y-axis is zoomed."
            if all_targets_met
            else "Some W pools were insufficient; actual removal is clipped. Y-axis is zoomed."
        ),
        fontsize=8.5,
        color="#64748B",
    )
    stem = f"{model}__{dataset}__weak_pruning_a1_vs_a2"
    figure.savefig(figure_dir / f"{stem}.png", dpi=250, facecolor="white")
    figure.savefig(figure_dir / f"{stem}.pdf", facecolor="white")
    plt.close(figure)


def _overall_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize equal-weighted score drops across the 16 model-dataset runs."""
    baseline = frame.loc[frame["condition"] == "baseline", ["model", "dataset", "score_percent"]].rename(
        columns={"score_percent": "baseline_score_percent"}
    )
    if len(baseline) != 16 or baseline.duplicated(["model", "dataset"]).any():
        raise ValueError("Overall comparison requires one baseline for each of 16 model-dataset runs.")
    comparisons = frame.loc[frame["method"].notna()].merge(
        baseline, on=["model", "dataset"], how="left", validate="many_to_one"
    )
    if comparisons["baseline_score_percent"].isna().any():
        raise ValueError("An A1/A2 comparison has no matching baseline.")
    comparisons["score_drop_pp"] = comparisons["baseline_score_percent"] - comparisons["score_percent"]
    overall = (
        comparisons.groupby(["method", "target_percent"], as_index=False)["score_drop_pp"]
        .agg(mean_drop_pp="mean", sd_drop_pp="std", run_count="size")
        .sort_values(["method", "target_percent"])
    )
    if len(overall) != 2 * len(TARGET_PERCENTS) or not (overall["run_count"] == 16).all():
        raise ValueError("Overall comparison needs 16 runs for each method and target percentage.")
    zero = pd.DataFrame(
        {
            "method": ["A1", "A2"],
            "target_percent": [0, 0],
            "mean_drop_pp": [0.0, 0.0],
            "sd_drop_pp": [0.0, 0.0],
            "run_count": [16, 16],
        }
    )
    return pd.concat([zero, overall], ignore_index=True).sort_values(["method", "target_percent"])


def _plot_overall(overall: pd.DataFrame, figure_dir: Path) -> None:
    """Draw the two mean removal curves with descriptive between-run SD bands."""
    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(9.4, 6.0), facecolor="white")
    figure.subplots_adjust(left=0.12, right=0.94, bottom=0.23, top=0.79)
    figure.suptitle("Overall weak-neuron removal: A1 vs A2", fontsize=18, fontweight="bold", y=0.98)
    figure.text(
        0.53,
        0.90,
        "4 models × 4 datasets | 1,000 frozen samples per run",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    figure.text(
        0.53,
        0.85,
        "Equal-weighted mean score drop from each run's own baseline",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    styles = {"A1": ("#397DB5", "o", "A1: separate V / T / C"), "A2": ("#D78A49", "s", "A2: pooled input tokens")}
    for method, (color, marker, label) in styles.items():
        series = overall.loc[overall["method"] == method].sort_values("target_percent")
        x = series["target_percent"].to_numpy(dtype=float)
        mean = series["mean_drop_pp"].to_numpy(dtype=float)
        sd = series["sd_drop_pp"].to_numpy(dtype=float)
        axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.17, linewidth=0)
        axis.plot(x, mean, color=color, marker=marker, lw=2.6, ms=7, label=label)
        axis.annotate(
            f"{mean[-1]:.2f} pp",
            (x[-1], mean[-1]),
            xytext=(-8, 10 if method == "A2" else -16),
            textcoords="offset points",
            color=color,
            fontsize=10,
            ha="right",
        )
    axis.axhline(0, color="#788493", ls="--", lw=1.2)
    axis.set_xlim(-0.5, 20.5)
    axis.set_xticks([0, *TARGET_PERCENTS], ["0%", *[f"{percent}%" for percent in TARGET_PERCENTS]])
    axis.set_xlabel("Target removal per layer (% of all neurons)", labelpad=10)
    axis.set_ylabel("Score drop from baseline (percentage points)", labelpad=10)
    axis.grid(color="#E4EAF0", lw=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#CBD5E1")
    axis.tick_params(length=0, pad=6)
    axis.legend(loc="upper left", fontsize=10, framealpha=0.96, edgecolor="#E2E8F0")
    figure.text(
        0.12,
        0.12,
        "Shaded bands: ±1 SD across 16 model–dataset runs; descriptive spread, not confidence intervals.",
        fontsize=8.8,
        color="#64748B",
    )
    figure.text(
        0.12,
        0.08,
        "Positive drop = worse performance. Different datasets use their established scoring metrics.",
        fontsize=8.8,
        color="#64748B",
    )
    stem = "weak_pruning_a1_vs_a2_overall"
    figure.savefig(figure_dir / f"{stem}.png", dpi=250, facecolor="white")
    figure.savefig(figure_dir / f"{stem}.pdf", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    figure_dir = args.output_dir / "figures" / "weak_pruning_a1_vs_a2"
    figure_dir.mkdir(parents=True, exist_ok=True)
    frames = [_load_one(args.input_root, model, dataset) for model in MODELS for dataset in DATASETS]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.output_dir / "weak_pruning_a1_vs_a2_conditions.csv", index=False)
    for model in MODELS:
        for dataset in DATASETS:
            _plot_one(frame[(frame.model == model) & (frame.dataset == dataset)], model, dataset, figure_dir)
    overall = _overall_summary(frame)
    overall.to_csv(args.output_dir / "weak_pruning_a1_vs_a2_overall.csv", index=False)
    _plot_overall(overall, figure_dir)
    print(json.dumps({"runs": 16, "predictions": 176000, "figures": 17}, indent=2))


if __name__ == "__main__":
    main()
