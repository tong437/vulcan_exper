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

"""Build the advisor's response-state ablation figures from completed GPU summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MODEL_LABELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_2b_instruct": "Qwen3-VL-2B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "gemma_4_e2b_it": "Gemma-4-E2B-it",
}
DATASET_LABELS = {"mme": "MME", "mmbench": "MMBench", "textvqa": "TextVQA", "chartqa": "ChartQA"}
METRIC_LABELS = {
    "mme": "Accuracy",
    "mmbench": "Accuracy",
    "textvqa": "Soft accuracy",
    "chartqa": "Relaxed accuracy",
}
STATES = ("V", "T", "M", "W")
METHODS = ("random", "max", "min")
METHOD_COLORS = {"random": "#4C78A8", "max": "#DC8456", "min": "#64A18C"}


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    return plt


def _condition_lookup(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup = {str(row["condition"]): row for row in summary["conditions"]}
    required = {"baseline", "global_random"} | {
        f"{state}_{method}" for state in STATES for method in METHODS
    }
    missing = sorted(required - set(lookup))
    if missing:
        raise ValueError(f"Ablation summary is missing conditions: {missing}")
    return lookup


def _plot_one(
    summary: dict[str, Any], model: str, dataset: str, figure_dir: Path
) -> tuple[Path, Path]:
    plt = _load_pyplot()
    lookup = _condition_lookup(summary)
    means = np.array(
        [[float(lookup[f"{state}_{method}"]["drop_mean"]) * 100.0 for method in METHODS] for state in STATES]
    )
    standard_deviations = np.array(
        [[float(lookup[f"{state}_{method}"]["drop_std"]) * 100.0 for method in METHODS] for state in STATES]
    )
    reference = float(lookup["global_random"]["drop_mean"]) * 100.0
    reference_std = float(lookup["global_random"]["drop_std"]) * 100.0
    baseline = float(summary["baseline_score"]) * 100.0
    removal_percent = float(summary["removal_fraction"]) * 100.0

    figure, axis = plt.subplots(figsize=(9, 5.5), facecolor="white")
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.22, top=0.73)
    figure.suptitle("Response-state ablation", fontsize=19, fontweight="bold", y=0.98)
    figure.text(
        0.545,
        0.901,
        f"{MODEL_LABELS[model]} | {DATASET_LABELS[dataset]} | {removal_percent:g}% per layer, all layers",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    figure.text(
        0.545,
        0.855,
        f"Baseline {METRIC_LABELS[dataset].lower()}: {baseline:.1f}%",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    x = np.arange(len(STATES))
    width = 0.23
    bottom = min(0.0, float((means - standard_deviations).min()), reference - reference_std)
    top = max(0.0, float((means + standard_deviations).max()), reference + reference_std)
    span = max(top - bottom, 1.0)
    axis.axhspan(reference - reference_std, reference + reference_std, color="#796A85", alpha=0.11, zorder=0)
    axis.axhline(reference, color="#796A85", linestyle=(0, (5, 3)), linewidth=1.6, zorder=2)
    axis.axhline(0, color="#334155", linewidth=1.1, zorder=2)
    for method_index, method in enumerate(METHODS):
        positions = x + (method_index - 1) * width
        axis.bar(
            positions,
            means[:, method_index],
            width=width * 0.9,
            color=METHOD_COLORS[method],
            zorder=3,
        )
        if (standard_deviations[:, method_index] > 0).any():
            axis.errorbar(
                positions,
                means[:, method_index],
                yerr=standard_deviations[:, method_index],
                fmt="none",
                ecolor="#334155",
                capsize=3,
                linewidth=1,
                zorder=4,
            )
        for state_index, position in enumerate(positions):
            value = means[state_index, method_index]
            offset = standard_deviations[state_index, method_index] + span * 0.025
            axis.text(
                position,
                value + (offset if value >= 0 else -offset),
                f"{value:.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=10,
                color="#334155",
            )
    handles = [
        Patch(facecolor=METHOD_COLORS[method], label=f"{method.title()} within state") for method in METHODS
    ]
    handles.append(
        Line2D([0], [0], color="#796A85", linestyle="--", linewidth=1.6, label=f"Global random ({reference:.1f} pp)")
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.545, 0.824),
        ncol=2,
        frameon=False,
        fontsize=10,
        columnspacing=2,
    )
    axis.set_xticks(x, STATES)
    axis.set_xlabel("Response state of ablated neurons", labelpad=10)
    axis.set_ylabel("Score drop from baseline (percentage points)", labelpad=10)
    axis.set_ylim(bottom - span * 0.13, top + span * 0.22)
    axis.set_xlim(-0.6, 3.6)
    axis.grid(axis="y", color="#E4EAF0", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#CBD5E1")
    axis.tick_params(length=0, pad=7)
    figure.text(
        0.12,
        0.103,
        "Zero line = baseline. Positive drop = worse performance; negative drop = improvement.",
        fontsize=9,
        color="#64748B",
    )
    figure.text(
        0.12,
        0.062,
        "Error bars / band: +/- 1 SD across 10 random runs.",
        fontsize=9,
        color="#64748B",
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model}__{dataset}__response_state_ablation"
    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=250, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_LABELS))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_LABELS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures/response_state_ablation"
    rows = []
    figure_rows = []
    models = (args.model,) if args.model else tuple(MODEL_LABELS)
    datasets = tuple(args.datasets) if args.datasets else tuple(DATASET_LABELS)
    for model in models:
        for dataset in datasets:
            summary_path = args.input_dir / f"{model}__{dataset}" / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Ablation summary is missing: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            png_path, pdf_path = _plot_one(summary, model, dataset, figure_dir)
            for condition in summary["conditions"]:
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "metric": METRIC_LABELS[dataset],
                        **condition,
                    }
                )
            figure_rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "dataset": DATASET_LABELS[dataset],
                    "png": str(png_path),
                    "pdf": str(pdf_path),
                }
            )
    frame = pd.DataFrame(rows)
    csv_path = args.output_dir / "advisor_2026_09_response_state_ablation.csv"
    json_path = args.output_dir / "advisor_2026_09_response_state_ablation.json"
    report_path = args.output_dir / "advisor_2026_09_response_state_ablation.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Response-state ablation (1% per layer)",
        "",
        "Scores use the same frozen 1,000 examples as the response-state study. Positive drops indicate worse performance.",
        "",
        "| Model | Dataset | Figure |",
        "|---|---|---|",
    ]
    for row in figure_rows:
        png_link = Path(row["png"]).relative_to(args.output_dir)
        pdf_link = Path(row["pdf"]).relative_to(args.output_dir)
        lines.append(f"| {row['model']} | {row['dataset']} | [PNG]({png_link}) / [PDF]({pdf_link}) |")
    for model in models:
        model_rows = frame.loc[frame["model"] == model].set_index(["dataset", "condition"])
        lines.extend(
            [
                "",
                f"## {MODEL_LABELS[model]} numerical results",
                "",
                "Baseline scores use each dataset's own metric. All other cells are score drops from baseline in percentage points; random conditions show mean +/- 1 SD across 10 runs.",
                "",
                "| Condition | MME | MMBench | TextVQA | ChartQA |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        condition_order = ["baseline"] + [
            f"{state}_{method}" for method in METHODS for state in STATES
        ] + ["global_random"]
        for condition in condition_order:
            cells = []
            for dataset in DATASET_LABELS:
                record = model_rows.loc[(dataset, condition)]
                if condition == "baseline":
                    cell = f"{float(record['score_mean']) * 100:.2f}%"
                else:
                    cell = f"{float(record['drop_mean']) * 100:+.2f}"
                    if int(record["runs"]) > 1:
                        cell += f" +/- {float(record['drop_std']) * 100:.2f}"
                cells.append(cell)
            lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    lines.extend(["", f"Machine-readable results: [CSV]({csv_path.name}) / [JSON]({json_path.name})", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"runs": len(models) * len(datasets), "figures": len(figure_rows), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
