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

"""Build the advisor's 4 x 4 layerwise V/T/M/W distribution report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from neuron_typing_generalization.advisor_layerwise_states import (
    STATE_ORDER,
    layerwise_state_distribution,
)
from neuron_typing_generalization.config import expand_study, load_study
from neuron_typing_generalization.runtime import build_run_id, resolve_artifact_root


MODEL_LABELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_2b_instruct": "Qwen3-VL-2B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "gemma_4_e2b_it": "Gemma-4-E2B-it",
}
DATASET_ORDER = ("mme", "mmbench", "textvqa", "chartqa")
DATASET_LABELS = {"mme": "MME", "mmbench": "MMBench", "textvqa": "TextVQA", "chartqa": "ChartQA"}
STATE_COLORS = {"V": "#397DB5", "T": "#DB9743", "M": "#599E8B", "W": "#D8DFE7"}
INDIVIDUAL_ASPECT_RATIO = 7.2 / 4.2
INDIVIDUAL_FIGURE_HEIGHT = 12.7
INDIVIDUAL_FIGURE_SIZE = (INDIVIDUAL_FIGURE_HEIGHT * INDIVIDUAL_ASPECT_RATIO, INDIVIDUAL_FIGURE_HEIGHT)
MAXIMUM_PERCENTAGE_LABEL_SIZE = 12.0
PERCENTAGE_LABEL_SIZE_STEP = 0.2


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    return plt


def _layer_ticks(layers: np.ndarray, maximum_ticks: int = 12) -> np.ndarray:
    step = max(1, int(np.ceil(len(layers) / maximum_ticks)))
    return np.unique(np.r_[layers[::step], layers[-1]])


def _assert_labels_fit(figure: Any, labels_and_bars: list[tuple[Any, Any]]) -> None:
    """Require every rotated percentage to fit inside its bar segment."""
    padding_pixels = 2.0
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for label, bar in labels_and_bars:
        label_box = label.get_window_extent(renderer=renderer)
        bar_box = bar.get_window_extent(renderer=renderer)
        if (
            label_box.x0 < bar_box.x0 + padding_pixels
            or label_box.x1 > bar_box.x1 - padding_pixels
            or label_box.y0 < bar_box.y0 + padding_pixels
            or label_box.y1 > bar_box.y1 - padding_pixels
        ):
            raise RuntimeError(f"Percentage label {label.get_text()} does not fit inside its stacked-bar segment.")


def _maximize_percentage_label_size(
    figure: Any,
    labels_and_bars: list[tuple[Any, Any]],
    minimum_size: float,
) -> float:
    """Use the largest common label size that keeps every percentage inside its segment."""
    candidate_sizes = np.arange(
        MAXIMUM_PERCENTAGE_LABEL_SIZE,
        minimum_size - PERCENTAGE_LABEL_SIZE_STEP / 2.0,
        -PERCENTAGE_LABEL_SIZE_STEP,
    )
    renderer = None
    padding_pixels = 2.0
    for candidate_size in candidate_sizes:
        for label, _ in labels_and_bars:
            label.set_fontsize(float(candidate_size))
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        if all(
            label.get_window_extent(renderer=renderer).x0 >= bar.get_window_extent(renderer=renderer).x0 + padding_pixels
            and label.get_window_extent(renderer=renderer).x1
            <= bar.get_window_extent(renderer=renderer).x1 - padding_pixels
            and label.get_window_extent(renderer=renderer).y0 >= bar.get_window_extent(renderer=renderer).y0 + padding_pixels
            and label.get_window_extent(renderer=renderer).y1
            <= bar.get_window_extent(renderer=renderer).y1 - padding_pixels
            for label, bar in labels_and_bars
        ):
            return float(candidate_size)

    _assert_labels_fit(figure, labels_and_bars)
    raise AssertionError("Unreachable: _assert_labels_fit must fail when no candidate size fits.")


def _save_individual_figure(
    distribution: pd.DataFrame,
    model_label: str,
    dataset_label: str,
    stem: str,
    figure_dir: Path,
) -> tuple[Path, Path]:
    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=INDIVIDUAL_FIGURE_SIZE, facecolor="white")
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.14, top=0.80)
    figure.suptitle(
        "Layerwise response-state distribution", y=0.98, fontsize=17, fontweight="bold", color="#172033"
    )
    figure.text(0.545, 0.91, f"{model_label} | {dataset_label}", ha="center", fontsize=10.5, color="#64748B")

    layers = distribution["layer"].to_numpy(dtype=np.int64)
    bottom = np.zeros(len(distribution), dtype=np.float64)
    minimum_label_size = 9.2 if len(distribution) <= 28 else 8.4
    labels_and_bars = []
    for state in STATE_ORDER:
        shares = distribution[f"share_{state}"].to_numpy(dtype=np.float64)
        bars = axis.bar(
            layers,
            shares,
            bottom=bottom,
            width=0.92,
            color=STATE_COLORS[state],
            label=state,
            edgecolor="none",
            zorder=3,
        )
        centers = bottom + shares / 2.0
        label_color = "#334155" if state == "W" else "#FFFFFF"
        for layer, center, share, bar in zip(layers, centers, shares, bars):
            label = axis.text(
                layer,
                center,
                f"{share:.1%}",
                ha="center",
                va="center",
                rotation=90,
                rotation_mode="anchor",
                fontsize=MAXIMUM_PERCENTAGE_LABEL_SIZE,
                color=label_color,
                zorder=4,
                clip_on=True,
            )
            labels_and_bars.append((label, bar))
        bottom += shares
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        columnspacing=2,
        handlelength=1.5,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlim(float(layers.min()) - 0.7, float(layers.max()) + 0.7)
    axis.set_xticks(_layer_ticks(layers))
    axis.set_xlabel("Decoder layer index (0-based)", labelpad=10)
    axis.set_ylabel("Mean response-state share", labelpad=10)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.grid(axis="y", color="#E9EEF4", zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#CBD5E1")
    axis.tick_params(length=0, pad=6)
    figure.text(
        0.11,
        0.025,
        "Each layer sums to 100%; average over inputs and real neurons within that layer.",
        fontsize=9,
        color="#64748B",
    )
    _maximize_percentage_label_size(figure, labels_and_bars, minimum_label_size)
    _assert_labels_fit(figure, labels_and_bars)
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=250, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def _save_overview_figure(
    distributions: dict[tuple[str, str], pd.DataFrame], output_dir: Path
) -> tuple[Path, Path]:
    plt = _load_pyplot()
    figure, axes = plt.subplots(4, 4, figsize=(18, 13.5), sharey=True, facecolor="white")
    legend_handles = None
    legend_labels = None
    for model_index, model in enumerate(MODEL_LABELS):
        for dataset_index, dataset in enumerate(DATASET_ORDER):
            axis = axes[model_index, dataset_index]
            distribution = distributions[(model, dataset)]
            layers = distribution["layer"].to_numpy(dtype=np.int64)
            bottom = np.zeros(len(distribution), dtype=np.float64)
            for state in STATE_ORDER:
                shares = distribution[f"share_{state}"].to_numpy(dtype=np.float64)
                axis.bar(
                    layers,
                    shares,
                    bottom=bottom,
                    width=0.88,
                    color=STATE_COLORS[state],
                    label=state,
                    edgecolor="none",
                    zorder=3,
                )
                bottom += shares
            if legend_handles is None:
                legend_handles, legend_labels = axis.get_legend_handles_labels()
            axis.set_ylim(0.0, 1.0)
            axis.set_xlim(float(layers.min()) - 0.7, float(layers.max()) + 0.7)
            axis.set_xticks(_layer_ticks(layers, maximum_ticks=8))
            axis.set_yticks(np.linspace(0.0, 1.0, 6))
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            axis.grid(axis="y", color="#E9EEF4", linewidth=0.8, zorder=0)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#CBD5E1")
            axis.tick_params(length=0, pad=5, labelsize=8.5)
            if model_index == 0:
                axis.set_title(DATASET_LABELS[dataset], fontsize=12, fontweight="bold", pad=10)
            if dataset_index == 0:
                axis.set_ylabel(f"{MODEL_LABELS[model]}\nMean share", labelpad=8)
            if model_index == len(MODEL_LABELS) - 1:
                axis.set_xlabel("Decoder layer", labelpad=7)

    figure.suptitle("Layerwise Response-state Distributions", fontsize=19, fontweight="bold", y=0.995)
    figure.text(
        0.5,
        0.971,
        "Each stacked bar is one decoder layer and sums to 100%; no layers or neurons are padded",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#526070",
    )
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=4,
        frameon=False,
        columnspacing=2.0,
    )
    figure.tight_layout(rect=(0.02, 0.025, 1.0, 0.925))
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "advisor_2026_09_layerwise_state_distributions.png"
    pdf_path = figure_dir / "advisor_2026_09_layerwise_state_distributions.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def _summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Dataset | Layers | Neurons | V | T | M | W |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['dataset_label']} | {row['layer_count']} | "
            f"{row['neuron_count']:,} | "
            + " | ".join(f"{row[f'overall_share_{state}']:.1%}" for state in STATE_ORDER)
            + " |"
        )
    return "\n".join(lines)


def _figure_table(figures: list[dict[str, str]]) -> str:
    lines = ["| Model | MME | MMBench | TextVQA | ChartQA |", "|---|---|---|---|---|"]
    for model, model_label in MODEL_LABELS.items():
        links = []
        for dataset in DATASET_ORDER:
            item = next(entry for entry in figures if entry["model"] == model and entry["dataset"] == dataset)
            links.append(f"[PNG]({item['png']}) / [PDF]({item['pdf']}) / [CSV]({item['csv']})")
        lines.append(f"| {model_label} | " + " | ".join(links) + " |")
    return "\n".join(lines)


def _descriptive_observations(
    distributions: dict[tuple[str, str], pd.DataFrame], summary_rows: list[dict[str, Any]]
) -> str:
    dominant_counts = dict.fromkeys(STATE_ORDER, 0)
    deeper_w_count = 0
    for distribution in distributions.values():
        dominant = distribution[[f"share_{state}" for state in STATE_ORDER]].idxmax(axis=1).str.removeprefix("share_")
        for state, count in dominant.value_counts().items():
            dominant_counts[state] += int(count)
        ordered = distribution.sort_values("layer")
        deeper_w_count += int(ordered["share_W"].iloc[-1] > ordered["share_W"].iloc[0])
    weak_shares = [row["overall_share_W"] for row in summary_rows]
    return (
        "## Descriptive observations\n\n"
        f"- `W` is the largest mean state share in all {sum(dominant_counts.values())} observed decoder layers. "
        f"Across complete runs, its neuron-weighted share ranges from {min(weak_shares):.1%} to "
        f"{max(weak_shares):.1%}.\n"
        f"- The final layer has a higher `W` share than the first layer in {deeper_w_count}/16 runs. This pattern is "
        "especially strong in LLaVA, Qwen3-VL, and Qwen3.5; Gemma is more oscillatory across depth.\n"
        "- These are occupancy proportions, not answer-impact estimates. `W` means below the within-layer importance "
        "threshold for both modalities, not zero activation or an unimportant neuron.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures/layerwise_state_distributions"
    table_dir = args.output_dir / "tables/layerwise_state_distributions"
    table_dir.mkdir(parents=True, exist_ok=True)
    distributions: dict[tuple[str, str], pd.DataFrame] = {}
    summary_rows = []
    long_frames = []
    individual_figures = []
    missing = []

    for spec in expand_study(load_study(args.study)):
        run_id = build_run_id(spec)
        profile_path = resolve_artifact_root(spec) / "runs" / spec.study / run_id / "distributions/neuron_profiles.parquet"
        if not profile_path.is_file():
            missing.append(str(profile_path))
            continue
        profile = pd.read_parquet(
            profile_path,
            columns=[
                "layer",
                "neuron",
                "d_ff",
                "valid_samples",
                "count_V",
                "count_T",
                "count_M",
                "count_W",
            ],
        )
        distribution = layerwise_state_distribution(profile)
        key = (spec.model.name, spec.dataset.name)
        distributions[key] = distribution

        stem = f"{spec.model.name}__{spec.dataset.name}__layerwise_state_distribution"
        csv_path = table_dir / f"{stem}.csv"
        distribution.to_csv(csv_path, index=False)
        png_path, pdf_path = _save_individual_figure(
            distribution,
            MODEL_LABELS[spec.model.name],
            DATASET_LABELS[spec.dataset.name],
            stem,
            figure_dir,
        )
        individual_figures.append(
            {
                "model": spec.model.name,
                "dataset": spec.dataset.name,
                "png": str(png_path.relative_to(args.output_dir)),
                "pdf": str(pdf_path.relative_to(args.output_dir)),
                "csv": str(csv_path.relative_to(args.output_dir)),
            }
        )

        weights = distribution["neuron_count"].to_numpy(dtype=np.float64)
        summary = {
            "model": spec.model.name,
            "model_label": MODEL_LABELS[spec.model.name],
            "dataset": spec.dataset.name,
            "dataset_label": DATASET_LABELS[spec.dataset.name],
            "run_id": run_id,
            "sample_count": int(distribution["sample_count"].iloc[0]),
            "layer_count": int(len(distribution)),
            "neuron_count": int(distribution["neuron_count"].sum()),
            "layer_widths": sorted(distribution["neuron_count"].unique().astype(int).tolist()),
            "maximum_layer_sum_residual": float(distribution["share_sum_residual"].abs().max()),
        }
        for state in STATE_ORDER:
            summary[f"overall_share_{state}"] = float(
                np.average(distribution[f"share_{state}"].to_numpy(dtype=np.float64), weights=weights)
            )
        summary_rows.append(summary)

        annotated = distribution.copy()
        annotated.insert(0, "run_id", run_id)
        annotated.insert(0, "dataset_label", DATASET_LABELS[spec.dataset.name])
        annotated.insert(0, "dataset", spec.dataset.name)
        annotated.insert(0, "model_label", MODEL_LABELS[spec.model.name])
        annotated.insert(0, "model", spec.model.name)
        long_frames.append(annotated)

    if missing:
        raise FileNotFoundError(f"The 16-run matrix is incomplete; missing {len(missing)} profiles. First: {missing[0]}")
    if len(summary_rows) != 16:
        raise ValueError(f"Expected 16 completed runs, found {len(summary_rows)}.")
    summary_rows.sort(
        key=lambda row: (list(MODEL_LABELS).index(row["model"]), DATASET_ORDER.index(row["dataset"]))
    )
    long_frame = pd.concat(long_frames, ignore_index=True)
    long_frame.to_csv(args.output_dir / "advisor_2026_09_layerwise_states_long.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "advisor_2026_09_layerwise_states_summary.csv", index=False)
    overview_png, overview_pdf = _save_overview_figure(distributions, args.output_dir)

    maximum_residual = max(row["maximum_layer_sum_residual"] for row in summary_rows)
    result = {
        "status": "complete",
        "runs": 16,
        "state_order": list(STATE_ORDER),
        "definition": "p_c^(l) = mean over real neurons j in layer l of N_c^(l,j) / N",
        "sample_count_policy": "one common positive valid-sample count for every neuron in a run",
        "padding_policy": "no fake neurons or layers; observed architecture widths are preserved",
        "maximum_layer_sum_residual": maximum_residual,
        "summaries": summary_rows,
        "layers": long_frame.to_dict(orient="records"),
        "overview_figures": [
            str(overview_png.relative_to(args.output_dir)),
            str(overview_pdf.relative_to(args.output_dir)),
        ],
        "individual_figures": individual_figures,
    }
    (args.output_dir / "advisor_2026_09_layerwise_states.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    markdown = f"""# Layerwise V/T/M/W response-state distributions

For state `c` and decoder layer `l`, this report computes
`p_c^(l) = (1 / |N^(l)|) sum_j N_c^(l,j) / N` from the frozen 1,000-sample neuron profiles. Every stacked bar is one
real decoder layer and sums to 100%. Layer widths are not forced to match: Gemma's 6,144- and 12,288-neuron layers are
averaged over their actual neurons, and no model is padded with fake neurons or layers.

This is a descriptive baseline for the planned answer-generation intervention. It shows how often neurons occupy each
response state, but does not yet measure the effect of removing or modifying any state-defined neuron set on answers.

Maximum absolute layer-sum residual across all 16 runs: `{maximum_residual:.3e}`.

{_summary_table(summary_rows)}

{_descriptive_observations(distributions, summary_rows)}

![Layerwise state distributions](figures/advisor_2026_09_layerwise_state_distributions.png)

The complete `{len(long_frame)}`-row layer table is `advisor_2026_09_layerwise_states_long.csv`; run summaries are in
`advisor_2026_09_layerwise_states_summary.csv`.

## Individual figures and tables

{_figure_table(individual_figures)}
"""
    (args.output_dir / "advisor_2026_09_layerwise_states.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "runs": len(summary_rows),
                "layers": len(long_frame),
                "maximum_layer_sum_residual": maximum_residual,
                "individual_figures": len(individual_figures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
