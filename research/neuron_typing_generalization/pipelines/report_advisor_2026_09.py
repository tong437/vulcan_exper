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

"""Build the advisor's 4 x 4 change-rate and effective-category reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from neuron_typing_generalization.advisor_interpretability import (
    EFFECTIVE_BINS,
    REFERENCE_BANDS,
    smooth_histogram,
    summarize_effective_categories,
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


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_effective_category_figure(
    histograms: dict[tuple[str, str], np.ndarray], summaries: dict[tuple[str, str], dict[str, Any]], output_dir: Path
) -> list[Path]:
    plt = _load_pyplot()
    centers = (EFFECTIVE_BINS[:-1] + EFFECTIVE_BINS[1:]) / 2
    maximum = max(
        max(float(histogram.max()), float(smooth_histogram(histogram).max())) for histogram in histograms.values()
    )
    y_max = max(5.0, np.ceil(maximum * 1.12 / 5.0) * 5.0)
    figure, axes = plt.subplots(4, 4, figsize=(17, 13), sharex=True, sharey=True)
    for row, model in enumerate(MODEL_LABELS):
        for column, dataset in enumerate(DATASET_ORDER):
            axis = axes[row, column]
            histogram = histograms[(model, dataset)]
            summary = summaries[(model, dataset)]
            axis.bar(
                centers,
                histogram,
                width=0.044,
                color="#4C78A8",
                alpha=0.78,
                edgecolor="none",
                label="Histogram (bin = 0.05)",
            )
            axis.plot(centers, smooth_histogram(histogram), color="#E07A5F", linewidth=1.8, label="Smoothed density")
            for reference in (1, 2, 3, 4):
                axis.axvline(reference, color="#A9B4C2", linewidth=0.8, linestyle=(0, (2, 3)), zorder=0)
            axis.text(
                0.98,
                0.95,
                f"Median {summary['effective_category_median']:.2f}\n"
                f"IQR {summary['effective_category_q25']:.2f}–{summary['effective_category_q75']:.2f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#526070",
            )
            if row == 0:
                axis.set_title(DATASET_LABELS[dataset], fontsize=12, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{MODEL_LABELS[model]}\nNeurons per bin (%)")
            if row == len(MODEL_LABELS) - 1:
                axis.set_xlabel(r"Effective category count $K_{eff}$")
            axis.set_xlim(1.0, 4.0)
            axis.set_ylim(0.0, y_max)
            axis.set_xticks([1, 2, 3, 4])
            axis.grid(axis="y", alpha=0.18, linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    figure.suptitle("Distribution of Effective Category Counts", fontsize=18, fontweight="bold", y=0.995)
    figure.text(
        0.5,
        0.973,
        r"All decoder MLP neurons; 1,000 samples per run; $K_{eff}=\exp(S)=4^{H_{norm}}$",
        ha="center",
        va="top",
        fontsize=10,
        color="#526070",
    )
    figure.tight_layout(rect=(0.02, 0.03, 1.0, 0.94))
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / "advisor_2026_09_effective_category_distributions.png",
        figure_dir / "advisor_2026_09_effective_category_distributions.pdf",
    ]
    figure.savefig(paths[0], dpi=240, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _save_reference_style_figures(
    histograms: dict[tuple[str, str], np.ndarray], summaries: dict[tuple[str, str], dict[str, Any]], output_dir: Path
) -> list[dict[str, str]]:
    """Write one advisor-reference-style effective-category plot per run."""
    plt = _load_pyplot()
    centers = (EFFECTIVE_BINS[:-1] + EFFECTIVE_BINS[1:]) / 2
    global_maximum = max(
        max(float(histogram.max()), float(smooth_histogram(histogram).max())) for histogram in histograms.values()
    )
    y_max = max(10.0, np.ceil(global_maximum * 1.30 / 5.0) * 5.0)
    figure_dir = output_dir / "figures/effective_category_distributions"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, str]] = []
    band_centers = (1.25, 2.0, 3.0, 3.75)

    for model in MODEL_LABELS:
        for dataset in DATASET_ORDER:
            summary = summaries[(model, dataset)]
            histogram = histograms[(model, dataset)]
            density = smooth_histogram(histogram, bandwidth=0.12)
            figure = plt.figure(figsize=(19, 11.2), facecolor="#FFFFFF")
            axis = figure.add_axes([0.095, 0.19, 0.86, 0.61], facecolor="#F8FAFC")

            figure.text(
                0.095,
                0.945,
                "Distribution of Effective Category Numbers",
                ha="left",
                va="top",
                fontsize=30,
                fontweight="bold",
                color="#172033",
            )
            figure.text(
                0.095,
                0.885,
                f"{MODEL_LABELS[model]} × {DATASET_LABELS[dataset]}   |   "
                f"Neurons: n = {summary['neuron_count']:,}   |   Samples per neuron: 1,000   |   "
                f"Median = {summary['effective_category_median']:.2f}, "
                f"IQR = {summary['effective_category_q25']:.2f}–{summary['effective_category_q75']:.2f}",
                ha="left",
                va="top",
                fontsize=15,
                color="#526070",
            )

            bars = axis.bar(
                centers,
                histogram,
                width=0.044,
                color="#4C78A8",
                edgecolor="none",
                label="Histogram (bin = 0.05)",
                zorder=2,
            )
            (line,) = axis.plot(
                centers,
                density,
                color="#E07A5F",
                linewidth=3.0,
                label="Smoothed density",
                zorder=3,
            )
            for reference in (1, 2, 3, 4):
                axis.axvline(reference, color="#A9B4C2", linewidth=1.1, linestyle=(0, (4, 5)), zorder=1)

            for center, (band_name, _, _) in zip(band_centers, REFERENCE_BANDS):
                display_name = band_name.replace("_", " ").title()
                share = summary[f"{band_name}_proportion"]
                axis.text(
                    center,
                    y_max * 0.91,
                    f"{display_name}\n{share:.1%}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#526070",
                    linespacing=1.5,
                    bbox={
                        "boxstyle": "round,pad=0.7",
                        "facecolor": "#FFFFFF",
                        "edgecolor": "#C8D1DC",
                        "linewidth": 1.0,
                    },
                    zorder=5,
                )

            axis.set_xlim(1.0, 4.0)
            axis.set_ylim(0.0, y_max)
            axis.set_xticks(np.arange(1.0, 4.01, 0.5))
            axis.set_xticklabels([f"{value:.1f}" for value in np.arange(1.0, 4.01, 0.5)])
            axis.set_yticks(np.arange(0.0, y_max + 0.1, 5.0))
            axis.set_yticklabels([f"{value:.0f}%" for value in np.arange(0.0, y_max + 0.1, 5.0)])
            axis.set_xlabel(
                r"Effective Number of Categories, $K_{eff}=\exp(S)=4^{H_{norm}}$", fontsize=14, labelpad=18
            )
            axis.set_ylabel("Percentage of Neurons per Bin", fontsize=14, labelpad=18)
            axis.tick_params(axis="both", labelsize=12, colors="#526070")
            axis.grid(axis="y", color="#D6DEE8", linewidth=1.0, zorder=0)
            for spine in axis.spines.values():
                spine.set_color("#8593A3")
                spine.set_linewidth(1.2)
            figure.legend(
                [bars, line],
                ["Histogram (bin = 0.05)", "Smoothed density"],
                loc="upper right",
                bbox_to_anchor=(0.955, 0.845),
                ncol=2,
                frameon=False,
                fontsize=12,
                handlelength=2.4,
            )
            figure.text(
                0.955,
                0.075,
                "Reference bands summarize the distribution; they are not clustering thresholds.",
                ha="right",
                va="center",
                fontsize=11,
                color="#526070",
            )

            stem = f"{model}__{dataset}__effective_category_distribution"
            png_path = figure_dir / f"{stem}.png"
            pdf_path = figure_dir / f"{stem}.pdf"
            figure.savefig(png_path, dpi=180, facecolor=figure.get_facecolor())
            figure.savefig(pdf_path, facecolor=figure.get_facecolor())
            plt.close(figure)
            outputs.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "png": str(png_path.relative_to(output_dir)),
                    "pdf": str(pdf_path.relative_to(output_dir)),
                }
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    specs = expand_study(load_study(args.study))
    rows = []
    interpretability_rows = []
    histograms: dict[tuple[str, str], np.ndarray] = {}
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    missing = []
    for spec in specs:
        run_dir = resolve_artifact_root(spec) / "runs" / spec.study / build_run_id(spec)
        profile_path = run_dir / "distributions/neuron_profiles.parquet"
        if not profile_path.is_file():
            missing.append(str(profile_path))
            continue
        profiles = pd.read_parquet(
            profile_path,
            columns=[
                "valid_samples",
                "invalid_samples",
                "change_rate_distinct_pairs",
                "entropy",
                "normalized_entropy",
                "effective_class_count",
            ],
        )
        rates = profiles["change_rate_distinct_pairs"].dropna()
        if rates.empty:
            raise ValueError(f"No defined neuron rates in {profile_path}.")
        rows.append(
            {
                "model": spec.model.name,
                "model_label": MODEL_LABELS[spec.model.name],
                "dataset": spec.dataset.name,
                "overall_change_rate": float(rates.mean()),
                "included_neurons": int(len(rates)),
                "excluded_neurons": int(len(profiles) - len(rates)),
                "invalid_observations": int(profiles["invalid_samples"].sum()),
                "run_id": build_run_id(spec),
            }
        )
        summary, histogram = summarize_effective_categories(profiles)
        key = (spec.model.name, spec.dataset.name)
        summaries[key] = summary
        histograms[key] = histogram
        interpretability_rows.append(
            {
                "model": spec.model.name,
                "model_label": MODEL_LABELS[spec.model.name],
                "dataset": spec.dataset.name,
                "dataset_label": DATASET_LABELS[spec.dataset.name],
                "run_id": build_run_id(spec),
                "sample_count": int(profiles["valid_samples"].max()),
                **summary,
            }
        )
    if missing:
        raise FileNotFoundError(
            f"The 16-run matrix is incomplete; missing {len(missing)} profiles. First: {missing[0]}"
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 16:
        raise ValueError(f"Expected 16 completed runs, found {len(frame)}.")
    matrix = frame.pivot(index="model_label", columns="dataset", values="overall_change_rate").reindex(
        index=list(MODEL_LABELS.values()), columns=DATASET_ORDER
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "advisor_2026_09_rates_long.csv", index=False)
    matrix.to_csv(args.output_dir / "advisor_2026_09_rates_matrix.csv")
    result = {"estimand": "arithmetic mean of per-neuron all-distinct-sample change rates", "runs": rows}
    (args.output_dir / "advisor_2026_09_rates.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    headers = "| Model / Dataset | MME | MMBench | TextVQA | ChartQA |\n|---|---:|---:|---:|---:|"
    body = "\n".join(
        f"| {model} | " + " | ".join(f"{matrix.loc[model, dataset]:.6f}" for dataset in DATASET_ORDER) + " |"
        for model in matrix.index
    )
    (args.output_dir / "advisor_2026_09_rates.md").write_text(headers + "\n" + body + "\n", encoding="utf-8")

    interpretation = pd.DataFrame(interpretability_rows)
    interpretation.to_csv(args.output_dir / "advisor_2026_09_entropy_effective_long.csv", index=False)
    median_matrix = interpretation.pivot(
        index="model_label", columns="dataset", values="effective_category_median"
    ).reindex(index=list(MODEL_LABELS.values()), columns=DATASET_ORDER)
    median_matrix.to_csv(args.output_dir / "advisor_2026_09_effective_median_matrix.csv")
    entropy_matrix = interpretation.pivot(
        index="model_label", columns="dataset", values="normalized_entropy_mean"
    ).reindex(index=list(MODEL_LABELS.values()), columns=DATASET_ORDER)
    entropy_matrix.to_csv(args.output_dir / "advisor_2026_09_entropy_mean_matrix.csv")
    figure_paths = _save_effective_category_figure(histograms, summaries, args.output_dir)
    individual_figures = _save_reference_style_figures(histograms, summaries, args.output_dir)
    interpretability_result = {
        "status": "complete",
        "runs": 16,
        "entropy_definition": "H_norm = -sum_c(p_c * log(p_c)) / log(4)",
        "effective_category_definition": "K_eff = exp(S) = 4 ** H_norm, where S is unnormalized Shannon entropy",
        "reference_bands": {
            "near_1": "[1.0, 1.5)",
            "near_2": "[1.5, 2.5)",
            "near_3": "[2.5, 3.5)",
            "near_4": "[3.5, 4.0]",
        },
        "reference_band_note": "Descriptive intervals only; they are not clustering or hypothesis-test thresholds.",
        "rows": interpretability_rows,
        "figures": [str(path.relative_to(args.output_dir)) for path in figure_paths],
        "individual_figures": individual_figures,
    }
    (args.output_dir / "advisor_2026_09_interpretability.json").write_text(
        json.dumps(interpretability_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    effective_headers = "| Model / Dataset | MME | MMBench | TextVQA | ChartQA |\n|---|---:|---:|---:|---:|"
    effective_body = "\n".join(
        f"| {model} | "
        + " | ".join(f"{median_matrix.loc[model, dataset]:.3f}" for dataset in DATASET_ORDER)
        + " |"
        for model in median_matrix.index
    )
    individual_rows = []
    for model_label, model in zip(MODEL_LABELS.values(), MODEL_LABELS):
        links = []
        for dataset in DATASET_ORDER:
            match = next(
                item for item in individual_figures if item["model"] == model and item["dataset"] == dataset
            )
            links.append(f"[{DATASET_LABELS[dataset]} PNG]({match['png']}) / [PDF]({match['pdf']})")
        individual_rows.append(f"| {model_label} | " + " | ".join(links) + " |")
    individual_table = (
        "| Model | MME | MMBench | TextVQA | ChartQA |\n|---|---|---|---|---|\n" + "\n".join(individual_rows)
    )
    interpretability_markdown = f"""# State entropy and effective category counts

For each neuron, let `S = -sum_c p_c log(p_c)` and `H_norm = S / log(4)`. The normalized entropy lies in `[0, 1]`.
The effective category count is `K_eff = exp(S) = 4 ** H_norm`, so one/two/three/four equally occupied states map
exactly to `1/2/3/4`. The table reports the median across all decoder MLP neurons in each run.

{effective_headers}
{effective_body}

The complete mean, quantile, and descriptive reference-band summaries are stored in
`advisor_2026_09_entropy_effective_long.csv`. The intervals around 1/2/3/4 in that file are descriptive only and are
not clustering thresholds.

![Effective category distributions](figures/advisor_2026_09_effective_category_distributions.png)

## Individual reference-style figures

{individual_table}
"""
    (args.output_dir / "advisor_2026_09_interpretability.md").write_text(
        interpretability_markdown, encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "complete", "runs": 16, "overview_figures": len(figure_paths), "individual_figures": 16},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
