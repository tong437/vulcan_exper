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

"""Build the advisor's 4 x 4 state-pair decomposition report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter
from neuron_typing_generalization.advisor_state_pairs import PAIR_LABELS, summarize_state_pair_contributions
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
PAIR_COLORS = {
    "VT": "#8064A2",
    "VM": "#3979B7",
    "VW": "#78ACCF",
    "TM": "#D68B35",
    "TW": "#E5B969",
    "MW": "#599986",
}


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _shared_y_max(rows: list[dict[str, Any]]) -> float:
    maximum = max(max(row[f"share_{pair}"] for pair in PAIR_LABELS) for row in rows)
    return float(min(1.0, max(0.60, np.ceil((maximum + 0.06) / 0.15) * 0.15)))


def _save_individual_figures(
    rows: list[dict[str, Any]], output_dir: Path, y_max: float
) -> list[dict[str, str]]:
    plt = _load_pyplot()
    figure_dir = output_dir / "figures/state_pair_composition"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    x = np.arange(len(PAIR_LABELS))
    colors = [PAIR_COLORS[pair] for pair in PAIR_LABELS]

    for row in rows:
        shares = np.array([row[f"share_{pair}"] for pair in PAIR_LABELS], dtype=np.float64)
        figure, axis = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
        figure.subplots_adjust(left=0.13, right=0.97, bottom=0.19, top=0.76)
        figure.suptitle("State-pair composition", y=0.97, fontsize=17, fontweight="bold", color="#172033")
        figure.text(
            0.55,
            0.858,
            f"{row['model_label']} | {row['dataset_label']}",
            ha="center",
            fontsize=10.5,
            color="#64748B",
        )
        axis.bar(x, shares, width=0.72, color=colors, edgecolor="none", zorder=3)
        axis.set_ylim(0.0, y_max)
        axis.set_xlim(-0.65, 5.65)
        axis.set_xticks(x, PAIR_LABELS)
        axis.set_xlabel("Unordered state pair", labelpad=10)
        axis.set_ylabel("Share of all differences", labelpad=10)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.yaxis.set_major_locator(MultipleLocator(0.15))
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.9, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#CBD5E1")
        axis.tick_params(axis="both", length=0, pad=7)
        label_offset = y_max * 0.018
        for index, value in enumerate(shares):
            axis.text(
                index,
                value + label_offset,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                color="#475569",
            )
        axis.text(
            0.98,
            0.97,
            f"R = {row['overall_change_rate']:.1%}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            color="#475569",
        )
        stem = f"{row['model']}__{row['dataset']}__state_pair_composition"
        png_path = figure_dir / f"{stem}.png"
        pdf_path = figure_dir / f"{stem}.pdf"
        figure.savefig(png_path, dpi=300, facecolor="white")
        figure.savefig(pdf_path, facecolor="white")
        plt.close(figure)
        outputs.append(
            {
                "model": row["model"],
                "dataset": row["dataset"],
                "png": str(png_path.relative_to(output_dir)),
                "pdf": str(pdf_path.relative_to(output_dir)),
            }
        )
    return outputs


def _save_overview_figure(rows: list[dict[str, Any]], output_dir: Path, y_max: float) -> list[Path]:
    plt = _load_pyplot()
    row_by_key = {(row["model"], row["dataset"]): row for row in rows}
    x = np.arange(len(PAIR_LABELS))
    colors = [PAIR_COLORS[pair] for pair in PAIR_LABELS]
    figure, axes = plt.subplots(4, 4, figsize=(17, 12.8), sharex=True, sharey=True, facecolor="white")
    for model_index, model in enumerate(MODEL_LABELS):
        for dataset_index, dataset in enumerate(DATASET_ORDER):
            axis = axes[model_index, dataset_index]
            row = row_by_key[(model, dataset)]
            shares = [row[f"share_{pair}"] for pair in PAIR_LABELS]
            axis.bar(x, shares, width=0.74, color=colors, edgecolor="none", zorder=3)
            axis.set_ylim(0.0, y_max)
            axis.set_xticks(x, PAIR_LABELS)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            axis.yaxis.set_major_locator(MultipleLocator(0.15))
            axis.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#CBD5E1")
            axis.tick_params(axis="both", length=0, pad=5, labelsize=9)
            axis.text(
                0.97,
                0.94,
                f"R = {row['overall_change_rate']:.1%}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#475569",
            )
            if model_index == 0:
                axis.set_title(DATASET_LABELS[dataset], fontsize=12, fontweight="bold", pad=10)
            if dataset_index == 0:
                axis.set_ylabel(f"{MODEL_LABELS[model]}\nShare", labelpad=9)
            if model_index == len(MODEL_LABELS) - 1:
                axis.set_xlabel("State pair", labelpad=7)
    figure.suptitle("State-pair Composition Across Models and Datasets", fontsize=19, fontweight="bold", y=0.995)
    figure.text(
        0.5,
        0.971,
        "Bars are shares of overall change rate R; all panels use one y-axis scale",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#526070",
    )
    figure.tight_layout(rect=(0.02, 0.025, 1.0, 0.95))
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / "advisor_2026_09_state_pair_composition.png",
        figure_dir / "advisor_2026_09_state_pair_composition.pdf",
    ]
    figure.savefig(paths[0], dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return paths


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| Model | Dataset | R | VT | VM | VW | TM | TW | MW | Dominant pair |\n"
    divider = "|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
    body = []
    for row in rows:
        dominant = max(PAIR_LABELS, key=lambda pair: row[f"share_{pair}"])
        shares = " | ".join(f"{row[f'share_{pair}']:.1%}" for pair in PAIR_LABELS)
        body.append(
            f"| {row['model_label']} | {row['dataset_label']} | {row['overall_change_rate']:.6f} | "
            f"{shares} | {dominant} |"
        )
    return header + divider + "\n".join(body)


def _figure_table(figures: list[dict[str, str]]) -> str:
    lines = ["| Model | MME | MMBench | TextVQA | ChartQA |", "|---|---|---|---|---|"]
    for model, model_label in MODEL_LABELS.items():
        links = []
        for dataset in DATASET_ORDER:
            match = next(item for item in figures if item["model"] == model and item["dataset"] == dataset)
            links.append(f"[PNG]({match['png']}) / [PDF]({match['pdf']})")
        lines.append(f"| {model_label} | " + " | ".join(links) + " |")
    return "\n".join(lines)


def _descriptive_observations(rows: list[dict[str, Any]]) -> str:
    dominant_counts = dict.fromkeys(PAIR_LABELS, 0)
    exceptions = []
    weak_involving = []
    for row in rows:
        dominant = max(PAIR_LABELS, key=lambda pair: row[f"share_{pair}"])
        dominant_counts[dominant] += 1
        if dominant != "TW":
            exceptions.append(f"{row['model_label']} × {row['dataset_label']} ({dominant})")
        weak_involving.append(row["share_VW"] + row["share_TW"] + row["share_MW"])
    equal_run_means = {
        pair: float(np.mean([row[f"share_{pair}"] for row in rows])) for pair in PAIR_LABELS
    }
    ordered = sorted(PAIR_LABELS, key=equal_run_means.get, reverse=True)
    mean_text = ", ".join(f"{pair} {equal_run_means[pair]:.1%}" for pair in ordered)
    return (
        "## Descriptive observations\n\n"
        f"- `TW` is the largest component in {dominant_counts['TW']}/16 runs. The two exceptions are "
        f"{', '.join(exceptions)}, where `VW` is largest.\n"
        f"- Giving each run equal descriptive weight, the mean composition is {mean_text}. This is a summary of "
        "the 16 normalized panels, not a new pooled-neuron estimand.\n"
        f"- Pairs involving `W` (`VW + TW + MW`) account for {np.mean(weak_involving):.1%} on average across runs "
        f"(range {np.min(weak_involving):.1%}–{np.max(weak_involving):.1%}). Thus most observed differences "
        "involve a sample switching between weak-on-both and one of the other response states.\n"
        "- These pairs are unordered: for example, `TW` does not distinguish `T→W` from `W→T`, and none of the "
        "components is a causal transition effect.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    long_rows = []
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
                "valid_samples",
                "count_V",
                "count_T",
                "count_M",
                "count_W",
                "change_rate_distinct_pairs",
            ],
        )
        summary = summarize_state_pair_contributions(profile)
        row = {
            "model": spec.model.name,
            "model_label": MODEL_LABELS[spec.model.name],
            "dataset": spec.dataset.name,
            "dataset_label": DATASET_LABELS[spec.dataset.name],
            "run_id": run_id,
            "included_neurons": summary["included_neurons"],
            "excluded_neurons": summary["excluded_neurons"],
            "overall_change_rate": summary["overall_change_rate"],
            "maximum_neuron_identity_residual": summary["maximum_neuron_identity_residual"],
        }
        for pair in PAIR_LABELS:
            row[f"R_{pair}"] = summary["absolute_contributions"][pair]
            row[f"share_{pair}"] = summary["composition_shares"][pair]
            long_rows.append(
                {
                    "model": spec.model.name,
                    "model_label": MODEL_LABELS[spec.model.name],
                    "dataset": spec.dataset.name,
                    "dataset_label": DATASET_LABELS[spec.dataset.name],
                    "run_id": run_id,
                    "pair": pair,
                    "absolute_contribution": summary["absolute_contributions"][pair],
                    "composition_share": summary["composition_shares"][pair],
                    "overall_change_rate": summary["overall_change_rate"],
                    "included_neurons": summary["included_neurons"],
                }
            )
        rows.append(row)

    if missing:
        raise FileNotFoundError(f"The 16-run matrix is incomplete; missing {len(missing)} profiles. First: {missing[0]}")
    if len(rows) != 16:
        raise ValueError(f"Expected 16 completed runs, found {len(rows)}.")
    rows.sort(key=lambda row: (list(MODEL_LABELS).index(row["model"]), DATASET_ORDER.index(row["dataset"])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "advisor_2026_09_state_pair_summary.csv", index=False)
    pd.DataFrame(long_rows).to_csv(args.output_dir / "advisor_2026_09_state_pair_long.csv", index=False)
    y_max = _shared_y_max(rows)
    individual_figures = _save_individual_figures(rows, args.output_dir, y_max)
    overview_figures = _save_overview_figure(rows, args.output_dir, y_max)

    maximum_identity_residual = max(abs(row["maximum_neuron_identity_residual"]) for row in rows)
    result = {
        "status": "complete",
        "runs": 16,
        "pair_order": list(PAIR_LABELS),
        "per_neuron_definition": "R_ab = 2 * N_a * N_b / (N * (N - 1))",
        "overall_definition": "R_ab = arithmetic mean over included neurons of the absolute per-neuron R_ab",
        "composition_definition": "share_ab = overall R_ab / sum_ab(overall R_ab); normalization follows averaging",
        "identity": "R = R_VT + R_VM + R_VW + R_TM + R_TW + R_MW",
        "maximum_neuron_identity_residual": maximum_identity_residual,
        "rows": rows,
        "overview_figures": [str(path.relative_to(args.output_dir)) for path in overview_figures],
        "individual_figures": individual_figures,
    }
    (args.output_dir / "advisor_2026_09_state_pairs.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    markdown = f"""# State-pair decomposition

For each neuron and unordered state pair `ab`, the absolute contribution is
`R_ab = 2 N_a N_b / (N(N-1))`. Each reported `R_ab` is first averaged over all included decoder MLP neurons. The
six averaged contributions sum to the previously reported overall change rate `R`; the displayed percentages are then
computed as `R_ab / R`. This order avoids giving a low-change neuron the same composition weight as a high-change
neuron and avoids the undefined per-neuron composition when `R=0`.

Maximum absolute neuron-level identity residual across all 16 runs: `{maximum_identity_residual:.3e}`.

{_markdown_table(rows)}

{_descriptive_observations(rows)}

![State-pair composition overview](figures/advisor_2026_09_state_pair_composition.png)

Absolute contributions and normalized shares are stored in `advisor_2026_09_state_pair_long.csv`; the one-row-per-run
table is `advisor_2026_09_state_pair_summary.csv`.

## Individual figures

{_figure_table(individual_figures)}
"""
    (args.output_dir / "advisor_2026_09_state_pairs.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "runs": len(rows),
                "pairs": len(long_rows),
                "maximum_neuron_identity_residual": maximum_identity_residual,
                "individual_figures": len(individual_figures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
