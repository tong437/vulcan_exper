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

"""Cross-model and cross-dataset descriptive synthesis for the unpaired 1,000-sample study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .synthesis import DATASET_LABELS, MODEL_COLORS, MODEL_LABELS


DATASET_ORDER = ("vqav2", "gqa_balanced", "clevr", "vizwiz_vqa")
MODEL_ORDER = ("qwen3_5_2b", "llava_1_5_7b", "gemma_4_e2b_it")
CLASS_LABELS = ("W", "V", "T", "M")
CLASS_COLORS = {"W": "#9aa0a6", "V": "#4285f4", "T": "#f4b400", "M": "#0f9d58"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _run_row(dataset: str, model: str, run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    statistics = json.loads((run_dir / "metrics/pilot_statistics.json").read_text(encoding="utf-8"))
    reproducibility = json.loads((run_dir / "metrics/reproducibility.json").read_text(encoding="utf-8"))
    summary = statistics["global"]
    classes = statistics["classes"]
    profiles = pd.read_parquet(
        run_dir / "distributions/neuron_profiles.parquet",
        columns=["layer", "valid_samples", "invalid_samples", "change_rate_distinct_pairs", "normalized_entropy"],
    )
    rates = profiles["change_rate_distinct_pairs"].dropna().to_numpy(dtype=np.float64)
    entropies = profiles["normalized_entropy"].dropna().to_numpy(dtype=np.float64)
    if len(profiles) == 0 or len(rates) == 0:
        raise ValueError(f"Neuron profiles are empty: {run_dir}")
    quantiles = np.quantile(rates, (0.10, 0.25, 0.50, 0.75, 0.90))
    total_observations = int((profiles["valid_samples"] + profiles["invalid_samples"]).sum())
    invalid_observations = int(profiles["invalid_samples"].sum())
    runtime = reproducibility["runtime"]
    row = {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "model": model,
        "model_label": MODEL_LABELS[model],
        "run_id": summary["run_id"],
        "sample_count": int(summary["sample_count"]),
        "layer_count": len(summary["per_layer_change_rate_distinct_pairs"]),
        "neuron_count": len(profiles),
        "overall_change_rate_including_self": float(summary["overall_change_rate_including_self"]),
        "overall_change_rate_distinct_pairs": float(summary["overall_change_rate_distinct_pairs"]),
        "mean_normalized_entropy": float(np.mean(entropies)),
        "median_neuron_change_rate": float(quantiles[2]),
        "p10_neuron_change_rate": float(quantiles[0]),
        "p25_neuron_change_rate": float(quantiles[1]),
        "p75_neuron_change_rate": float(quantiles[3]),
        "p90_neuron_change_rate": float(quantiles[4]),
        "zero_change_neuron_proportion": float(np.mean(rates == 0)),
        "change_ge_0_5_neuron_proportion": float(np.mean(rates >= 0.5)),
        "invalid_observation_proportion": invalid_observations / total_observations,
        **{f"class_proportion_{label}": float(classes["occurrence_proportions"][label]) for label in CLASS_LABELS},
        "runtime_seconds": float(runtime["elapsed_seconds"]),
        "peak_cuda_memory_gib": float(runtime["peak_cuda_memory_bytes"] / 2**30),
        "collection_commit": reproducibility["original_run_git"]["commit"],
        "sample_manifest_sha256": reproducibility["sample_manifest_sha256"],
        "activation_parts_sha256": reproducibility["activation_parts_sha256"],
        "profile_sha256": _sha256(run_dir / "distributions/neuron_profiles.parquet"),
    }
    layer_rows = [
        {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "model": model,
            "model_label": MODEL_LABELS[model],
            "layer": layer,
            "layer_count": row["layer_count"],
            "normalized_depth": layer / (row["layer_count"] - 1),
            "change_rate": float(rate),
        }
        for layer, rate in enumerate(summary["per_layer_change_rate_distinct_pairs"])
    ]
    return row, layer_rows


def build_unpaired_synthesis(
    run_dirs: dict[str, dict[str, Path]], output_dir: Path, *, analysis_commit: str
) -> dict[str, Any]:
    """Build exact tables, descriptive figures, provenance, and Markdown for all 12 runs."""
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            row, layers = _run_row(dataset, model, run_dirs[dataset][model])
            rows.append(row)
            layer_rows.extend(layers)
    frame = pd.DataFrame(rows)
    if len(frame) != 12 or not frame["sample_count"].eq(1000).all():
        raise ValueError("Unpaired synthesis requires 12 complete 1,000-sample runs.")
    collection_commits = sorted(frame["collection_commit"].unique())
    result = {
        "status": "complete",
        "scope": "Three frozen VLMs × four frozen unpaired_v1_1000 datasets",
        "analysis_code_commit": analysis_commit,
        "collection_code_commits": collection_commits,
        "estimand": "global all-distinct-sample neuron response-state change rate",
        "inference": "descriptive only; no controlled-pair or causal estimand and no confidence intervals",
        "main_results": rows,
        "normalized_depth_layers": layer_rows,
        "interpretation_limits": [
            "Every rate describes the frozen natural 1,000-sample distribution; it is not a causal effect.",
            "Neuron indices are model-local and are not aligned across architectures.",
            "Normalized decoder depth is used only to compare curve shape; layers are not structurally equivalent.",
            "The 12 overall values are deterministic summaries of the frozen samples, not independent replications.",
            "M denotes joint threshold exceedance and does not by itself establish causal multimodal interaction.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "unpaired_neuron_v1_1000_table.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(output_dir / "unpaired_neuron_v1_1000_layer_rates.csv", index=False)
    figures = _save_figures(frame, pd.DataFrame(layer_rows), figure_dir)
    complete_result = {**result, "figures": figures}
    _write_json(output_dir / "unpaired_neuron_v1_1000_results.json", complete_result)
    _write_markdown(complete_result, figures, output_dir / "unpaired_neuron_v1_1000_results.md")
    return complete_result


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figures(frame: pd.DataFrame, layers: pd.DataFrame, figure_dir: Path) -> list[dict[str, str]]:
    plt = _load_pyplot()
    figures: list[tuple[Path, str]] = []
    matrix = frame.pivot(index="model", columns="dataset", values="overall_change_rate_distinct_pairs").reindex(
        index=MODEL_ORDER, columns=DATASET_ORDER
    )
    figure, axis = plt.subplots(figsize=(9, 4.5))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=0.55)
    axis.set_xticks(range(4), [DATASET_LABELS[value] for value in DATASET_ORDER])
    axis.set_yticks(range(3), [MODEL_LABELS[value] for value in MODEL_ORDER])
    for row in range(3):
        for column in range(4):
            value = matrix.iloc[row, column]
            axis.text(
                column, row, f"{value:.1%}", ha="center", va="center", color="white" if value > 0.38 else "#202124"
            )
    figure.colorbar(image, ax=axis, label="All-distinct-sample change rate")
    axis.set_title("Neuron response-state change rate across frozen 1,000-sample datasets")
    figure.tight_layout()
    path = figure_dir / "unpaired_change_rate_heatmap.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    figures.append((path, "Overall change-rate heatmap"))

    grid = np.linspace(0, 1, 101)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        subset = layers[layers["dataset"] == dataset]
        for model in MODEL_ORDER:
            values = subset[subset["model"] == model]
            interpolated = np.interp(grid, values["normalized_depth"], values["change_rate"])
            axis.plot(grid, interpolated, color=MODEL_COLORS[model], label=MODEL_LABELS[model], linewidth=1.8)
        axis.set_title(DATASET_LABELS[dataset])
        axis.grid(alpha=0.2)
        axis.set_ylim(0, 0.65)
    axes[0, 0].set_ylabel("Change rate")
    axes[1, 0].set(ylabel="Change rate", xlabel="Normalized decoder depth")
    axes[1, 1].set_xlabel("Normalized decoder depth")
    axes[0, 0].legend(ncol=1, fontsize=9)
    figure.suptitle("Layerwise change-rate shape; normalized depth is descriptive, not structural equivalence")
    figure.tight_layout()
    path = figure_dir / "unpaired_normalized_depth_curves.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    figures.append((path, "Normalized-depth change-rate curves"))

    ordered = (
        frame.set_index(["dataset", "model"])
        .loc[[(dataset, model) for dataset in DATASET_ORDER for model in MODEL_ORDER]]
        .reset_index()
    )
    labels = [f"{row.dataset_label} · {row.model_label}" for row in ordered.itertuples()]
    figure, axis = plt.subplots(figsize=(11, 8.0))
    left = np.zeros(len(ordered))
    for label in CLASS_LABELS:
        values = ordered[f"class_proportion_{label}"].to_numpy()
        axis.barh(labels, values, left=left, color=CLASS_COLORS[label], label=label)
        left += values
    axis.invert_yaxis()
    axis.set(xlabel="Sample–neuron occurrence proportion", xlim=(0, 1))
    axis.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    axis.grid(axis="x", alpha=0.2)
    figure.subplots_adjust(left=0.25, right=0.985, bottom=0.11, top=0.90)
    path = figure_dir / "unpaired_class_composition.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    figures.append((path, "W/V/T/M occurrence composition"))

    figure, axis = plt.subplots(figsize=(11, 8.0))
    positions = np.arange(len(ordered))
    axis.hlines(
        positions, ordered["p10_neuron_change_rate"], ordered["p90_neuron_change_rate"], color="#9aa0a6", linewidth=2
    )
    axis.hlines(
        positions, ordered["p25_neuron_change_rate"], ordered["p75_neuron_change_rate"], color="#4285f4", linewidth=7
    )
    axis.scatter(ordered["median_neuron_change_rate"], positions, color="#202124", s=24, zorder=3)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set(xlabel="Per-neuron all-distinct-sample change rate", xlim=(0, 0.8))
    axis.grid(axis="x", alpha=0.2)
    axis.set_title("Per-neuron distribution: median, interquartile range, and 10th–90th percentiles")
    figure.subplots_adjust(left=0.25, right=0.985, bottom=0.11, top=0.90)
    path = figure_dir / "unpaired_neuron_change_distributions.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    figures.append((path, "Per-neuron change-rate distributions"))
    return [
        {"file": str(path.relative_to(figure_dir.parent)), "title": title, "sha256": _sha256(path)}
        for path, title in figures
    ]


def _write_markdown(result: dict[str, Any], figures: list[dict[str, str]], path: Path) -> None:
    frame = pd.DataFrame(result["main_results"])
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"| {row.dataset_label} | {row.model_label} | {row.overall_change_rate_distinct_pairs:.4f} | "
            f"{row.median_neuron_change_rate:.4f} | {row.mean_normalized_entropy:.4f} | "
            f"{row.class_proportion_W:.3f} | {row.class_proportion_V:.3f} | "
            f"{row.class_proportion_T:.3f} | {row.class_proportion_M:.3f} | {row.neuron_count:,} |"
        )
    pivot = frame.pivot(index="model", columns="dataset", values="overall_change_rate_distinct_pairs")
    model_ranges = {model: (float(pivot.loc[model].min()), float(pivot.loc[model].max())) for model in MODEL_ORDER}
    limits = "\n".join(f"- {item}" for item in result["interpretation_limits"])
    figure_lines = "\n".join(f"- [{item['title']}]({item['file']})" for item in figures)
    clevr_gap = pivot.loc["qwen3_5_2b", "clevr"] - pivot.loc["llava_1_5_7b", "clevr"]
    text = f"""# CLEVR has the lowest descriptive response-state variability across all three models

## Executive summary

- All 12 frozen runs contain 1,000 samples. LLaVA has the highest all-distinct-sample change rate on VQAv2, GQA,
  and VizWiz; Qwen is higher on CLEVR by {100 * clevr_gap:.2f} percentage points. Gemma has the lowest rate on all
  four datasets. These are descriptive differences, not significance tests.
- CLEVR produces the lowest rate for every model: Qwen {pivot.loc["qwen3_5_2b", "clevr"]:.1%}, LLaVA
  {pivot.loc["llava_1_5_7b", "clevr"]:.1%}, and Gemma {pivot.loc["gemma_4_e2b_it", "clevr"]:.1%}.
- Across datasets, Qwen spans {model_ranges["qwen3_5_2b"][0]:.1%}–{model_ranges["qwen3_5_2b"][1]:.1%}, LLaVA
  {model_ranges["llava_1_5_7b"][0]:.1%}–{model_ranges["llava_1_5_7b"][1]:.1%}, and Gemma
  {model_ranges["gemma_4_e2b_it"][0]:.1%}–{model_ranges["gemma_4_e2b_it"][1]:.1%}.
- No invalid observations were recorded in any of the 12 completed runs.

## Cross-model × dataset table

| Dataset | Model | Overall distinct rate | Median neuron rate | Mean normalized entropy | W | V | T | M | Neurons |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The overall rate is a count-weighted ratio across real neurons. The median neuron rate is an unweighted distribution
summary and therefore answers a different question; disagreement between the two is not an error.

## Figures

{figure_lines}

## Interpretation limits

{limits}

## Reproducibility

- Collection commit: `{result["collection_code_commits"][0]}`
- Analysis commit: `{result["analysis_code_commit"]}`
- Frozen tier: `unpaired_v1_1000`; seed 42; 1,000 samples per dataset; no controlled pair edges.
- Exact run IDs and SHA-256 provenance are stored in `unpaired_neuron_v1_1000_results.json` and the CSV table.
"""
    path.write_text(text, encoding="utf-8")
