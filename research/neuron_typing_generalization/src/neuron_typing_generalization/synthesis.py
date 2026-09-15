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

"""Cross-model paired inference and normalized-depth study synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import paired_bootstrap_ratio_difference


MODEL_LABELS = {
    "gemma_4_e2b_it": "Gemma-4-E2B-IT",
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_5_2b": "Qwen3.5-2B",
}
DATASET_LABELS = {
    "vqav2": "VQAv2",
    "gqa_balanced": "GQA Balanced",
    "clevr": "CLEVR",
    "vizwiz_vqa": "VizWiz VQA",
}
MODEL_COLORS = {
    "gemma_4_e2b_it": "#0f9d58",
    "llava_1_5_7b": "#db4437",
    "qwen3_5_2b": "#4285f4",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_run(run_dir: Path) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "statistics": json.loads((run_dir / "metrics/pilot_statistics.json").read_text(encoding="utf-8")),
        "reproducibility": json.loads((run_dir / "metrics/reproducibility.json").read_text(encoding="utf-8")),
    }


def _cluster_counts(run_dir: Path) -> pd.DataFrame:
    """Aggregate overall controlled-edge counts to independent manifest bootstrap units."""
    changes = pd.read_parquet(run_dir / "distributions/pairwise_changes.parquet")
    changes = changes[(changes["pair_scope"] == "controlled") & changes["layer"].isna()].copy()
    manifest = pd.read_parquet(run_dir / "samples/pair_manifest.parquet")
    keys = ["left_sample_id", "right_sample_id"]
    changes[keys] = changes[keys].astype(str)
    manifest[keys] = manifest[keys].astype(str)
    merged = changes.merge(manifest[keys + ["pair_group_id"]], on=keys, how="left", validate="one_to_one")
    if merged["pair_group_id"].isna().any() or len(merged) != len(manifest):
        raise ValueError(f"Controlled changes do not align with the pair manifest: {run_dir}")
    return merged.groupby("pair_group_id", sort=True)[["changed_neurons", "valid_neurons"]].sum().astype(np.int64)


def paired_model_differences(
    runs: dict[str, Path],
    comparisons: list[tuple[str, str]],
    *,
    replicates: int = 20_000,
    seed: int = 42,
    familywise_comparisons: int = 1,
) -> list[dict[str, Any]]:
    """Estimate paired left-minus-right model differences on shared clusters."""
    if familywise_comparisons < 1:
        raise ValueError("familywise_comparisons must be positive.")
    familywise_confidence_level = 1.0 - 0.05 / familywise_comparisons
    counts = {model: _cluster_counts(path) for model, path in runs.items()}
    reference_index = next(iter(counts.values())).index
    if any(not frame.index.equals(reference_index) for frame in counts.values()):
        raise ValueError("Models do not share the same ordered bootstrap clusters.")
    rows = []
    for left_model, right_model in comparisons:
        left, right = counts[left_model], counts[right_model]
        result = paired_bootstrap_ratio_difference(
            left["changed_neurons"].to_numpy(),
            left["valid_neurons"].to_numpy(),
            right["changed_neurons"].to_numpy(),
            right["valid_neurons"].to_numpy(),
            replicates=replicates,
            seed=seed,
        )
        familywise_result = paired_bootstrap_ratio_difference(
            left["changed_neurons"].to_numpy(),
            left["valid_neurons"].to_numpy(),
            right["changed_neurons"].to_numpy(),
            right["valid_neurons"].to_numpy(),
            replicates=replicates,
            seed=seed,
            confidence_level=familywise_confidence_level,
        )
        rows.append(
            {
                "left_model": left_model,
                "right_model": right_model,
                "contrast": f"{left_model} - {right_model}",
                "estimate": float(result.estimate[0]),
                "standard_error": float(result.standard_error[0]),
                "ci_lower": float(result.lower[0]),
                "ci_upper": float(result.upper[0]),
                "confidence_level": result.confidence_level,
                "bootstrap_replicates": result.replicates,
                "bootstrap_seed": result.seed,
                "bootstrap_unit_count": len(reference_index),
                "ci_excludes_zero": bool(result.lower[0] > 0 or result.upper[0] < 0),
                "familywise_ci_lower": float(familywise_result.lower[0]),
                "familywise_ci_upper": float(familywise_result.upper[0]),
                "familywise_confidence_level": familywise_confidence_level,
                "familywise_ci_excludes_zero": bool(familywise_result.lower[0] > 0 or familywise_result.upper[0] < 0),
            }
        )
    return rows


def normalized_depth_curves(
    runs_by_dataset: dict[str, dict[str, dict[str, Any]]], grid_size: int = 101
) -> list[dict[str, Any]]:
    """Interpolate per-layer rates onto a common relative-depth grid for visualization only."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least two.")
    grid = np.linspace(0.0, 1.0, grid_size)
    rows = []
    for dataset, model_runs in runs_by_dataset.items():
        for model, run in model_runs.items():
            controlled = run["statistics"]["controlled"]
            if controlled["status"] == "complete":
                raw = np.asarray([item["estimate"] for item in controlled["per_layer"]], dtype=np.float64)
                estimand = "controlled_pair_change_rate"
            else:
                raw = np.asarray(run["statistics"]["global"]["per_layer_change_rate_distinct_pairs"], dtype=np.float64)
                estimand = "global_distinct_pair_change_rate"
            source_depth = np.linspace(0.0, 1.0, len(raw))
            interpolated = np.interp(grid, source_depth, raw)
            rows.extend(
                {
                    "dataset": dataset,
                    "model": model,
                    "estimand": estimand,
                    "normalized_depth": float(depth),
                    "change_rate": float(rate),
                    "source_layer_count": len(raw),
                }
                for depth, rate in zip(grid, interpolated)
            )
    return rows


def build_synthesis_report(
    run_dirs: dict[str, dict[str, Path]],
    output_dir: Path,
    *,
    analysis_commit: str,
    comparisons: list[tuple[str, str]],
    bootstrap_replicates: int = 20_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Build the final four-dataset main table, paired contrasts, curves, figures, and report."""
    loaded = {
        dataset: {model: _load_run(path) for model, path in model_runs.items()}
        for dataset, model_runs in run_dirs.items()
    }
    main_rows = []
    for dataset, model_runs in loaded.items():
        for model, run in model_runs.items():
            statistics = run["statistics"]
            controlled = statistics["controlled"]
            if controlled["status"] == "complete":
                overall = controlled["overall"]
                estimand = "controlled_pair_change_rate"
                estimate, lower, upper = overall["estimate"], overall["ci_lower"], overall["ci_upper"]
                units = len(_cluster_counts(run["run_dir"]))
            else:
                estimand = "global_distinct_pair_change_rate_descriptive"
                estimate = statistics["global"]["overall_change_rate_distinct_pairs"]
                lower = upper = None
                units = 0
            classes = statistics["classes"]
            main_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "estimand": estimand,
                    "sample_count": statistics["global"]["sample_count"],
                    "bootstrap_unit_count": units,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "mean_normalized_entropy": classes["mean_normalized_entropy"],
                    **{f"proportion_{label}": classes["occurrence_proportions"][label] for label in "WVTM"},
                }
            )

    contrasts = []
    familywise_comparisons = len(comparisons) * 3
    for dataset in ("vqav2", "gqa_balanced", "clevr"):
        dataset_rows = paired_model_differences(
            {model: value["run_dir"] for model, value in loaded[dataset].items()},
            comparisons,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            familywise_comparisons=familywise_comparisons,
        )
        contrasts.extend({"dataset": dataset, **row} for row in dataset_rows)
    curves = normalized_depth_curves(loaded)
    result = {
        "status": "complete",
        "scope": "Three frozen VLMs across VQAv2, GQA Balanced, CLEVR audited_v2_111, and VizWiz VQA",
        "analysis_code_commit": analysis_commit,
        "paired_bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "familywise_comparisons": familywise_comparisons,
            "familywise_confidence_level": 1.0 - 0.05 / familywise_comparisons,
            "multiplicity": "Bonferroni 95% familywise intervals across all nine prespecified contrasts",
        },
        "main_results": main_rows,
        "paired_model_contrasts": contrasts,
        "normalized_depth_curves": curves,
        "interpretation_limits": [
            "Paired model contrasts are valid only within a dataset because pairing interventions differ by dataset.",
            "Both unadjusted pairwise 95% intervals and Bonferroni 95% familywise intervals are reported.",
            "Normalized-depth interpolation is descriptive and does not assert layer-to-layer architectural equivalence.",
            "VizWiz has no controlled pair estimand and is excluded from model-difference inference.",
            "Response classes are input-conditional states and do not establish causal multimodal interaction.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "main_synthesis_results.json", result)
    pd.DataFrame(main_rows).to_csv(output_dir / "main_results_table.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output_dir / "paired_model_contrasts.csv", index=False)
    _save_synthesis_figures(result, output_dir / "figures")
    _write_synthesis_markdown(result, output_dir / "main_synthesis_results.md")
    return result


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_synthesis_figures(result: dict[str, Any], figure_dir: Path) -> None:
    plt = _load_pyplot()
    figure_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.DataFrame(result["normalized_depth_curves"])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, dataset in zip(axes.flat, ("vqav2", "gqa_balanced", "clevr", "vizwiz_vqa")):
        for model, values in curves[curves["dataset"] == dataset].groupby("model", sort=True):
            axis.plot(
                values["normalized_depth"],
                values["change_rate"],
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
        estimand = "global descriptive" if dataset == "vizwiz_vqa" else "controlled"
        axis.set(title=f"{DATASET_LABELS[dataset]} ({estimand})", ylim=(0, 0.65))
        axis.grid(alpha=0.25)
    axes[1, 0].set(xlabel="Normalized decoder depth", ylabel="Class-change rate")
    axes[1, 1].set(xlabel="Normalized decoder depth")
    axes[0, 0].set_ylabel("Class-change rate")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "normalized_depth_change_curves.png", dpi=200)
    plt.close(figure)

    contrasts = result["paired_model_contrasts"]
    labels = [
        f"{DATASET_LABELS[row['dataset']]}: {MODEL_LABELS[row['left_model']]} - {MODEL_LABELS[row['right_model']]}"
        for row in contrasts
    ]
    estimates = np.asarray([row["estimate"] for row in contrasts])
    lowers = np.asarray([row["familywise_ci_lower"] for row in contrasts])
    uppers = np.asarray([row["familywise_ci_upper"] for row in contrasts])
    positions = np.arange(len(contrasts))
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.errorbar(
        estimates,
        positions,
        xerr=[estimates - lowers, uppers - estimates],
        fmt="o",
        capsize=4,
        color="#4285f4",
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.set(xlabel="Paired controlled-rate difference (Bonferroni familywise 95% CI)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "paired_model_difference_forest.png", dpi=200)
    plt.close(figure)


def _write_synthesis_markdown(result: dict[str, Any], path: Path) -> None:
    main_rows = []
    for row in result["main_results"]:
        interval = "descriptive only" if row["ci_lower"] is None else f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        main_rows.append(
            f"| {DATASET_LABELS[row['dataset']]} | {MODEL_LABELS[row['model']]} | {row['estimate']:.4f} | "
            f"{interval} | {row['mean_normalized_entropy']:.4f} | {row['bootstrap_unit_count']} |"
        )
    contrast_rows = []
    for row in result["paired_model_contrasts"]:
        contrast_rows.append(
            f"| {DATASET_LABELS[row['dataset']]} | {MODEL_LABELS[row['left_model']]} − "
            f"{MODEL_LABELS[row['right_model']]} | {row['estimate']:+.4f} | "
            f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}] | "
            f"[{row['familywise_ci_lower']:+.4f}, {row['familywise_ci_upper']:+.4f}] | "
            f"{row['bootstrap_unit_count']} |"
        )
    limits = "\n".join(f"- {item}" for item in result["interpretation_limits"])
    text = f"""# Four-dataset main synthesis

Scope: {result["scope"]}.

- Analysis commit: `{result["analysis_code_commit"]}`
- Paired bootstrap: {result["paired_bootstrap"]["replicates"]:,} replicates, seed {result["paired_bootstrap"]["seed"]}
- Multiplicity: {result["paired_bootstrap"]["multiplicity"]}

## Main results

| Dataset | Model | Primary rate | 95% CI | Mean normalized entropy | Independent bootstrap units |
|---|---|---:|---:|---:|---:|
{chr(10).join(main_rows)}

VizWiz reports the global all-distinct-sample rate because it has no controlled pairs. The other datasets report their
dataset-specific controlled estimand.

## Within-dataset paired model contrasts

| Dataset | Contrast | Difference | Pairwise 95% CI | Bonferroni familywise 95% CI | Units |
|---|---|---:|---:|---:|---:|
{chr(10).join(contrast_rows)}

Familywise intervals excluding zero provide multiplicity-protected evidence of a model difference on the same frozen
pair units. The intervals are effect-size intervals, not a cross-dataset model leaderboard.

## Interpretation limits

{limits}
"""
    path.write_text(text, encoding="utf-8")
