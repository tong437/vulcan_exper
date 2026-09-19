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

"""Within-run distributions of per-neuron response-state change rates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .synthesis import DATASET_LABELS, MODEL_COLORS, MODEL_LABELS
from .unpaired_synthesis import DATASET_ORDER, MODEL_ORDER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def theoretical_change_rate_maximum(sample_count: int, class_count: int = 4) -> float:
    """Return the largest distinct-ordered-pair rate for balanced integer class counts."""
    if sample_count < 2:
        raise ValueError("At least two samples are required for a distinct-pair rate.")
    quotient, remainder = divmod(sample_count, class_count)
    squared_counts = remainder * (quotient + 1) ** 2 + (class_count - remainder) * quotient**2
    return (sample_count**2 - squared_counts) / (sample_count * (sample_count - 1))


def summarize_change_rates(profiles: pd.DataFrame) -> dict[str, Any]:
    """Summarize the full per-neuron distribution in one frozen run."""
    required = {"layer", "neuron", "valid_samples", "invalid_samples", "change_rate_distinct_pairs"}
    missing = sorted(required.difference(profiles.columns))
    if missing:
        raise ValueError(f"Neuron profiles are missing variability columns: {missing}")
    valid = profiles.loc[np.isfinite(profiles["change_rate_distinct_pairs"])].copy()
    if valid.empty:
        raise ValueError("No finite neuron change rates are available.")
    if valid["valid_samples"].nunique() != 1 or valid["invalid_samples"].ne(0).any():
        raise ValueError("Within-run summary requires equally weighted, fully observed neurons.")
    sample_count = int(valid.iloc[0]["valid_samples"])
    rates = valid["change_rate_distinct_pairs"].astype(float)
    nonzero = rates[rates > 0]
    maximum = float(rates.max())
    maximum_rows = valid.loc[rates == maximum].sort_values(["layer", "neuron"], kind="stable")
    quantiles = rates.quantile([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    return {
        "sample_count": sample_count,
        "neuron_count": len(valid),
        "minimum": float(rates.min()),
        "minimum_nonzero": float(nonzero.min()) if not nonzero.empty else None,
        "p01": float(quantiles.loc[0.01]),
        "p10": float(quantiles.loc[0.10]),
        "p25": float(quantiles.loc[0.25]),
        "mean": float(rates.mean()),
        "standard_deviation": float(rates.std(ddof=0)),
        "median": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.90]),
        "p99": float(quantiles.loc[0.99]),
        "maximum": maximum,
        "theoretical_maximum": theoretical_change_rate_maximum(sample_count),
        "maximum_gap_from_theory": theoretical_change_rate_maximum(sample_count) - maximum,
        "maximum_tie_count": len(maximum_rows),
        "maximum_layer": int(maximum_rows.iloc[0]["layer"]),
        "maximum_neuron": int(maximum_rows.iloc[0]["neuron"]),
        "zero_count": int((rates == 0).sum()),
        "zero_proportion": float((rates == 0).mean()),
        "ge_0_25_count": int((rates >= 0.25).sum()),
        "ge_0_25_proportion": float((rates >= 0.25).mean()),
        "ge_0_50_count": int((rates >= 0.50).sum()),
        "ge_0_50_proportion": float((rates >= 0.50).mean()),
        "ge_0_70_count": int((rates >= 0.70).sum()),
        "ge_0_70_proportion": float((rates >= 0.70).mean()),
    }


def build_within_run_variability_report(
    run_dirs: dict[str, dict[str, Path]], report_dir: Path, *, analysis_commit: str
) -> dict[str, Any]:
    """Build tables, figures, and provenance for all 12 within-run distributions."""
    rows: list[dict[str, Any]] = []
    distributions: dict[tuple[str, str], np.ndarray] = {}
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            run_dir = run_dirs[dataset][model]
            profile_path = run_dir / "distributions/neuron_profiles.parquet"
            profiles = pd.read_parquet(
                profile_path,
                columns=[
                    "run_id",
                    "layer",
                    "neuron",
                    "valid_samples",
                    "invalid_samples",
                    "change_rate_distinct_pairs",
                ],
            )
            if profiles["run_id"].nunique() != 1 or str(profiles.iloc[0]["run_id"]) != run_dir.name:
                raise ValueError(f"Profile run identity does not match directory: {run_dir}")
            summary = summarize_change_rates(profiles)
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "run_id": run_dir.name,
                    **summary,
                    "profile_sha256": _sha256(profile_path),
                }
            )
            distributions[(dataset, model)] = profiles["change_rate_distinct_pairs"].to_numpy(dtype=float)
    frame = pd.DataFrame(rows)
    if len(frame) != 12 or not frame["sample_count"].eq(1000).all():
        raise ValueError("Expected 12 complete 1,000-sample distributions.")
    report_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_dir / "within_run_variability_v1_1000.csv", index=False)
    figure_dir = report_dir / "figures"
    figures = _save_figures(frame, distributions, figure_dir)
    result = {
        "status": "complete",
        "analysis_code_commit": analysis_commit,
        "estimand": "distribution of exact per-neuron all-distinct-sample state-change rates within each frozen run",
        "rows": rows,
        "figures": figures,
        "interpretation_limits": [
            "Minimum and maximum are extreme order statistics over architecture-dependent neuron counts.",
            "Neuron indices are model-local and are not aligned across architectures.",
            "The summaries are descriptive for the frozen 1,000 samples and are not uncertainty intervals.",
            "A zero rate means no state transition in this sample set, not universal invariance.",
        ],
    }
    _write_json(report_dir / "within_run_variability_v1_1000.json", result)
    _write_markdown(result, report_dir / "within_run_variability_v1_1000.md")
    return result


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figures(
    frame: pd.DataFrame, distributions: dict[tuple[str, str], np.ndarray], figure_dir: Path
) -> list[dict[str, str]]:
    plt = _load_pyplot()
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[Path, str]] = []
    bins = np.linspace(0, theoretical_change_rate_maximum(1000), 76)
    centers = (bins[:-1] + bins[1:]) / 2
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, dataset in zip(axes.flat, DATASET_ORDER):
        for model in MODEL_ORDER:
            values = distributions[(dataset, model)]
            counts, _ = np.histogram(values, bins=bins)
            axis.plot(centers, 100 * counts / len(values), color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        axis.set_title(DATASET_LABELS[dataset])
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 0.76)
    axes[0, 0].set_ylabel("Neurons per 0.01 bin (%)")
    axes[1, 0].set(ylabel="Neurons per 0.01 bin (%)", xlabel="Per-neuron change rate")
    axes[1, 1].set_xlabel("Per-neuron change rate")
    axes[0, 0].legend(fontsize=9)
    figure.suptitle("Within-run distributions across all real decoder MLP neurons")
    figure.tight_layout()
    path = figure_dir / "within_run_change_rate_distributions.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    outputs.append((path, "Within-run change-rate distributions"))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, column, title, maximum in (
        (axes[0], "zero_proportion", "Exactly stable neurons", 0.08),
        (axes[1], "ge_0_50_proportion", "Neurons with change rate ≥ 0.5", 0.55),
    ):
        matrix = frame.pivot(index="model", columns="dataset", values=column).reindex(
            index=MODEL_ORDER, columns=DATASET_ORDER
        )
        image = axis.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=maximum)
        axis.set_xticks(range(4), [DATASET_LABELS[value] for value in DATASET_ORDER], rotation=20, ha="right")
        axis.set_yticks(range(3), [MODEL_LABELS[value] for value in MODEL_ORDER])
        for row in range(3):
            for col in range(4):
                value = matrix.iloc[row, col]
                axis.text(col, row, f"{value:.1%}", ha="center", va="center")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path = figure_dir / "within_run_stable_and_high_change_heatmaps.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    outputs.append((path, "Stable and high-change neuron proportions"))
    return [
        {"file": str(path.relative_to(figure_dir.parent)), "title": title, "sha256": _sha256(path)}
        for path, title in outputs
    ]


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    frame = pd.DataFrame(result["rows"])
    natural = frame[frame["dataset"].isin(("vqav2", "gqa_balanced", "vizwiz_vqa"))].set_index(["model", "dataset"])

    def model_range(model: str, column: str) -> tuple[float, float]:
        values = natural.loc[model, column]
        return float(values.min()), float(values.max())

    llava_median = model_range("llava_1_5_7b", "median")
    qwen_median = model_range("qwen3_5_2b", "median")
    gemma_median = model_range("gemma_4_e2b_it", "median")
    llava_high = model_range("llava_1_5_7b", "ge_0_50_proportion")
    qwen_high = model_range("qwen3_5_2b", "ge_0_50_proportion")
    gemma_high = model_range("gemma_4_e2b_it", "ge_0_50_proportion")
    clevr = frame.set_index(["model", "dataset"])
    table_rows = []
    for row in frame.itertuples():
        table_rows.append(
            f"| {row.dataset_label} | {row.model_label} | {row.minimum:.4f} | {row.minimum_nonzero:.4f} | "
            f"{row.p10:.4f} | {row.median:.4f} | {row.p90:.4f} | {row.maximum:.4f} | "
            f"{row.zero_proportion:.2%} | {row.ge_0_50_proportion:.2%} | L{row.maximum_layer}:N{row.maximum_neuron} |"
        )
    figures = "\n".join(f"- [{item['title']}]({item['file']})" for item in result["figures"])
    limits = "\n".join(f"- {item}" for item in result["interpretation_limits"])
    text = f"""# Within-run neuron variability over 1,000 frozen samples

All 12 runs have an observed minimum of 0, a minimum nonzero rate of 0.002, and a maximum close to the finite-sample
four-state ceiling of {theoretical_change_rate_maximum(1000):.6f}. Consequently, central quantiles and prevalence
statistics are more informative for model and dataset comparison than raw extrema.

## Main findings

- On VQAv2, GQA, and VizWiz, median rates span {llava_median[0]:.3f}–{llava_median[1]:.3f} for LLaVA,
  {qwen_median[0]:.3f}–{qwen_median[1]:.3f} for Qwen, and {gemma_median[0]:.3f}–{gemma_median[1]:.3f} for Gemma.
- On those same datasets, the proportion at or above 0.5 is {llava_high[0]:.1%}–{llava_high[1]:.1%} for LLaVA,
  {qwen_high[0]:.1%}–{qwen_high[1]:.1%} for Qwen, and {gemma_high[0]:.1%}–{gemma_high[1]:.1%} for Gemma.
- For CLEVR, all three distributions lie closer to stability. Rates at or above 0.5 are
  {clevr.loc[("qwen3_5_2b", "clevr"), "ge_0_50_proportion"]:.1%} for Qwen,
  {clevr.loc[("llava_1_5_7b", "clevr"), "ge_0_50_proportion"]:.1%} for LLaVA, and
  {clevr.loc[("gemma_4_e2b_it", "clevr"), "ge_0_50_proportion"]:.1%} for Gemma; exact-zero rates rise to
  {clevr.loc[("qwen3_5_2b", "clevr"), "zero_proportion"]:.1%},
  {clevr.loc[("llava_1_5_7b", "clevr"), "zero_proportion"]:.1%}, and
  {clevr.loc[("gemma_4_e2b_it", "clevr"), "zero_proportion"]:.1%}, respectively.
- These are broad within-model distributions, not a claim that every neuron follows the model-level mean.

## Distribution table

| Dataset | Model | Min | Min nonzero | P10 | Median | P90 | Max | Zero | Rate ≥ 0.5 | Max neuron |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table_rows)}

`Lx:Ny` denotes model-local layer `x`, neuron `y`; indices are not aligned between models.

## Figures

{figures}

## Interpretation limits

{limits}

## Reproducibility

- Analysis commit: `{result["analysis_code_commit"]}`
- Exact percentiles, counts, thresholds, run IDs, source hashes, and figure hashes are stored in the CSV and JSON.
"""
    path.write_text(text, encoding="utf-8")
