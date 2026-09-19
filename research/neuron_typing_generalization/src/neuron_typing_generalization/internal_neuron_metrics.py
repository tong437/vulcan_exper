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

"""Per-neuron entropy, effective-state, and activation-RMS analysis within frozen runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .synthesis import DATASET_LABELS, MODEL_LABELS
from .unpaired_synthesis import DATASET_ORDER, MODEL_ORDER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _update_moments(
    sums: np.ndarray,
    squared_sums: np.ndarray,
    minima: np.ndarray,
    maxima: np.ndarray,
    channel: int,
    values: np.ndarray,
) -> None:
    values64 = values.astype(np.float64, copy=False)
    sums[channel] += values64
    squared_sums[channel] += values64 * values64
    minima[channel] = np.minimum(minima[channel], values64)
    maxima[channel] = np.maximum(maxima[channel], values64)


def collect_activation_rms_profile(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Stream one run once and write exact per-neuron RMS moments for visual, text, and pooled tokens."""
    score_manifest_path = run_dir / "scores/manifest.json"
    source_sha256 = _sha256(score_manifest_path)
    destination = output_dir / run_dir.name
    profile_path = destination / "activation_rms_neuron_profiles.parquet"
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file() and profile_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source_score_manifest_sha256") == source_sha256:
            return existing

    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    widths = [int(value) for value in score_manifest["d_ff_by_layer"]]
    sums = [np.zeros((3, width), dtype=np.float64) for width in widths]
    squared_sums = [np.zeros((3, width), dtype=np.float64) for width in widths]
    minima = [np.full((3, width), np.inf, dtype=np.float64) for width in widths]
    maxima = [np.full((3, width), -np.inf, dtype=np.float64) for width in widths]
    counts = np.zeros((len(widths), 3), dtype=np.int64)
    parts = score_manifest["parts"]
    for sample_index, part in enumerate(parts):
        with np.load(run_dir / part["file"], allow_pickle=False) as arrays:
            rms = arrays["activation_rms"]
            token_count = arrays["token_count"]
            valid = arrays["valid"] if "valid" in arrays else token_count > 0
            for layer, width in enumerate(widths):
                for modality in range(2):
                    if bool(valid[layer, modality]):
                        _update_moments(
                            sums[layer],
                            squared_sums[layer],
                            minima[layer],
                            maxima[layer],
                            modality,
                            rms[layer, modality, :width],
                        )
                        counts[layer, modality] += 1
                if bool(valid[layer, 0]) and bool(valid[layer, 1]):
                    visual_tokens, text_tokens = int(token_count[layer, 0]), int(token_count[layer, 1])
                    pooled = np.sqrt(
                        (
                            visual_tokens * np.square(rms[layer, 0, :width].astype(np.float64))
                            + text_tokens * np.square(rms[layer, 1, :width].astype(np.float64))
                        )
                        / (visual_tokens + text_tokens)
                    )
                    _update_moments(sums[layer], squared_sums[layer], minima[layer], maxima[layer], 2, pooled)
                    counts[layer, 2] += 1
        if (sample_index + 1) % 250 == 0 or sample_index + 1 == len(parts):
            print(f"{run_dir.name}: RMS {sample_index + 1}/{len(parts)}", flush=True)

    layer_values = np.concatenate([np.full(width, layer, dtype=np.int16) for layer, width in enumerate(widths)])
    neuron_values = np.concatenate([np.arange(width, dtype=np.int32) for width in widths])
    data: dict[str, Any] = {"run_id": run_dir.name, "layer": layer_values, "neuron": neuron_values}
    labels = ("visual", "text", "pooled")
    for channel, label in enumerate(labels):
        means, standard_deviations, mins, maxs, valid_counts = [], [], [], [], []
        for layer, width in enumerate(widths):
            count = int(counts[layer, channel])
            if count == 0:
                raise ValueError(f"No valid {label} RMS samples in layer {layer}: {run_dir}")
            mean = sums[layer][channel] / count
            variance = np.maximum(squared_sums[layer][channel] / count - mean * mean, 0.0)
            means.append(mean)
            standard_deviations.append(np.sqrt(variance))
            mins.append(minima[layer][channel])
            maxs.append(maxima[layer][channel])
            valid_counts.append(np.full(width, count, dtype=np.int32))
        data[f"activation_rms_{label}_valid_samples"] = np.concatenate(valid_counts)
        data[f"activation_rms_{label}_mean"] = np.concatenate(means)
        data[f"activation_rms_{label}_std"] = np.concatenate(standard_deviations)
        data[f"activation_rms_{label}_min"] = np.concatenate(mins)
        data[f"activation_rms_{label}_max"] = np.concatenate(maxs)
    frame = pd.DataFrame(data)
    frame["activation_rms_pooled_cv"] = np.divide(
        frame["activation_rms_pooled_std"],
        frame["activation_rms_pooled_mean"],
        out=np.full(len(frame), np.nan),
        where=frame["activation_rms_pooled_mean"].to_numpy() > 0,
    )
    state_profile = pd.read_parquet(
        run_dir / "distributions/neuron_profiles.parquet",
        columns=[
            "layer",
            "neuron",
            "change_rate_distinct_pairs",
            "normalized_entropy",
            "effective_class_count",
            "dominant_class",
            "dominant_proportion",
        ],
    )
    frame = frame.merge(state_profile, on=["layer", "neuron"], how="left", validate="one_to_one")
    if frame["normalized_entropy"].isna().any():
        raise ValueError(f"RMS and state profiles do not align: {run_dir}")
    destination.mkdir(parents=True, exist_ok=True)
    temporary = profile_path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(profile_path)
    result = {
        "status": "complete",
        "run_id": run_dir.name,
        "samples": len(parts),
        "layers": len(widths),
        "neurons": len(frame),
        "source_score_manifest_sha256": source_sha256,
        "source_state_profile_sha256": _sha256(run_dir / "distributions/neuron_profiles.parquet"),
        "profile": str(profile_path.relative_to(output_dir)),
        "profile_sha256": _sha256(profile_path),
        "pooled_definition": "token-count-weighted RMS across visual and text token groups per sample",
    }
    _write_json(manifest_path, result)
    return result


def _quantile_summary(values: pd.Series, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)].astype(float)
    quantiles = finite.quantile([0.10, 0.50, 0.90])
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_p10": float(quantiles.loc[0.10]),
        f"{prefix}_median": float(quantiles.loc[0.50]),
        f"{prefix}_p90": float(quantiles.loc[0.90]),
    }


def build_internal_metric_report(
    run_dirs: dict[str, dict[str, Path]], artifact_root: Path, report_dir: Path, *, analysis_commit: str
) -> dict[str, Any]:
    """Collect missing RMS profiles and report entropy/effective-state/RMS distributions for all runs."""
    manifests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    histogram_data: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            run_dir = run_dirs[dataset][model]
            manifest = collect_activation_rms_profile(run_dir, artifact_root)
            manifests.append(manifest)
            profile = pd.read_parquet(artifact_root / manifest["profile"])
            entropy = profile["normalized_entropy"].clip(0, 1)
            effective = profile["effective_class_count"].clip(1, 4)
            pooled_mean = profile["activation_rms_pooled_mean"]
            pooled_std = profile["activation_rms_pooled_std"]
            pooled_cv = profile["activation_rms_pooled_cv"]
            row = {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "run_id": run_dir.name,
                "sample_count": manifest["samples"],
                "neuron_count": len(profile),
                **_quantile_summary(entropy, "normalized_entropy"),
                **_quantile_summary(effective, "effective_class_count"),
                **_quantile_summary(pooled_mean, "activation_rms_pooled_mean"),
                **_quantile_summary(pooled_std, "activation_rms_pooled_std"),
                **_quantile_summary(pooled_cv, "activation_rms_pooled_cv"),
                "entropy_zero_proportion": float((entropy == 0).mean()),
                "entropy_ge_0_8_proportion": float((entropy >= 0.8).mean()),
                "effective_ge_3_proportion": float((effective >= 3).mean()),
                "entropy_rms_strength_spearman": float(entropy.corr(pooled_mean, method="spearman")),
                "entropy_rms_fluctuation_spearman": float(entropy.corr(pooled_std, method="spearman")),
                "rms_profile_sha256": manifest["profile_sha256"],
            }
            rows.append(row)
            for layer, values in profile.groupby("layer", sort=True):
                layer_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "layer": int(layer),
                        "normalized_depth": int(layer) / (manifest["layers"] - 1),
                        "neuron_count": len(values),
                        "normalized_entropy_median": float(values["normalized_entropy"].median()),
                        "effective_class_count_median": float(values["effective_class_count"].median()),
                        "activation_rms_pooled_mean_median": float(values["activation_rms_pooled_mean"].median()),
                        "activation_rms_pooled_std_median": float(values["activation_rms_pooled_std"].median()),
                    }
                )
            log_strength = np.log10(pooled_mean[pooled_mean > 0])
            log_fluctuation = np.log10(pooled_std[pooled_std > 0])
            aligned = profile[(pooled_mean > 0) & (pooled_std > 0)]
            x = np.log10(aligned["activation_rms_pooled_mean"].to_numpy())
            y = np.log10(aligned["activation_rms_pooled_std"].to_numpy())
            x_range = tuple(np.quantile(x, [0.005, 0.995]))
            y_range = tuple(np.quantile(y, [0.005, 0.995]))
            histogram, x_edges, y_edges = np.histogram2d(x, y, bins=55, range=(x_range, y_range))
            histogram_data[(dataset, model)] = {
                "entropy": np.histogram(entropy, bins=np.linspace(0, 1, 51))[0],
                "effective": np.histogram(effective, bins=np.linspace(1, 4, 61))[0],
                "rms_histogram": histogram,
                "rms_x_edges": x_edges,
                "rms_y_edges": y_edges,
                "log_strength_mean": float(log_strength.mean()),
                "log_fluctuation_mean": float(log_fluctuation.mean()),
            }
    frame = pd.DataFrame(rows)
    if len(frame) != 12 or not frame["sample_count"].eq(1000).all():
        raise ValueError("Expected 12 complete 1,000-sample internal-metric summaries.")
    report_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_dir / "internal_neuron_metrics_v1_1000.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(report_dir / "internal_neuron_metrics_v1_1000_layers.csv", index=False)
    figures = _save_figures(histogram_data, report_dir / "figures")
    result = {
        "status": "complete",
        "analysis_code_commit": analysis_commit,
        "scope": "within each frozen model × dataset run; no cross-model neuron alignment",
        "pooled_rms_definition": "per-sample token-count-weighted RMS across visual and text token groups",
        "entropy_effective_count_identity": "effective_class_count = 4 ** normalized_entropy",
        "rows": rows,
        "layers": layer_rows,
        "artifact_root": str(artifact_root),
        "run_manifests": manifests,
        "figures": figures,
        "interpretation_limits": [
            "Raw RMS scale is architecture- and layer-dependent; use it primarily within a run or layer.",
            "Entropy and effective class count are two representations of the same state-occupancy information.",
            "Association between entropy and RMS does not identify a causal mechanism.",
            "All summaries describe the frozen 1,000 samples and are not population guarantees.",
        ],
    }
    _write_json(report_dir / "internal_neuron_metrics_v1_1000.json", result)
    _write_json(artifact_root / "manifest.json", result)
    _write_markdown(result, report_dir / "internal_neuron_metrics_v1_1000.md")
    return result


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figures(data: dict[tuple[str, str], dict[str, Any]], figure_dir: Path) -> list[dict[str, str]]:
    plt = _load_pyplot()
    from matplotlib.colors import LogNorm

    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[Path, str]] = []
    entropy_centers = (np.linspace(0, 1, 51)[:-1] + np.linspace(0, 1, 51)[1:]) / 2
    figure, axes = plt.subplots(3, 4, figsize=(15, 10), sharex="col")
    for row, model in enumerate(MODEL_ORDER):
        for column, dataset in enumerate(DATASET_ORDER):
            axis = axes[row, column]
            values = data[(dataset, model)]
            entropy_counts = values["entropy"] / values["entropy"].sum() * 100
            axis.plot(entropy_centers, entropy_counts, color="#4285f4", label="Normalized entropy")
            twin = axis.twiny()
            if row == 0:
                axis.set_title(DATASET_LABELS[dataset])
            if column == 0:
                axis.set_ylabel(f"{MODEL_LABELS[model]}\nNeurons/bin (%)")
            if row == 2:
                axis.set_xlabel("Normalized entropy H / log 4")
            axis.grid(alpha=0.2)
            axis.set_xlim(0, 1)
            twin.set_xlim(0, 1)
            if row == 0:
                twin.set_xticks(np.log([1, 2, 3, 4]) / np.log(4), ["1", "2", "3", "4"])
                twin.set_xlabel("Effective class count")
            else:
                twin.set_xticks([])
    figure.suptitle("Within-run neuron state diversity over 1,000 samples")
    figure.tight_layout()
    path = figure_dir / "internal_normalized_entropy_effective_states.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    outputs.append((path, "Normalized entropy and effective class count distributions"))

    maximum_count = max(float(values["rms_histogram"].max()) for values in data.values())
    figure, axes = plt.subplots(3, 4, figsize=(15, 10))
    image = None
    for row, model in enumerate(MODEL_ORDER):
        for column, dataset in enumerate(DATASET_ORDER):
            axis = axes[row, column]
            values = data[(dataset, model)]
            histogram = np.ma.masked_less_equal(values["rms_histogram"].T, 0)
            image = axis.pcolormesh(
                values["rms_x_edges"],
                values["rms_y_edges"],
                histogram,
                shading="auto",
                cmap="viridis",
                norm=LogNorm(vmin=1, vmax=maximum_count),
            )
            if row == 0:
                axis.set_title(DATASET_LABELS[dataset])
            if column == 0:
                axis.set_ylabel(f"{MODEL_LABELS[model]}\nlog10 RMS standard deviation")
            if row == 2:
                axis.set_xlabel("log10 mean pooled activation RMS")
            axis.grid(alpha=0.12)
    if image is not None:
        color_axis = figure.add_axes([0.91, 0.18, 0.015, 0.64])
        figure.colorbar(image, cax=color_axis, label="Neuron count (log color scale)")
    figure.suptitle("Per-neuron activation strength and across-sample fluctuation within each run")
    figure.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.92, hspace=0.3, wspace=0.28)
    path = figure_dir / "internal_activation_rms_strength_fluctuation.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    outputs.append((path, "Activation RMS strength versus fluctuation"))
    return [
        {"file": str(path.relative_to(figure_dir.parent)), "title": title, "sha256": _sha256(path)}
        for path, title in outputs
    ]


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    frame = pd.DataFrame(result["rows"])
    clevr = frame[frame["dataset"] == "clevr"]
    other = frame[frame["dataset"] != "clevr"]
    rows = []
    for row in result["rows"]:
        rows.append(
            f"| {row['dataset_label']} | {row['model_label']} | {row['normalized_entropy_median']:.3f} | "
            f"{row['effective_class_count_median']:.2f} | {row['activation_rms_pooled_mean_median']:.3e} | "
            f"{row['activation_rms_pooled_std_median']:.3e} | {row['activation_rms_pooled_cv_median']:.3f} | "
            f"{row['entropy_rms_strength_spearman']:.3f} | {row['entropy_rms_fluctuation_spearman']:.3f} |"
        )
    findings = [
        (
            f"Across the 12 within-run distributions, the median effective state count ranges from "
            f"{frame['effective_class_count_median'].min():.2f} to "
            f"{frame['effective_class_count_median'].max():.2f}; the typical neuron therefore occupies "
            "substantially fewer than four states effectively."
        ),
        (
            f"CLEVR median effective counts are {clevr['effective_class_count_median'].min():.2f}–"
            f"{clevr['effective_class_count_median'].max():.2f}, compared with "
            f"{other['effective_class_count_median'].min():.2f}–"
            f"{other['effective_class_count_median'].max():.2f} in the other frozen datasets."
        ),
        (
            f"Median pooled-RMS coefficients of variation are "
            f"{clevr['activation_rms_pooled_cv_median'].min():.3f}–"
            f"{clevr['activation_rms_pooled_cv_median'].max():.3f} for CLEVR and "
            f"{other['activation_rms_pooled_cv_median'].min():.3f}–"
            f"{other['activation_rms_pooled_cv_median'].max():.3f} otherwise."
        ),
        (
            f"Spearman associations between entropy and RMS strength range from "
            f"{frame['entropy_rms_strength_spearman'].min():.3f} to "
            f"{frame['entropy_rms_strength_spearman'].max():.3f}; associations with RMS fluctuation range from "
            f"{frame['entropy_rms_fluctuation_spearman'].min():.3f} to "
            f"{frame['entropy_rms_fluctuation_spearman'].max():.3f}. State diversity is therefore not reducible "
            "to activation magnitude alone."
        ),
    ]
    findings_text = "\n".join(f"- {item}" for item in findings)
    figures = "\n".join(f"- [{item['title']}]({item['file']})" for item in result["figures"])
    limits = "\n".join(f"- {item}" for item in result["interpretation_limits"])
    text = f"""# Within-run entropy, effective states, and activation RMS

Each row summarizes all real decoder MLP neurons in one frozen 1,000-sample run. Pooled activation RMS is calculated
within each sample using visual/text token counts, then its per-neuron mean and standard deviation are taken over the
1,000 samples. The table reports medians across neurons; complete P10/P50/P90 values are in the CSV and JSON.

Normalized entropy and effective class count are not independent metrics:

`effective_class_count = 4 ** normalized_entropy`.

## Main findings

{findings_text}

## Internal-distribution summary

| Dataset | Model | Entropy median | Effective states median | RMS strength median | RMS fluctuation median | RMS CV median | rho(H,strength) | rho(H,fluctuation) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Figures

{figures}

## Interpretation limits

{limits}

## Reproducibility

- Analysis commit: `{result["analysis_code_commit"]}`
- Full per-neuron RMS/state profiles and their SHA-256 manifests are stored under `{result["artifact_root"]}`.
- Exact run and layer summaries are stored in the CSV and JSON outputs.
"""
    path.write_text(text, encoding="utf-8")
