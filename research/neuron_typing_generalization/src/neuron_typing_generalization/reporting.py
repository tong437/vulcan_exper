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

"""Reproducible pilot statistics, figures, and cross-model reports."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .analysis import normal_precision_pair_count, pair_level_bootstrap_ratio


CLASS_LABELS = ("W", "V", "T", "M")
CLASS_COLORS = ("#9aa0a6", "#4285f4", "#f4b400", "#0f9d58")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _controlled_arrays(changes: pd.DataFrame) -> tuple[list[int], np.ndarray, np.ndarray, pd.DataFrame]:
    controlled = changes[changes["pair_scope"] == "controlled"].copy()
    overall = controlled[controlled["layer"].isna()].sort_values(["left_sample_id", "right_sample_id"])
    layers = sorted(int(value) for value in controlled["layer"].dropna().unique())
    if overall.empty:
        return (
            layers,
            np.empty((0, len(layers) + 1), dtype=np.int64),
            np.empty((0, len(layers) + 1), dtype=np.int64),
            overall,
        )
    keys = ["left_sample_id", "right_sample_id"]
    base_index = pd.MultiIndex.from_frame(overall[keys])
    matrices = []
    for column in ("changed_neurons", "valid_neurons"):
        layer_values = controlled[controlled["layer"].notna()].copy()
        layer_values["layer"] = layer_values["layer"].astype(int)
        pivot = layer_values.pivot(index=keys, columns="layer", values=column).reindex(
            index=base_index, columns=layers
        )
        if pivot.isna().any().any():
            raise ValueError("Controlled pair table is incomplete across decoder layers.")
        matrices.append(np.column_stack([overall[column].to_numpy(dtype=np.int64), pivot.to_numpy(dtype=np.int64)]))
    return layers, matrices[0], matrices[1], overall


def summarize_controlled_pairs(
    changes: pd.DataFrame,
    pair_manifest: pd.DataFrame | None = None,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42,
    confidence_level: float = 0.95,
    precision_half_widths: tuple[float, ...] = (0.005, 0.01, 0.02),
) -> dict[str, Any]:
    """Compute ratio-of-sums point estimates, pair bootstrap CIs, and precision scenarios."""
    layers, changed, valid, overall = _controlled_arrays(changes)
    bootstrap_unit = "manifest_pair"
    if pair_manifest is not None and "pair_group_id" in pair_manifest and len(overall):
        keys = ["left_sample_id", "right_sample_id"]
        mapping = pair_manifest[keys + ["pair_group_id"]].copy()
        mapping[keys] = mapping[keys].astype(str)
        edge_keys = overall[keys].astype(str)
        merged = edge_keys.merge(mapping, on=keys, how="left", validate="one_to_one")
        if merged["pair_group_id"].isna().any():
            raise ValueError("Pairwise changes cannot be matched to pair-manifest bootstrap groups.")
        group_values = merged["pair_group_id"].astype(str).to_numpy()
        unique_groups = sorted(set(group_values))
        if len(unique_groups) < len(group_values):
            bootstrap_unit = "pair_group_id"
        grouped_changed = np.stack([changed[group_values == group].sum(axis=0) for group in unique_groups])
        grouped_valid = np.stack([valid[group_values == group].sum(axis=0) for group in unique_groups])
    else:
        unique_groups = list(range(len(overall)))
        grouped_changed = changed
        grouped_valid = valid
    if len(unique_groups) < 2:
        return {"status": "insufficient_pairs", "pair_count": len(overall), "per_layer": []}
    bootstrap = pair_level_bootstrap_ratio(
        grouped_changed,
        grouped_valid,
        replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
    )

    def metric(index: int) -> dict[str, Any]:
        return {
            "estimate": float(bootstrap.estimate[index]),
            "standard_error": float(bootstrap.standard_error[index]),
            "ci_lower": float(bootstrap.lower[index]),
            "ci_upper": float(bootstrap.upper[index]),
            "ci_half_width": float((bootstrap.upper[index] - bootstrap.lower[index]) / 2),
            "changed_neurons": int(changed[:, index].sum()),
            "valid_neurons": int(valid[:, index].sum()),
        }

    pair_rates = grouped_changed[:, 0] / grouped_valid[:, 0]
    precision = {
        f"half_width_{width:.3f}": {
            "target_half_width": width,
            "estimated_required_pairs": normal_precision_pair_count(pair_rates, width),
        }
        for width in precision_half_widths
    }
    return {
        "status": "complete",
        "method": "manifest-cluster percentile bootstrap of the ratio of summed changed to valid neurons",
        "pair_count": len(overall),
        "bootstrap_unit": bootstrap_unit,
        "bootstrap_unit_count": len(unique_groups),
        "bootstrap_replicates": bootstrap.replicates,
        "bootstrap_seed": bootstrap.seed,
        "confidence_level": bootstrap.confidence_level,
        "overall": metric(0),
        "per_layer": [{"layer": layer, **metric(index + 1)} for index, layer in enumerate(layers)],
        "empirical_pair_rate_standard_deviation": float(np.std(pair_rates, ddof=1)),
        "precision_scenarios_normal_approximation": precision,
    }


def _class_summary(profiles: pd.DataFrame) -> dict[str, Any]:
    valid = profiles[profiles["valid_samples"] > 0]
    counts = {label: int(valid[f"count_{label}"].sum()) for label in CLASS_LABELS}
    total = sum(counts.values())
    return {
        "occurrence_counts": counts,
        "occurrence_proportions": {label: counts[label] / total for label in CLASS_LABELS},
        "mean_normalized_entropy": float(valid["normalized_entropy"].mean()),
        "dominant_class_proportions": valid["dominant_class"].value_counts(normalize=True).sort_index().to_dict(),
        "profile_proportions": valid["profile"].value_counts(normalize=True).sort_index().to_dict(),
    }


def _reproducibility_record(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    runtime_path = run_dir / "logs/gpu_collection_summary.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
    activation_path = run_dir / "activations/manifest.json"
    activation = json.loads(activation_path.read_text(encoding="utf-8")) if activation_path.is_file() else {}

    def sidecar(name: str) -> str | None:
        path = run_dir / "samples" / name
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None

    model = config.get("model", {})
    dataset = config.get("dataset", {})
    study_options = config.get("study_options", {})
    return {
        "run_id": activation.get("run_id") or metadata.get("run_id"),
        "model": model.get("name"),
        "model_revision": model.get("revision"),
        "dataset": dataset.get("name"),
        "dataset_revision": dataset.get("revision"),
        "dataset_config": dataset.get("config"),
        "dataset_split": dataset.get("split"),
        "seed": config.get("seed"),
        "manifest_tier": study_options.get("sampling", {}).get("manifest_tier"),
        "sample_manifest_sha256": sidecar("sample_manifest.parquet.sha256"),
        "pair_manifest_sha256": sidecar("pair_manifest.parquet.sha256"),
        "activation_parts_sha256": activation.get("parts_sha256"),
        "token_mask_policy": model.get("token_roles"),
        "importance_policy": config.get("metrics", [{}])[0] if config.get("metrics") else None,
        "runtime": runtime,
        "original_run_git": metadata.get("git"),
        "resolved_config_sha256": _sha256(config_path) if config_path.is_file() else None,
    }


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_run_figures(
    run_dir: Path,
    summary: dict[str, Any],
    changes: pd.DataFrame,
    profiles: pd.DataFrame,
    transitions: np.ndarray,
    controlled: dict[str, Any],
) -> list[dict[str, str]]:
    plt = _load_pyplot()
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, str]] = []

    layers = np.arange(len(summary["per_layer_change_rate_distinct_pairs"]))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        layers, summary["per_layer_change_rate_distinct_pairs"], marker="o", markersize=3, label="All distinct pairs"
    )
    if controlled.get("status") == "complete":
        axis.plot(
            layers,
            [row["estimate"] for row in controlled["per_layer"]],
            marker="o",
            markersize=3,
            label="Controlled pairs",
        )
    axis.set(xlabel="Decoder layer (zero-indexed)", ylabel="Neuron class-change rate", ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = figure_dir / "layer_change_rates.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figures.append((path.name, "Global and controlled class-change rates by decoder layer"))

    classes = _class_summary(profiles)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(CLASS_LABELS, [classes["occurrence_proportions"][label] for label in CLASS_LABELS], color=CLASS_COLORS)
    axis.set(xlabel="Response class", ylabel="Sample-neuron occurrence proportion", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = figure_dir / "class_proportions.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figures.append((path.name, "V/T/M/W sample-neuron occurrence proportions"))

    overall_pairs = changes[(changes["pair_scope"] == "controlled") & changes["layer"].isna()]
    if len(overall_pairs):
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(
            overall_pairs["change_rate"], bins=min(25, max(5, int(math.sqrt(len(overall_pairs))))), color="#4285f4"
        )
        axis.set(xlabel="Controlled-pair class-change rate", ylabel="Pair count")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = figure_dir / "controlled_pair_distribution.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        figures.append((path.name, "Distribution of overall controlled-pair change rates"))

    matrix = transitions.sum(axis=0, dtype=np.int64)
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(4), CLASS_LABELS)
    axis.set_yticks(range(4), CLASS_LABELS)
    axis.set(xlabel="Destination class", ylabel="Source class")
    figure.colorbar(image, ax=axis, label="Row-normalized transition proportion")
    figure.tight_layout()
    path = figure_dir / "symmetrized_transitions.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figures.append((path.name, "Symmetrized controlled transitions, aggregated over layers"))

    return [
        {"file": f"figures/{name}", "title": title, "sha256": _sha256(figure_dir / name)} for name, title in figures
    ]


def evaluate_pilot_run(
    run_dir: Path,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Create complete statistics, provenance, and real figures for one analyzed run."""
    summary = json.loads((run_dir / "distributions/analysis_summary.json").read_text(encoding="utf-8"))
    changes = pd.read_parquet(run_dir / "distributions/pairwise_changes.parquet")
    pair_manifest = pd.read_parquet(run_dir / "samples/pair_manifest.parquet")
    profiles = pd.read_parquet(run_dir / "distributions/neuron_profiles.parquet")
    with np.load(run_dir / "distributions/transition_matrices.npz", allow_pickle=False) as stored:
        symmetrized = stored["symmetrized"]
    controlled = summarize_controlled_pairs(
        changes,
        pair_manifest,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    statistics = {
        "run_id": summary["run_id"],
        "global": summary,
        "controlled": controlled,
        "classes": _class_summary(profiles),
    }
    _write_json(run_dir / "metrics/pilot_statistics.json", statistics)
    reproducibility = _reproducibility_record(run_dir)
    _write_json(run_dir / "metrics/reproducibility.json", reproducibility)
    figures = _save_run_figures(run_dir, summary, changes, profiles, symmetrized, controlled)
    figure_manifest = {"run_id": summary["run_id"], "figures": figures, "status": "complete"}
    _write_json(run_dir / "figures/manifest.json", figure_manifest)
    return {"statistics": statistics, "reproducibility": reproducibility, "figures": figure_manifest}


def build_unified_vqav2_report(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    collection_commit: str,
    analysis_commit: str,
    target_half_width: float = 0.01,
) -> dict[str, Any]:
    """Combine completed model runs into reviewable JSON, Markdown, and comparison figures."""
    records = []
    for run_dir in run_dirs:
        statistics = json.loads((run_dir / "metrics/pilot_statistics.json").read_text(encoding="utf-8"))
        reproducibility = json.loads((run_dir / "metrics/reproducibility.json").read_text(encoding="utf-8"))
        records.append(
            {"run_dir": str(run_dir.resolve()), "statistics": statistics, "reproducibility": reproducibility}
        )
    records.sort(key=lambda row: str(row["reproducibility"]["model"]))
    half_widths = [row["statistics"]["controlled"]["overall"]["ci_half_width"] for row in records]
    requires_larger = any(value > target_half_width for value in half_widths)
    result = {
        "status": "complete",
        "scope": "VQAv2 frozen pilot: 500 samples / 250 same-question different-image pairs / seed 42",
        "collection_code_commit": collection_commit,
        "analysis_code_commit": analysis_commit,
        "primary_precision_target_half_width": target_half_width,
        "larger_tier_required_for_primary_controlled_mean": requires_larger,
        "larger_tier_decision": (
            "required because at least one model exceeds the target confidence half-width"
            if requires_larger
            else "not required for the primary controlled mean; create a new tier only for narrower precision or subgroup claims"
        ),
        "models": records,
        "interpretation_limits": [
            "Response classes are sample-conditional descriptive states, not immutable neuron identities.",
            "M does not establish causal cross-modal interaction; W does not mean inactive.",
            "The controlled VQAv2 comparison isolates image context only subject to the frozen question/sample audit.",
            "One seed and one dataset do not establish cross-dataset or cross-seed generalization.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "vqav2_pilot_results.json", result)
    _save_unified_figures(records, output_dir / "figures")
    _write_unified_markdown(result, output_dir / "vqav2_pilot_results.md")
    return result


def _save_unified_figures(records: list[dict[str, Any]], figure_dir: Path) -> None:
    plt = _load_pyplot()
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = [row["reproducibility"]["model"] for row in records]

    estimates = [row["statistics"]["controlled"]["overall"]["estimate"] for row in records]
    lowers = [row["statistics"]["controlled"]["overall"]["ci_lower"] for row in records]
    uppers = [row["statistics"]["controlled"]["overall"]["ci_upper"] for row in records]
    figure, axis = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(records))
    axis.errorbar(
        positions, estimates, yerr=[np.subtract(estimates, lowers), np.subtract(uppers, estimates)], fmt="o", capsize=5
    )
    axis.set_xticks(positions, names, rotation=15, ha="right")
    axis.set(ylabel="Controlled image-context change rate", ylim=(0, max(uppers) * 1.25))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "controlled_change_bootstrap_ci.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(len(records), 1, figsize=(8, 3 * len(records)), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, record, name in zip(axes, records, names):
        global_rates = record["statistics"]["global"]["per_layer_change_rate_distinct_pairs"]
        controlled_rates = [row["estimate"] for row in record["statistics"]["controlled"]["per_layer"]]
        axis.plot(global_rates, label="All distinct pairs")
        axis.plot(controlled_rates, label="Controlled pairs")
        axis.set(title=name, ylabel="Change rate", ylim=(0, 1))
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Decoder layer (zero-indexed)")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "layer_change_rates.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(records))
    for label, color in zip(CLASS_LABELS, CLASS_COLORS):
        values = [row["statistics"]["classes"]["occurrence_proportions"][label] for row in records]
        axis.bar(names, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set(ylabel="Sample-neuron occurrence proportion", ylim=(0, 1))
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(figure_dir / "class_proportions.png", dpi=200)
    plt.close(figure)


def _write_unified_markdown(result: dict[str, Any], path: Path) -> None:
    rows = []
    for record in result["models"]:
        provenance = record["reproducibility"]
        controlled = record["statistics"]["controlled"]
        classes = record["statistics"]["classes"]
        overall = controlled["overall"]
        rows.append(
            f"| {provenance['model']} | {overall['estimate']:.4f} | "
            f"[{overall['ci_lower']:.4f}, {overall['ci_upper']:.4f}] | "
            f"{classes['mean_normalized_entropy']:.4f} | {controlled['pair_count']} |"
        )
    text = f"""# VQAv2 three-model frozen-pilot report

Scope: {result["scope"]}.

- Collection code commit: `{result["collection_code_commit"]}`
- Analysis code commit: `{result["analysis_code_commit"]}`
- Primary precision target: 95% CI half-width <= {result["primary_precision_target_half_width"]:.3f}
- Larger-tier decision: {result["larger_tier_decision"]}.

## Controlled same-question, different-image result

| Model | Change rate | Pair-bootstrap 95% CI | Mean normalized entropy | Pairs |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

The bootstrap resampling unit is one frozen manifest pair. Each replicate recomputes the ratio of summed changed-neuron
counts to summed valid-neuron counts; neurons are not treated as independent observations.

## Reproducibility

The companion JSON contains model and dataset revisions, split/config, manifest hashes, activation partition hashes,
seed, token-mask policy, runtime versions, explicit denominators, class distributions, and precision scenarios for every
model. Large activations and per-sample results remain in the ignored local artifact directories recorded there.

## Interpretation limits

""" + "".join(f"- {item}\n" for item in result["interpretation_limits"])
    path.write_text(text, encoding="utf-8")
