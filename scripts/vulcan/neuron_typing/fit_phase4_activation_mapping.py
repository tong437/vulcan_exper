# Copyright 2026 the LlamaFactory team.
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

"""Fit and gate visual-to-text activation mappings for Phase 4."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase4_mapping_utils import (  # noqa: E402
    fit_linear_mapper,
    fit_randomized_projector,
    fit_target_components,
    prepare_linear_mapper,
    regression_metrics,
    topk_trigger_metrics,
)
from run_phase2_ablation import infer_score_columns, read_score_table  # noqa: E402


FEATURE_SPACES = ("question", "vision", "vision_question")
MODEL_TYPES = ("ridge", "reduced_rank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Phase-4 visual-to-text activation mappings.")
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_rank", type=int, default=128)
    parser.add_argument("--target_rank", type=int, default=64)
    parser.add_argument("--alphas", default="0.1,1,10,100")
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--null_permutations", type=int, default=20)
    parser.add_argument("--primary_model", choices=MODEL_TYPES, default="reduced_rank")
    parser.add_argument("--min_mapping_r2", type=float, default=0.01)
    parser.add_argument("--min_incremental_r2", type=float, default=0.002)
    parser.add_argument("--max_null_p", type=float, default=0.05)
    parser.add_argument("--min_positive_layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2029)
    return parser.parse_args()


def _parse_alphas(value: str) -> list[float]:
    alphas = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("alphas must contain non-negative comma-separated values.")
    return alphas


def load_activation_chunks(
    activation_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Load a completed collection while validating contiguous chunk alignment."""
    root = Path(activation_dir)
    state = json.loads((root / "collection_state.json").read_text(encoding="utf-8"))
    if not state.get("complete"):
        raise RuntimeError("Phase-4 activation collection is incomplete.")
    vision_parts = []
    question_parts = []
    delta_parts: dict[int, list[np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    expected_start = 0
    for path in sorted((root / "chunks").glob("chunk_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        start = int(payload["start_index"])
        end = int(payload["end_index"])
        if start != expected_start or end <= start:
            raise ValueError(f"Invalid Phase-4 chunk range in {path}: [{start}, {end})")
        if len(payload["rows"]) != end - start:
            raise ValueError(f"Chunk row count mismatch in {path}.")
        vision_parts.append(payload["vision_features"].float().numpy())
        question_parts.append(payload["question_features"].float().numpy())
        for layer, values in payload["delta_activations"].items():
            delta_parts.setdefault(int(layer), []).append(values.numpy())
        rows.extend(payload["rows"])
        expected_start = end
    if expected_start != int(state["completed_rows"]):
        raise ValueError("Phase-4 chunk coverage differs from collection_state.json.")
    if not rows:
        raise ValueError("No Phase-4 activation chunks were found.")

    vision = np.concatenate(vision_parts, axis=0)
    question = np.concatenate(question_parts, axis=0)
    delta = {layer: np.concatenate(parts, axis=0) for layer, parts in delta_parts.items()}
    if any(len(values) != len(rows) for values in (vision, question, *delta.values())):
        raise ValueError("Phase-4 features, targets, and metadata are not row-aligned.")
    return vision, question, delta, rows, state


def _split_indices(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    result = {
        split: np.asarray([index for index, row in enumerate(rows) if row["split"] == split], dtype=np.int64)
        for split in ("train", "validation", "test")
    }
    if any(len(indices) < 2 for indices in result.values()):
        raise ValueError(f"Every mapping split needs at least two rows: {result}")
    image_sets = {
        split: {rows[index]["image_id"] for index in indices.tolist()} for split, indices in result.items()
    }
    if image_sets["train"] & image_sets["validation"] or image_sets["train"] & image_sets["test"]:
        raise ValueError("Training images overlap validation or test images.")
    if image_sets["validation"] & image_sets["test"]:
        raise ValueError("Validation and test images overlap.")
    return result


def _feature_matrices(vision: np.ndarray, question: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "question": question,
        "vision": vision,
        "vision_question": np.concatenate((vision, question), axis=1),
    }


def _metric_summary(metrics: dict[str, Any], trigger_metrics: dict[str, float] | None = None) -> dict[str, Any]:
    result = {
        key: value
        for key, value in metrics.items()
        if key not in {"per_neuron_r2", "per_neuron_correlation"}
    }
    if trigger_metrics is not None:
        result["trigger_prediction"] = trigger_metrics
    return result


def _select_alpha(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_targets: np.ndarray,
    validation_targets: np.ndarray,
    *,
    alphas: list[float],
    target_components: np.ndarray | None,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    candidates = []
    workspace = prepare_linear_mapper(
        train_features,
        train_targets,
        seed=seed,
        target_components=target_components,
    )
    for alpha in alphas:
        mapper = workspace.fit(alpha)
        validation_metrics = regression_metrics(validation_targets, mapper.predict(validation_features))
        candidates.append({"alpha": alpha, "validation_r2": validation_metrics["variance_weighted_r2"]})
    best = max(candidates, key=lambda row: (row["validation_r2"], -row["alpha"]))
    return float(best["alpha"]), candidates


def _prepare_feature_projections(
    feature_values: dict[str, np.ndarray],
    split_indices: dict[str, np.ndarray],
    *,
    rank: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    train = split_indices["train"]
    train_validation = np.concatenate((train, split_indices["validation"]))
    result = {}
    for feature_index, (name, values) in enumerate(feature_values.items()):
        tuning_projector = fit_randomized_projector(values[train], rank=rank, seed=seed + feature_index)
        final_projector = fit_randomized_projector(
            values[train_validation],
            rank=rank,
            seed=seed + 100 + feature_index,
        )
        result[name] = {
            "tuning_train": tuning_projector.transform(values[train]),
            "tuning_validation": tuning_projector.transform(values[split_indices["validation"]]),
            "final_train_validation": final_projector.transform(values[train_validation]),
            "final_test": final_projector.transform(values[split_indices["test"]]),
            "tuning_rank": int(tuning_projector.components.shape[0]),
            "final_rank": int(final_projector.components.shape[0]),
        }
    return result


def _safe_float(value: float) -> float | None:
    return None if not math.isfinite(float(value)) else float(value)


def add_phase4_ranking_columns(table: pd.DataFrame, layer_col: str) -> pd.DataFrame:
    """Add finite mapping/q ranking columns while preserving raw q values for auditing."""
    result = table.copy()
    result["mapping_signal"] = (
        result["mapping_r2"].clip(lower=0.0).fillna(0.0) * result["image_induced_mean_abs_delta"]
    )
    result["mapping_signal_percentile"] = result.groupby(layer_col)["mapping_signal"].rank(
        method="average",
        pct=True,
    )
    # q is undefined for a handful of Phase-1 dead neurons. Preserve the raw
    # NaN, but encode absence of q-response evidence as zero for ranking.
    result["q_multimodal_rank_value"] = (
        pd.to_numeric(result["q_multimodal"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    result["q_multimodal_percentile"] = result.groupby(layer_col)["q_multimodal_rank_value"].rank(
        method="average",
        pct=True,
    )
    result["combined_protection"] = (
        result["mapping_signal_percentile"] + result["q_multimodal_percentile"]
    ) / 2.0
    ranking_columns = (
        "mapping_signal",
        "mapping_signal_percentile",
        "q_multimodal_rank_value",
        "q_multimodal_percentile",
        "combined_protection",
    )
    non_finite = {
        column: int((~np.isfinite(result[column].astype(float))).sum()) for column in ranking_columns
    }
    if any(non_finite.values()):
        raise ValueError(f"Phase-4 ranking columns contain non-finite values: {non_finite}")
    return result


def fit_mapping(args: argparse.Namespace) -> dict[str, Any]:
    if args.feature_rank < 1 or args.target_rank < 1 or args.top_k < 1:
        raise ValueError("feature_rank, target_rank, and top_k must be positive.")
    minimum_nulls = math.ceil(1.0 / args.max_null_p) - 1
    if args.null_permutations < minimum_nulls:
        raise ValueError(
            f"At least {minimum_nulls} null permutations are required to resolve max_null_p={args.max_null_p}."
        )
    alphas = _parse_alphas(args.alphas)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Phase-4 fit output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    vision, question, layer_targets, rows, collection_state = load_activation_chunks(args.activation_dir)
    indices = _split_indices(rows)
    train = indices["train"]
    validation = indices["validation"]
    train_validation = np.concatenate((train, validation))
    test = indices["test"]
    feature_values = _feature_matrices(vision, question)
    projections = _prepare_feature_projections(
        feature_values,
        indices,
        rank=args.feature_rank,
        seed=args.seed,
    )

    layer_results: dict[str, Any] = {}
    primary_neuron_metrics: dict[int, dict[str, np.ndarray]] = {}
    null_by_layer: dict[int, list[float]] = {}
    for layer_position, layer in enumerate(sorted(layer_targets)):
        print(f"Fitting Phase-4 layer {layer}", flush=True)
        targets = np.asarray(layer_targets[layer], dtype=np.float64)
        train_targets = targets[train]
        validation_targets = targets[validation]
        train_validation_targets = targets[train_validation]
        test_targets = targets[test]
        tuning_target_components = fit_target_components(
            train_targets,
            args.target_rank,
            args.seed + 1000 + layer,
        )
        final_target_components = fit_target_components(
            train_validation_targets,
            args.target_rank,
            args.seed + 2000 + layer,
        )
        layer_output: dict[str, Any] = {"models": {}}

        for model_type in MODEL_TYPES:
            layer_output["models"][model_type] = {}
            tuning_components = tuning_target_components if model_type == "reduced_rank" else None
            final_components = final_target_components if model_type == "reduced_rank" else None
            for feature_name in FEATURE_SPACES:
                feature_projection = projections[feature_name]
                best_alpha, tuning = _select_alpha(
                    feature_projection["tuning_train"],
                    feature_projection["tuning_validation"],
                    train_targets,
                    validation_targets,
                    alphas=alphas,
                    target_components=tuning_components,
                    seed=args.seed + layer,
                )
                selection_mapper = fit_linear_mapper(
                    feature_projection["tuning_train"],
                    train_targets,
                    alpha=best_alpha,
                    seed=args.seed + layer,
                    target_components=tuning_components,
                )
                selection_predictions = selection_mapper.predict(feature_projection["tuning_validation"])
                selection_metrics = regression_metrics(validation_targets, selection_predictions)
                mapper = fit_linear_mapper(
                    feature_projection["final_train_validation"],
                    train_validation_targets,
                    alpha=best_alpha,
                    seed=args.seed + layer,
                    target_components=final_components,
                )
                predictions = mapper.predict(feature_projection["final_test"])
                metrics = regression_metrics(test_targets, predictions)
                trigger = topk_trigger_metrics(test_targets, predictions, args.top_k)
                result_row = {
                    "selected_alpha": best_alpha,
                    "alpha_tuning": tuning,
                    "feature_rank": feature_projection["final_rank"],
                    "target_rank": (
                        int(final_components.shape[0]) if final_components is not None else targets.shape[1]
                    ),
                    "selection_validation": _metric_summary(selection_metrics),
                    "test": _metric_summary(metrics, trigger),
                }
                layer_output["models"][model_type][feature_name] = result_row

                if model_type == args.primary_model and feature_name == "vision_question":
                    primary_neuron_metrics[layer] = {
                        "r2": selection_metrics["per_neuron_r2"],
                        "correlation": selection_metrics["per_neuron_correlation"],
                        "mean_abs_delta": np.mean(np.abs(validation_targets), axis=0),
                    }
                    null_values = []
                    rng = np.random.default_rng(args.seed + 10000 + layer_position)
                    for permutation_index in range(args.null_permutations):
                        permuted_targets = train_validation_targets[rng.permutation(len(train_validation_targets))]
                        null_mapper = fit_linear_mapper(
                            feature_projection["final_train_validation"],
                            permuted_targets,
                            alpha=best_alpha,
                            seed=args.seed + permutation_index,
                            target_components=final_components,
                        )
                        null_metrics = regression_metrics(
                            test_targets,
                            null_mapper.predict(feature_projection["final_test"]),
                        )
                        null_values.append(float(null_metrics["variance_weighted_r2"]))
                    observed = float(metrics["variance_weighted_r2"])
                    null_p = (1 + sum(value >= observed for value in null_values)) / (len(null_values) + 1)
                    result_row["permutation_null"] = {
                        "values": null_values,
                        "mean": float(np.mean(null_values)),
                        "max": float(np.max(null_values)),
                        "p_greater_equal": float(null_p),
                        "permutations": args.null_permutations,
                    }
                    null_by_layer[layer] = null_values

        primary_models = layer_output["models"][args.primary_model]
        vq_r2 = primary_models["vision_question"]["test"]["variance_weighted_r2"]
        q_r2 = primary_models["question"]["test"]["variance_weighted_r2"]
        v_r2 = primary_models["vision"]["test"]["variance_weighted_r2"]
        layer_output["primary_comparison"] = {
            "vision_question_r2": vq_r2,
            "question_only_r2": q_r2,
            "vision_only_r2": v_r2,
            "incremental_r2_over_question": vq_r2 - q_r2,
        }
        layer_results[str(layer)] = layer_output

    observed_by_layer = np.asarray(
        [
            layer_results[str(layer)]["primary_comparison"]["vision_question_r2"]
            for layer in sorted(layer_targets)
        ]
    )
    question_by_layer = np.asarray(
        [
            layer_results[str(layer)]["primary_comparison"]["question_only_r2"]
            for layer in sorted(layer_targets)
        ]
    )
    null_matrix = np.asarray([null_by_layer[layer] for layer in sorted(layer_targets)])
    aggregate_null = null_matrix.mean(axis=0)
    observed_mean = float(observed_by_layer.mean())
    aggregate_null_p = float(
        (1 + np.sum(aggregate_null >= observed_mean)) / (args.null_permutations + 1)
    )
    mean_incremental = float(np.mean(observed_by_layer - question_by_layer))
    positive_layers = int(np.sum(observed_by_layer > 0.0))
    gate_a = (
        observed_mean >= args.min_mapping_r2
        and aggregate_null_p <= args.max_null_p
        and positive_layers >= args.min_positive_layers
    )
    gate_b = mean_incremental >= args.min_incremental_r2

    score_table = read_score_table(args.score_file).copy()
    layer_col, neuron_col, _, _ = infer_score_columns(score_table, None)
    if "q_multimodal" not in score_table:
        raise ValueError("The Phase-4 score file must contain q_multimodal.")
    metric_rows = []
    for layer, metrics in primary_neuron_metrics.items():
        width = len(metrics["r2"])
        for neuron in range(width):
            metric_rows.append(
                {
                    layer_col: layer,
                    neuron_col: neuron,
                    "mapping_r2": _safe_float(metrics["r2"][neuron]),
                    "mapping_correlation": _safe_float(metrics["correlation"][neuron]),
                    "image_induced_mean_abs_delta": float(metrics["mean_abs_delta"][neuron]),
                }
            )
    mapping_table = pd.DataFrame(metric_rows)
    augmented = score_table.merge(mapping_table, on=[layer_col, neuron_col], how="left", validate="one_to_one")
    if augmented["mapping_r2"].isna().all():
        raise ValueError("No Phase-4 neuron mapping metrics aligned with the q/r score table.")
    augmented = add_phase4_ranking_columns(augmented, layer_col)
    augmented_path = output_dir / "neuron_scores_with_mapping.parquet"
    augmented.to_parquet(augmented_path, index=False)

    rank_correlations = {}
    for layer, group in augmented.groupby(layer_col):
        rank_correlations[str(int(layer))] = float(
            group["q_multimodal"].corr(group["mapping_signal"], method="spearman")
        )
    rank_correlations["all_layers"] = float(
        augmented["q_multimodal"].corr(augmented["mapping_signal"], method="spearman")
    )

    gates = {
        "gate_a_mapping_exists": {
            "passed": gate_a,
            "observed_mean_vq_r2": observed_mean,
            "minimum_mean_vq_r2": args.min_mapping_r2,
            "aggregate_null_mean": float(aggregate_null.mean()),
            "aggregate_null_p_greater_equal": aggregate_null_p,
            "maximum_null_p": args.max_null_p,
            "positive_layers": positive_layers,
            "minimum_positive_layers": args.min_positive_layers,
        },
        "gate_b_incremental_value": {
            "passed": gate_b,
            "mean_incremental_r2_over_question": mean_incremental,
            "minimum_incremental_r2": args.min_incremental_r2,
        },
        "phase4_group_causal_ablation_allowed": bool(gate_a and gate_b),
    }
    ablation_plan = {
        "allowed": bool(gate_a and gate_b),
        "score_file": str(augmented_path),
        "reference_mask": "rank_band:multimodal:0.05:0.20",
        "conditions": [
            "rank_band:multimodal:0.05:0.20",
            "matched_score:mapping_signal:highest:rank_band:multimodal:0.05:0.20",
            "matched_score:combined_protection:lowest:rank_band:multimodal:0.05:0.20",
        ],
        "interpretation": {
            "mapping_signal_highest": "Tests whether predictable, image-induced neurons are causally important.",
            "combined_protection_lowest": (
                "A provisional deletion candidate only; it must pass matched hook safety before structural pruning."
            ),
        },
    }
    (output_dir / "phase4_ablation_plan.json").write_text(
        json.dumps(ablation_plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result = {
        "complete": True,
        "config": vars(args),
        "collection_state": collection_state,
        "dataset": {
            "rows": len(rows),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "primary_model": args.primary_model,
        "layer_results": layer_results,
        "aggregate": {
            "mean_vision_question_r2": observed_mean,
            "mean_question_only_r2": float(question_by_layer.mean()),
            "mean_incremental_r2_over_question": mean_incremental,
            "aggregate_null_r2_values": aggregate_null.tolist(),
            "q_mapping_spearman": rank_correlations,
        },
        "ranking_definition": {
            "mapping_signal": "max(validation_neuron_r2, 0) * validation_mean_abs_delta",
            "q_rank_input": "q_multimodal with non-finite Phase-1 dead-neuron values encoded as 0",
            "raw_q_preserved": True,
            "combined_protection": "0.5 * mapping_signal_percentile + 0.5 * q_multimodal_percentile",
            "ranking_split": "validation",
        },
        "gates": gates,
        "outputs": {
            "augmented_score_file": str(augmented_path),
            "ablation_plan": str(output_dir / "phase4_ablation_plan.json"),
        },
    }
    (output_dir / "mapping_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    result = fit_mapping(parse_args())
    print(json.dumps(result["gates"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
