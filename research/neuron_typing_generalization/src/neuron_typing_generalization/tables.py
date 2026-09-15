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

"""Tabular adapters connecting GPU-produced scores to CPU analysis functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import (
    change_rates_from_counts,
    maximum_change_pair,
    neuron_stability,
    pair_change_rate,
    transition_matrix,
)
from .importance import CLASS_LABELS, INVALID_CLASS


SCORE_KEYS = ["run_id", "sample_id", "model", "dataset", "layer", "neuron"]


def _finite_float_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def classify_score_table(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize long-form importance rows and create sample-conditional classes."""
    required = set(SCORE_KEYS) | {
        "modality",
        "token_count",
        "activation_rms",
        "down_projection_norm",
        "importance",
        "valid",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Score table is missing columns: {missing}")
    if not set(scores["modality"]).issubset({"visual", "text"}):
        raise ValueError("Score modalities must be exactly visual or text.")
    if scores.duplicated(SCORE_KEYS + ["modality"]).any():
        raise ValueError("Score table has duplicate sample/layer/neuron/modality rows.")

    normalized = scores.copy()
    group_keys = ["run_id", "sample_id", "model", "dataset", "layer", "modality"]
    denominator = normalized.groupby(group_keys, sort=False)["importance"].transform("sum")
    slice_valid = normalized.groupby(group_keys, sort=False)["valid"].transform("all") & denominator.gt(0)
    normalized["valid"] = normalized["valid"].astype(bool) & slice_valid
    normalized["q"] = normalized["importance"].div(denominator).where(normalized["valid"])

    visual = normalized[normalized["modality"] == "visual"][SCORE_KEYS + ["q", "valid"]].rename(
        columns={"q": "q_visual", "valid": "visual_valid"}
    )
    text = normalized[normalized["modality"] == "text"][SCORE_KEYS + ["q", "valid"]].rename(
        columns={"q": "q_text", "valid": "text_valid"}
    )
    classes = visual.merge(text, on=SCORE_KEYS, how="outer", validate="one_to_one")
    if classes[["visual_valid", "text_valid"]].isna().any().any():
        raise ValueError("Every neuron must have both visual and text score rows.")

    slice_keys = ["run_id", "sample_id", "model", "dataset", "layer"]
    neuron_count = classes.groupby(slice_keys, sort=False)["neuron"].transform("nunique")
    expected_count = classes.groupby(slice_keys, sort=False)["neuron"].transform("max") + 1
    if not neuron_count.eq(expected_count).all():
        raise ValueError("Neuron IDs must be contiguous from zero in every sample/layer slice.")
    threshold = 1.0 / neuron_count
    classes["valid"] = classes["visual_valid"] & classes["text_valid"]
    classes["visual_strong"] = classes["q_visual"].gt(threshold) & classes["valid"]
    classes["text_strong"] = classes["q_text"].gt(threshold) & classes["valid"]
    class_ids = classes["visual_strong"].astype(np.int8) + 2 * classes["text_strong"].astype(np.int8)
    classes["class"] = [
        CLASS_LABELS[value] if valid else "INVALID" for value, valid in zip(class_ids, classes["valid"])
    ]
    columns = SCORE_KEYS + ["q_visual", "q_text", "visual_strong", "text_strong", "class", "valid"]
    return normalized, classes[columns].sort_values(SCORE_KEYS).reset_index(drop=True)


def class_tensor_from_table(
    classes: pd.DataFrame,
    sample_ids: list[str],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Build a dense [sample, layer, neuron] tensor after completeness checks."""
    layers = sorted(int(value) for value in classes["layer"].unique())
    neurons = sorted(int(value) for value in classes["neuron"].unique())
    if layers != list(range(len(layers))) or neurons != list(range(len(neurons))):
        raise ValueError("Layer and neuron IDs must be contiguous from zero.")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be unique and ordered by the frozen manifest.")
    class_to_id = {label: index for index, label in enumerate(CLASS_LABELS)}
    class_to_id["INVALID"] = INVALID_CLASS
    indexed = classes.set_index(["sample_id", "layer", "neuron"])
    expected = pd.MultiIndex.from_product([sample_ids, layers, neurons], names=["sample_id", "layer", "neuron"])
    if indexed.index.has_duplicates or not indexed.index.equals(expected):
        indexed = indexed.reindex(expected)
    if indexed["class"].isna().any():
        raise ValueError("Class table is incomplete for the frozen sample/layer/neuron grid.")
    tensor = (
        indexed["class"].map(class_to_id).to_numpy(dtype=np.int8).reshape(len(sample_ids), len(layers), len(neurons))
    )
    return tensor, layers, neurons


def analyze_class_tensor(
    tensor: np.ndarray,
    sample_ids: list[str],
    run_id: str,
    pair_manifest: pd.DataFrame,
    chunk_size: int = 64,
) -> tuple[dict, pd.DataFrame, dict[str, np.ndarray]]:
    """Analyze a compact [sample, layer, neuron] class tensor."""
    if tensor.ndim != 3 or tensor.shape[0] != len(sample_ids):
        raise ValueError("Compact classes must have shape [sample, layer, neuron].")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be unique and ordered by the frozen manifest.")
    layers = list(range(tensor.shape[1]))
    neurons = list(range(tensor.shape[2]))
    rates = change_rates_from_counts(tensor)
    maximum = maximum_change_pair(tensor, chunk_size=chunk_size)
    stability = neuron_stability(tensor)
    summary = {
        "run_id": run_id,
        "sample_count": len(sample_ids),
        "overall_change_rate_including_self": _finite_float_or_none(rates.including_self),
        "overall_change_rate_distinct_pairs": _finite_float_or_none(rates.distinct_pairs),
        "per_layer_change_rate_including_self": [
            _finite_float_or_none(value) for value in rates.per_layer_including_self
        ],
        "per_layer_change_rate_distinct_pairs": [
            _finite_float_or_none(value) for value in rates.per_layer_distinct_pairs
        ],
        "overall_changed_ordered_comparisons": rates.including_self_changed,
        "overall_valid_ordered_comparisons_including_self": rates.including_self_comparisons,
        "overall_changed_distinct_ordered_comparisons": rates.distinct_pairs_changed,
        "overall_valid_distinct_ordered_comparisons": rates.distinct_pair_comparisons,
        "per_layer_changed_ordered_comparisons": rates.per_layer_including_self_changed.tolist(),
        "per_layer_valid_ordered_comparisons_including_self": rates.per_layer_including_self_comparisons.tolist(),
        "per_layer_changed_distinct_ordered_comparisons": rates.per_layer_distinct_pairs_changed.tolist(),
        "per_layer_valid_distinct_ordered_comparisons": rates.per_layer_distinct_pair_comparisons.tolist(),
        "maximum_pair": {
            "left_sample_id": sample_ids[maximum.left],
            "right_sample_id": sample_ids[maximum.right],
            "change_rate": maximum.rate,
            "valid_neurons": maximum.valid_neurons,
        },
    }

    profile_rows = []
    for layer in layers:
        for neuron in neurons:
            counts = stability.counts[layer, neuron]
            proportions = stability.proportions[layer, neuron]
            dominant = int(stability.dominant_class[layer, neuron])
            profile_rows.append(
                {
                    "run_id": run_id,
                    "layer": layer,
                    "neuron": neuron,
                    "valid_samples": int(counts.sum()),
                    **{f"count_{label}": int(counts[index]) for index, label in enumerate(CLASS_LABELS)},
                    **{f"proportion_{label}": float(proportions[index]) for index, label in enumerate(CLASS_LABELS)},
                    "entropy": float(stability.entropy[layer, neuron]),
                    "normalized_entropy": float(stability.normalized_entropy[layer, neuron]),
                    "dominant_class": CLASS_LABELS[dominant] if dominant != INVALID_CLASS else "INVALID",
                    "profile": str(stability.profiles[layer, neuron]),
                }
            )

    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if len(pair_manifest):
        pair_indices = np.asarray(
            [
                [sample_index[str(row.left_sample_id)], sample_index[str(row.right_sample_id)]]
                for row in pair_manifest.itertuples()
            ],
            dtype=np.int64,
        )
        transitions = {
            "directed": transition_matrix(tensor, pair_indices, symmetric=False),
            "symmetrized": transition_matrix(tensor, pair_indices, symmetric=True),
        }
    else:
        empty = np.zeros((len(layers), len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)
        transitions = {"directed": empty, "symmetrized": empty.copy()}
    return summary, pd.DataFrame(profile_rows), transitions


def controlled_pair_change_tensor(
    tensor: np.ndarray,
    sample_ids: list[str],
    run_id: str,
    pair_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Compute controlled-pair changes directly from compact classes."""
    columns = [
        "run_id",
        "left_sample_id",
        "right_sample_id",
        "pair_scope",
        "layer",
        "changed_neurons",
        "valid_neurons",
        "change_rate",
    ]
    if not len(pair_manifest):
        return pd.DataFrame(columns=columns)
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    rows = []
    for pair in pair_manifest.itertuples():
        left_id = str(pair.left_sample_id)
        right_id = str(pair.right_sample_id)
        left = tensor[sample_index[left_id]]
        right = tensor[sample_index[right_id]]
        for layer in [None, *range(tensor.shape[1])]:
            left_values = left if layer is None else left[layer]
            right_values = right if layer is None else right[layer]
            rate, valid_neurons = pair_change_rate(left_values, right_values)
            changed_neurons = int(
                np.count_nonzero(
                    (left_values != right_values) & (left_values != INVALID_CLASS) & (right_values != INVALID_CLASS)
                )
            )
            rows.append(
                {
                    "run_id": run_id,
                    "left_sample_id": left_id,
                    "right_sample_id": right_id,
                    "pair_scope": "controlled",
                    "layer": layer,
                    "changed_neurons": changed_neurons,
                    "valid_neurons": valid_neurons,
                    "change_rate": rate if np.isfinite(rate) else None,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def analyze_class_table(
    classes: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    pair_manifest: pd.DataFrame,
    chunk_size: int = 64,
) -> tuple[dict, pd.DataFrame, dict[str, np.ndarray]]:
    """Produce run summary, neuron profiles, and controlled transition matrices."""
    sample_ids = [str(value) for value in sample_manifest["sample_id"]]
    tensor, _, _ = class_tensor_from_table(classes, sample_ids)
    run_id = str(classes["run_id"].iloc[0])
    return analyze_class_tensor(tensor, sample_ids, run_id, pair_manifest, chunk_size=chunk_size)


def controlled_pair_change_table(
    classes: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    pair_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Compute overall and per-layer rates only for explicit controlled pair edges."""
    sample_ids = [str(value) for value in sample_manifest["sample_id"]]
    tensor, _, _ = class_tensor_from_table(classes, sample_ids)
    run_id = str(classes["run_id"].iloc[0])
    return controlled_pair_change_tensor(tensor, sample_ids, run_id, pair_manifest)
