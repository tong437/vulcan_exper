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

"""Pairwise input-dependence and cross-sample neuron stability analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .importance import CLASS_LABELS, INVALID_CLASS


ClassArray = NDArray[np.integer]


@dataclass(frozen=True)
class ChangeRateSummary:
    """Overall and per-layer class-change rates."""

    including_self: float
    distinct_pairs: float
    per_layer_including_self: NDArray[np.float64]
    per_layer_distinct_pairs: NDArray[np.float64]
    including_self_changed: int
    including_self_comparisons: int
    distinct_pairs_changed: int
    distinct_pair_comparisons: int
    per_layer_including_self_changed: NDArray[np.int64]
    per_layer_including_self_comparisons: NDArray[np.int64]
    per_layer_distinct_pairs_changed: NDArray[np.int64]
    per_layer_distinct_pair_comparisons: NDArray[np.int64]


@dataclass(frozen=True)
class MaximumPair:
    """The deterministic maximum-Hamming-distance sample pair."""

    left: int
    right: int
    rate: float
    valid_neurons: int


@dataclass(frozen=True)
class StabilitySummary:
    """Per-neuron class occupancy, entropy, and derived profile labels."""

    counts: NDArray[np.int64]
    proportions: NDArray[np.float64]
    entropy: NDArray[np.float64]
    normalized_entropy: NDArray[np.float64]
    dominant_class: NDArray[np.int8]
    profiles: NDArray[np.object_]


def _validate_classes(classes: ClassArray) -> NDArray[np.int64]:
    values = np.asarray(classes, dtype=np.int64)
    if values.ndim != 3:
        raise ValueError("classes must have shape [samples, layers, d_ff].")
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("classes dimensions must be non-empty.")
    valid_values = (values == INVALID_CLASS) | ((0 <= values) & (values < len(CLASS_LABELS)))
    if not np.all(valid_values):
        raise ValueError("classes contain unknown class IDs.")
    return values


def change_rates_from_counts(classes: ClassArray) -> ChangeRateSummary:
    """Compute study-wide mismatch rates without materializing all N-squared pairs."""
    values = _validate_classes(classes)
    valid_counts = np.count_nonzero(values != INVALID_CLASS, axis=0).astype(np.int64)
    class_counts = np.stack(
        [np.count_nonzero(values == class_id, axis=0) for class_id in range(len(CLASS_LABELS))], axis=-1
    ).astype(np.int64)
    squared_counts = np.square(class_counts).sum(axis=-1)
    mismatch_counts = np.square(valid_counts) - squared_counts
    including_denominators = np.square(valid_counts)
    distinct_denominators = valid_counts * (valid_counts - 1)
    per_layer_changed = mismatch_counts.sum(axis=1, dtype=np.int64)
    per_layer_including_denominators = including_denominators.sum(axis=1, dtype=np.int64)
    per_layer_distinct_denominators = distinct_denominators.sum(axis=1, dtype=np.int64)
    per_layer_including = np.divide(
        per_layer_changed,
        per_layer_including_denominators,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=per_layer_including_denominators > 0,
    )
    per_layer_distinct = np.divide(
        per_layer_changed,
        per_layer_distinct_denominators,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=per_layer_distinct_denominators > 0,
    )
    including_changed = int(per_layer_changed.sum())
    including_comparisons = int(per_layer_including_denominators.sum())
    distinct_comparisons = int(per_layer_distinct_denominators.sum())
    return ChangeRateSummary(
        including_self=including_changed / including_comparisons if including_comparisons else float("nan"),
        distinct_pairs=including_changed / distinct_comparisons if distinct_comparisons else float("nan"),
        per_layer_including_self=per_layer_including,
        per_layer_distinct_pairs=per_layer_distinct,
        including_self_changed=including_changed,
        including_self_comparisons=including_comparisons,
        distinct_pairs_changed=including_changed,
        distinct_pair_comparisons=distinct_comparisons,
        per_layer_including_self_changed=per_layer_changed,
        per_layer_including_self_comparisons=per_layer_including_denominators,
        per_layer_distinct_pairs_changed=per_layer_changed.copy(),
        per_layer_distinct_pair_comparisons=per_layer_distinct_denominators,
    )


def pair_change_rate(left: ClassArray, right: ClassArray) -> tuple[float, int]:
    """Return Hamming change rate over neurons valid in both samples."""
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.shape != right_values.shape:
        raise ValueError("Pair class arrays must have identical shapes.")
    valid = (left_values != INVALID_CLASS) & (right_values != INVALID_CLASS)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return float("nan"), 0
    return float(np.count_nonzero((left_values != right_values) & valid) / valid_count), valid_count


def maximum_change_pair(
    classes: ClassArray,
    chunk_size: int = 64,
    candidate_pairs: NDArray[np.integer] | None = None,
) -> MaximumPair:
    """Find the most different pair in bounded memory with deterministic tie-breaking."""
    values = _validate_classes(classes)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    sample_count = values.shape[0]
    if sample_count < 2:
        raise ValueError("At least two samples are required.")

    flattened = values.reshape(sample_count, -1)
    valid = flattened != INVALID_CLASS
    safe = np.where(valid, flattened, 0).astype(np.uint8)
    low_bits = np.packbits(safe & 1, axis=1, bitorder="little")
    high_bits = np.packbits((safe >> 1) & 1, axis=1, bitorder="little")
    valid_bits = np.packbits(valid, axis=1, bitorder="little")
    best: MaximumPair | None = None

    def consider_blocks(left_indices: NDArray[np.int64], right_indices: NDArray[np.int64]) -> None:
        nonlocal best
        pair_valid = valid_bits[left_indices] & valid_bits[right_indices]
        differing = (low_bits[left_indices] ^ low_bits[right_indices]) | (
            high_bits[left_indices] ^ high_bits[right_indices]
        )
        valid_counts = np.bitwise_count(pair_valid).sum(axis=1)
        mismatch_counts = np.bitwise_count(differing & pair_valid).sum(axis=1)
        rates = np.divide(
            mismatch_counts,
            valid_counts,
            out=np.full(len(left_indices), np.nan, dtype=np.float64),
            where=valid_counts > 0,
        )
        for offset in np.flatnonzero(np.isfinite(rates)):
            candidate = MaximumPair(
                int(left_indices[offset]), int(right_indices[offset]), float(rates[offset]), int(valid_counts[offset])
            )
            if (
                best is None
                or candidate.rate > best.rate
                or (candidate.rate == best.rate and (candidate.left, candidate.right) < (best.left, best.right))
            ):
                best = candidate

    if candidate_pairs is not None:
        pairs = np.asarray(candidate_pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("candidate_pairs must have shape [pair_count, 2].")
        if np.any(pairs < 0) or np.any(pairs >= sample_count):
            raise IndexError("candidate_pairs reference a sample outside classes.")
        pairs = np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1)
        pairs = np.unique(pairs, axis=0)
        for start in range(0, len(pairs), chunk_size):
            block = pairs[start : start + chunk_size]
            consider_blocks(block[:, 0], block[:, 1])
    else:
        for left in range(sample_count - 1):
            for start in range(left + 1, sample_count, chunk_size):
                right_indices = np.arange(start, min(start + chunk_size, sample_count), dtype=np.int64)
                left_indices = np.full(len(right_indices), left, dtype=np.int64)
                consider_blocks(left_indices, right_indices)

    if best is None:
        raise ValueError("No pair has any mutually valid neurons.")
    return best


def neuron_stability(classes: ClassArray) -> StabilitySummary:
    """Summarize each neuron's class distribution across samples."""
    values = _validate_classes(classes)
    _, layer_count, d_ff = values.shape
    counts = np.zeros((layer_count, d_ff, len(CLASS_LABELS)), dtype=np.int64)
    for class_id in range(len(CLASS_LABELS)):
        counts[..., class_id] = np.count_nonzero(values == class_id, axis=0)
    totals = counts.sum(axis=-1, keepdims=True)
    proportions = np.divide(
        counts,
        totals,
        out=np.full(counts.shape, np.nan, dtype=np.float64),
        where=totals > 0,
    )
    log_proportions = np.zeros_like(proportions)
    np.log(proportions, out=log_proportions, where=proportions > 0)
    entropy = -(proportions * log_proportions).sum(axis=-1)
    entropy = np.where(totals[..., 0] > 0, entropy, np.nan)
    normalized_entropy = entropy / np.log(len(CLASS_LABELS))
    dominant = np.argmax(counts, axis=-1).astype(np.int8)
    dominant = np.where(totals[..., 0] > 0, dominant, INVALID_CLASS).astype(np.int8)

    profiles = np.empty((layer_count, d_ff), dtype=object)
    constant_names = {
        0: "constant_weak",
        1: "constant_visual",
        2: "constant_text",
        3: "constant_multimodal",
    }
    for layer in range(layer_count):
        for neuron in range(d_ff):
            present = tuple(np.flatnonzero(counts[layer, neuron] > 0).tolist())
            if not present:
                profile = "invalid"
            elif len(present) == 1:
                profile = constant_names[present[0]]
            elif present == (1, 2):
                profile = "dynamic_v_t"
            elif present == (1, 3):
                profile = "dynamic_v_m"
            elif present == (2, 3):
                profile = "dynamic_t_m"
            elif 0 in present:
                profile = "dynamic_with_weak"
            else:
                profile = "dynamic_multi_state"
            profiles[layer, neuron] = profile
    return StabilitySummary(counts, proportions, entropy, normalized_entropy, dominant, profiles)


def transition_matrix(
    classes: ClassArray,
    pairs: NDArray[np.integer],
    symmetric: bool = False,
) -> NDArray[np.int64]:
    """Count per-layer class transitions along explicit manifest pair edges."""
    values = _validate_classes(classes)
    pair_values = np.asarray(pairs, dtype=np.int64)
    if pair_values.ndim != 2 or pair_values.shape[1] != 2:
        raise ValueError("pairs must have shape [pair_count, 2].")
    if np.any(pair_values < 0) or np.any(pair_values >= values.shape[0]):
        raise IndexError("pairs reference a sample outside classes.")

    matrix = np.zeros((values.shape[1], len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)
    for left, right in pair_values:
        left_values = values[left]
        right_values = values[right]
        for layer in range(values.shape[1]):
            valid = (left_values[layer] != INVALID_CLASS) & (right_values[layer] != INVALID_CLASS)
            np.add.at(matrix[layer], (left_values[layer, valid], right_values[layer, valid]), 1)
            if symmetric:
                np.add.at(matrix[layer], (right_values[layer, valid], left_values[layer, valid]), 1)
    return matrix
