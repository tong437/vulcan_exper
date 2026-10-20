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

"""Sample-conditional A1/A2 weak-neuron selection for graded removal."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .advisor_ablation import per_layer_deletion_count


TARGET_PERCENTS = (1, 5, 10, 15, 20)


def classify_pooled_weak(q_all: NDArray[np.floating]) -> NDArray[np.bool_]:
    """A2 weak means pooled-input share is at most the uniform layer share."""
    values = np.asarray(q_all)
    if (
        values.ndim != 1
        or len(values) < 1
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Pooled shares must be a finite, nonempty one-dimensional array.")
    threshold = np.asarray(1.0 / len(values), dtype=values.dtype)
    return values <= threshold


def select_weak_indices(
    q_three: NDArray[np.floating],
    classes_three: NDArray[np.integer],
    q_all: NDArray[np.floating],
    widths: Sequence[int],
    *,
    method: str,
    removal_percent: float,
) -> tuple[dict[int, NDArray[np.int64]], int, int]:
    """Select the lowest-strength W neurons, clipping each layer to its W pool.

    A1 ranks neurons by qV + qT + qC, after requiring each share to be
    no greater than 1/d_ff. A2 ranks by q_all computed directly over the
    union of visual, question-text, and control input positions. Ties go to
    the smaller neuron index.
    """
    if method not in {"A1", "A2"} or not 0.0 < removal_percent <= 100.0:
        raise ValueError("Method must be A1/A2 and removal_percent must lie in (0, 100].")
    q_three = np.asarray(q_three)
    classes_three = np.asarray(classes_three)
    q_all = np.asarray(q_all)
    widths = tuple(int(width) for width in widths)
    if q_three.ndim != 3 or q_three.shape[1] != 3:
        raise ValueError("Three-role shares must have shape [layers, 3, maximum_d_ff].")
    layers, _, maximum_width = q_three.shape
    if (
        classes_three.shape != (layers, maximum_width)
        or q_all.shape != (layers, maximum_width)
        or len(widths) != layers
        or any(width < 1 or width > maximum_width for width in widths)
    ):
        raise ValueError("Classes, pooled shares, and real layer widths must align.")

    selected: dict[int, NDArray[np.int64]] = {}
    shortfall_layers = 0
    available_neurons = 0
    for layer, width in enumerate(widths):
        threshold = 1.0 / width
        three_raw = q_three[layer, :, :width]
        pooled_raw = q_all[layer, :width]
        three = three_raw.astype(np.float64)
        pooled = pooled_raw.astype(np.float64)
        if not np.isfinite(three).all() or not np.isfinite(pooled).all():
            raise ValueError(f"Layer {layer} contains non-finite real-neuron shares.")
        if method == "A1":
            weak = np.all(three_raw <= np.asarray(threshold, dtype=three_raw.dtype), axis=0)
            if not np.array_equal(weak, classes_three[layer, :width] == 4):
                raise ValueError(f"Layer {layer} A1 classes disagree with three-role shares.")
            strengths = three.sum(axis=0)
        else:
            weak = classify_pooled_weak(pooled_raw)
            strengths = pooled
        candidates = np.flatnonzero(weak).astype(np.int64)
        target = per_layer_deletion_count(width, removal_percent / 100.0)
        count = min(target, len(candidates))
        available_neurons += len(candidates)
        shortfall_layers += int(count < target)
        order = np.lexsort((candidates, strengths[candidates]))
        selected[layer] = np.sort(candidates[order[:count]])
    return selected, shortfall_layers, available_neurons
