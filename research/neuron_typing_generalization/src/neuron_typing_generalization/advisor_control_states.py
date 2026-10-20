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

"""Five-way visual/text/mixed/control/weak response-state utilities."""

from __future__ import annotations

import numpy as np


CONTROL_STATE_ORDER = ("V", "T", "M", "C", "W")


def classify_control_shares(q_visual: np.ndarray, q_text: np.ndarray, q_control: np.ndarray) -> np.ndarray:
    """Classify neuron shares, carving C only out of the former weak state.

    Equality to ``1 / d_ff`` is weak. V/T/M retain their original definitions;
    control strength is consulted only when both visual and text are weak.
    """
    arrays = tuple(np.asarray(value) for value in (q_visual, q_text, q_control))
    if any(value.shape != arrays[0].shape for value in arrays[1:]) or arrays[0].ndim != 1:
        raise ValueError("q_visual, q_text, and q_control must be aligned one-dimensional arrays.")
    if not len(arrays[0]) or not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Response shares must be nonempty and finite.")
    threshold = 1.0 / len(arrays[0])
    visual_strong, text_strong, control_strong = (value > threshold for value in arrays)
    classes = np.full(len(arrays[0]), 4, dtype=np.int8)  # W
    classes[~visual_strong & ~text_strong & control_strong] = 3  # C
    classes[visual_strong & ~text_strong] = 0  # V
    classes[~visual_strong & text_strong] = 1  # T
    classes[visual_strong & text_strong] = 2  # M
    return classes


def validate_state_counts(counts: np.ndarray, completed_samples: int, d_ff_by_layer: list[int]) -> None:
    """Validate a resumable ``[state, layer, maximum_d_ff]`` count tensor."""
    if counts.ndim != 3 or counts.shape[0] != len(CONTROL_STATE_ORDER):
        raise ValueError("Five-state counts must have shape [5, layers, maximum_d_ff].")
    if counts.shape[1] != len(d_ff_by_layer) or counts.shape[2] < max(d_ff_by_layer):
        raise ValueError("Five-state count shape disagrees with d_ff_by_layer.")
    if completed_samples < 0 or np.any(counts < 0):
        raise ValueError("Sample and state counts must be nonnegative.")
    for layer, width in enumerate(d_ff_by_layer):
        if not np.all(counts[:, layer, :width].sum(axis=0) == completed_samples):
            raise ValueError(f"Layer {layer} real-neuron counts do not sum to completed_samples.")
        if np.any(counts[:, layer, width:] != 0):
            raise ValueError(f"Layer {layer} contains counts beyond d_ff={width}.")
