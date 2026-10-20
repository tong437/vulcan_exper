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

"""Tests for separate-role and pooled-input W selection."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from neuron_typing_generalization.interfaces import FFNBinding, TokenRoleMasks
from neuron_typing_generalization.weak_pruning import classify_pooled_weak, select_weak_indices
from neuron_typing_generalization.weak_pruning_runner import _pooled_shares, _summarize


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_three = np.array(
        [
            [
                [0.04, 0.14, 0.16, 0.20, 0.22, 0.24],
                [0.05, 0.13, 0.16, 0.20, 0.22, 0.24],
                [0.06, 0.12, 0.16, 0.20, 0.22, 0.24],
            ]
        ],
        dtype=np.float32,
    )
    classes = np.array([[4, 4, 4, 3, 3, 3]], dtype=np.int8)
    pooled = np.array([[0.01, 0.07, 0.17, 0.20, 0.25, 0.30]], dtype=np.float32)
    return q_three, classes, pooled


def test_pooled_weak_uses_inclusive_uniform_threshold() -> None:
    values = np.array([0.1, 0.25, 0.25, 0.4])
    np.testing.assert_array_equal(classify_pooled_weak(values), [True, True, True, False])


def test_float32_threshold_equality_matches_frozen_a1_classification() -> None:
    width = 12288
    threshold = np.float32(1 / width)
    q_three = np.full((1, 3, width), threshold, dtype=np.float32)
    classes = np.full((1, width), 4, dtype=np.int8)
    pooled = np.full((1, width), threshold, dtype=np.float32)
    a1, _, _ = select_weak_indices(q_three, classes, pooled, (width,), method="A1", removal_percent=1)
    a2, _, _ = select_weak_indices(q_three, classes, pooled, (width,), method="A2", removal_percent=1)
    assert len(a1[0]) == len(a2[0]) == 122
    assert classify_pooled_weak(pooled[0]).all()


def test_a1_a2_clip_to_available_and_rank_differently() -> None:
    q_three, classes, pooled = _fixture()
    a1, a1_shortfall, a1_available = select_weak_indices(
        q_three, classes, pooled, (6,), method="A1", removal_percent=20
    )
    a2, a2_shortfall, a2_available = select_weak_indices(
        q_three, classes, pooled, (6,), method="A2", removal_percent=20
    )
    np.testing.assert_array_equal(a1[0], [0])
    np.testing.assert_array_equal(a2[0], [0])
    assert (a1_shortfall, a1_available) == (0, 3)
    assert (a2_shortfall, a2_available) == (0, 2)

    a1, a1_shortfall, _ = select_weak_indices(q_three, classes, pooled, (6,), method="A1", removal_percent=80)
    a2, a2_shortfall, _ = select_weak_indices(q_three, classes, pooled, (6,), method="A2", removal_percent=80)
    np.testing.assert_array_equal(a1[0], [0, 1, 2])
    np.testing.assert_array_equal(a2[0], [0, 1])
    assert (a1_shortfall, a2_shortfall) == (1, 1)


def test_a1_class_must_agree_with_three_role_shares() -> None:
    q_three, classes, pooled = _fixture()
    classes[0, 0] = 3
    with pytest.raises(ValueError, match="disagree"):
        select_weak_indices(q_three, classes, pooled, (6,), method="A1", removal_percent=1)


def test_weak_sets_can_differ_between_separate_and_pooled() -> None:
    q_three, classes, pooled = _fixture()
    pooled[0] = [0.01, 0.18, 0.15, 0.16, 0.20, 0.30]
    a1, _, _ = select_weak_indices(q_three, classes, pooled, (6,), method="A1", removal_percent=50)
    a2, _, _ = select_weak_indices(q_three, classes, pooled, (6,), method="A2", removal_percent=50)
    np.testing.assert_array_equal(a1[0], [0, 1, 2])
    np.testing.assert_array_equal(a2[0], [0, 2, 3])


def test_pooled_shares_are_computed_from_union_rms_not_sum_of_separate_shares() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_projection = torch.nn.Linear(3, 2, bias=False)
            with torch.no_grad():
                self.down_projection.weight.fill_(1.0)

        def forward(self, x: torch.Tensor, **_kwargs: object) -> torch.Tensor:
            return self.down_projection(x)

    model = TinyModel()
    activations = torch.tensor([[[9.0, 1.0, 1.0], [1.0, 4.0, 1.0], [1.0, 1.0, 2.0]]])
    roles = TokenRoleMasks(
        visual=torch.tensor([True, False, False]),
        text=torch.tensor([False, True, False]),
        control=torch.tensor([False, False, True]),
        padding=torch.tensor([False, False, False]),
    )
    binding = FFNBinding(0, model.down_projection, 2, 3)
    norm = torch.linalg.vector_norm(model.down_projection.weight, ord=2, dim=0)
    pooled = _pooled_shares(model, {"x": activations}, roles, (binding,), {0: norm}, (3,))
    rms = np.sqrt(np.mean(activations[0].numpy() ** 2, axis=0))
    np.testing.assert_allclose(pooled[0], rms / rms.sum(), atol=1e-6)
    separate = activations[0].numpy() / activations[0].numpy().sum(axis=1, keepdims=True)
    assert not np.allclose(pooled[0], separate.mean(axis=0))


def test_summary_records_weighted_actual_removal_and_all_conditions() -> None:
    rows = []
    for sample_index in range(2):
        rows.append(
            {
                "sample_index": sample_index,
                "condition": "baseline",
                "method": None,
                "removal_percent": 0,
                "score": 1.0,
                "selected_neurons": 0,
                "target_neurons": 0,
                "available_weak_neurons": 0,
            }
        )
        for method in ("A1", "A2"):
            for percent in (1, 5, 10, 15, 20):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "condition": f"{method}_{percent:02d}pct",
                        "method": method,
                        "removal_percent": percent,
                        "score": 0.5,
                        "selected_neurons": 1,
                        "target_neurons": 1,
                        "available_weak_neurons": 2,
                    }
                )
    summary = _summarize(rows, expected_samples=2, widths=(6,))
    assert summary["total_neurons"] == 6
    assert len(summary["conditions"]) == 11
    selected = next(row for row in summary["conditions"] if row["condition"] == "A1_01pct")
    assert selected["actual_removal_percent"] == pytest.approx(100 / 6)
