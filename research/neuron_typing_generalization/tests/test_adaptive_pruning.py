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

"""Regression tests for exact-knot adaptive A1/A2 selection."""

import numpy as np
import pytest
from neuron_typing_generalization.adaptive_pruning import (
    adaptive_selections,
    nonuniform_selection,
    single_layer_selections,
)


def test_uniform_scores_have_zero_gap_and_budget_limited_rate() -> None:
    q = np.full((2, 3, 10), 0.1, dtype=np.float32)
    plans = adaptive_selections(q, (10, 10), epsilon=0.05)
    assert plans["A1"].candidate_ratio == 0
    assert plans["A1"].actual_ratio == 0
    assert plans["A2"].candidate_ratio == pytest.approx(0.05)
    assert plans["A2"].actual_ratio == 0  # floor(10 * 0.05) in each layer
    assert plans["A2"].actual_score_share == 0


def test_knee_and_budget_use_layer_equal_mean_not_pooled_neurons() -> None:
    q = np.zeros((2, 3, 10), dtype=np.float32)
    q[0, :, :] = 0.1  # Uniform first layer.
    q[1, :, -1] = 1.0  # All mass on one neuron in second layer.
    plans = adaptive_selections(q, (10, 10), epsilon=0.05)
    assert plans["A1"].candidate_ratio == pytest.approx(0.9)
    assert plans["A1"].selected_per_layer == (9, 9)
    assert plans["A1"].actual_score_share == pytest.approx(0.45)
    assert plans["A2"].candidate_ratio == pytest.approx(0.1)
    assert plans["A2"].selected_per_layer == (1, 1)
    assert plans["A2"].actual_score_share == pytest.approx(0.05)
    np.testing.assert_array_equal(plans["A2"].selected[1], [0])


def test_unequal_widths_and_invalid_scores() -> None:
    q = np.full((2, 3, 6), np.nan, dtype=np.float32)
    q[0, :, :4] = 0.25
    q[1, :, :6] = 1 / 6
    plans = adaptive_selections(q, (4, 6), epsilon=0.05)
    assert plans["A1"].candidate_ratio == 0
    assert plans["A2"].actual_score_share <= 0.05
    q[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid"):
        adaptive_selections(q, (4, 6))


def test_nonuniform_matches_exhaustive_optimum_and_nested_sets() -> None:
    from itertools import product

    widths = (3, 4)
    q = np.full((2, 3, 4), np.nan)
    q[0, :, :3] = [0, 2, 8]
    q[1, :, :4] = [1, 2, 3, 4]
    costs = np.array([0, 0.1, 0.4, 0.05, 0.1, 0.15, 0.2])
    previous = set()
    for epsilon in (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.8):
        plan = nonuniform_selection(q, widths, epsilon=epsilon)
        optimum = max(sum(mask) for mask in product((0, 1), repeat=7) if np.dot(mask, costs) <= epsilon + 1e-12)
        assert sum(plan.selected_per_layer) == optimum
        assert plan.actual_score_share <= epsilon + 1e-10
        current = {(l, int(j)) for l, indices in plan.selected.items() for j in indices}
        assert previous <= current
        assert all(0 <= j < widths[l] for l, j in current)
        previous = current
        uniform = adaptive_selections(q, widths, epsilon=epsilon)["A2"]
        assert sum(plan.selected_per_layer) >= sum(uniform.selected_per_layer)


def test_nonuniform_deterministic_ties_and_invalid_input() -> None:
    q = np.ones((2, 3, 4))
    plan = nonuniform_selection(q, (4, 4), epsilon=0.25)
    assert plan.selected_per_layer == (2, 0)
    np.testing.assert_array_equal(plan.selected[0], [0, 1])
    assert nonuniform_selection(q, (4, 4), epsilon=0).actual_ratio == 0
    for epsilon in (-0.1, 1, np.nan):
        with pytest.raises(ValueError):
            nonuniform_selection(q, (4, 4), epsilon=epsilon)
    q[0] = 0
    with pytest.raises(ValueError, match="zero"):
        nonuniform_selection(q, (4, 4), epsilon=0.1)


def test_matched_count_is_exact_and_minimum_cost() -> None:
    from itertools import combinations

    widths = (3, 4)
    q = np.full((2, 3, 4), np.nan)
    q[0, :, :3] = [0, 2, 8]
    q[1, :, :4] = [1, 2, 3, 4]
    costs = np.array([0, 0.1, 0.4, 0.05, 0.1, 0.15, 0.2])
    for eps in (0, 0.05, 0.1, 0.2, 0.3, 0.8):
        u = adaptive_selections(q, widths, epsilon=eps)["A2"]
        k = sum(u.selected_per_layer)
        n = nonuniform_selection(q, widths, epsilon=eps, target_count=k)
        assert n.method == "N_match"
        assert sum(n.selected_per_layer) == k
        assert n.actual_ratio == u.actual_ratio
        optimal = min(costs[list(indices)].sum() for indices in combinations(range(7), k))
        assert n.actual_score_share == pytest.approx(optimal)
        assert n.actual_score_share <= u.actual_score_share + 1e-10
    with pytest.raises(ValueError, match="disagrees"):
        nonuniform_selection(q, widths, epsilon=0.2, target_count=7)
    with pytest.raises(ValueError, match="integer"):
        nonuniform_selection(q, widths, epsilon=0.2, target_count=1.5)


def test_single_layer_masks_leave_other_layers_intact():
    q = np.full((2, 3, 6), np.nan)
    q[0, :, :4] = [4, 1, 3, 2]
    q[1, :, :6] = [6, 5, 4, 3, 2, 1]
    plans = single_layer_selections(q, (4, 6), (0, 1), (0, 0.1, 0.5))
    assert len(plans) == 6
    p = plans["L00_r0.50"]
    np.testing.assert_array_equal(p.selected[0], [1, 3])
    assert len(p.selected[1]) == 0
    assert p.actual_score_share * 2 == pytest.approx(0.3)
    assert p.actual_ratio == pytest.approx(0.2)
    p = plans["L01_r0.50"]
    np.testing.assert_array_equal(p.selected[1], [3, 4, 5])
    assert len(p.selected[0]) == 0
    assert not sum(plans["L01_r0.00"].selected_per_layer)
    with pytest.raises(ValueError):
        single_layer_selections(q, (4, 6), (2,), (0.5,))
