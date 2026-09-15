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

import itertools

import numpy as np
import pytest
from neuron_typing_generalization.analysis import (
    change_rates_from_counts,
    maximum_change_pair,
    neuron_stability,
    normal_precision_pair_count,
    pair_change_rate,
    pair_level_bootstrap_ratio,
    transition_matrix,
)


def _brute_force(classes: np.ndarray, include_self: bool) -> float:
    rates = []
    for left, right in itertools.product(range(len(classes)), repeat=2):
        if not include_self and left == right:
            continue
        rates.append(np.not_equal(classes[left], classes[right]).mean())
    return float(np.mean(rates))


def test_count_formula_matches_brute_force_with_and_without_self_pairs():
    classes = np.array([[[0, 1], [2, 3]], [[0, 2], [2, 3]], [[1, 2], [0, 3]]])
    result = change_rates_from_counts(classes)

    assert result.including_self == pytest.approx(_brute_force(classes, include_self=True))
    assert result.distinct_pairs == pytest.approx(_brute_force(classes, include_self=False))
    assert result.per_layer_including_self.shape == (2,)
    assert result.including_self_changed == 12
    assert result.including_self_comparisons == 36
    assert result.distinct_pair_comparisons == 24


def test_count_formula_uses_total_valid_comparisons_when_invalid_counts_differ():
    classes = np.array([[[0, 0]], [[1, -1]], [[1, -1]]])
    result = change_rates_from_counts(classes)

    # Neuron 0: 4 changed / 9 including-self comparisons. Neuron 1: 0 / 1.
    assert result.including_self_changed == 4
    assert result.including_self_comparisons == 10
    assert result.including_self == pytest.approx(0.4)
    # Only neuron 0 has distinct valid comparisons: 4 changed / 6 comparisons.
    assert result.distinct_pair_comparisons == 6
    assert result.distinct_pairs == pytest.approx(2 / 3)


def test_count_formula_marks_a_layer_without_valid_comparisons_as_nan():
    classes = np.array([[[0], [-1]], [[1], [-1]]])
    result = change_rates_from_counts(classes)

    assert result.per_layer_including_self[0] == pytest.approx(0.5)
    assert np.isnan(result.per_layer_including_self[1])
    assert result.per_layer_including_self_comparisons.tolist() == [4, 0]
    assert result.per_layer_distinct_pair_comparisons.tolist() == [2, 0]


def test_pair_change_rate_excludes_invalid_neurons():
    rate, valid = pair_change_rate(np.array([[0, -1, 1]]), np.array([[1, 3, 1]]))
    assert valid == 2
    assert rate == 0.5


def test_chunked_maximum_is_chunk_invariant_and_uses_lexicographic_tie_break():
    classes = np.array([[[0, 0]], [[1, 1]], [[2, 2]], [[0, 0]]])
    first = maximum_change_pair(classes, chunk_size=1)
    second = maximum_change_pair(classes, chunk_size=3)

    assert first == second
    assert (first.left, first.right, first.rate) == (0, 1, 1.0)


def test_maximum_can_be_restricted_to_manifest_pairs():
    classes = np.array([[[0, 0]], [[1, 1]], [[0, 1]]])
    result = maximum_change_pair(classes, candidate_pairs=np.array([[0, 2], [1, 2]]))
    assert (result.left, result.right, result.rate) == (0, 2, 0.5)


def test_stability_profiles_proportions_and_entropy():
    classes = np.array([[[1, 1, 0]], [[1, 2, 3]], [[1, 1, 0]]])
    result = neuron_stability(classes)

    assert result.profiles.tolist() == [["constant_visual", "dynamic_v_t", "dynamic_with_weak"]]
    np.testing.assert_allclose(result.proportions[0, 0], [0.0, 1.0, 0.0, 0.0])
    assert result.entropy[0, 0] == pytest.approx(0.0)
    assert 0 < result.normalized_entropy[0, 1] < 1


def test_transition_matrix_uses_explicit_direction_and_optional_symmetry():
    classes = np.array([[[1, 2]], [[2, 3]]])
    directed = transition_matrix(classes, np.array([[0, 1]]))
    symmetric = transition_matrix(classes, np.array([[0, 1]]), symmetric=True)

    assert directed[0, 1, 2] == 1
    assert directed[0, 2, 3] == 1
    assert symmetric[0, 2, 1] == 1
    assert symmetric[0, 3, 2] == 1


def test_pair_level_bootstrap_is_deterministic_and_uses_ratio_of_sums():
    changed = np.array([[1, 2], [2, 1], [8, 4]])
    valid = np.array([[10, 4], [10, 4], [20, 8]])
    first = pair_level_bootstrap_ratio(changed, valid, replicates=200, seed=7, batch_size=31)
    second = pair_level_bootstrap_ratio(changed, valid, replicates=200, seed=7, batch_size=31)

    np.testing.assert_allclose(first.estimate, [11 / 40, 7 / 16])
    np.testing.assert_array_equal(first.lower, second.lower)
    np.testing.assert_array_equal(first.upper, second.upper)
    assert np.all(first.lower <= first.estimate)
    assert np.all(first.upper >= first.estimate)


def test_precision_pair_count_decreases_with_wider_target():
    rates = np.array([0.1, 0.2, 0.3, 0.4])
    assert normal_precision_pair_count(rates, 0.01) > normal_precision_pair_count(rates, 0.02)
