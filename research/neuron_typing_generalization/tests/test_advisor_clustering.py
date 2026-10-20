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

import numpy as np
import pandas as pd
import pytest
from neuron_typing_generalization.advisor_clustering import (
    PROBABILITY_COLUMNS,
    average_linkage_clusters,
    condensed_distance_bytes,
    condensed_hellinger,
    hellinger_distance,
    stratified_neuron_sample,
)


def test_hellinger_distance_endpoints_and_symmetry() -> None:
    visual = np.array([1.0, 0.0, 0.0, 0.0])
    text = np.array([0.0, 1.0, 0.0, 0.0])
    assert hellinger_distance(visual, visual) == pytest.approx(0.0)
    assert hellinger_distance(visual, text) == pytest.approx(1.0)
    assert hellinger_distance(visual, text) == pytest.approx(hellinger_distance(text, visual))
    assert condensed_hellinger(np.stack([visual, text])).tolist() == pytest.approx([1.0])


def test_average_linkage_separates_two_probability_families() -> None:
    probabilities = np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.98, 0.02, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.02, 0.98, 0.00, 0.00],
        ]
    )
    labels, linkage_matrix, clusters = average_linkage_clusters(probabilities, threshold=0.20)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert linkage_matrix.shape == (3, 4)
    assert [row["neuron_count"] for row in clusters] == [2, 2]


def test_layer_stratified_sample_is_deterministic_and_proportional() -> None:
    rows = []
    for layer, count in ((0, 10), (1, 30)):
        for neuron in range(count):
            rows.append(
                {"layer": layer, "neuron": neuron, **dict(zip(PROBABILITY_COLUMNS, [1.0, 0.0, 0.0, 0.0]))}
            )
    profile = pd.DataFrame(rows)
    first = stratified_neuron_sample(profile, maximum_neurons=20, seed=7)
    second = stratified_neuron_sample(profile, maximum_neurons=20, seed=7)
    assert first[["layer", "neuron"]].equals(second[["layer", "neuron"]])
    assert first.groupby("layer").size().to_dict() == {0: 5, 1: 15}


def test_condensed_distance_storage_formula() -> None:
    assert condensed_distance_bytes(4096) == 4096 * 4095 // 2 * 8
