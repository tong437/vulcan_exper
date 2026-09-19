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

import pandas as pd
from neuron_typing_generalization.representative_neurons import select_representative_neurons


def test_select_representative_neurons_uses_independent_deterministic_criteria():
    profiles = pd.DataFrame(
        [
            {
                "layer": 1,
                "neuron": 4,
                "valid_samples": 1000,
                "invalid_samples": 0,
                "change_rate_distinct_pairs": 0.8,
                "normalized_entropy": 0.9,
                "dominant_class": "W",
                "dominant_proportion": 0.4,
                "visual_margin_mean": 0.1,
                "text_margin_mean": 0.2,
                "count_W": 400,
                "count_V": 200,
                "count_T": 200,
                "count_M": 200,
            },
            {
                "layer": 2,
                "neuron": 5,
                "valid_samples": 1000,
                "invalid_samples": 0,
                "change_rate_distinct_pairs": 0.7,
                "normalized_entropy": 1.0,
                "dominant_class": "V",
                "dominant_proportion": 0.25,
                "visual_margin_mean": 0.2,
                "text_margin_mean": -0.1,
                "count_W": 250,
                "count_V": 250,
                "count_T": 250,
                "count_M": 250,
            },
            {
                "layer": 0,
                "neuron": 3,
                "valid_samples": 1000,
                "invalid_samples": 0,
                "change_rate_distinct_pairs": 0.0,
                "normalized_entropy": 0.0,
                "dominant_class": "T",
                "dominant_proportion": 1.0,
                "visual_margin_mean": -0.3,
                "text_margin_mean": 0.4,
                "count_W": 0,
                "count_V": 0,
                "count_T": 1000,
                "count_M": 0,
            },
            {
                "layer": 0,
                "neuron": 0,
                "valid_samples": 999,
                "invalid_samples": 1,
                "change_rate_distinct_pairs": 0.99,
                "normalized_entropy": 0.99,
                "dominant_class": "M",
                "dominant_proportion": 0.3,
                "visual_margin_mean": 0.1,
                "text_margin_mean": 0.1,
                "count_W": 233,
                "count_V": 233,
                "count_T": 233,
                "count_M": 300,
            },
        ]
    )

    selected = select_representative_neurons(profiles).set_index("criterion")

    assert (selected.loc["high_change", ["layer", "neuron"]].tolist()) == [1, 4]
    assert (selected.loc["high_entropy", ["layer", "neuron"]].tolist()) == [2, 5]
    assert (selected.loc["stable", ["layer", "neuron"]].tolist()) == [0, 3]
    assert selected["candidate_count"].tolist() == [3, 3, 3]
