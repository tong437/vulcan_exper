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
import pytest
from neuron_typing_generalization.within_run_variability import (
    summarize_change_rates,
    theoretical_change_rate_maximum,
)


def test_theoretical_change_rate_maximum_uses_balanced_integer_counts():
    assert theoretical_change_rate_maximum(1000) == pytest.approx(750000 / 999000)
    assert theoretical_change_rate_maximum(5) == pytest.approx(18 / 20)


def test_summarize_change_rates_reports_extrema_quantiles_and_prevalence():
    profiles = pd.DataFrame(
        {
            "layer": [1, 0, 2, 3],
            "neuron": [3, 4, 5, 6],
            "valid_samples": [1000] * 4,
            "invalid_samples": [0] * 4,
            "change_rate_distinct_pairs": [0.5, 0.0, 0.75, 0.25],
        }
    )

    result = summarize_change_rates(profiles)

    assert result["minimum"] == 0.0
    assert result["minimum_nonzero"] == 0.25
    assert result["mean"] == pytest.approx(0.375)
    assert result["median"] == pytest.approx(0.375)
    assert result["maximum"] == 0.75
    assert (result["maximum_layer"], result["maximum_neuron"]) == (2, 5)
    assert result["zero_count"] == 1
    assert result["zero_proportion"] == 0.25
    assert result["ge_0_50_count"] == 2
