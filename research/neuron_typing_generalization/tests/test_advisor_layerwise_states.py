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
from neuron_typing_generalization.advisor_layerwise_states import (
    SHARE_COLUMNS,
    layerwise_state_distribution,
)


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "layer": [0, 0, 1, 1],
            "neuron": [0, 1, 0, 1],
            "d_ff": [2, 2, 2, 2],
            "valid_samples": [10, 10, 10, 10],
            "count_V": [10, 0, 2, 4],
            "count_T": [0, 4, 2, 2],
            "count_M": [0, 2, 2, 2],
            "count_W": [0, 4, 4, 2],
        }
    )


def test_layerwise_distribution_averages_neuron_frequencies() -> None:
    result = layerwise_state_distribution(_profile())
    assert result["layer"].tolist() == [0, 1]
    assert result["neuron_count"].tolist() == [2, 2]
    assert result["sample_count"].tolist() == [10, 10]
    assert result.loc[0, "share_V"] == pytest.approx(0.5)
    assert result.loc[0, "share_T"] == pytest.approx(0.2)
    assert result.loc[0, "share_M"] == pytest.approx(0.1)
    assert result.loc[0, "share_W"] == pytest.approx(0.2)
    assert result.loc[1, list(SHARE_COLUMNS)].sum() == pytest.approx(1.0)
    assert result["share_sum_residual"].abs().max() == pytest.approx(0.0)


def test_layerwise_distribution_supports_variable_real_layer_widths() -> None:
    profile = _profile().drop(columns="d_ff")
    extra = profile.iloc[[2]].copy()
    extra["neuron"] = 2
    profile = pd.concat([profile, extra], ignore_index=True)
    result = layerwise_state_distribution(profile)
    assert result["neuron_count"].tolist() == [2, 3]


def test_layerwise_distribution_rejects_different_sample_counts() -> None:
    profile = _profile()
    profile.loc[0, "valid_samples"] = 11
    profile.loc[0, "count_V"] = 11
    with pytest.raises(ValueError, match="same positive"):
        layerwise_state_distribution(profile)


def test_layerwise_distribution_rejects_padding_or_missing_neurons() -> None:
    profile = _profile().iloc[:-1].copy()
    with pytest.raises(ValueError, match="exactly d_ff real neurons"):
        layerwise_state_distribution(profile)


def test_layerwise_distribution_rejects_inconsistent_counts() -> None:
    profile = _profile()
    profile.loc[0, "count_W"] = 1
    with pytest.raises(ValueError, match="sum to valid_samples"):
        layerwise_state_distribution(profile)
