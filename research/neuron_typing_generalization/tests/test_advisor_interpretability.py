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
from neuron_typing_generalization.advisor_interpretability import (
    EFFECTIVE_BINS,
    smooth_histogram,
    summarize_effective_categories,
)


def _profile() -> pd.DataFrame:
    effective = np.array([1.0, 2.0, 3.0, 4.0])
    entropy = np.log(effective)
    return pd.DataFrame(
        {
            "valid_samples": [1000] * 4,
            "entropy": entropy,
            "normalized_entropy": entropy / np.log(4.0),
            "effective_class_count": effective,
        }
    )


def test_effective_category_summary_respects_hill_number_identity() -> None:
    summary, histogram = summarize_effective_categories(_profile())
    assert summary["neuron_count"] == 4
    assert summary["effective_category_median"] == pytest.approx(2.5)
    assert summary["normalized_entropy_median"] == pytest.approx(np.log(np.sqrt(6.0)) / np.log(4.0))
    assert histogram.sum() == pytest.approx(100.0)
    assert summary["near_1_proportion"] == pytest.approx(0.25)
    assert summary["near_2_proportion"] == pytest.approx(0.25)
    assert summary["near_3_proportion"] == pytest.approx(0.25)
    assert summary["near_4_proportion"] == pytest.approx(0.25)


def test_effective_category_summary_rejects_inconsistent_formula() -> None:
    profile = _profile()
    profile.loc[1, "effective_class_count"] = np.exp(profile.loc[1, "normalized_entropy"])
    with pytest.raises(ValueError, match="inconsistent with exp"):
        summarize_effective_categories(profile)


def test_histogram_smoothing_has_expected_shape_and_constant_invariance() -> None:
    values = np.full(len(EFFECTIVE_BINS) - 1, 2.0)
    smoothed = smooth_histogram(values)
    assert smoothed.shape == values.shape
    assert np.allclose(smoothed, values)
