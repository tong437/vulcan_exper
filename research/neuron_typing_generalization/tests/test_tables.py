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

import numpy as np
import pandas as pd
import pytest
from neuron_typing_generalization.tables import (
    analyze_class_table,
    analyze_class_tensor,
    classify_score_table,
    cohort_summary_tensor,
)


def _score_rows() -> pd.DataFrame:
    rows = []
    values = {
        "a": {"visual": [2.0, 1.0], "text": [3.0, 1.0]},
        "b": {"visual": [1.0, 2.0], "text": [3.0, 1.0]},
    }
    for sample_id, modalities in values.items():
        for modality, importance in modalities.items():
            for neuron, value in enumerate(importance):
                rows.append(
                    {
                        "run_id": "run",
                        "sample_id": sample_id,
                        "model": "fake",
                        "dataset": "fake",
                        "layer": 0,
                        "neuron": neuron,
                        "modality": modality,
                        "token_count": 1,
                        "activation_rms": value,
                        "down_projection_norm": 1.0,
                        "importance": value,
                        "valid": True,
                    }
                )
    return pd.DataFrame(rows)


def test_score_to_class_to_analysis_cpu_end_to_end():
    normalized, classes = classify_score_table(_score_rows())
    assert normalized.groupby(["sample_id", "layer", "modality"])["q"].sum().eq(1).all()
    assert classes["class"].tolist() == ["M", "W", "T", "V"]

    samples = pd.DataFrame({"sample_id": ["a", "b"]})
    pairs = pd.DataFrame(
        {
            "left_sample_id": ["a"],
            "right_sample_id": ["b"],
            "directional": [False],
        }
    )
    summary, profiles, transitions = analyze_class_table(classes, samples, pairs, chunk_size=1)
    assert summary["overall_change_rate_including_self"] == 0.5
    assert summary["overall_change_rate_distinct_pairs"] == 1.0
    assert summary["maximum_pair"]["change_rate"] == 1.0
    assert profiles["profile"].tolist() == ["dynamic_t_m", "dynamic_with_weak"]
    assert profiles["change_rate_distinct_pairs"].tolist() == [1.0, 1.0]
    assert profiles["valid_distinct_ordered_pairs"].tolist() == [2, 2]
    assert profiles["dominant_proportion"].tolist() == [0.5, 0.5]
    assert transitions["directed"].sum() == 2
    assert transitions["symmetrized"].sum() == 4


def test_unpaired_profile_analysis_can_skip_quadratic_maximum_pair_search():
    tensor = np.array([[[0, 1]], [[1, 1]], [[1, 2]]], dtype=np.int8)

    summary, profiles, transitions = analyze_class_tensor(
        tensor,
        ["a", "b", "c"],
        "run",
        pd.DataFrame(),
        compute_maximum_pair=False,
    )

    assert summary["maximum_pair"] is None
    assert profiles["change_rate_distinct_pairs"].tolist() == pytest.approx([2 / 3, 2 / 3])
    assert transitions["directed"].sum() == 0


def test_real_neuron_with_no_valid_observations_is_retained():
    tensor = np.array([[[0, -1]], [[1, -1]]], dtype=np.int8)

    _, profiles, _ = analyze_class_tensor(
        tensor,
        ["a", "b"],
        "run",
        pd.DataFrame(),
        d_ff_by_layer=[2],
        compute_maximum_pair=False,
    )

    invalid = profiles.loc[profiles["neuron"] == 1].iloc[0]
    assert len(profiles) == 2
    assert invalid["valid_samples"] == 0
    assert invalid["invalid_samples"] == 2
    assert invalid["dominant_class"] == "INVALID"
    assert invalid["profile"] == "invalid"
    assert np.isnan(invalid["change_rate_distinct_pairs"])


def test_natural_cohort_summary_is_descriptive_and_layer_explicit():
    tensor = np.array([[[0, 1]], [[2, 3]], [[0, 3]]], dtype=np.int8)
    samples = pd.DataFrame({"sample_id": ["a", "b", "c"], "stratum": ["answerable", "unanswerable", "answerable"]})
    result = cohort_summary_tensor(tensor, samples, "run")

    assert set(result["cohort"]) == {"answerable", "unanswerable"}
    overall = result[(result["cohort"] == "answerable") & result["layer"].isna()].iloc[0]
    assert overall["sample_count"] == 2
    assert overall["valid_observations"] == 4
    assert overall["count_W"] == 2
    assert overall["count_V"] == 1
    assert overall["count_M"] == 1
