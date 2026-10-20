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
from neuron_typing_generalization.advisor_state_pairs import (
    PAIR_COLUMNS,
    compute_state_pair_contributions,
    summarize_state_pair_contributions,
)


def _profile(rows: list[tuple[int, int, int, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["count_V", "count_T", "count_M", "count_W"])
    frame["valid_samples"] = frame[["count_V", "count_T", "count_M", "count_W"]].sum(axis=1)
    counts = frame[["count_V", "count_T", "count_M", "count_W"]].to_numpy(dtype=np.float64)
    n = frame["valid_samples"].to_numpy(dtype=np.float64)
    frame["change_rate_distinct_pairs"] = (np.square(n) - np.square(counts).sum(axis=1)) / (n * (n - 1))
    return frame


def test_two_state_pair_reproduces_change_rate() -> None:
    values = compute_state_pair_contributions(_profile([(5, 5, 0, 0)]))
    assert values.loc[0, "R_VT"] == pytest.approx(50 / 90)
    assert values.loc[0, list(PAIR_COLUMNS)[1:]].sum() == pytest.approx(0.0)
    assert values.loc[0, "R_sum"] == pytest.approx(50 / 90)
    assert values.loc[0, "identity_residual"] == pytest.approx(0.0)


def test_four_state_pair_identity_holds() -> None:
    values = compute_state_pair_contributions(_profile([(2, 3, 4, 5)]))
    assert values.loc[0, "R_sum"] == pytest.approx(_profile([(2, 3, 4, 5)]).loc[0, "change_rate_distinct_pairs"])


def test_summary_normalizes_only_after_averaging_absolute_contributions() -> None:
    summary = summarize_state_pair_contributions(_profile([(5, 5, 0, 0), (9, 0, 1, 0)]))
    expected_vt = (50 / 90) / 2
    expected_vm = (18 / 90) / 2
    assert summary["absolute_contributions"]["VT"] == pytest.approx(expected_vt)
    assert summary["absolute_contributions"]["VM"] == pytest.approx(expected_vm)
    assert summary["composition_shares"]["VT"] == pytest.approx(expected_vt / (expected_vt + expected_vm))
    assert sum(summary["composition_shares"].values()) == pytest.approx(1.0)


def test_neuron_with_fewer_than_two_samples_is_undefined() -> None:
    frame = pd.DataFrame(
        {
            "valid_samples": [1],
            "count_V": [1],
            "count_T": [0],
            "count_M": [0],
            "count_W": [0],
            "change_rate_distinct_pairs": [np.nan],
        }
    )
    values = compute_state_pair_contributions(frame)
    assert values[list(PAIR_COLUMNS)].isna().all(axis=None)


def test_counts_must_sum_to_valid_samples() -> None:
    frame = _profile([(5, 5, 0, 0)])
    frame.loc[0, "valid_samples"] = 11
    with pytest.raises(ValueError, match="sum to valid_samples"):
        compute_state_pair_contributions(frame)
