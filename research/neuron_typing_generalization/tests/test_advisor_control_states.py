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
import pytest
from neuron_typing_generalization.advisor_control_states import classify_control_shares, validate_state_counts


def test_classify_control_shares_uses_strict_threshold_and_only_splits_weak() -> None:
    threshold = 0.2
    q_visual = np.array([0.3, 0.1, 0.3, threshold, 0.1])
    q_text = np.array([0.1, 0.3, 0.3, threshold, 0.1])
    q_control = np.array([0.4, 0.4, 0.4, 0.3, 0.1])
    assert classify_control_shares(q_visual, q_text, q_control).tolist() == [0, 1, 2, 3, 4]


def test_validate_state_counts_rejects_incomplete_real_neurons() -> None:
    counts = np.zeros((5, 1, 2), dtype=np.int32)
    counts[0, 0, 0] = 2
    counts[1, 0, 1] = 1
    with pytest.raises(ValueError, match="do not sum"):
        validate_state_counts(counts, completed_samples=2, d_ff_by_layer=[2])
