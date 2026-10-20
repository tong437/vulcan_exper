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
from neuron_typing_generalization.control_ablation_runner import (
    build_control_ablation_conditions,
    select_control_indices,
)


def test_control_conditions_expand_to_35_runs() -> None:
    conditions = build_control_ablation_conditions(10)
    assert len(conditions) == 35
    assert sum(item.condition == "C_random" for item in conditions) == 10


def test_control_selection_deletes_all_when_pool_is_short() -> None:
    q = np.full((1, 3, 100), 0.01, dtype=np.float32)
    classes = np.full((1, 100), 4, dtype=np.int8)
    classes[0, :2] = 3
    selected, shortfall = select_control_indices(q, classes, (100,), fraction=0.05, state="C", method="max", seed=42)
    assert selected[0].tolist() == [0, 1]
    assert shortfall == 1
