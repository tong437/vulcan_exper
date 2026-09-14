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
from neuron_typing_generalization.schema_validation import validate_frame


def test_class_schema_rejects_semantically_inconsistent_strong_bits():
    frame = pd.DataFrame(
        [
            {
                "run_id": "run",
                "sample_id": "sample",
                "model": "model",
                "dataset": "dataset",
                "layer": 0,
                "neuron": 0,
                "q_visual": 0.8,
                "q_text": 0.2,
                "visual_strong": True,
                "text_strong": False,
                "class": "T",
                "valid": True,
            }
        ]
    )
    with pytest.raises(ValueError, match="disagrees"):
        validate_frame(frame, "neuron_classes.schema.json")


def test_pairwise_schema_rejects_rate_that_disagrees_with_counts():
    frame = pd.DataFrame(
        [
            {
                "run_id": "run",
                "left_sample_id": "a",
                "right_sample_id": "b",
                "pair_scope": "controlled",
                "layer": None,
                "changed_neurons": 1,
                "valid_neurons": 2,
                "change_rate": 0.25,
            }
        ]
    )
    with pytest.raises(ValueError, match="disagrees"):
        validate_frame(frame, "pairwise_changes.schema.json")
