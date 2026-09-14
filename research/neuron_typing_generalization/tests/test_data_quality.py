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
from neuron_typing_generalization.data_quality import pairing_violations


def test_vqav2_pair_policy_detects_same_image_and_question_mismatch():
    samples = pd.DataFrame(
        [
            {"sample_id": "a", "image_id": "1", "question": "What color?", "pair_group_id": "g"},
            {"sample_id": "b", "image_id": "1", "question": "How many?", "pair_group_id": "g"},
        ]
    )
    pairs = pd.DataFrame(
        [{"left_sample_id": "a", "right_sample_id": "b", "pair_group_id": "g", "audit_status": "passed"}]
    )
    rules = {item["rule"] for item in pairing_violations("vqav2", samples, pairs)}
    assert rules == {"same_normalized_question", "different_image"}
