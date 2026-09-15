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
from neuron_typing_generalization.reporting import summarize_controlled_pairs


def test_controlled_pair_bootstrap_uses_pair_edges_and_ratio_of_sums():
    rows = []
    for left, right, changed, valid in (("a", "b", 1, 10), ("c", "d", 8, 20)):
        for layer, scale in ((None, 1), (0, 1), (1, 2)):
            rows.append(
                {
                    "run_id": "run",
                    "left_sample_id": left,
                    "right_sample_id": right,
                    "pair_scope": "controlled",
                    "layer": layer,
                    "changed_neurons": changed * scale,
                    "valid_neurons": valid * scale,
                    "change_rate": changed / valid,
                }
            )
    result = summarize_controlled_pairs(pd.DataFrame(rows), bootstrap_replicates=200, bootstrap_seed=9)

    assert result["status"] == "complete"
    assert result["pair_count"] == 2
    assert result["overall"]["estimate"] == pytest.approx(9 / 30)
    assert len(result["per_layer"]) == 2
    assert result["precision_scenarios_normal_approximation"]["half_width_0.010"]["estimated_required_pairs"] > 2


def test_bootstrap_clusters_reused_pair_edges_by_manifest_group():
    changes = pd.DataFrame(
        {
            "run_id": ["run"] * 4,
            "left_sample_id": ["a", "a", "d", "d"],
            "right_sample_id": ["b", "c", "e", "f"],
            "pair_scope": ["controlled"] * 4,
            "layer": [None] * 4,
            "changed_neurons": [1, 2, 7, 8],
            "valid_neurons": [10] * 4,
            "change_rate": [0.1, 0.2, 0.7, 0.8],
        }
    )
    manifest = changes[["left_sample_id", "right_sample_id"]].copy()
    manifest["pair_group_id"] = ["image-1", "image-1", "image-2", "image-2"]
    result = summarize_controlled_pairs(changes, manifest, bootstrap_replicates=100, bootstrap_seed=3)

    assert result["pair_count"] == 4
    assert result["bootstrap_unit"] == "pair_group_id"
    assert result["bootstrap_unit_count"] == 2
    assert result["overall"]["estimate"] == pytest.approx(18 / 40)
