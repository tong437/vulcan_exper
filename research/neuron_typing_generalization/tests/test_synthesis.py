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

from pathlib import Path

import pandas as pd
import pytest
from neuron_typing_generalization.synthesis import normalized_depth_curves, paired_model_differences


def _write_pair_inputs(run_dir: Path, changed: list[int]) -> None:
    (run_dir / "distributions").mkdir(parents=True)
    (run_dir / "samples").mkdir()
    edges = [("a", "b", "g1"), ("a", "c", "g1"), ("d", "e", "g2")]
    pd.DataFrame(
        [
            {
                "run_id": "run",
                "left_sample_id": left,
                "right_sample_id": right,
                "pair_scope": "controlled",
                "layer": None,
                "changed_neurons": value,
                "valid_neurons": 10,
                "change_rate": value / 10,
            }
            for (left, right, _), value in zip(edges, changed)
        ]
    ).to_parquet(run_dir / "distributions/pairwise_changes.parquet", index=False)
    pd.DataFrame(
        [{"left_sample_id": left, "right_sample_id": right, "pair_group_id": group} for left, right, group in edges]
    ).to_parquet(run_dir / "samples/pair_manifest.parquet", index=False)


def test_paired_model_difference_clusters_shared_edges(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    _write_pair_inputs(left, [4, 4, 2])
    _write_pair_inputs(right, [2, 2, 2])

    result = paired_model_differences({"left": left, "right": right}, [("left", "right")], replicates=200, seed=3)[0]

    assert result["bootstrap_unit_count"] == 2
    assert result["estimate"] == pytest.approx(4 / 30)


def test_normalized_depth_curves_use_controlled_when_available_and_global_otherwise():
    controlled = {
        "statistics": {
            "controlled": {"status": "complete", "per_layer": [{"estimate": 0.1}, {"estimate": 0.3}]},
            "global": {"per_layer_change_rate_distinct_pairs": [0.5, 0.6]},
        }
    }
    descriptive = {
        "statistics": {
            "controlled": {"status": "insufficient_pairs", "per_layer": []},
            "global": {"per_layer_change_rate_distinct_pairs": [0.2, 0.4, 0.6]},
        }
    }

    rows = normalized_depth_curves({"controlled": {"m": controlled}, "natural": {"m": descriptive}}, grid_size=3)

    controlled_rows = [row for row in rows if row["dataset"] == "controlled"]
    natural_rows = [row for row in rows if row["dataset"] == "natural"]
    assert [row["change_rate"] for row in controlled_rows] == pytest.approx([0.1, 0.2, 0.3])
    assert [row["change_rate"] for row in natural_rows] == pytest.approx([0.2, 0.4, 0.6])
    assert {row["estimand"] for row in natural_rows} == {"global_distinct_pair_change_rate"}
