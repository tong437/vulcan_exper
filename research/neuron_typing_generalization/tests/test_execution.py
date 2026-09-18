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

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from neuron_typing_generalization.execution import classify_run, compare_run, evaluate_run


def test_cpu_postprocessing_stages_write_contract_outputs(tmp_path: Path):
    for directory in ("scores", "classifications", "samples", "distributions", "metrics", "figures"):
        (tmp_path / directory).mkdir()
    rows = []
    importance = {
        "a": {"visual": [2.0, 1.0], "text": [3.0, 1.0]},
        "b": {"visual": [1.0, 2.0], "text": [3.0, 1.0]},
    }
    for sample_id, modalities in importance.items():
        for modality, values in modalities.items():
            for neuron, value in enumerate(values):
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
    pd.DataFrame(rows).to_parquet(tmp_path / "scores/neuron_scores.parquet", index=False)
    pd.DataFrame({"sample_id": ["a", "b"]}).to_parquet(tmp_path / "samples/sample_manifest.parquet", index=False)
    pd.DataFrame({"left_sample_id": ["a"], "right_sample_id": ["b"], "directional": [False]}).to_parquet(
        tmp_path / "samples/pair_manifest.parquet", index=False
    )

    classify_run(tmp_path)
    compare_run(tmp_path, chunk_size=1)
    evaluate_run(tmp_path)

    assert (tmp_path / "classifications/neuron_classes.parquet").is_file()
    assert (tmp_path / "distributions/pairwise_changes.parquet").is_file()
    assert (tmp_path / "distributions/neuron_profiles.parquet").is_file()
    profiles = pd.read_parquet(tmp_path / "distributions/neuron_profiles.parquet")
    assert profiles["change_rate_distinct_pairs"].tolist() == [1.0, 1.0]
    assert profiles["q_visual_mean"].tolist() == pytest.approx([0.5, 0.5])
    assert profiles["q_text_mean"].tolist() == pytest.approx([0.75, 0.25])
    transition_path = tmp_path / "distributions/transition_matrices.npz"
    assert transition_path.is_file()
    with np.load(transition_path, allow_pickle=False) as transitions:
        assert transitions["directed"].sum() == 2
        assert transitions["symmetrized"].sum() == 4
        assert transitions["class_labels"].tolist() == ["W", "V", "T", "M"]
    assert (tmp_path / "distributions/transition_manifest.json").is_file()
    summary = json.loads((tmp_path / "metrics/summary.json").read_text())
    assert summary["overall_change_rate_distinct_pairs"] == 1.0
    assert summary["overall_changed_ordered_comparisons"] == 4
    assert summary["overall_valid_ordered_comparisons_including_self"] == 8
    figures = json.loads((tmp_path / "figures/manifest.json").read_text())
    assert figures["status"] == "complete"
    assert len(figures["figures"]) >= 3
    assert (tmp_path / "metrics/pilot_statistics.json").is_file()
    assert (tmp_path / "metrics/reproducibility.json").is_file()
