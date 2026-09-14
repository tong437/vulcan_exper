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

import json

import pandas as pd
import pytest
from neuron_typing_generalization.collection import collect_synthetic_fixture
from neuron_typing_generalization.execution import (
    classify_run,
    compare_run,
    compute_importance_run,
    evaluate_run,
)


def test_manifest_to_hook_to_importance_to_analysis_cpu_fixture(tmp_path):
    for directory in ("activations", "scores", "classifications", "samples", "distributions", "metrics", "figures"):
        (tmp_path / directory).mkdir()
    samples = pd.DataFrame({"sample_id": ["alpha", "beta"], "dataset": ["synthetic", "synthetic"]})
    samples.to_parquet(tmp_path / "samples/sample_manifest.parquet", index=False)
    pd.DataFrame({"left_sample_id": ["alpha"], "right_sample_id": ["beta"], "directional": [False]}).to_parquet(
        tmp_path / "samples/pair_manifest.parquet", index=False
    )

    activation_manifest = collect_synthetic_fixture(tmp_path, layers=2, d_ff=4)
    importance_result = compute_importance_run(tmp_path)
    classify_run(tmp_path)
    compare_run(tmp_path, chunk_size=1)
    evaluate_run(tmp_path)

    assert activation_manifest["rows"] == 2 * 2 * 4 * 2
    assert activation_manifest["raw_token_activations_retained"] is False
    assert importance_result["score_rows"] == activation_manifest["rows"]
    scores = pd.read_parquet(tmp_path / "scores/neuron_scores.parquet")
    assert scores.groupby(["sample_id", "layer", "modality"])["q"].sum().eq(1).all()
    summary = json.loads((tmp_path / "metrics/summary.json").read_text())
    assert summary["overall_valid_ordered_comparisons_including_self"] == 2 * 4 * 2**2
    assert summary["overall_valid_distinct_ordered_comparisons"] == 2 * 4 * 2


def test_compute_importance_rejects_tampered_activation_artifact(tmp_path):
    for directory in ("activations", "scores", "samples"):
        (tmp_path / directory).mkdir()
    pd.DataFrame({"sample_id": ["alpha"], "dataset": ["synthetic"]}).to_parquet(
        tmp_path / "samples/sample_manifest.parquet", index=False
    )
    collect_synthetic_fixture(tmp_path, layers=1, d_ff=2)
    manifest_path = tmp_path / "activations/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["statistics_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="digest"):
        compute_importance_run(tmp_path)
