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

import numpy as np
import pandas as pd
import pytest
import torch
from neuron_typing_generalization.collection import collect_reduced_activations, collect_synthetic_fixture
from neuron_typing_generalization.execution import (
    classify_run,
    compare_run,
    compute_importance_run,
    evaluate_run,
)
from neuron_typing_generalization.hooks import InjectedModelAdapter, masks_from_spans
from neuron_typing_generalization.interfaces import PreparedInputs


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
    score_manifest = json.loads((tmp_path / "scores/manifest.json").read_text())
    for part in score_manifest["parts"]:
        with np.load(tmp_path / part["file"], allow_pickle=False) as arrays:
            assert np.allclose(arrays["q"].sum(axis=-1), 1)
    summary = json.loads((tmp_path / "metrics/summary.json").read_text())
    assert summary["overall_valid_ordered_comparisons_including_self"] == 2 * 4 * 2**2
    assert summary["overall_valid_distinct_ordered_comparisons"] == 2 * 4 * 2
    with np.load(tmp_path / "distributions/transition_matrices.npz", allow_pickle=False) as transitions:
        assert transitions["symmetrized"].sum() == 2 * transitions["directed"].sum()


def test_compute_importance_rejects_tampered_activation_artifact(tmp_path):
    for directory in ("activations", "scores", "samples"):
        (tmp_path / directory).mkdir()
    pd.DataFrame({"sample_id": ["alpha"], "dataset": ["synthetic"]}).to_parquet(
        tmp_path / "samples/sample_manifest.parquet", index=False
    )
    collect_synthetic_fixture(tmp_path, layers=1, d_ff=2)
    manifest_path = tmp_path / "activations/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parts"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="digest"):
        compute_importance_run(tmp_path)


def test_compact_collection_resumes_completed_sample_parts(tmp_path):
    for directory in ("activations", "samples"):
        (tmp_path / directory).mkdir()
    pd.DataFrame({"sample_id": ["alpha"], "dataset": ["synthetic"]}).to_parquet(
        tmp_path / "samples/sample_manifest.parquet", index=False
    )
    first = collect_synthetic_fixture(tmp_path, layers=1, d_ff=2)
    second = collect_synthetic_fixture(tmp_path, layers=1, d_ff=2)
    assert first["resumed_samples"] == 0
    assert second["resumed_samples"] == 1
    assert second["parts_sha256"] == first["parts_sha256"]


class _VariableMLP(torch.nn.Module):
    def __init__(self, hidden, width):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden, width, bias=False)
        self.down_proj = torch.nn.Linear(width, hidden, bias=False)
        with torch.no_grad():
            self.up_proj.weight.fill_(1)
            self.down_proj.weight.fill_(1)

    def forward(self, values):
        return self.down_proj(torch.relu(self.up_proj(values)))


class _VariableLayer(torch.nn.Module):
    def __init__(self, hidden, width):
        super().__init__()
        self.mlp = _VariableMLP(hidden, width)

    def forward(self, values):
        return self.mlp(values)


class _VariableDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_VariableLayer(2, 2), _VariableLayer(2, 4)])

    def forward(self, input_values):
        for layer in self.layers:
            input_values = layer(input_values)
        return input_values


def test_compact_pipeline_supports_layer_specific_d_ff(tmp_path):
    for directory in ("activations", "scores", "classifications", "samples", "distributions"):
        (tmp_path / directory).mkdir()
    samples = pd.DataFrame({"sample_id": ["alpha", "beta"], "dataset": ["synthetic", "synthetic"]})
    samples.to_parquet(tmp_path / "samples/sample_manifest.parquet", index=False)
    pd.DataFrame({"left_sample_id": ["alpha"], "right_sample_id": ["beta"], "directional": [False]}).to_parquet(
        tmp_path / "samples/pair_manifest.parquet", index=False
    )
    model = _VariableDecoder()

    def prepare(_sample):
        roles = masks_from_spans(torch.ones(2), [(0, 1)], [(1, 2)])
        return PreparedInputs({"input_values": torch.ones(2, 2)}, roles)

    adapter = InjectedModelAdapter("synthetic", model, prepare)
    manifest = collect_reduced_activations(adapter, samples, tmp_path, "run", "synthetic", "synthetic")
    assert manifest["d_ff"] == 4
    assert manifest["d_ff_by_layer"] == [2, 4]
    compute_importance_run(tmp_path)
    classify_run(tmp_path)
    class_manifest = json.loads((tmp_path / "classifications/manifest.json").read_text())
    with np.load(tmp_path / class_manifest["parts"][0]["file"], allow_pickle=False) as arrays:
        assert arrays["classes"].shape == (2, 4)
        assert arrays["classes"][0, 2:].tolist() == [-1, -1]
    compare_run(tmp_path, chunk_size=1)
    summary = json.loads((tmp_path / "distributions/analysis_summary.json").read_text())
    assert summary["overall_valid_ordered_comparisons_including_self"] == (2 + 4) * 2**2
