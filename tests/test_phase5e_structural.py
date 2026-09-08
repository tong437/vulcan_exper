# Copyright 2026 the LlamaFactory team.
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
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from build_phase5e_structural_artifact import build_artifact  # noqa: E402
from phase3_structural_utils import canonical_json_sha256  # noqa: E402


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_phase5e_artifact_from_static_frontier(tmp_path):
    mask = {"0": [1], "1": [0, 2]}
    mask_path = tmp_path / "masks" / "candidate.json"
    _write(mask_path, mask)
    row = {
        "deletion_budget": 3,
        "mask_file": str(mask_path),
        "mask_hash": canonical_json_sha256(mask),
        "automatic_semantic_pass": True,
        "parameter_summary": {"removed_parameters": 90},
        "gold_proxy": {"mean_nll": 1.0},
        "generation": {
            "final_caption": "Several bicycles are inside a train car.",
            "semantic": {"contract_version": "phase5e_bicycles_in_train_v2"},
        },
        "human_confirmation": None,
    }
    frontier = {
        "complete": True,
        "config": {
            "trace_target": "gold_caption",
            "config_path": str(tmp_path / "config.yaml"),
            "model_name_or_path": "model",
            "reference_caption": "A group of bicycles on a subway train.",
        },
        "layer_widths": {"0": 4, "1": 4},
        "conditions": {"candidate": row},
        "best_semantic_pass": {"taylor": {"condition": "candidate", "deletion_budget": 3}},
    }
    (tmp_path / "config.yaml").write_text("model_name_or_path: model\n", encoding="utf-8")
    frontier_path = tmp_path / "static_frontier.json"
    _write(frontier_path, frontier)
    neuron_ids, clusters, metadata = build_artifact(frontier_path)
    assert neuron_ids == mask
    assert metadata["phase"] == "5E-C"
    assert metadata["deletion_budget"] == 3
    assert metadata["target_layer_dims"] == {0: 3, 1: 2}
    assert len(clusters[0]) == 3
    assert len(clusters[1]) == 2


def test_build_phase5e_artifact_rejects_stale_semantic_contract(tmp_path):
    mask = {"0": [1]}
    mask_path = tmp_path / "mask.json"
    _write(mask_path, mask)
    frontier = {
        "complete": True,
        "config": {
            "trace_target": "gold_caption",
            "config_path": str(tmp_path / "config.yaml"),
            "model_name_or_path": "model",
            "reference_caption": "A group of bicycles on a subway train.",
        },
        "layer_widths": {"0": 4},
        "conditions": {
            "candidate": {
                "deletion_budget": 1,
                "mask_file": str(mask_path),
                "mask_hash": canonical_json_sha256(mask),
                "automatic_semantic_pass": True,
                "parameter_summary": {"removed_parameters": 30},
                "gold_proxy": {"mean_nll": 1.0},
                "generation": {"semantic": {"contract_version": "phase5e_bicycles_in_train_v1"}},
            }
        },
        "best_semantic_pass": {"taylor": {"condition": "candidate", "deletion_budget": 1}},
    }
    (tmp_path / "config.yaml").write_text("model_name_or_path: model\n", encoding="utf-8")
    frontier_path = tmp_path / "static_frontier.json"
    _write(frontier_path, frontier)

    with pytest.raises(ValueError, match="stale semantic contract"):
        build_artifact(frontier_path)
