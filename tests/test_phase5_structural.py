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
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from build_phase5_atomic_structural_artifact import build_atomic_artifact  # noqa: E402
from build_phase5_core_extension_artifact import build_core_extension_artifact  # noqa: E402
from build_phase5_nested_probe import extend_deletion_masks  # noqa: E402
from build_phase5_structural_artifact import build_artifact  # noqa: E402
from phase3_structural_utils import canonical_json_sha256, sha256_file  # noqa: E402
from phase5_structural_utils import (  # noqa: E402
    build_partial_singleton_cluster_idx,
    compare_logits,
    target_layer_dims,
    validate_partial_singleton_cluster_idx,
    validate_phase5_deletion_masks,
)
from run_phase2_ablation import MLPNeuronAblator  # noqa: E402

from llamafactory.train.vulcan.pruning import pruning_mlp  # noqa: E402


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(3, 4, bias=False)
        self.gate_proj = torch.nn.Linear(3, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 3, bias=False)

    def forward(self, inputs):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs))


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()

    def forward(self, inputs):
        return inputs + self.mlp(inputs)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def test_partial_singleton_clusters_preserve_unchanged_layers():
    masks = {
        0: torch.tensor([False, False, False, False]),
        1: torch.tensor([False, True, False, True]),
    }
    clusters = build_partial_singleton_cluster_idx(masks)
    assert clusters[0] is None
    assert clusters[1] == [{"anchor": 0, "neuron": [0]}, {"anchor": 2, "neuron": [2]}]
    assert validate_partial_singleton_cluster_idx(clusters, masks)["validated"]
    assert target_layer_dims(masks) == {0: 4, 1: 2}


def test_partial_singleton_structural_pruning_matches_hook_ablation():
    torch.manual_seed(17)
    hook_model = TinyModel().eval().requires_grad_(False)
    structural_model = TinyModel().eval().requires_grad_(False)
    structural_model.load_state_dict(hook_model.state_dict())
    masks = {
        0: torch.tensor([False, False, False, False]),
        1: torch.tensor([False, True, False, True]),
    }
    inputs = torch.randn(2, 3, 3)
    with MLPNeuronAblator(hook_model, masks):
        hook_output = hook_model(inputs)
    pruning_mlp(structural_model, build_partial_singleton_cluster_idx(masks))
    assert torch.allclose(hook_output, structural_model(inputs), atol=1e-6, rtol=1e-6)


def test_phase5_mask_validation_requires_a_real_but_not_complete_deletion():
    with pytest.raises(ValueError, match="does not prune"):
        validate_phase5_deletion_masks({0: torch.zeros(3, dtype=torch.bool)})
    with pytest.raises(ValueError, match="keep at least one"):
        validate_phase5_deletion_masks({0: torch.ones(3, dtype=torch.bool)})


def test_logit_comparison_checks_shape_and_reports_exactness():
    values = torch.tensor([[1.0, 2.0]])
    assert compare_logits(values, values)["max_abs"] == 0.0
    with pytest.raises(ValueError, match="shapes differ"):
        compare_logits(values, torch.ones(3))


def test_phase5_artifact_builder_freezes_best_feasible_run(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name_or_path: model\n", encoding="utf-8")
    static_path = tmp_path / "static.json"
    static = {
        "layer_widths": {"0": 4, "1": 4},
        "config": {"model_name_or_path": "model", "config_path": str(config_path)},
    }
    static_path.write_text(json.dumps(static), encoding="utf-8")
    mask = {"0": [], "1": [1, 3]}
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps(mask), encoding="utf-8")
    run = {
        "feasible": True,
        "deletion_budget": 2,
        "mask_file": str(mask_path),
        "mask_hash": canonical_json_sha256(mask),
        "parameter_summary": {"removed_parameters": 18},
        "teacher_fidelity": {"mean_kl": 0.0},
        "generation": {"exact_match": True},
    }
    learned = {
        "complete": True,
        "config": {"static_frontier": str(static_path), "kl_tolerance": 0.001},
        "best_feasible": {"run": "best"},
        "runs": {"best": run},
    }
    learned_path = tmp_path / "learned.json"
    learned_path.write_text(json.dumps(learned), encoding="utf-8")

    neuron_ids, clusters, metadata = build_artifact(learned_path)
    assert neuron_ids == mask
    assert clusters[0] is None
    assert metadata["deletion_budget"] == 2
    assert metadata["target_layer_dims"] == {0: 4, 1: 2}
    assert metadata["theoretical_reduction"]["removed_parameters"] == 18


def test_phase5_artifact_builder_requires_opt_in_for_behavioral_candidate(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name_or_path: model\n", encoding="utf-8")
    static_path = tmp_path / "static.json"
    static_path.write_text(
        json.dumps(
            {
                "layer_widths": {"0": 4},
                "config": {"model_name_or_path": "model", "config_path": str(config_path)},
            }
        ),
        encoding="utf-8",
    )
    mask = {"0": [1]}
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps(mask), encoding="utf-8")
    learned = {
        "complete": True,
        "config": {"static_frontier": str(static_path), "kl_tolerance": 0.001},
        "best_feasible": None,
        "runs": {
            "behavioral": {
                "feasible": False,
                "deletion_budget": 1,
                "mask_file": str(mask_path),
                "mask_hash": canonical_json_sha256(mask),
                "parameter_summary": {"removed_parameters": 9},
                "teacher_fidelity": {"mean_kl": 0.0005},
                "generation": {"exact_match": True},
            }
        },
    }
    learned_path = tmp_path / "learned.json"
    learned_path.write_text(json.dumps(learned), encoding="utf-8")
    with pytest.raises(ValueError, match="strict-feasible"):
        build_artifact(learned_path, "behavioral")
    _, _, metadata = build_artifact(learned_path, "behavioral", allow_behavioral=True)
    assert metadata["source_gate"] == "behavioral"


def test_phase5_artifact_builder_accepts_cached_frontier_schema(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name_or_path: model\n", encoding="utf-8")
    static_path = tmp_path / "static.json"
    static_path.write_text(
        json.dumps(
            {
                "layer_widths": {"0": 4},
                "config": {"model_name_or_path": "model", "config_path": str(config_path)},
            }
        ),
        encoding="utf-8",
    )
    mask = {"0": [2]}
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps(mask), encoding="utf-8")
    run = {
        "strict_feasible": True,
        "behavioral_feasible": True,
        "deletion_budget": 1,
        "restart": 0,
        "mask_file": str(mask_path),
        "mask_hash": canonical_json_sha256(mask),
        "parameter_summary": {"removed_parameters": 9},
        "cached_fidelity": {"mean_kl": 0.0004},
        "generation": {"exact_match": True},
    }
    learned_path = tmp_path / "cached.json"
    learned_path.write_text(
        json.dumps(
            {
                "complete": True,
                "config": {"static_frontier": str(static_path), "kl_tolerance": 0.001},
                "best_strict_feasible": {"run": "cached"},
                "runs": {"cached": run},
            }
        ),
        encoding="utf-8",
    )
    _, _, metadata = build_artifact(learned_path)
    assert metadata["source_fidelity_path"] == "cached"
    assert metadata["source_strict_feasible"]


def test_phase5_artifact_builder_requires_explicit_physical_probe_opt_in(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name_or_path: model\n", encoding="utf-8")
    static_path = tmp_path / "static.json"
    static_path.write_text(
        json.dumps(
            {
                "layer_widths": {"0": 4},
                "config": {"model_name_or_path": "model", "config_path": str(config_path)},
            }
        ),
        encoding="utf-8",
    )
    mask = {"0": [2]}
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps(mask), encoding="utf-8")
    run = {
        "strict_feasible": False,
        "behavioral_feasible": False,
        "deletion_budget": 1,
        "mask_file": str(mask_path),
        "mask_hash": canonical_json_sha256(mask),
        "parameter_summary": {"removed_parameters": 9},
        "cached_fidelity": {
            "mean_kl": 0.002,
            "token_agreement": 1.0,
            "generated_token_agreement": 1.0,
        },
        "generation": {"exact_match": True},
    }
    learned_path = tmp_path / "cached.json"
    learned_path.write_text(
        json.dumps(
            {
                "complete": True,
                "config": {"static_frontier": str(static_path), "kl_tolerance": 0.001},
                "runs": {"probe": run},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict-feasible"):
        build_artifact(learned_path, "probe")
    _, _, metadata = build_artifact(learned_path, "probe", allow_physical_probe=True)
    assert metadata["source_gate"] == "physical_probe"
    assert metadata["source_physical_probe_eligible"]


def test_atomic_artifact_builder_freezes_best_strict_candidate(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name_or_path: model\n", encoding="utf-8")
    static_path = tmp_path / "static.json"
    static_path.write_text(
        json.dumps(
            {
                "layer_widths": {"0": 4, "1": 4},
                "parameter_scope": {"removable_by_layer": {"0": 36, "1": 36}},
                "config": {
                    "model_name_or_path": "model",
                    "config_path": str(config_path),
                    "kl_tolerance": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )
    mask = {"0": [2], "1": [1, 3]}
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps(mask), encoding="utf-8")
    candidate = {
        "strict_feasible": True,
        "behavior_preserved": True,
        "total_deletion_count": 3,
        "deletions_by_layer": {"0": 1, "1": 2},
        "deleted_neuron_ids_by_layer": {"0": [2], "1": [1, 3]},
        "mask_file": str(mask_path),
        "mask_sha256": canonical_json_sha256(mask),
        "structure_checks": {"artifact_valid": True},
        "cached_fidelity": {"mean_kl": 0.0005, "token_agreement": 1.0},
        "generation": {"exact_match": True},
    }
    atomic_path = tmp_path / "atomic.json"
    atomic_path.write_text(
        json.dumps(
            {
                "complete": True,
                "config": {"static_frontier": str(static_path), "max_new_tokens": 512},
                "best_strict_candidate": {"candidate": "best"},
                "candidates": {"best": candidate},
            }
        ),
        encoding="utf-8",
    )

    neuron_ids, clusters, metadata = build_atomic_artifact(atomic_path)
    assert neuron_ids == mask
    assert len(clusters[0]) == 3
    assert len(clusters[1]) == 2
    assert metadata["phase"] == "5E-atomic"
    assert metadata["source_gate"] == "structural_strict"
    assert metadata["kl_tolerance"] == 0.001
    assert metadata["frozen_search_horizon"] == 512
    assert metadata["theoretical_reduction"]["removed_parameters"] == 27


def test_atomic_artifact_builder_rejects_non_strict_candidate(tmp_path: Path):
    atomic_path = tmp_path / "atomic.json"
    atomic_path.write_text(
        json.dumps(
            {
                "complete": True,
                "best_strict_candidate": {"candidate": "failed"},
                "candidates": {"failed": {"strict_feasible": False}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict atomic candidate"):
        build_atomic_artifact(atomic_path)


def test_core_extension_artifact_builder_freezes_best_candidate(tmp_path: Path):
    static_path = tmp_path / "static.json"
    static_path.write_text(json.dumps({"config": {}}), encoding="utf-8")
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_mask = {"0": [1], "1": []}
    (base_dir / "mask.json").write_text(json.dumps(base_mask), encoding="utf-8")
    base_metadata = {
        "phase": "5E-atomic",
        "run": "core",
        "layer_dims": {"0": 4, "1": 4},
        "deletion_budget": 1,
        "mask_sha256": canonical_json_sha256(base_mask),
        "static_frontier": str(static_path),
        "theoretical_reduction": {"hidden_size": 3},
        "model_name_or_path": "model",
        "config_path": "config.yaml",
    }
    base_metadata_path = base_dir / "metadata.json"
    base_metadata_path.write_text(json.dumps(base_metadata), encoding="utf-8")
    extension_mask = {"0": [1], "1": [2]}
    extension_mask_path = tmp_path / "extension_mask.json"
    extension_mask_path.write_text(json.dumps(extension_mask), encoding="utf-8")
    candidate = {
        "strict_feasible": True,
        "behavior_preserved": True,
        "total_deletion_count": 2,
        "added_layer": 1,
        "added_neuron": 2,
        "candidate_rank": 1,
        "deleted_neuron_ids_by_layer": {"0": [1], "1": [2]},
        "mask_file": str(extension_mask_path),
        "mask_sha256": canonical_json_sha256(extension_mask),
        "structure_checks": {"artifact_valid": True},
        "cached_fidelity": {"mean_kl": 0.0005, "token_agreement": 1.0},
        "generation": {"exact_match": True},
    }
    frontier_path = tmp_path / "extension.json"
    frontier_path.write_text(
        json.dumps(
            {
                "complete": True,
                "config": {
                    "base_artifact_dir": str(base_dir),
                    "base_metadata_sha256": sha256_file(base_metadata_path),
                    "static_frontier": str(static_path),
                    "kl_tolerance": 0.001,
                    "max_new_tokens": 512,
                },
                "best_strict_candidate": {"candidate": "best"},
                "candidates": {"best": candidate},
            }
        ),
        encoding="utf-8",
    )

    neuron_ids, clusters, metadata = build_core_extension_artifact(frontier_path)
    assert neuron_ids == extension_mask
    assert len(clusters[0]) == 3
    assert len(clusters[1]) == 3
    assert metadata["source_gate"] == "structural_strict"
    assert metadata["deletion_budget"] == 2
    assert metadata["deletions_by_layer"] == {"0": 1, "1": 1}
    assert metadata["theoretical_reduction"]["removed_parameters"] == 18


def test_nested_probe_preserves_base_and_adds_lowest_normalized_scores():
    base = {
        0: torch.tensor([True, False, False]),
        1: torch.tensor([False, False, False]),
    }
    scores = {
        0: torch.tensor([9.0, 2.0, 1.0]),
        1: torch.tensor([3.0, 4.0, 5.0]),
    }
    extended, additions = extend_deletion_masks(base, scores, 3, normalization="none")
    assert additions == [(0, 2), (0, 1)]
    assert torch.equal(extended[0], torch.tensor([True, True, True]))
    assert torch.equal(extended[1], base[1])
    assert bool((extended[0] >= base[0]).all())
