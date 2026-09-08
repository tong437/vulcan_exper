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

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from scan_phase5_atomic_physical import (  # noqa: E402
    StopAfterFirstDivergence,
    build_atomic_masks,
    build_combined_masks,
    combination_name,
    parse_combinations,
    parse_csv_ints,
    refresh_summary,
    select_lowest_neurons,
    temporary_structural_pruning,
)

from llamafactory.train.vulcan.modeling import find_mlp_layers, get_intermediate_size  # noqa: E402


class TinyMLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.intermediate_size = 4
        self.gate_proj = torch.nn.Linear(3, 4, bias=False)
        self.up_proj = torch.nn.Linear(3, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 3, bias=False)


class TinyLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = TinyMLP(config)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        text_config = SimpleNamespace(intermediate_size=4)
        self.config = SimpleNamespace(intermediate_size=4, text_config=text_config)
        self.layers = torch.nn.ModuleList([TinyLayer(text_config)])


def test_select_lowest_neurons_is_nested_and_stable():
    scores = torch.tensor([2.0, 1.0, 1.0, 3.0])
    assert select_lowest_neurons(scores, 1) == [1]
    assert select_lowest_neurons(scores, 3) == [1, 2, 0]
    with pytest.raises(ValueError, match="deletion_count"):
        select_lowest_neurons(scores, 4)


def test_build_atomic_masks_changes_only_target_layer():
    masks = build_atomic_masks({0: 4, 1: 3}, 1, [0, 2])
    assert masks[0].tolist() == [False, False, False, False]
    assert masks[1].tolist() == [True, False, True]
    with pytest.raises(ValueError, match="duplicates"):
        build_atomic_masks({0: 4}, 0, [1, 1])


def test_parse_and_build_combined_masks():
    assert parse_combinations("8:1+23:2,19:1+23:2") == [{8: 1, 23: 2}, {19: 1, 23: 2}]
    assert combination_name({23: 2, 8: 1}) == "layers_08x1-23x2"
    masks = build_combined_masks({0: 3, 1: 4}, {0: [2], 1: [0, 3]})
    assert masks[0].tolist() == [False, False, True]
    assert masks[1].tolist() == [True, False, False, True]
    with pytest.raises(ValueError, match="at least two"):
        parse_combinations("8:1")


def test_stop_after_first_divergence():
    criterion = StopAfterFirstDivergence(2, torch.tensor([7, 8, 9]))
    assert not criterion(torch.tensor([[1, 2, 7]]), torch.empty(0))
    assert criterion(torch.tensor([[1, 2, 7, 4]]), torch.empty(0))
    assert not criterion(torch.tensor([[1, 2, 7, 8]]), torch.empty(0))


def test_parse_csv_ints_deduplicates_and_rejects_negative_values():
    assert parse_csv_ints("1,2,1") == [1, 2]
    with pytest.raises(ValueError, match="non-negative"):
        parse_csv_ints("1,-1")


def test_temporary_structural_pruning_restores_modules_width_and_config():
    model = TinyModel()
    mlp = find_mlp_layers(model)[0].mlp
    original_modules = (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    clusters = [[{"anchor": index, "neuron": [index]} for index in (0, 2, 3)]]
    with temporary_structural_pruning(model, clusters, 0) as pruned:
        assert pruned[0] is mlp
        assert get_intermediate_size(mlp) == 3
        assert model.config.text_config.intermediate_size == 3
        assert model.config.vulcan_intermediate_sizes == [3]
    assert get_intermediate_size(mlp) == 4
    assert (mlp.gate_proj, mlp.up_proj, mlp.down_proj) == original_modules
    assert model.config.text_config.intermediate_size == 4
    assert not hasattr(model.config, "vulcan_intermediate_sizes")


def test_refresh_summary_prefers_largest_exact_deletion_count():
    result = {
        "candidates": {
            "small": {
                "deletion_count": 1,
                "generation": {"exact_match": True, "common_prefix_tokens": 512},
                "cached_fidelity": {"mean_kl": 0.0001},
                "strict_feasible": True,
            },
            "large": {
                "total_deletion_count": 3,
                "generation": {"exact_match": True, "common_prefix_tokens": 512},
                "cached_fidelity": {"mean_kl": 0.0005},
                "strict_feasible": True,
            },
        }
    }
    refresh_summary(result)
    assert result["best_exact_candidate"]["candidate"] == "large"
    assert result["best_strict_candidate"]["candidate"] == "large"
    assert result["best_prefix"]["candidate"] == "large"
