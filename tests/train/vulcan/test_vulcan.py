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

from types import SimpleNamespace

import pytest
import torch
from transformers import TrainingArguments

from llamafactory.train.vulcan import (
    build_layerwise_cluster_idx,
    build_third_keep_ratios,
    find_mlp_layers,
    get_cluster_greedy_match,
    get_collapse_lambdas,
    init_collapse_lambdas,
    weight_collapse_loss,
)
from llamafactory.train.trainer_utils import create_custom_optimizer
from llamafactory.train.vulcan.pruning import pruning_mlp


class TinyMLP(torch.nn.Module):
    def __init__(self, hidden_size: int = 2, intermediate_size: int = 4):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * self.gate_proj(x))


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()

    def forward(self, x):
        return self.mlp(x)


class TinyModel(torch.nn.Module):
    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyLayer() for _ in range(num_layers)])
        self.config = type("Config", (), {"intermediate_size": 4})()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.mark.runs_on(["cpu", "mps"])
def test_find_mlp_layers():
    model = TinyModel(num_layers=3)
    mlp_layers = find_mlp_layers(model)
    assert len(mlp_layers) == 3
    assert [layer.index for layer in mlp_layers] == [0, 1, 2]


@pytest.mark.runs_on(["cpu", "mps"])
def test_greedy_match_cluster_count():
    activation = torch.tensor([0.1, 0.9, 0.2, 0.8])
    vectors = torch.tensor([[0.0], [0.1], [10.0], [10.1]])
    clusters = get_cluster_greedy_match(activation, vectors, n_clusters=2)
    assert len(clusters) == 2
    assert sorted(sum((cluster["neuron"] for cluster in clusters), [])) == [0, 1, 2, 3]
    assert sorted(cluster["anchor"] for cluster in clusters) == [1, 3]


@pytest.mark.runs_on(["cpu", "mps"])
def test_layerwise_cluster_idx_uses_per_layer_keep_ratios():
    model = TinyModel(num_layers=3)
    activations = torch.ones(3, 4)
    cluster_idx = build_layerwise_cluster_idx(model, activations, keep_ratios=[1.0, 0.75, 0.5])

    assert cluster_idx[0] is None
    assert len(cluster_idx[1]) == 3
    assert len(cluster_idx[2]) == 2


@pytest.mark.runs_on(["cpu", "mps"])
def test_third_keep_ratios():
    keep_ratios = build_third_keep_ratios(
        num_layers=6,
        first_keep_ratio=1.0,
        middle_keep_ratio=0.75,
        last_keep_ratio=0.5,
    )
    assert keep_ratios == [1.0, 1.0, 0.75, 0.75, 0.5, 0.5]


@pytest.mark.runs_on(["cpu", "mps"])
def test_learnable_collapse_lambdas_init_to_config_values():
    model = TinyModel(num_layers=1)
    finetuning_args = SimpleNamespace(
        collapse_learnable_lambda=True,
        collapse_lambda1=0.0,
        collapse_lambda2=0.0,
    )
    init_collapse_lambdas(model, finetuning_args)
    lambda1, lambda2 = get_collapse_lambdas(model, finetuning_args)

    assert isinstance(model.vulcan_lambda1, torch.nn.Parameter)
    assert isinstance(model.vulcan_lambda2, torch.nn.Parameter)
    assert torch.equal(lambda1.detach(), torch.tensor(0.0))
    assert torch.equal(lambda2.detach(), torch.tensor(0.0))


@pytest.mark.runs_on(["cpu", "mps"])
def test_learnable_collapse_lambdas_use_separate_lr(tmp_path):
    model = TinyModel(num_layers=1)
    finetuning_args = SimpleNamespace(
        collapse_learnable_lambda=True,
        collapse_lambda1=0.0,
        collapse_lambda2=0.0,
        collapse_lambda_lr=-1.0,
        use_galore=False,
        use_apollo=False,
        loraplus_lr_ratio=None,
        use_badam=False,
        use_adam_mini=False,
        use_muon=False,
    )
    init_collapse_lambdas(model, finetuning_args)
    training_args = TrainingArguments(output_dir=str(tmp_path), learning_rate=1e-5, optim="adamw_torch")
    optimizer = create_custom_optimizer(model, training_args, finetuning_args)

    lambda_group = next(
        group for group in optimizer.param_groups if any(param is model.vulcan_lambda1 for param in group["params"])
    )
    assert lambda_group["lr"] == -1.0
    assert lambda_group["weight_decay"] == 0.0


@pytest.mark.runs_on(["cpu", "mps"])
def test_weight_collapse_loss_and_proxy_backward():
    model = TinyModel(num_layers=1)
    with torch.no_grad():
        model.layers[0].mlp.up_proj.weight.copy_(torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]))
        model.layers[0].mlp.gate_proj.weight.copy_(
            torch.tensor([[1.0, 1.0], [1.0, 3.0], [1.0, 5.0], [1.0, 7.0]])
        )

    cluster_idx = [[{"anchor": 0, "neuron": [0, 1]}, {"anchor": 2, "neuron": [2, 3]}]]
    loss = weight_collapse_loss(
        model,
        cluster_idx,
        lambda1=torch.tensor(0.5),
        lambda2=torch.tensor(0.25),
        use_weight_proxy=True,
    )
    assert torch.isclose(loss, torch.tensor(5.5))
    loss.backward()
    assert model.layers[0].mlp.up_proj.weight.grad is not None
    assert model.layers[0].mlp.gate_proj.weight.grad is not None


@pytest.mark.runs_on(["cpu", "mps"])
def test_pruning_mlp_preserves_output_for_identical_clusters():
    model = TinyModel(num_layers=1)
    mlp = model.layers[0].mlp
    with torch.no_grad():
        mlp.up_proj.weight.copy_(torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]))
        mlp.gate_proj.weight.copy_(torch.tensor([[2.0, 1.0], [2.0, 1.0], [4.0, 3.0], [4.0, 3.0]]))
        mlp.down_proj.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]))

    x = torch.tensor([[0.25, 0.5], [1.0, -0.25]])
    expected = model(x)
    summary = pruning_mlp(model, [[{"anchor": 0, "neuron": [0, 1]}, {"anchor": 2, "neuron": [2, 3]}]])
    actual = model(x)

    assert summary.original_intermediate_size == 4
    assert summary.pruned_intermediate_size == 2
    assert model.config.intermediate_size == 2
    assert torch.allclose(actual, expected, atol=1e-6)
