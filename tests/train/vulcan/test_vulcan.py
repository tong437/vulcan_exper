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

import pytest
import torch

from llamafactory.train.vulcan import find_mlp_layers, get_cluster_greedy_match, weight_collapse_loss
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
