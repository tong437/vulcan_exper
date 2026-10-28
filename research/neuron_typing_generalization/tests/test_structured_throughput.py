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

"""Physical slicing equivalence and alignment on asymmetric, biased MLPs."""

import copy

import numpy as np
import pytest
import torch
from neuron_typing_generalization.advisor_ablation import DownProjectionAblation
from neuron_typing_generalization.hooks import discover_decoder_ffns
from neuron_typing_generalization.structured_throughput import make_plan, physically_prune, removed_indices
from torch import nn


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate_proj = nn.Linear(7, width)
        self.up_proj = nn.Linear(7, width)
        self.down_proj = nn.Linear(width, 7)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({"mlp": MLP(w)}) for w in (16, 32)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer["mlp"](x)
        return x


@pytest.mark.parametrize("method", ["original", "uniform", "nonuniform"])
@pytest.mark.parametrize("batch", [1, 4])
def test_physical_matches_masked_with_bias(method, batch):
    torch.manual_seed(10)
    model = Network().eval()
    sliced = copy.deepcopy(model)
    costs = [np.linspace(0.1, 2, w) / w for w in (16, 32)]
    plan = make_plan(costs, 0 if method == "original" else 0.5, method)
    x = torch.randn(batch, 5, 7)
    with DownProjectionAblation(discover_decoder_ffns(model), removed_indices(plan)):
        expected = model(x)
    counts = physically_prune(sliced, plan)
    torch.testing.assert_close(sliced(x), expected, atol=1e-6, rtol=1e-6)
    assert [b.d_ff for b in discover_decoder_ffns(sliced)] == plan["aligned_widths"]
    assert counts["parameters_after"] <= counts["parameters_before"]


def test_alignment_restores_highest_deleted_and_matches_precount():
    costs = [np.arange(16, dtype=float), np.arange(32, dtype=float) + 0.5]
    u, n = [make_plan(costs, 0.3, method) for method in ("uniform", "nonuniform")]
    assert sum(u["pre_alignment_deleted"]) == sum(n["pre_alignment_deleted"]) == 13
    for plan in (u, n):
        for w, keep, scores in zip(plan["aligned_widths"], plan["kept"], costs):
            assert w % 8 == 0 and w >= 8
            assert keep == list(range(len(scores) - w, len(scores)))
        assert plan["actual_rate"] <= 0.3


def test_zero_width_is_restored_to_eight():
    plan = make_plan([np.zeros(16), np.ones(16)], 0.5, "nonuniform")
    assert plan["pre_alignment_deleted"] == [16, 0]
    assert plan["aligned_widths"] == [8, 16]


def test_unsupported_original_width_rejected():
    with pytest.raises(ValueError):
        make_plan([np.ones(17)], 0.1, "uniform")
