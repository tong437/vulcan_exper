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

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from run_phase5_learned_gate import (  # noqa: E402
    ExactBudgetGate,
    is_better_feasible_candidate,
    normalized_initial_logits,
    parse_csv_ints,
    temperature_at_step,
)


def test_best_feasible_candidate_prefers_budget_then_lower_kl():
    assert is_better_feasible_candidate(None, deletion_budget=500, mean_kl=0.001)
    current = {"deletion_budget": 500, "mean_kl": 0.0008}
    assert is_better_feasible_candidate(current, deletion_budget=750, mean_kl=0.01)
    assert is_better_feasible_candidate(current, deletion_budget=500, mean_kl=0.0007)
    assert not is_better_feasible_candidate(current, deletion_budget=500, mean_kl=0.0009)
    assert not is_better_feasible_candidate(current, deletion_budget=250, mean_kl=0.0001)


def test_deletion_budget_parsing_and_temperature_schedule():
    assert parse_csv_ints("10, 20,10") == [10, 20]
    with pytest.raises(ValueError, match="positive"):
        parse_csv_ints("0")
    assert temperature_at_step(2.0, 0.5, 0, 3) == pytest.approx(2.0)
    assert temperature_at_step(2.0, 0.5, 1, 3) == pytest.approx(1.0)
    assert temperature_at_step(2.0, 0.5, 2, 3) == pytest.approx(0.5)


def test_normalized_initial_logits_are_reproducible():
    scores = {0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])}
    first = normalized_initial_logits(
        scores, normalization="layer_mean", noise_std=0.1, seed=7, device=torch.device("cpu")
    )
    second = normalized_initial_logits(
        scores, normalization="layer_mean", noise_std=0.1, seed=7, device=torch.device("cpu")
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_exact_budget_gate_has_exact_hard_count_and_ste_gradients():
    torch.manual_seed(3)
    modules = [torch.nn.Linear(4, 3, bias=False), torch.nn.Linear(4, 3, bias=False)]
    initial = [torch.tensor([-2.0, -1.0, 1.0, 2.0]), torch.tensor([-3.0, 0.0, 3.0, 4.0])]
    controller = ExactBudgetGate(modules, initial, deletion_budget=3)
    masks = controller.hard_deletion_masks()
    assert sum(int(mask.sum()) for mask in masks.values()) == 3

    inputs = [torch.randn(2, 4), torch.randn(2, 4)]
    with controller:
        controller.prepare_ste_gates(temperature=1.0)
        loss = sum(module(inputs[index]).square().mean() for index, module in enumerate(modules))
        loss.backward()
    assert all(parameter.grad is not None for parameter in controller.logits)
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in controller.logits)
