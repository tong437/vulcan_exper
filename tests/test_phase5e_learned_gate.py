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

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from run_phase5_learned_gate import ExactBudgetGate  # noqa: E402
from run_phase5_single_sample_frontier import collect_teacher_trace  # noqa: E402
from run_phase5e_learned_gate import optimize_gold_gate  # noqa: E402

from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402


class TinyMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        intermediate = torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(intermediate)


class TinyBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.mlp = TinyMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        return hidden_states + self.mlp(hidden_states)


class TinyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 11, hidden_size: int = 5, intermediate_size: int = 4):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            [TinyBlock(hidden_size, intermediate_size), TinyBlock(hidden_size, intermediate_size)]
        )
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        hidden_states = self.embed(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


def test_optimize_gold_gate_preserves_exact_budget_and_records_losses():
    torch.manual_seed(19)
    model = TinyLM().eval().requires_grad_(False)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]]),
    }
    modules = [layer.mlp.down_proj for layer in model.layers]
    trace, saliency = collect_teacher_trace(model, batch, modules)
    controller = ExactBudgetGate(modules, [saliency["taylor"][i] for i in range(2)], deletion_budget=2)
    masks, training = optimize_gold_gate(
        model,
        batch,
        trace,
        controller,
        steps=3,
        learning_rate=0.01,
        temperature_start=2.0,
        temperature_end=0.5,
        gold_ce_weight=1.0,
        reference_kl_weight=1.0,
        gradient_clip=1.0,
        history_interval=1,
    )
    assert sum(int(mask.sum()) for mask in masks.values()) == 2
    assert training["best_step"] is not None
    assert len(training["history"]) == 3
    assert all("gold_ce" in row and "reference_kl" in row for row in training["history"])
