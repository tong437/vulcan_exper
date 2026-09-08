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

from phase5e_proxy import build_exact_deletion_masks, evaluate_gold_proxy, parse_deletion_budgets  # noqa: E402
from run_phase5_single_sample_frontier import collect_teacher_trace  # noqa: E402

from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(13, 5)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.SiLU(), torch.nn.Linear(4, 5))
        self.lm_head = torch.nn.Linear(5, 13, bias=False)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.lm_head(self.mlp(self.embed(input_ids))))


def test_parse_deletion_budgets():
    assert parse_deletion_budgets("100, 250,100") == [100, 250]
    with pytest.raises(ValueError, match="positive"):
        parse_deletion_budgets("0")


def test_exact_static_masks_are_nested_and_stable():
    scores = {0: torch.tensor([0.4, 0.1, 0.3]), 1: torch.tensor([0.2, 0.8, 0.5])}
    one = build_exact_deletion_masks(scores, 1, global_normalization="none")
    three = build_exact_deletion_masks(scores, 3, global_normalization="none")
    assert one[0].tolist() == [False, True, False]
    assert all(bool((one[layer] <= three[layer]).all()) for layer in one)
    assert sum(int(mask.sum()) for mask in three.values()) == 3


def test_gold_proxy_is_identity_for_reference_model():
    torch.manual_seed(11)
    model = TinyLM().eval().requires_grad_(False)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]]),
    }
    trace, _ = collect_teacher_trace(model, batch, [model.mlp[2]])
    metrics = evaluate_gold_proxy(model, batch, trace)
    assert metrics["num_gold_tokens"] == 3
    assert metrics["delta_nll"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["mean_kl"] == pytest.approx(0.0, abs=1e-6)
    assert len(metrics["per_token_kl"]) == 3
    assert len(metrics["per_token_gold_logit_margin"]) == 3
