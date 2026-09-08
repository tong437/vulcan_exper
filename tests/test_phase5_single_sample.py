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

from run_phase5_single_sample_frontier import (  # noqa: E402
    build_deletion_masks,
    build_prompt_inputs,
    build_teacher_rollout_batch,
    collect_teacher_trace,
    compare_generation,
    evaluate_teacher_fidelity,
    merge_resume_result,
    parse_csv_floats,
    summarize_masks,
)

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


def test_parse_keep_ratios_rejects_invalid_values():
    assert parse_csv_floats("0.5, 0.25,0.5,0") == [0.5, 0.25, 0.0]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_csv_floats("1.1")


def test_build_deletion_masks_supports_uniform_and_global_budgets():
    scores = {0: torch.tensor([1.0, 4.0, 2.0, 3.0]), 1: torch.tensor([8.0, 5.0, 7.0, 6.0])}
    uniform = build_deletion_masks(scores, 0.5, "per_layer")
    assert (~uniform[0]).nonzero().flatten().tolist() == [1, 3]
    assert (~uniform[1]).nonzero().flatten().tolist() == [0, 2]
    assert summarize_masks(uniform)["kept_neurons"] == 4

    global_masks = build_deletion_masks(scores, 0.25, "global", global_normalization="layer_rank")
    assert (~global_masks[0]).nonzero().flatten().tolist() == [1]
    assert (~global_masks[1]).nonzero().flatten().tolist() == [0]
    assert summarize_masks(global_masks)["kept_neurons"] == 2

    nonuniform_scores = {0: torch.tensor([100.0, 0.0, 0.0, 0.0]), 1: torch.tensor([4.0, 3.0, 2.0, 1.0])}
    nonuniform = build_deletion_masks(nonuniform_scores, 0.5, "global", global_normalization="layer_mean")
    assert int((~nonuniform[0]).sum()) == 1
    assert int((~nonuniform[1]).sum()) == 3

    empty = build_deletion_masks(scores, 0.0, "per_layer")
    assert all(bool(mask.all()) for mask in empty.values())


def test_teacher_trace_produces_all_saliency_scores_and_identity_fidelity():
    torch.manual_seed(4)
    model = TinyLM().eval()
    model.requires_grad_(False)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]]),
    }
    modules = [layer.mlp.down_proj for layer in model.layers]
    trace, saliency = collect_teacher_trace(model, batch, modules)

    assert trace["num_label_tokens"] == 3
    assert set(saliency) == {"activation", "contribution", "taylor"}
    for method in saliency.values():
        assert set(method) == {0, 1}
        assert all(values.shape == (4,) for values in method.values())
        assert all(bool(torch.isfinite(values).all()) for values in method.values())
    fidelity = evaluate_teacher_fidelity(model, batch, trace)
    assert fidelity["token_agreement"] == 1.0
    assert fidelity["mean_kl"] == pytest.approx(0.0, abs=1e-6)
    assert fidelity["delta_nll"] == pytest.approx(0.0, abs=1e-6)


def test_prompt_truncation_preserves_nonsequence_image_tensors():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "position_ids": torch.arange(15).view(1, 3, 5),
        "pixel_values": torch.randn(2, 4),
        "rope_deltas": torch.tensor([[0]]),
        "labels": torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5]]),
    }
    prompt, length = build_prompt_inputs(batch)
    assert length == 3
    assert prompt["input_ids"].shape[-1] == 3
    assert prompt["position_ids"].shape[-1] == 3
    assert prompt["pixel_values"] is batch["pixel_values"]
    assert "labels" not in prompt
    assert "rope_deltas" not in prompt

    rollout = build_teacher_rollout_batch(prompt, torch.tensor([7, 8]))
    assert rollout["input_ids"].tolist() == [[1, 2, 3, 7, 8]]
    assert rollout["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert rollout["labels"].tolist() == [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 7, 8]]
    assert rollout["pixel_values"] is batch["pixel_values"]
    assert "position_ids" not in rollout


def test_generation_comparison_reports_first_divergence():
    teacher = {"token_ids": torch.tensor([1, 2, 3]), "text": "teacher"}
    exact = compare_generation(teacher, {"token_ids": torch.tensor([1, 2, 3]), "text": "teacher"})
    changed = compare_generation(teacher, {"token_ids": torch.tensor([1, 4, 3]), "text": "student"})
    assert exact["exact_match"]
    assert exact["first_divergence"] is None
    assert not changed["exact_match"]
    assert changed["common_prefix_tokens"] == 1
    assert changed["first_divergence"] == 1


def test_resume_merge_extends_grid_and_rejects_identity_changes():
    config = {
        "model_name_or_path": "model",
        "dataset": "data",
        "dataset_stage": "sft",
        "sample_offset": 2,
        "global_normalization": "layer_mean",
        "max_new_tokens": 8,
        "trace_target": "teacher_generation",
        "kl_tolerance": 0.001,
        "seed": 7,
        "methods": ["activation"],
        "selections": ["per_layer"],
        "keep_ratios": [0.5],
    }
    existing = {
        "complete": True,
        "config": config,
        "teacher": {"generated_token_ids": [1, 2]},
        "parameter_scope": {},
        "layer_widths": {},
    }
    fresh = {
        "config": {**config, "methods": ["taylor"], "keep_ratios": [0.75]},
        "teacher": {"generated_token_ids": [1, 2]},
        "parameter_scope": {"total_model_parameters": 10},
        "layer_widths": {"0": 4},
    }
    merged = merge_resume_result(existing, fresh)
    assert not merged["complete"]
    assert merged["config"]["methods"] == ["activation", "taylor"]
    assert merged["config"]["keep_ratios"] == [0.5, 0.75]

    changed = {**fresh, "config": {**fresh["config"], "sample_offset": 3}}
    with pytest.raises(ValueError, match="mismatched fields"):
        merge_resume_result(existing, changed)
