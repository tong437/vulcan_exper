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

from llamafactory.train.vulcan import ActivationAligner, find_mlp_layers


class TinyMLP(torch.nn.Module):
    def __init__(self, hidden_size: int = 2, intermediate_size: int = 4):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()

    def forward(self, x):
        return self.mlp(x)


class TinyModel(torch.nn.Module):
    def __init__(self, num_layers: int = 2, image_token_id: int = 99):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyLayer() for _ in range(num_layers)])
        self.config = type("Config", (), {"intermediate_size": 4, "image_token_id": image_token_id})()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _make_finetuning_args(**kwargs):
    defaults = dict(
        align_lambda=1.0,
        align_temperature=0.1,
        align_quantile=0.8,
        align_pool_type="mean",
        align_loss_type="l1",
        align_text_mode="answer",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.runs_on(["cpu", "mps"])
def test_activation_aligner_hooks_registered():
    model = TinyModel(num_layers=3)
    aligner = ActivationAligner(model, _make_finetuning_args(), image_token_id=99)
    assert len(aligner._hooks) == 3
    aligner.remove_hooks()
    assert len(aligner._hooks) == 0


@pytest.mark.runs_on(["cpu", "mps"])
def test_activation_aligner_l1_loss_scalar():
    model = TinyModel(num_layers=2)
    aligner = ActivationAligner(model, _make_finetuning_args(), image_token_id=99)

    input_ids = torch.tensor([[99, 99, 1, 1], [99, 1, 1, 1]])
    aligner.set_input_ids(input_ids)
    x = torch.randn(2, 4, 2)
    model(x)
    loss = aligner.compute_alignment_loss()

    assert loss.shape == ()
    assert loss.item() >= 0
    aligner.remove_hooks()


@pytest.mark.runs_on(["cpu", "mps"])
def test_activation_aligner_answer_text_mode_uses_labels():
    model = TinyModel(num_layers=1)
    aligner = ActivationAligner(model, _make_finetuning_args(), image_token_id=99)

    input_ids = torch.tensor([[99, 10, 11, 12]])
    labels = torch.tensor([[-100, -100, 11, -100]])
    attention_mask = torch.ones_like(input_ids)
    aligner.set_batch(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    x = torch.randn(1, 4, 2)
    model(x)
    loss = aligner.compute_alignment_loss()
    logs = aligner.get_log()

    assert loss.shape == ()
    assert logs["align_visual_tokens"] == 1.0
    assert logs["align_text_tokens"] == 1.0
    aligner.remove_hooks()


@pytest.mark.runs_on(["cpu", "mps"])
def test_activation_aligner_soft_iou_loss():
    model = TinyModel(num_layers=1)
    aligner = ActivationAligner(model, _make_finetuning_args(align_loss_type="soft_iou"), image_token_id=99)

    input_ids = torch.tensor([[99, 1]])
    labels = torch.tensor([[-100, 1]])
    aligner.set_batch(input_ids=input_ids, labels=labels, attention_mask=torch.ones_like(input_ids))
    x = torch.randn(1, 2, 2)
    model(x)
    loss = aligner.compute_alignment_loss()

    assert loss.shape == ()
    assert loss.item() >= 0
    aligner.remove_hooks()


@pytest.mark.runs_on(["cpu", "mps"])
def test_activation_aligner_neg_iou_loss():
    model = TinyModel(num_layers=2)
    aligner = ActivationAligner(model, _make_finetuning_args(align_loss_type="neg_iou"), image_token_id=99)

    input_ids = torch.tensor([[99, 99, 1, 1]])
    aligner.set_input_ids(input_ids)
    x = torch.randn(1, 4, 2)
    model(x)
    loss = aligner.compute_alignment_loss()

    assert loss.shape == ()
    aligner.remove_hooks()


@pytest.mark.runs_on(["cpu", "mps"])
def test_activation_aligner_gradient_flows():
    model = TinyModel(num_layers=1)
    aligner = ActivationAligner(model, _make_finetuning_args(), image_token_id=99)

    input_ids = torch.tensor([[99, 1]])
    aligner.set_input_ids(input_ids)
    x = torch.randn(1, 2, 2)
    model(x)
    loss = aligner.compute_alignment_loss()
    loss.backward()

    assert model.layers[0].mlp.up_proj.weight.grad is not None
    assert model.layers[0].mlp.gate_proj.weight.grad is not None
    assert model.layers[0].mlp.down_proj.weight.grad is not None
    aligner.remove_hooks()


@pytest.mark.runs_on(["cpu", "mps"])
def test_activation_aligner_no_visual_tokens_returns_zero():
    model = TinyModel(num_layers=1)
    aligner = ActivationAligner(model, _make_finetuning_args(), image_token_id=99)

    input_ids = torch.tensor([[1, 1, 1]])
    aligner.set_input_ids(input_ids)
    x = torch.randn(1, 3, 2)
    model(x)
    loss = aligner.compute_alignment_loss()

    assert loss.item() == 0.0
    aligner.remove_hooks()


@pytest.mark.runs_on(["cpu", "mps"])
def test_activation_aligner_no_text_tokens_returns_zero():
    model = TinyModel(num_layers=1)
    aligner = ActivationAligner(model, _make_finetuning_args(), image_token_id=99)

    input_ids = torch.tensor([[99, 99, 99]])
    aligner.set_input_ids(input_ids)
    x = torch.randn(1, 3, 2)
    model(x)
    loss = aligner.compute_alignment_loss()

    assert loss.item() == 0.0
    aligner.remove_hooks()


@pytest.mark.runs_on(["cpu", "mps"])
def test_activation_aligner_max_pool():
    model = TinyModel(num_layers=1)
    aligner = ActivationAligner(model, _make_finetuning_args(align_pool_type="max"), image_token_id=99)

    input_ids = torch.tensor([[99, 1, 1]])
    aligner.set_input_ids(input_ids)
    x = torch.randn(1, 3, 2)
    model(x)
    loss = aligner.compute_alignment_loss()

    assert loss.shape == ()
    assert loss.item() >= 0
    aligner.remove_hooks()


@pytest.mark.runs_on(["cpu", "mps"])
def test_activation_aligner_lambda_scales_loss():
    model = TinyModel(num_layers=1)

    loss_lambda1 = None
    for lam in [0.5, 2.0]:
        aligner = ActivationAligner(model, _make_finetuning_args(align_lambda=lam), image_token_id=99)
        input_ids = torch.tensor([[99, 1]])
        aligner.set_input_ids(input_ids)
        x = torch.randn(1, 2, 2)
        model(x)
        loss = aligner.compute_alignment_loss()
        if loss_lambda1 is None:
            loss_lambda1 = loss.item()
        else:
            assert abs(loss.item() - lam / 0.5 * loss_lambda1) < 1e-5
        aligner.remove_hooks()


@pytest.mark.runs_on(["cpu", "mps"])
def test_activation_aligner_act_store_cleared_after_loss():
    model = TinyModel(num_layers=1)
    aligner = ActivationAligner(model, _make_finetuning_args(), image_token_id=99)

    input_ids = torch.tensor([[99, 1]])
    aligner.set_input_ids(input_ids)
    x = torch.randn(1, 2, 2)
    model(x)
    aligner.compute_alignment_loss()

    assert len(aligner._act_store) == 0
    assert aligner._input_ids is None
    aligner.remove_hooks()
