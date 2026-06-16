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
from torch.utils.checkpoint import checkpoint
from transformers import TrainingArguments

from llamafactory.train.trainer_utils import create_custom_optimizer
from llamafactory.train.vulcan import (
    ActivationAligner,
    build_layerwise_cluster_idx,
    build_multimodal_cluster_idx,
    build_third_keep_ratios,
    collect_multimodal_mlp_activations,
    find_mlp_layers,
    get_cluster_greedy_match,
    get_collapse_lambdas,
    get_collapse_schedule_factor,
    init_collapse_lambdas,
    weight_collapse_loss,
)
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


class TinyTokenModel(TinyModel):
    def __init__(self, num_layers: int = 2):
        super().__init__(num_layers=num_layers)
        self.embed_tokens = torch.nn.Embedding(128, 2)

    def forward(self, input_ids, labels=None, attention_mask=None):
        return super().forward(self.embed_tokens(input_ids))


class CheckpointTinyModel(TinyModel):
    def __init__(self, use_reentrant: bool):
        super().__init__(num_layers=1)
        self.use_reentrant = use_reentrant

    def forward(self, x):
        return checkpoint(self.layers[0], x, use_reentrant=self.use_reentrant)


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
def test_multimodal_activation_collection_and_clustering():
    model = TinyTokenModel(num_layers=1)
    batches = {
        "modality": [
            {
                "input_ids": torch.tensor([[99, 10, 11, 20]]),
                "labels": torch.tensor([[-100, -100, -100, 20]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }
        ],
        "organ": [
            {
                "input_ids": torch.tensor([[99, 99, 12, 21]]),
                "labels": torch.tensor([[-100, -100, -100, 21]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }
        ],
    }
    stats = collect_multimodal_mlp_activations(
        model,
        batches,
        image_token_id=99,
        special_token_ids=set(),
        image_weight=0.4,
        question_weight=0.4,
        prediction_weight=0.2,
    )
    cluster_idx = build_multimodal_cluster_idx(
        model,
        stats,
        keep_ratios=[0.5],
        activation_distance_weight=0.25,
    )

    assert stats.anchor_scores.shape == (1, 4)
    assert stats.signatures.shape == (1, 4, 3)
    assert len(cluster_idx[0]) == 2


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
    assert lambda_group["lr"] == 1.0
    assert lambda_group.get("weight_decay", 0.0) == 0.0


@pytest.mark.runs_on(["cpu", "mps"])
def test_collapse_schedule_factor():
    assert get_collapse_schedule_factor(0, warmup_steps=2, ramp_steps=4) == 0.0
    assert get_collapse_schedule_factor(1, warmup_steps=2, ramp_steps=4) == 0.0
    assert get_collapse_schedule_factor(2, warmup_steps=2, ramp_steps=4) == 0.25
    assert get_collapse_schedule_factor(5, warmup_steps=2, ramp_steps=4) == 1.0


@pytest.mark.runs_on(["cpu", "mps"])
def test_weight_collapse_loss_and_proxy_backward():
    model = TinyModel(num_layers=1)
    with torch.no_grad():
        model.layers[0].mlp.up_proj.weight.copy_(torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]))
        model.layers[0].mlp.gate_proj.weight.copy_(torch.tensor([[1.0, 1.0], [1.0, 3.0], [1.0, 5.0], [1.0, 7.0]]))

    cluster_idx = [[{"anchor": 0, "neuron": [0, 1]}, {"anchor": 2, "neuron": [2, 3]}]]
    loss = weight_collapse_loss(
        model,
        cluster_idx,
        lambda1=torch.tensor(0.5),
        lambda2=torch.tensor(0.25),
        use_weight_proxy=True,
    )
    assert torch.isclose(loss, torch.tensor(2.75))
    loss.backward()
    assert model.layers[0].mlp.up_proj.weight.grad is not None
    assert model.layers[0].mlp.gate_proj.weight.grad is not None


@pytest.mark.runs_on(["cpu", "mps"])
def test_normalized_weight_collapse_loss_is_elementwise_mean():
    model = TinyModel(num_layers=1)
    with torch.no_grad():
        model.layers[0].mlp.up_proj.weight.zero_()
        model.layers[0].mlp.gate_proj.weight.zero_()
        model.layers[0].mlp.up_proj.weight[1].fill_(2.0)
        model.layers[0].mlp.gate_proj.weight[1].fill_(2.0)

    cluster_idx = [[{"anchor": 0, "neuron": [0, 1]}, {"anchor": 2, "neuron": [2, 3]}]]
    loss = weight_collapse_loss(
        model,
        cluster_idx,
        lambda1=torch.tensor(1.0),
        lambda2=torch.tensor(0.0),
        use_weight_proxy=False,
        reduction="normalized",
    )

    assert torch.isclose(loss, torch.tensor(1.0))


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


@pytest.mark.runs_on(["cpu", "mps"])
def test_pruning_mlp_supports_layerwise_intermediate_sizes():
    model = TinyModel(num_layers=2)
    second_mlp = model.layers[1].mlp
    with torch.no_grad():
        second_mlp.up_proj.weight[1].copy_(second_mlp.up_proj.weight[0])
        second_mlp.gate_proj.weight[1].copy_(second_mlp.gate_proj.weight[0])
        second_mlp.up_proj.weight[3].copy_(second_mlp.up_proj.weight[2])
        second_mlp.gate_proj.weight[3].copy_(second_mlp.gate_proj.weight[2])

    cluster_idx = [None, [{"anchor": 0, "neuron": [0, 1]}, {"anchor": 2, "neuron": [2, 3]}]]
    summary = pruning_mlp(model, cluster_idx)

    assert summary.pruned_intermediate_size == [4, 2]
    assert model.config.vulcan_intermediate_sizes == [4, 2]
    assert model.layers[0].mlp.up_proj.out_features == 4
    assert model.layers[1].mlp.up_proj.out_features == 2


def _make_align_finetuning_args(**overrides):
    defaults = dict(
        align_mode="neuron",
        align_lambda=0.05,
        align_temperature=0.05,
        align_quantile=0.8,
        align_pool_type="mean",
        align_loss_type="soft_iou",
        align_margin=0.0,
        align_text_mode="qa",
        align_question_weight=1.0,
        align_answer_weight=0.2,
        align_cluster_temperature=1.0,
        align_cluster_question_weight=1.0,
        align_cluster_answer_weight=0.5,
        align_layer_start_ratio=0.0,
        align_layer_end_ratio=1.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


IMAGE_TOKEN_ID = 99


def _make_aligner(model, cluster_idx=None, **overrides):
    finetuning_args = _make_align_finetuning_args(**overrides)
    return ActivationAligner(model, finetuning_args, IMAGE_TOKEN_ID, cluster_idx=cluster_idx)


@pytest.mark.runs_on(["cpu", "mps"])
def test_align_loss_requires_grad():
    model = TinyModel(num_layers=2)
    aligner = _make_aligner(model)
    input_ids = torch.tensor([[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4]])

    aligner.set_batch(input_ids=input_ids, labels=labels)
    x = torch.randn(1, 6, 2, requires_grad=True)
    model(x)

    loss = aligner.compute_alignment_loss()
    assert loss.requires_grad, "alignment loss must have requires_grad=True"
    assert loss.grad_fn is not None, "alignment loss must have a grad_fn"


@pytest.mark.runs_on(["cpu", "mps"])
def test_align_loss_can_return_raw_loss():
    model = TinyModel(num_layers=1)
    aligner = _make_aligner(model, align_lambda=0.5)
    input_ids = torch.tensor([[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4]])

    aligner.set_batch(input_ids=input_ids, labels=labels)
    model(torch.randn(1, 6, 2, requires_grad=True))
    weighted_loss, raw_loss = aligner.compute_alignment_loss(return_raw_loss=True)

    assert weighted_loss.requires_grad
    assert raw_loss.requires_grad
    assert torch.allclose(weighted_loss, 0.5 * raw_loss)


@pytest.mark.runs_on(["cpu", "mps"])
def test_align_loss_backward_produces_param_gradients():
    model = TinyModel(num_layers=1)
    aligner = _make_aligner(model)
    input_ids = torch.tensor([[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4]])

    aligner.set_batch(input_ids=input_ids, labels=labels)
    x = torch.randn(1, 6, 2, requires_grad=True)
    model(x)

    loss = aligner.compute_alignment_loss()
    loss.backward()

    mlp = model.layers[0].mlp
    assert mlp.up_proj.weight.grad is not None and mlp.up_proj.weight.grad.abs().sum() > 0
    assert mlp.gate_proj.weight.grad is not None and mlp.gate_proj.weight.grad.abs().sum() > 0


@pytest.mark.runs_on(["cpu", "mps"])
def test_align_loss_supports_non_reentrant_gradient_checkpointing():
    model = CheckpointTinyModel(use_reentrant=False)
    aligner = _make_aligner(model)
    input_ids = torch.tensor([[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4]])

    aligner.set_batch(input_ids=input_ids, labels=labels)
    model(torch.randn(1, 6, 2, requires_grad=True))
    loss = aligner.compute_alignment_loss()
    loss.backward()

    mlp = model.layers[0].mlp
    assert loss.requires_grad
    assert mlp.up_proj.weight.grad is not None and mlp.up_proj.weight.grad.abs().sum() > 0
    assert not aligner._act_store


@pytest.mark.runs_on(["cpu", "mps"])
def test_align_loss_rejects_reentrant_gradient_checkpointing():
    model = CheckpointTinyModel(use_reentrant=True)
    aligner = _make_aligner(model)
    input_ids = torch.tensor([[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4]])

    aligner.set_batch(input_ids=input_ids, labels=labels)
    model(torch.randn(1, 6, 2, requires_grad=True))

    with pytest.raises(RuntimeError, match="requires_grad=False"):
        aligner.compute_alignment_loss()


@pytest.mark.runs_on(["cpu", "mps"])
def test_align_fail_fast_on_detached_activations():
    model = TinyModel(num_layers=2)
    aligner = _make_aligner(model)
    input_ids = torch.tensor([[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4]])

    aligner.set_batch(input_ids=input_ids, labels=labels)
    x = torch.randn(1, 6, 2)
    with torch.no_grad():
        model(x)

    with pytest.raises(RuntimeError, match="requires_grad=False"):
        aligner.compute_alignment_loss()

    aligner.remove_hooks()


@pytest.mark.runs_on(["cpu", "mps"])
def test_align_loss_allows_no_grad_evaluation():
    model = TinyModel(num_layers=1).eval()
    aligner = _make_aligner(model)
    input_ids = torch.tensor([[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4]])

    aligner.set_batch(input_ids=input_ids, labels=labels)
    with torch.no_grad():
        model(torch.randn(1, 6, 2))
        loss = aligner.compute_alignment_loss()

    assert not loss.requires_grad


@pytest.mark.runs_on(["cpu", "mps"])
def test_cluster_align_separates_question_and_answer_distributions():
    model = TinyModel(num_layers=1)
    cluster_idx = [[{"anchor": 0, "neuron": [0, 1]}, {"anchor": 2, "neuron": [2, 3]}]]
    aligner = _make_aligner(
        model,
        cluster_idx=cluster_idx,
        align_mode="cluster",
        align_lambda=0.1,
        align_cluster_question_weight=1.0,
        align_cluster_answer_weight=0.5,
    )
    input_ids = torch.tensor([[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 10, 11, 1, 2]])
    labels = torch.tensor([[-100, -100, -100, -100, 1, 2]])
    activations = torch.tensor(
        [
            [
                [8.0, 8.0, 1.0, 1.0],
                [8.0, 8.0, 1.0, 1.0],
                [1.0, 1.0, 8.0, 8.0],
                [1.0, 1.0, 8.0, 8.0],
                [8.0, 8.0, 1.0, 1.0],
                [8.0, 8.0, 1.0, 1.0],
            ]
        ],
        requires_grad=True,
    )

    aligner.set_batch(input_ids=input_ids, labels=labels)
    aligner._act_store[0] = activations
    weighted_loss, raw_loss = aligner.compute_alignment_loss(return_raw_loss=True)
    logs = aligner.get_log()

    assert weighted_loss.requires_grad
    assert torch.allclose(weighted_loss, 0.1 * raw_loss)
    assert logs["align_cluster_js_question"] > 0.0
    assert logs["align_cluster_js_answer"] == pytest.approx(0.0, abs=1e-7)
    assert logs["align_cluster_layers"] == 1.0
    weighted_loss.backward()
    assert activations.grad is not None and activations.grad.abs().sum() > 0


@pytest.mark.runs_on(["cpu", "mps"])
def test_cluster_align_uses_mean_salience_per_cluster():
    model = TinyModel(num_layers=1)
    cluster_idx = [[{"anchor": 0, "neuron": [0]}, {"anchor": 1, "neuron": [1, 2, 3]}]]
    aligner = _make_aligner(model, cluster_idx=cluster_idx, align_mode="cluster")

    distribution = aligner._cluster_distribution(torch.ones(1, 4), layer_position=0)

    assert torch.allclose(distribution, torch.tensor([[0.5, 0.5]]))


@pytest.mark.runs_on(["cpu", "mps"])
def test_cluster_align_can_skip_early_layers():
    model = TinyModel(num_layers=2)
    clusters = [{"anchor": 0, "neuron": [0, 1]}, {"anchor": 2, "neuron": [2, 3]}]
    aligner = _make_aligner(
        model,
        cluster_idx=[None, clusters],
        align_mode="cluster",
        align_layer_start_ratio=0.5,
    )
    input_ids = torch.tensor([[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 10, 11, 1, 2]])
    labels = torch.tensor([[-100, -100, -100, -100, 1, 2]])

    aligner.set_batch(input_ids=input_ids, labels=labels)
    model(torch.randn(1, 6, 2, requires_grad=True))
    loss = aligner.compute_alignment_loss()

    assert loss.requires_grad
    assert aligner.get_log()["align_cluster_layers"] == 1.0
