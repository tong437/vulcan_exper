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

from phase5_cached_utils import cached_fidelity, detach_qwen_cache, generated_token_margins  # noqa: E402
from run_phase5_cached_gate import (  # noqa: E402
    build_token_loss_weights,
    cached_feasibility,
    cached_token_loss,
    is_better_cached_training_candidate,
    rebuild_best,
)


class FakeCache:
    def __init__(self):
        source = torch.ones(2, requires_grad=True) * 2
        self.key_cache = [source]
        self.value_cache = [source + 1]
        self.conv_states = [None]
        self.recurrent_states = [source + 2]


def test_qwen_cache_detach_breaks_autograd_history():
    cache = detach_qwen_cache(FakeCache())
    assert all(value.grad_fn is None for value in cache.key_cache + cache.value_cache if value is not None)
    assert cache.recurrent_states[0].grad_fn is None
    with pytest.raises(TypeError, match="Unsupported cache"):
        detach_qwen_cache(object())


def test_cached_token_loss_has_finite_gradient():
    student = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([2.0, 0.0, -2.0])
    objective, kl_loss, margin_loss, margin = cached_token_loss(
        student, teacher, torch.tensor(0), margin_weight=0.1, margin_target=0.5
    )
    objective.backward()
    assert float(kl_loss) > 0
    assert float(margin_loss) == 0
    assert float(margin) == pytest.approx(1.0)
    assert student.grad is not None and bool(torch.isfinite(student.grad).all())


def test_token_loss_weights_focus_interval_and_validate_bounds():
    weights = build_token_loss_weights(8, focus_start=3, focus_end=6, focus_weight=4.0)
    assert weights.tolist() == [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 1.0, 1.0]
    assert build_token_loss_weights(3, focus_start=0, focus_end=None, focus_weight=1.0).tolist() == [1.0] * 3
    with pytest.raises(ValueError, match="focus_token_start"):
        build_token_loss_weights(8, focus_start=8, focus_end=None, focus_weight=2.0)
    with pytest.raises(ValueError, match="focus_token_end"):
        build_token_loss_weights(8, focus_start=3, focus_end=9, focus_weight=2.0)
    with pytest.raises(ValueError, match="focus_token_weight"):
        build_token_loss_weights(8, focus_start=3, focus_end=6, focus_weight=0.0)


def test_token_loss_weights_include_teacher_vulnerabilities_without_multiplying_focus():
    margins = torch.tensor([1.0, 0.0, 1.0, 0.125, 1.0])
    weights = build_token_loss_weights(
        5,
        focus_start=2,
        focus_end=4,
        focus_weight=4.0,
        teacher_margins=margins,
        low_margin_threshold=0.25,
        low_margin_weight=3.0,
    )
    assert weights.tolist() == [1.0, 3.0, 4.0, 4.0, 1.0]


def test_generated_token_margins_exclude_label_from_competitors():
    logits = torch.tensor([[3.0, 1.0, 0.0], [2.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    labels = torch.tensor([0, 1, 1])
    assert generated_token_margins(logits, labels).tolist() == [2.0, 0.0, -1.0]


def test_cached_training_candidate_prioritizes_discrete_trajectory():
    current = {
        "generated_token_agreement_count": 7,
        "teacher_top_agreement_count": 8,
        "min_generated_token_margin": 0.5,
        "unweighted_mean_kl": 0.0001,
        "objective": 0.001,
    }
    more_generated_tokens = {
        **current,
        "generated_token_agreement_count": 8,
        "min_generated_token_margin": -0.1,
        "unweighted_mean_kl": 0.01,
    }
    assert is_better_cached_training_candidate(current, more_generated_tokens)
    assert not is_better_cached_training_candidate(more_generated_tokens, current)


def test_cached_fidelity_and_dual_feasibility():
    teacher = torch.tensor([[3.0, 1.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    exact = cached_fidelity(teacher, teacher.clone(), labels)
    assert exact["mean_kl"] == pytest.approx(0.0, abs=1e-7)
    assert exact["token_agreement"] == 1.0
    assert exact["min_generated_token_margin"] == pytest.approx(2.0)
    assert exact["min_generated_token_margin_indices"] == [0, 1]
    assert exact["tokens_below_margin_0_25"] == 0
    assert exact["tokens_below_margin_0_25_indices"] == []
    strict, behavioral = cached_feasibility(exact, {"exact_match": True}, kl_tolerance=0.001)
    assert strict and behavioral

    diagnostic_mismatch = dict(exact, token_agreement=0.5)
    strict, behavioral = cached_feasibility(diagnostic_mismatch, {"exact_match": True}, kl_tolerance=0.001)
    assert not strict and behavioral


def test_rebuild_best_prefers_budget_then_cached_kl():
    template = {
        "restart": 0,
        "mask_summary": {"pruning_ratio": 0.1},
        "parameter_summary": {"total_parameter_reduction_ratio": 0.01},
        "strict_feasible": True,
        "robust_feasible": True,
    }
    result = {
        "runs": {
            "small": {**template, "deletion_budget": 10, "cached_fidelity": {"mean_kl": 0.0001}},
            "large_bad_kl": {
                **template,
                "deletion_budget": 20,
                "cached_fidelity": {"mean_kl": 0.0009},
            },
            "large_good_kl": {
                **template,
                "deletion_budget": 20,
                "cached_fidelity": {"mean_kl": 0.0007},
            },
        }
    }
    assert rebuild_best(result, "strict_feasible")["run"] == "large_good_kl"


def test_rebuild_best_robust_prefers_minimum_margin_at_same_budget():
    template = {
        "deletion_budget": 20,
        "restart": 0,
        "mask_summary": {"pruning_ratio": 0.1},
        "parameter_summary": {"total_parameter_reduction_ratio": 0.01},
        "robust_feasible": True,
    }
    result = {
        "runs": {
            "low_kl": {
                **template,
                "cached_fidelity": {"mean_kl": 0.0001, "min_generated_token_margin": 0.3},
            },
            "high_margin": {
                **template,
                "cached_fidelity": {"mean_kl": 0.0002, "min_generated_token_margin": 0.6},
            },
        }
    }
    assert rebuild_best(result, "robust_feasible")["run"] == "high_margin"
