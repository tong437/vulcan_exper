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

import numpy as np
import pytest
import torch
from neuron_typing_generalization.ablation_runner import evaluation_prompt, summarize_ablation_rows
from neuron_typing_generalization.advisor_ablation import (
    DownProjectionAblation,
    build_ablation_conditions,
    chartqa_relaxed_accuracy,
    extract_choice,
    normalize_vqa_answer,
    per_layer_deletion_count,
    response_strength,
    score_prediction,
    select_ablation_indices,
    textvqa_soft_accuracy,
)
from neuron_typing_generalization.advisor_prompts import mmbench_selection_prompt
from neuron_typing_generalization.interfaces import FFNBinding


def test_condition_expansion_has_14_concepts_and_59_executions() -> None:
    conditions = build_ablation_conditions(random_repeats=10)
    assert len(conditions) == 59
    assert len({condition.condition for condition in conditions}) == 14
    assert sum(condition.random for condition in conditions) == 50


def test_selects_fixed_count_and_advisor_strength_order() -> None:
    q = np.array([[[0.8, 0.1, 0.7, 0.2], [0.1, 0.9, 0.3, 0.8]]], dtype=np.float64)
    classes = np.array([[1, 1, 1, 1]], dtype=np.int8)
    assert per_layer_deletion_count(4, 0.5) == 2
    np.testing.assert_allclose(response_strength(q[0], "V"), q[0, 0])
    selected = select_ablation_indices(
        q,
        classes,
        [4],
        fraction=0.5,
        state="V",
        method="max",
        seed=42,
    )
    np.testing.assert_array_equal(selected[0], [0, 2])
    selected = select_ablation_indices(
        q,
        classes,
        [4],
        fraction=0.5,
        state="V",
        method="min",
        seed=42,
    )
    np.testing.assert_array_equal(selected[0], [1, 3])


def test_state_pool_shortage_is_rejected() -> None:
    q = np.ones((1, 2, 4), dtype=np.float64)
    classes = np.array([[1, 0, 0, 0]], dtype=np.int8)
    with pytest.raises(ValueError, match="only 1 V candidates"):
        select_ablation_indices(q, classes, [4], fraction=0.5, state="V", method="random", seed=1)


@pytest.mark.parametrize("method", ["random", "min", "max"])
def test_state_pool_shortage_can_delete_all_candidates(method: str) -> None:
    q = np.ones((1, 2, 4), dtype=np.float64)
    classes = np.array([[1, 0, 0, 0]], dtype=np.int8)
    selected = select_ablation_indices(
        q, classes, [4], fraction=0.5, state="V", method=method, seed=1, insufficient_pool="all"
    )
    np.testing.assert_array_equal(selected[0], [0])


def test_state_pool_shortage_can_delete_zero_candidates() -> None:
    q = np.ones((1, 2, 4), dtype=np.float64)
    classes = np.zeros((1, 4), dtype=np.int8)
    selected = select_ablation_indices(
        q, classes, [4], fraction=0.5, state="V", method="min", seed=1, insufficient_pool="all"
    )
    assert selected[0].size == 0


def test_down_projection_hook_matches_zeroed_weight_columns() -> None:
    projection = torch.nn.Linear(4, 2, bias=False)
    projection.weight.data.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]))
    binding = FFNBinding(layer=0, down_projection=projection, d_model=2, d_ff=4)
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected_weight = projection.weight.detach().clone()
    expected_weight[:, [1, 3]] = 0
    with DownProjectionAblation([binding], {0: np.array([1, 3])}):
        actual = projection(values)
    torch.testing.assert_close(actual, values @ expected_weight.T)
    torch.testing.assert_close(projection.weight, torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]))


def test_official_style_vqa_normalization_and_leave_one_out_score() -> None:
    assert normalize_vqa_answer("The two, dogs!") == "2 dogs"
    answers = ["cat", "cat", "dog", "dog", "dog", "dog", "dog", "dog", "dog", "dog"]
    assert textvqa_soft_accuracy("cat", answers) == pytest.approx(0.6)
    assert textvqa_soft_accuracy("dog", answers) == 1.0


def test_chartqa_relaxed_accuracy_and_choice_extraction() -> None:
    assert chartqa_relaxed_accuracy("104", "100") == 1.0
    assert chartqa_relaxed_accuracy("106", "100") == 0.0
    assert chartqa_relaxed_accuracy("50%", "0.5") == 1.0
    assert chartqa_relaxed_accuracy("Blue", "blue") == 1.0
    assert extract_choice("The answer is (C).") == "C"


def test_textvqa_dispatch_accepts_manifest_numpy_answers() -> None:
    answers = np.array(["cat"] * 4 + ["dog"] * 6, dtype=object)
    assert score_prediction("textvqa", "cat", answers) == 1.0


def test_mmbench_selection_prompt_contains_only_question_and_valid_choices() -> None:
    sample = {"sample_id": "mmbench:1", "question": "Which one?"}
    replay = {"hint": "Inspect the diagram.", "choices": {"A": "Left", "B": "Right", "C": "nan"}}
    selection_prompt = mmbench_selection_prompt(sample, replay)
    assert selection_prompt == "Which one?\n\nA. Left\n\nB. Right"
    assert "Inspect the diagram." not in selection_prompt
    assert "Answer with only" not in selection_prompt

    generation_prompt = evaluation_prompt("mmbench", sample, replay)
    assert "Inspect the diagram." not in generation_prompt
    assert "A. Left" in generation_prompt and "B. Right" in generation_prompt and "C." not in generation_prompt
    assert generation_prompt.endswith("Answer with only the option letter.")


def test_ablation_summary_uses_run_level_standard_deviation() -> None:
    rows = []
    for condition, random, scores in (("baseline", False, [[1.0, 1.0]]), ("global_random", True, [[0.8, 0.6], [0.6, 0.4]])):
        for repeat, repeat_scores in enumerate(scores):
            for sample_index, score in enumerate(repeat_scores):
                rows.append(
                    {
                        "condition": condition,
                        "state": None,
                        "method": "baseline" if condition == "baseline" else "random",
                        "repeat": repeat,
                        "random": random,
                        "sample_index": sample_index,
                        "score": score,
                    }
                )
    summary = summarize_ablation_rows(rows, expected_samples=2, removal_fraction=0.01, random_repeats=2)
    lookup = {row["condition"]: row for row in summary["conditions"]}
    assert summary["baseline_score"] == 1.0
    assert lookup["global_random"]["score_mean"] == pytest.approx(0.6)
    assert lookup["global_random"]["score_std"] == pytest.approx(np.sqrt(0.02))


def test_ablation_summary_records_actual_deletion_shortfall() -> None:
    rows = []
    for condition, method, actual in (("baseline", "baseline", 0), ("T_min", "min", 1)):
        for sample_index in range(2):
            rows.append(
                {
                    "condition": condition,
                    "state": None if condition == "baseline" else "T",
                    "method": method,
                    "repeat": 0,
                    "random": False,
                    "sample_index": sample_index,
                    "score": 1.0,
                    "selected_neurons": actual,
                    "target_neurons": 0 if condition == "baseline" else 2,
                }
            )
    summary = summarize_ablation_rows(rows, expected_samples=2, removal_fraction=0.05, random_repeats=10)
    item = next(row for row in summary["conditions"] if row["condition"] == "T_min")
    assert item["selected_neurons_mean"] == 1.0
    assert item["target_neurons"] == 2
    assert item["shortfall_prediction_count"] == 2
