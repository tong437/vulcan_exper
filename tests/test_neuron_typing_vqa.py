from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_vqa import (  # noqa: E402
    binary_candidate_logprobs,
    build_shuffled_image_control,
    compute_random_control_comparisons,
    load_binary_records,
    paired_binary_analysis,
    select_binary_records,
)
from run_phase2_ablation import AblationSpec  # noqa: E402


def test_official_pope_json_lines_and_complete_image_selection(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for image_index in range(3):
        image_name = f"image_{image_index}.jpg"
        (image_root / image_name).touch()
        for question_index, label in enumerate(("yes", "no")):
            rows.append(
                {
                    "question_id": image_index * 2 + question_index,
                    "image": image_name,
                    "text": f"Question {question_index}?",
                    "label": label,
                }
            )
    pope_file = tmp_path / "pope.json"
    pope_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    records = load_binary_records(str(pope_file), str(image_root))
    selected, summary = select_binary_records(
        records,
        excluded_image_ids={"image_0.jpg"},
        max_images=1,
    )

    assert len(records) == 6
    assert len(selected) == 2
    assert {row["image_id"] if "image_id" in row else Path(row["images"][0]).name for row in selected} == {
        "image_1.jpg"
    }
    assert summary == {
        "input_rows": 6,
        "input_unique_images": 3,
        "excluded_rows": 2,
        "excluded_unique_images": 1,
        "eligible_rows": 4,
        "eligible_unique_images": 2,
        "selected_rows": 2,
        "selected_unique_images": 1,
        "selection_unit": "image",
        "yes_count": 1,
        "no_count": 1,
    }


def test_image_group_selection_rejects_question_slice_options():
    records = [{"images": ["a.jpg"], "answer": "yes"}]
    with pytest.raises(ValueError, match="not both"):
        select_binary_records(records, max_images=1, max_samples=1)


def test_shuffled_image_control_is_group_consistent_and_deranged():
    records = [
        {
            "source_index": index,
            "question_id": index,
            "images": [f"image_{index // 2}.jpg"],
            "question": "Question?",
            "answer": "yes",
        }
        for index in range(6)
    ]
    shuffled = build_shuffled_image_control(records, seed=7)

    mapping = {}
    for original, changed in zip(records, shuffled):
        source_image = original["images"][0]
        target_image = changed["images"][0]
        assert source_image != target_image
        mapping.setdefault(source_image, target_image)
        assert mapping[source_image] == target_image
    assert len(set(mapping.values())) == len(mapping)


def _prediction(index: int, image_id: str, answer: str, prediction: str) -> dict:
    return {
        "source_index": index,
        "image_id": image_id,
        "answer": answer,
        "prediction": prediction,
    }


def test_paired_binary_analysis_uses_image_cluster_bootstrap():
    baseline = [
        _prediction(0, "a.jpg", "yes", "yes"),
        _prediction(1, "a.jpg", "no", "no"),
        _prediction(2, "b.jpg", "yes", "yes"),
        _prediction(3, "b.jpg", "no", "no"),
    ]
    condition = [
        _prediction(0, "a.jpg", "yes", "yes"),
        _prediction(1, "a.jpg", "no", "no"),
        _prediction(2, "b.jpg", "yes", "no"),
        _prediction(3, "b.jpg", "no", "yes"),
    ]

    result = paired_binary_analysis(baseline, condition, num_bootstrap=500, seed=3)

    assert result["delta_accuracy"] == pytest.approx(-0.5)
    assert result["baseline_only_correct"] == 2
    assert result["condition_only_correct"] == 0
    assert result["bootstrap_unit"] == "image"
    assert result["num_images"] == 2
    assert result["delta_accuracy_ci95"] == pytest.approx([-1.0, 0.0])


def test_random_control_comparison_uses_lower_accuracy_as_more_damage():
    target = AblationSpec(name="multimodal", ratio=0.05)
    controls = [
        AblationSpec(name="matched_random", ratio=0.05, seed=1, reference=target),
        AblationSpec(name="matched_random", ratio=0.05, seed=2, reference=target),
    ]
    metrics = {
        target.result_name: {"delta_accuracy": -0.2, "delta_f1": -0.3},
        controls[0].result_name: {"delta_accuracy": -0.1, "delta_f1": -0.1},
        controls[1].result_name: {"delta_accuracy": 0.0, "delta_f1": -0.2},
    }

    result = compute_random_control_comparisons([target, *controls], metrics)[target.result_name]

    assert result["delta_accuracy"]["relative_damage"] == pytest.approx(0.15)
    assert result["delta_accuracy"]["empirical_p_more_damaging"] == pytest.approx(1 / 3)
    assert result["delta_f1"]["relative_damage"] == pytest.approx(0.15)


def test_binary_candidate_scores_one_forward_at_each_last_unpadded_token():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return {"yes": [1], "no": [2]}[text]

    class Model:
        calls = 0

        def __call__(self, input_ids, attention_mask, use_cache):
            self.calls += 1
            logits = torch.zeros((*input_ids.shape, 4))
            logits[0, 2, 1] = 3
            logits[0, 2, 2] = 1
            logits[1, 1, 1] = 1
            logits[1, 1, 2] = 3
            return SimpleNamespace(logits=logits)

    model = Model()
    inputs = {
        "input_ids": torch.tensor([[4, 5, 6], [7, 8, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
    }
    yes_scores, no_scores = binary_candidate_logprobs(model, Tokenizer(), inputs)

    assert model.calls == 1
    assert yes_scores[0] > no_scores[0]
    assert yes_scores[1] < no_scores[1]
