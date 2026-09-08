#!/usr/bin/env python3
"""Held-out binary VQA evaluation with image input and typed FFN ablation."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_guard import (  # noqa: E402
    assert_disjoint_manifests,
    build_dataset_manifest,
    normalize_image_id,
    save_manifest,
)
from run_phase2_ablation import (  # noqa: E402
    AblationSpec,
    MLPNeuronAblator,
    build_type_mask,
    get_layer_dims,
    infer_score_columns,
    parse_ablation_spec,
    read_score_table,
    summarize_masks,
    verify_mask_nesting,
    verify_matched_random_controls,
    verify_matched_score_controls,
)

from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary VQA forced-choice evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help="Optional model-path override, used by Phase-3 structural checkpoint evaluation.",
    )
    parser.add_argument("--adapter_name_or_path", default=None, help="Optional LoRA adapter-path override.")
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--vqa_file", required=True, help="JSON/JSONL with image, question, answer fields.")
    parser.add_argument("--image_root", default=None, help="Root directory for relative image paths.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Number of question rows after filtering.")
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Select complete image groups instead of truncating question rows (recommended for POPE).",
    )
    parser.add_argument("--image_offset", type=int, default=0)
    parser.add_argument("--ablation", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument(
        "--exclude_manifest",
        action="append",
        default=[],
        help="Additional manifest whose images must be excluded; repeatable.",
    )
    parser.add_argument(
        "--filter_manifest_overlaps",
        action="store_true",
        help="Remove all rows whose image appears in a comparison manifest before selecting the evaluation slice.",
    )
    parser.add_argument("--require_data_isolation", action="store_true")
    parser.add_argument("--max_image_repeat", type=int, default=6)
    parser.add_argument("--allow_excessive_image_repeats", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--include_shuffled_image_control",
        action="store_true",
        help="Evaluate the no-ablation model after deterministically assigning every question a different image.",
    )
    parser.add_argument("--shuffled_image_seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true", help="Resume completed conditions from output_file.")
    parser.add_argument(
        "--reuse_metrics_from",
        default=None,
        help=(
            "Seed a new evaluation with compatible completed conditions from another output. "
            "Dataset identities and every reused neuron mask are verified exactly."
        ),
    )
    return parser.parse_args()


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Official POPE files use JSON Lines despite their `.json` suffix.
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        for key in ("questions", "data", "annotations"):
            if isinstance(data.get(key), list):
                return data[key]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {source}.")
    return data


def _first_value(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return None


def normalize_binary_answer(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("answer") or value.get("multiple_choice_answer")
    if value is None:
        return None
    match = re.search(r"\b(yes|no)\b", str(value).strip().lower())
    return match.group(1) if match else None


def load_binary_records(
    path: str,
    image_root: str | None = None,
) -> list[dict[str, Any]]:
    records = _read_records(path)
    normalized = []
    for source_index, record in enumerate(records):
        image_value = _first_value(record, ("image", "image_path", "images", "file_name"))
        if isinstance(image_value, list):
            image_value = image_value[0] if image_value else None
        question = _first_value(record, ("question", "text", "prompt"))
        answer = normalize_binary_answer(_first_value(record, ("answer", "answers", "label")))
        if isinstance(record.get("messages"), list):
            messages = record["messages"]
            user_message = next((item for item in messages if item.get("role") == "user"), None)
            assistant_message = next((item for item in messages if item.get("role") == "assistant"), None)
            if user_message is not None:
                question = str(user_message.get("content", "")).replace("<image>", "").strip()
            if assistant_message is not None:
                answer = normalize_binary_answer(assistant_message.get("content"))
        if not image_value or not question or answer is None:
            raise ValueError(f"Invalid binary VQA record at source index {source_index}: {record}")
        image_path = Path(str(image_value))
        if not image_path.is_absolute() and image_root:
            image_path = Path(image_root) / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for source index {source_index}: {image_path}")
        normalized.append(
            {
                "source_index": source_index,
                "question_id": _first_value(record, ("question_id", "id")) or source_index,
                "images": [str(image_path)],
                "question": str(question),
                "answer": answer,
            }
        )
    return normalized


def _manifest_image_ids(paths: list[str]) -> set[str]:
    image_ids: set[str] = set()
    for path in paths:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        image_ids.update(normalize_image_id(value) for value in manifest.get("image_ids", []))
    return image_ids


def select_binary_records(
    records: list[dict[str, Any]],
    *,
    excluded_image_ids: set[str] | None = None,
    sample_offset: int = 0,
    max_samples: int | None = None,
    image_offset: int = 0,
    max_images: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter overlaps, then select either question rows or complete image groups."""
    if sample_offset < 0 or image_offset < 0:
        raise ValueError("sample_offset and image_offset must be non-negative.")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when provided.")
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive when provided.")
    if max_images is not None and (max_samples is not None or sample_offset != 0):
        raise ValueError("Use image_offset/max_images or sample_offset/max_samples, not both.")

    excluded = excluded_image_ids or set()
    eligible = [record for record in records if normalize_image_id(record["images"][0]) not in excluded]
    excluded_rows = len(records) - len(eligible)
    excluded_images = {
        normalize_image_id(record["images"][0])
        for record in records
        if normalize_image_id(record["images"][0]) in excluded
    }

    if max_images is not None:
        ordered_images = list(dict.fromkeys(normalize_image_id(record["images"][0]) for record in eligible))
        end = image_offset + max_images
        if end > len(ordered_images):
            raise ValueError(
                f"Requested image groups [{image_offset}:{end}], but only {len(ordered_images)} remain after filtering."
            )
        selected_ids = set(ordered_images[image_offset:end])
        selected = [record for record in eligible if normalize_image_id(record["images"][0]) in selected_ids]
        selection_unit = "image"
    else:
        end = len(eligible) if max_samples is None else sample_offset + max_samples
        if end > len(eligible):
            raise ValueError(
                f"Requested records [{sample_offset}:{end}], but only {len(eligible)} remain after filtering."
            )
        selected = eligible[sample_offset:end]
        selection_unit = "question"

    return selected, {
        "input_rows": len(records),
        "input_unique_images": len({normalize_image_id(record["images"][0]) for record in records}),
        "excluded_rows": excluded_rows,
        "excluded_unique_images": len(excluded_images),
        "eligible_rows": len(eligible),
        "eligible_unique_images": len({normalize_image_id(record["images"][0]) for record in eligible}),
        "selected_rows": len(selected),
        "selected_unique_images": len({normalize_image_id(record["images"][0]) for record in selected}),
        "selection_unit": selection_unit,
        "yes_count": sum(record["answer"] == "yes" for record in selected),
        "no_count": sum(record["answer"] == "no" for record in selected),
    }


def load_binary_dataset(
    path: str,
    offset: int,
    max_samples: int | None,
    image_root: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible row selection helper used by unit tests and callers."""
    records = load_binary_records(path, image_root)
    selected, _ = select_binary_records(records, sample_offset=offset, max_samples=max_samples)
    return selected


def load_ablation_masks(
    score_file: str,
    specs: list[str],
    seed: int,
) -> tuple[list[AblationSpec], dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    table = read_score_table(score_file)
    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(table, None)
    layer_dims = get_layer_dims(table, layer_col, neuron_col)
    parsed_specs = [parse_ablation_spec(text, seed) for text in specs]
    if any(spec.name == "none" for spec in parsed_specs):
        raise ValueError("Do not pass --ablation none; the evaluator always inserts its own baseline.")
    result: dict[str, dict[int, torch.Tensor]] = {}
    for spec in parsed_specs:
        result[spec.result_name] = build_type_mask(
            table,
            spec,
            layer_col,
            neuron_col,
            score_cols,
            activation_col,
            layer_dims,
            None,
            "per_layer",
            1.0,
            0.0,
        )
    matched_counts, matched_partitions = verify_matched_random_controls(parsed_specs, result)
    matched_scores = verify_matched_score_controls(parsed_specs, result)
    nesting: dict[str, dict[str, bool]] = {}
    for type_name in ("visual", "text", "multimodal", "unknown", "unknown_safe"):
        masks_for_type = {
            spec.ratio: result[spec.result_name] for spec in parsed_specs if spec.name == type_name and spec.ratio > 0
        }
        if len(masks_for_type) > 1:
            nesting[type_name] = verify_mask_nesting(masks_for_type)
    verification = {
        "mask_summaries": {name: summarize_masks(mask) for name, mask in result.items()},
        "nesting_verification": nesting,
        "matched_random_verification": matched_counts,
        "matched_partition_verification": matched_partitions,
        "matched_score_verification": matched_scores,
    }
    return parsed_specs, result, verification


def build_shuffled_image_control(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Assign each image group a different image while preserving questions and labels."""
    image_paths = list(dict.fromkeys(record["images"][0] for record in records))
    if len(image_paths) < 2:
        raise ValueError("A shuffled-image control requires at least two unique images.")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(1, len(image_paths)))
    shifted = image_paths[shift:] + image_paths[:shift]
    mapping = dict(zip(image_paths, shifted))
    shuffled = deepcopy(records)
    for record in shuffled:
        original = record["images"][0]
        record["original_image"] = original
        record["images"] = [mapping[original]]
        if normalize_image_id(original) == normalize_image_id(record["images"][0]):
            raise RuntimeError("Shuffled-image control left an image unchanged.")
    return shuffled


def prepare_model_inputs(processor, record: dict[str, Any], device: torch.device) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{record['question']} Answer with only yes or no."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(record["images"][0]) as image:
        inputs = processor(text=[prompt], images=[image.convert("RGB")], return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def prepare_model_batch(processor, records: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    prompts = []
    images = []
    for record in records:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{record['question']} Answer with only yes or no."},
                ],
            }
        ]
        prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        with Image.open(record["images"][0]) as image:
            images.append(image.convert("RGB"))
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


@torch.no_grad()
def candidate_logprob(model, tokenizer, inputs: dict[str, Any], candidate: str) -> float:
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if not candidate_ids:
        raise ValueError(f"Candidate {candidate!r} produced no tokens.")
    input_ids = inputs["input_ids"]
    candidate_tensor = torch.tensor([candidate_ids], device=input_ids.device, dtype=input_ids.dtype)
    full_ids = torch.cat([input_ids, candidate_tensor], dim=1)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    full_attention = torch.cat([attention_mask, torch.ones_like(candidate_tensor)], dim=1)
    model_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
    }
    outputs = model(input_ids=full_ids, attention_mask=full_attention, use_cache=False, **model_inputs)
    start = input_ids.shape[1] - 1
    logits = outputs.logits[:, start : start + len(candidate_ids), :].float()
    labels = candidate_tensor
    return float(F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).sum())


@torch.no_grad()
def binary_candidate_logprobs(model, tokenizer, inputs: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Score the single-token `yes` and `no` candidates in one batched forward pass."""
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(
            "The optimized binary evaluator requires single-token `yes` and `no` candidates; "
            f"got yes={yes_ids}, no={no_ids}."
        )
    outputs = model(**inputs, use_cache=False)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        last_positions = torch.full(
            (inputs["input_ids"].shape[0],),
            inputs["input_ids"].shape[1] - 1,
            device=inputs["input_ids"].device,
            dtype=torch.long,
        )
    else:
        last_positions = attention_mask.long().sum(dim=1) - 1
    batch_indices = torch.arange(inputs["input_ids"].shape[0], device=last_positions.device)
    next_token_logits = outputs.logits[batch_indices, last_positions].float()
    log_probs = F.log_softmax(next_token_logits, dim=-1)
    return log_probs[:, yes_ids[0]].tolist(), log_probs[:, no_ids[0]].tolist()


@torch.no_grad()
def evaluate_condition(model, tokenizer, processor, records, device, batch_size: int = 1) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    predictions = []
    tp = fp = tn = fn = 0
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        inputs = prepare_model_batch(processor, batch_records, device)
        yes_scores, no_scores = binary_candidate_logprobs(model, tokenizer, inputs)
        for record, yes_score, no_score in zip(batch_records, yes_scores, no_scores):
            prediction = "yes" if yes_score > no_score else "no"
            answer = record["answer"]
            tp += prediction == "yes" and answer == "yes"
            fp += prediction == "yes" and answer == "no"
            tn += prediction == "no" and answer == "no"
            fn += prediction == "no" and answer == "yes"
            predictions.append(
                {
                    "source_index": record["source_index"],
                    "question_id": record["question_id"],
                    "image": record["images"][0],
                    "image_id": normalize_image_id(record["images"][0]),
                    "question": record["question"],
                    "answer": answer,
                    "prediction": prediction,
                    "correct": prediction == answer,
                    "yes_logprob": yes_score,
                    "no_logprob": no_score,
                    "margin": yes_score - no_score,
                }
            )
    total = len(records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "total": total,
        "correct": tp + tn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "yes_ratio": (tp + fp) / total if total else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "predictions": predictions,
    }


def _binary_metrics(answers: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    tp = int(np.sum(predictions & answers))
    fp = int(np.sum(predictions & ~answers))
    fn = int(np.sum(~predictions & answers))
    accuracy = float(np.mean(predictions == answers)) if len(answers) else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return accuracy, f1


def _mcnemar_exact_p(baseline_only_correct: int, condition_only_correct: int) -> float:
    discordant = baseline_only_correct + condition_only_correct
    if discordant == 0:
        return 1.0
    tail = min(baseline_only_correct, condition_only_correct)
    log_probabilities = [
        math.lgamma(discordant + 1) - math.lgamma(k + 1) - math.lgamma(discordant - k + 1) - discordant * math.log(2)
        for k in range(tail + 1)
    ]
    max_log = max(log_probabilities)
    one_sided = math.exp(max_log) * sum(math.exp(value - max_log) for value in log_probabilities)
    return min(1.0, 2 * one_sided)


def paired_binary_analysis(
    baseline_predictions: list[dict[str, Any]],
    condition_predictions: list[dict[str, Any]],
    *,
    num_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Paired accuracy/F1 analysis with an image-cluster bootstrap."""
    if len(baseline_predictions) != len(condition_predictions):
        raise ValueError("Paired conditions have different numbers of predictions.")
    for baseline, condition in zip(baseline_predictions, condition_predictions):
        if baseline["source_index"] != condition["source_index"]:
            raise ValueError("Paired conditions are not aligned by source_index.")

    answers = np.array([row["answer"] == "yes" for row in baseline_predictions], dtype=bool)
    baseline = np.array([row["prediction"] == "yes" for row in baseline_predictions], dtype=bool)
    condition = np.array([row["prediction"] == "yes" for row in condition_predictions], dtype=bool)
    baseline_correct = baseline == answers
    condition_correct = condition == answers
    base_accuracy, base_f1 = _binary_metrics(answers, baseline)
    condition_accuracy, condition_f1 = _binary_metrics(answers, condition)
    baseline_only_correct = int(np.sum(baseline_correct & ~condition_correct))
    condition_only_correct = int(np.sum(~baseline_correct & condition_correct))

    result: dict[str, Any] = {
        "delta_accuracy": condition_accuracy - base_accuracy,
        "delta_f1": condition_f1 - base_f1,
        "baseline_only_correct": baseline_only_correct,
        "condition_only_correct": condition_only_correct,
        "unchanged_correctness": int(np.sum(baseline_correct == condition_correct)),
        "mcnemar_exact_p_two_sided": _mcnemar_exact_p(baseline_only_correct, condition_only_correct),
        "bootstrap_unit": "image",
        "num_images": len({row["image_id"] for row in baseline_predictions}),
        "num_questions": len(baseline_predictions),
    }
    if num_bootstrap <= 0:
        return result

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(baseline_predictions):
        groups[row["image_id"]].append(index)
    group_indices = [np.asarray(indices, dtype=np.int64) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    delta_accuracy = np.empty(num_bootstrap, dtype=np.float64)
    delta_f1 = np.empty(num_bootstrap, dtype=np.float64)
    for bootstrap_index in range(num_bootstrap):
        sampled_groups = rng.integers(0, len(group_indices), size=len(group_indices))
        indices = np.concatenate([group_indices[index] for index in sampled_groups])
        base_acc, base_sample_f1 = _binary_metrics(answers[indices], baseline[indices])
        condition_acc, condition_sample_f1 = _binary_metrics(answers[indices], condition[indices])
        delta_accuracy[bootstrap_index] = condition_acc - base_acc
        delta_f1[bootstrap_index] = condition_sample_f1 - base_sample_f1
    result.update(
        {
            "delta_accuracy_ci95": np.percentile(delta_accuracy, [2.5, 97.5]).tolist(),
            "delta_f1_ci95": np.percentile(delta_f1, [2.5, 97.5]).tolist(),
            "bootstrap_samples": num_bootstrap,
            "bootstrap_seed": seed,
        }
    )
    return result


def compute_random_control_comparisons(
    specs: list[AblationSpec],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compare typed accuracy/F1 changes with their exact-count random controls."""
    comparisons: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.name in {"none", "random", "layer_random", "matched_random", "matched_score"}:
            continue
        controls = [
            candidate.result_name
            for candidate in specs
            if candidate.name == "matched_random"
            and candidate.reference is not None
            and candidate.reference.result_name == spec.result_name
        ]
        if not controls:
            continue
        row: dict[str, Any] = {"control_type": "matched_random", "num_controls": len(controls)}
        for metric_name in ("delta_accuracy", "delta_f1"):
            typed_delta = float(metrics[spec.result_name][metric_name])
            random_deltas = [float(metrics[name][metric_name]) for name in controls]
            # Smaller accuracy/F1 delta means more damage.
            row[metric_name] = {
                "typed_delta": typed_delta,
                "mean_random_delta": float(np.mean(random_deltas)),
                "median_random_delta": float(np.median(random_deltas)),
                "min_random_delta": float(np.min(random_deltas)),
                "max_random_delta": float(np.max(random_deltas)),
                "relative_damage": float(np.mean(random_deltas) - typed_delta),
                "empirical_p_more_damaging": float(
                    (1 + sum(value <= typed_delta for value in random_deltas)) / (len(random_deltas) + 1)
                ),
            }
        comparisons[spec.result_name] = row
    return comparisons


def _evaluation_record_signatures(records: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            int(record["source_index"]),
            str(record["question_id"]),
            normalize_image_id(record["images"][0]),
            str(record["question"]),
            str(record["answer"]),
        )
        for record in records
    ]


def _prediction_record_signatures(predictions: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            int(record["source_index"]),
            str(record["question_id"]),
            normalize_image_id(record.get("image_id") or record["image"]),
            str(record["question"]),
            str(record["answer"]),
        )
        for record in predictions
    ]


def reuse_compatible_metrics(
    reuse_path: str | Path,
    *,
    task_name: str,
    records: list[dict[str, Any]],
    current_masks: dict[str, dict[int, torch.Tensor]],
    required_conditions: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Reuse prior metrics only after exact dataset and mask equivalence checks."""
    source_path = Path(reuse_path)
    previous = json.loads(source_path.read_text(encoding="utf-8"))
    if previous.get("task") != task_name:
        raise ValueError(f"Metric reuse task mismatch: {previous.get('task')!r} != {task_name!r}.")
    previous_metrics = previous.get("metrics", {})
    if "none" not in previous_metrics:
        raise ValueError("Metric reuse source has no baseline condition.")
    previous_predictions = previous_metrics["none"].get("predictions", [])
    if _prediction_record_signatures(previous_predictions) != _evaluation_record_signatures(records):
        raise ValueError("Metric reuse dataset/order mismatch.")

    previous_config = previous.get("config", {})
    previous_score_file = previous_config.get("score_file")
    previous_ablations = previous_config.get("ablation")
    previous_seed = previous_config.get("seed")
    if not previous_score_file or not isinstance(previous_ablations, list) or previous_seed is None:
        raise ValueError("Metric reuse source lacks score-file, ablation, or seed provenance.")
    _, previous_masks, _ = load_ablation_masks(previous_score_file, previous_ablations, int(previous_seed))

    reusable_names = sorted(required_conditions & set(previous_metrics))
    reused: dict[str, dict[str, Any]] = {}
    for name in reusable_names:
        if name != "none":
            if name not in current_masks or name not in previous_masks:
                raise ValueError(f"Metric reuse mask is unavailable for condition {name!r}.")
            current = current_masks[name]
            old = previous_masks[name]
            if set(current) != set(old) or any(not torch.equal(current[layer], old[layer]) for layer in current):
                raise ValueError(f"Metric reuse mask mismatch for condition {name!r}.")
        reused[name] = deepcopy(previous_metrics[name])
    return reused, {
        "source": str(source_path.resolve()),
        "reused_conditions": reusable_names,
        "reused_condition_count": len(reusable_names),
        "dataset_equivalent": True,
        "masks_equivalent": True,
    }


def run_evaluation(args: argparse.Namespace, task_name: str = "vqa") -> dict[str, Any]:
    torch.manual_seed(args.seed)
    comparison_paths = list(
        dict.fromkeys(
            path for path in (args.calibration_manifest, args.typing_manifest, *args.exclude_manifest) if path
        )
    )
    if args.require_data_isolation and not (args.calibration_manifest and args.typing_manifest):
        raise ValueError("Data isolation requires calibration and typing manifests.")
    excluded_image_ids = _manifest_image_ids(comparison_paths) if args.filter_manifest_overlaps else set()
    all_records = load_binary_records(args.vqa_file, args.image_root)
    records, selection_summary = select_binary_records(
        all_records,
        excluded_image_ids=excluded_image_ids,
        sample_offset=args.sample_offset,
        max_samples=args.max_samples,
        image_offset=args.image_offset,
        max_images=args.max_images,
    )
    manifest = build_dataset_manifest(
        records,
        [record["source_index"] for record in records],
        role=f"{task_name}_evaluation",
        dataset_name=args.vqa_file,
        tokenized_path=None,
        max_image_repeat=args.max_image_repeat,
        allow_excessive_image_repeats=args.allow_excessive_image_repeats,
    )
    isolation = assert_disjoint_manifests(manifest, comparison_paths) if comparison_paths else None

    specs, masks, mask_verification = load_ablation_masks(args.score_file, args.ablation, args.seed)
    output_path = Path(args.output_file)
    manifest_path = output_path.with_name(f"{output_path.stem}.sample_manifest.json")
    save_manifest(manifest, manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    previous: dict[str, Any] = {}
    reuse_provenance: dict[str, Any] | None = None
    if args.resume:
        if not output_path.exists():
            raise FileNotFoundError(f"Cannot resume because output_file does not exist: {output_path}")
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        previous_config = {key: value for key, value in previous.get("config", {}).items() if key != "resume"}
        current_config = {key: value for key, value in vars(args).items() if key != "resume"}
        if previous.get("task") != task_name or previous_config != current_config:
            differing_keys = sorted(
                key
                for key in set(previous_config) | set(current_config)
                if previous_config.get(key) != current_config.get(key)
            )
            raise ValueError(f"Resume configuration mismatch for keys: {differing_keys}.")

    conditions: dict[str, dict[str, Any]] = previous.get("metrics", {})
    required_conditions = {"none", *masks}
    if args.reuse_metrics_from:
        if args.resume:
            raise ValueError("Use either --resume or --reuse_metrics_from, not both.")
        conditions, reuse_provenance = reuse_compatible_metrics(
            args.reuse_metrics_from,
            task_name=task_name,
            records=records,
            current_masks=masks,
            required_conditions=required_conditions,
        )
        print(
            f"Reused {reuse_provenance['reused_condition_count']} verified conditions from {args.reuse_metrics_from}",
            flush=True,
        )
    shuffled_control = previous.get("image_control", {}).get("shuffled")

    def save_checkpoint(complete: bool) -> dict[str, Any]:
        result = {
            "task": task_name,
            "complete": complete,
            "config": vars(args),
            "manifest": str(manifest_path),
            "selection_summary": selection_summary,
            "data_isolation": isolation,
            "metric_reuse": reuse_provenance,
            **mask_verification,
            "metrics": conditions,
            "random_control_comparisons": (compute_random_control_comparisons(specs, conditions) if complete else {}),
            "image_control": {"shuffled": shuffled_control} if shuffled_control is not None else {},
        }
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    needs_model = not required_conditions.issubset(conditions) or (
        args.include_shuffled_image_control and shuffled_control is None
    )
    if needs_model:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        config.update({"do_train": False, "do_eval": False, "do_predict": False})
        config.pop("deepspeed", None)
        if getattr(args, "model_name_or_path", None) is not None:
            config["model_name_or_path"] = args.model_name_or_path
        if getattr(args, "adapter_name_or_path", None) is not None:
            config["adapter_name_or_path"] = args.adapter_name_or_path
        config.setdefault("output_dir", f"saves/neuron_typing/{task_name}_tmp")
        model_args, _, _, finetuning_args, _ = get_train_args(config)
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        processor = tokenizer_module.get("processor")
        if processor is None:
            raise RuntimeError("The configured vision-language model did not provide an image processor.")
        model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()

        if "none" not in conditions:
            print("Running condition: none", flush=True)
            conditions["none"] = evaluate_condition(model, tokenizer, processor, records, device, args.batch_size)
            save_checkpoint(complete=False)
        for name, mask in masks.items():
            if name in conditions:
                print(f"Resuming: skipped completed condition {name}", flush=True)
                continue
            print(f"Running condition: {name}", flush=True)
            with MLPNeuronAblator(model, mask):
                conditions[name] = evaluate_condition(model, tokenizer, processor, records, device, args.batch_size)
            conditions[name].update(
                paired_binary_analysis(
                    conditions["none"]["predictions"],
                    conditions[name]["predictions"],
                    num_bootstrap=args.bootstrap_samples,
                    seed=args.bootstrap_seed,
                )
            )
            print(
                f"  accuracy={conditions[name]['accuracy']:.4f} f1={conditions[name]['f1']:.4f} ",
                f"delta_accuracy={conditions[name]['delta_accuracy']:+.4f}",
                flush=True,
            )
            save_checkpoint(complete=False)

        if args.include_shuffled_image_control and shuffled_control is None:
            print("Running image control: shuffled", flush=True)
            shuffled_records = build_shuffled_image_control(records, args.shuffled_image_seed)
            shuffled_control = evaluate_condition(
                model, tokenizer, processor, shuffled_records, device, args.batch_size
            )
            shuffled_control.update(
                paired_binary_analysis(
                    conditions["none"]["predictions"],
                    shuffled_control["predictions"],
                    num_bootstrap=args.bootstrap_samples,
                    seed=args.bootstrap_seed,
                )
            )
            save_checkpoint(complete=False)

    result = save_checkpoint(complete=True)
    return result


def main() -> None:
    result = run_evaluation(parse_args(), task_name="vqa")
    for name, metrics in result["metrics"].items():
        print(f"{name:35s} accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
