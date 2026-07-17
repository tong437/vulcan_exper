#!/usr/bin/env python3
"""Held-out binary VQA evaluation with image input and typed FFN ablation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402

from dataset_guard import assert_disjoint_manifests, build_dataset_manifest, save_manifest  # noqa: E402
from run_phase2_ablation import (  # noqa: E402
    MLPNeuronAblator,
    build_type_mask,
    get_layer_dims,
    infer_score_columns,
    parse_ablation_spec,
    read_score_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary VQA forced-choice evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--vqa_file", required=True, help="JSON/JSONL with image, question, answer fields.")
    parser.add_argument("--image_root", default=None, help="Root directory for relative image paths.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--ablation", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--require_data_isolation", action="store_true")
    parser.add_argument("--max_image_repeat", type=int, default=5)
    parser.add_argument("--allow_excessive_image_repeats", action="store_true")
    return parser.parse_args()


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(source.read_text(encoding="utf-8"))
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


def load_binary_dataset(
    path: str,
    offset: int,
    max_samples: int,
    image_root: str | None = None,
) -> list[dict[str, Any]]:
    records = _read_records(path)
    end = offset + max_samples
    if end > len(records):
        raise ValueError(f"Requested records [{offset}:{end}], but {path} only contains {len(records)}.")

    normalized = []
    for source_index, record in enumerate(records[offset:end], start=offset):
        image_value = _first_value(record, ("image", "image_path", "images", "file_name"))
        if isinstance(image_value, list):
            image_value = image_value[0] if image_value else None
        question = _first_value(record, ("question", "text", "prompt"))
        answer = normalize_binary_answer(_first_value(record, ("answer", "answers", "label")))
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
                "images": [str(image_path)],
                "question": str(question),
                "answer": answer,
            }
        )
    return normalized


def load_ablation_masks(score_file: str, specs: list[str], seed: int) -> dict[str, dict[int, torch.Tensor]]:
    table = read_score_table(score_file)
    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(table, None)
    layer_dims = get_layer_dims(table, layer_col, neuron_col)
    result = {}
    for text in specs:
        spec = parse_ablation_spec(text, seed)
        result[spec.result_name] = build_type_mask(
            table, spec, layer_col, neuron_col, score_cols, activation_col,
            layer_dims, None, "per_layer", 1.0, 0.0,
        )
    return result


def prepare_model_inputs(processor, record: dict[str, Any], device: torch.device) -> dict[str, Any]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"{record['question']} Answer with only yes or no."},
        ],
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(record["images"][0]) as image:
        inputs = processor(text=[prompt], images=[image.convert("RGB")], return_tensors="pt")
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
        key: value for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
    }
    outputs = model(input_ids=full_ids, attention_mask=full_attention, use_cache=False, **model_inputs)
    start = input_ids.shape[1] - 1
    logits = outputs.logits[:, start:start + len(candidate_ids), :].float()
    labels = candidate_tensor
    return float(F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).sum())


@torch.no_grad()
def evaluate_condition(model, tokenizer, processor, records, device) -> dict[str, Any]:
    predictions = []
    tp = fp = tn = fn = 0
    for record in records:
        inputs = prepare_model_inputs(processor, record, device)
        yes_score = candidate_logprob(model, tokenizer, inputs, "yes")
        no_score = candidate_logprob(model, tokenizer, inputs, "no")
        prediction = "yes" if yes_score > no_score else "no"
        answer = record["answer"]
        tp += prediction == "yes" and answer == "yes"
        fp += prediction == "yes" and answer == "no"
        tn += prediction == "no" and answer == "no"
        fn += prediction == "no" and answer == "yes"
        predictions.append({
            "source_index": record["source_index"],
            "image": record["images"][0],
            "question": record["question"],
            "answer": answer,
            "prediction": prediction,
            "correct": prediction == answer,
            "yes_logprob": yes_score,
            "no_logprob": no_score,
            "margin": yes_score - no_score,
        })
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


def run_evaluation(args: argparse.Namespace, task_name: str = "vqa") -> dict[str, Any]:
    torch.manual_seed(args.seed)
    records = load_binary_dataset(args.vqa_file, args.sample_offset, args.max_samples, args.image_root)
    manifest = build_dataset_manifest(
        records,
        [record["source_index"] for record in records],
        role=f"{task_name}_evaluation",
        dataset_name=args.vqa_file,
        tokenized_path=None,
        max_image_repeat=args.max_image_repeat,
        allow_excessive_image_repeats=args.allow_excessive_image_repeats,
    )
    comparison_paths = [path for path in (args.calibration_manifest, args.typing_manifest) if path]
    if args.require_data_isolation and len(comparison_paths) != 2:
        raise ValueError("Data isolation requires calibration and typing manifests.")
    isolation = assert_disjoint_manifests(manifest, comparison_paths) if comparison_paths else None

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config.update({"do_train": False, "do_eval": False, "do_predict": False})
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

    masks = load_ablation_masks(args.score_file, args.ablation, args.seed)
    conditions: dict[str, dict[str, Any]] = {"none": evaluate_condition(model, tokenizer, processor, records, device)}
    for name, mask in masks.items():
        with MLPNeuronAblator(model, mask):
            conditions[name] = evaluate_condition(model, tokenizer, processor, records, device)
        conditions[name]["delta_accuracy"] = conditions[name]["accuracy"] - conditions["none"]["accuracy"]

    output_path = Path(args.output_file)
    manifest_path = output_path.with_name(f"{output_path.stem}.sample_manifest.json")
    save_manifest(manifest, manifest_path)
    result = {
        "task": task_name,
        "config": vars(args),
        "manifest": str(manifest_path),
        "data_isolation": isolation,
        "metrics": conditions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    result = run_evaluation(parse_args(), task_name="vqa")
    for name, metrics in result["metrics"].items():
        print(f"{name:35s} accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
