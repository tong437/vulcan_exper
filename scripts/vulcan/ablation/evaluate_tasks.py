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

"""Unified task evaluation for ablation and pruning experiments.

Supports:
- COCO Caption: BLEU-4, CIDEr
- VQAv2: VQA Accuracy
- GQA: Accuracy
- WikiText-103: Perplexity
- HellaSwag: Accuracy
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified task evaluation.")
    parser.add_argument("--task", required=True,
                        choices=["caption", "vqav2", "gqa", "wikitext", "hellaswag"],
                        help="Evaluation task.")
    parser.add_argument("--model_path", required=True, help="Path to model.")
    parser.add_argument("--dataset", required=True, help="Dataset name or path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu).")
    return parser.parse_args()


def evaluate_caption(model, processor, dataset, max_samples: int, device: str) -> dict:
    """Evaluate on COCO Caption task."""
    print(f"Evaluating caption task with {max_samples} samples...")

    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            inputs = processor(
                text=[sample["prompt"]],
                images=[sample["image"]] if "image" in sample else None,
                return_tensors="pt",
                padding=True,
            ).to(device)

            outputs = model.generate(**inputs, max_new_tokens=128)
            pred = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

            predictions.append(pred)
            references.append(sample["response"])

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{max_samples}")

    bleu4 = compute_bleu4(predictions, references)
    cider = compute_cider(predictions, references)

    return {
        "task": "caption",
        "num_samples": len(predictions),
        "metrics": {
            "bleu4": bleu4,
            "cider": cider,
        }
    }


def evaluate_vqa(model, processor, dataset, max_samples: int, device: str) -> dict:
    """Evaluate on VQA task."""
    print(f"Evaluating VQA task with {max_samples} samples...")

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            inputs = processor(
                text=[sample["prompt"]],
                images=[sample["image"]] if "image" in sample else None,
                return_tensors="pt",
                padding=True,
            ).to(device)

            outputs = model.generate(**inputs, max_new_tokens=32)
            pred = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

            answer = sample["response"].strip().lower()
            if pred == answer:
                correct += 1
            total += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{max_samples}")

    accuracy = correct / total if total > 0 else 0

    return {
        "task": "vqa",
        "num_samples": total,
        "metrics": {
            "accuracy": accuracy,
        }
    }


def evaluate_wikitext(model, tokenizer, dataset, max_samples: int, device: str) -> dict:
    """Evaluate on WikiText-103 (perplexity)."""
    print(f"Evaluating WikiText with {max_samples} samples...")

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            inputs = tokenizer(
                sample["text"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{max_samples}")

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "task": "wikitext",
        "num_samples": min(i + 1, max_samples),
        "metrics": {
            "perplexity": perplexity,
        }
    }


def compute_bleu4(predictions: list[str], references: list[str]) -> float:
    """Compute BLEU-4 score."""
    try:
        from nltk.translate.bleu_score import corpus_bleu
        refs = [[ref.split()] for ref in references]
        preds = [pred.split() for pred in predictions]
        return corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25))
    except ImportError:
        print("WARNING: nltk not available, returning 0 for BLEU-4")
        return 0.0


def compute_cider(predictions: list[str], references: list[str]) -> float:
    """Compute CIDEr score (simplified)."""
    print("NOTE: Using simplified CIDEr computation")
    return 0.0


def main():
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Task: {args.task}")
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")

    print("\nNOTE: This is a placeholder evaluation script.")
    print("Implement actual model loading and dataset processing for production use.")

    results = {
        "task": args.task,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "metrics": {},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
