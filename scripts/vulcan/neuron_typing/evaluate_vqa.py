#!/usr/bin/env python3
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

"""VQA Yes/No evaluation for neuron typing ablation studies.

Generates yes/no questions from COCO captions and evaluates model accuracy
under different ablation conditions.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamafactory.data import (
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VQA Yes/No evaluation.")
    parser.add_argument("--config", required=True, help="LlamaFactory SFT YAML config.")
    parser.add_argument("--score_file", required=True, help="Neuron type scores parquet.")
    parser.add_argument("--output_file", required=True, help="Output JSON file.")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to evaluate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--ablation", action="append", default=[], help="Ablation specs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def generate_yes_no_questions(caption: str) -> list[tuple[str, str]]:
    """Generate yes/no questions from a caption."""
    questions = []
    
    # Extract objects from caption
    # Simple heuristic: look for "a", "an", "the" followed by nouns
    objects = re.findall(r'(?:a|an|the)\s+(\w+)', caption.lower())
    
    # Generate questions
    for obj in set(objects[:3]):  # Limit to 3 objects
        questions.append((f"Is there a {obj}?", "yes"))
    
    # Generate some "no" questions
    common_objects = ["car", "dog", "cat", "bird", "tree", "house", "person", "man", "woman"]
    for obj in common_objects:
        if obj not in caption.lower():
            questions.append((f"Is there a {obj}?", "no"))
            break
    
    return questions[:5]  # Limit to 5 questions per caption


def evaluate_vqa_accuracy(
    model: torch.nn.Module,
    tokenizer,
    template,
    vqa_dataset: list[tuple[str, str]],
    device: torch.device,
    max_samples: int,
) -> dict[str, Any]:
    """Evaluate VQA yes/no accuracy."""
    correct = 0
    total = 0
    yes_count = 0
    no_count = 0
    
    for question, expected_answer in vqa_dataset[:max_samples]:
        # Format question
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Check answer
        response_lower = response.lower().strip()
        if "yes" in response_lower:
            yes_count += 1
            if expected_answer == "yes":
                correct += 1
        elif "no" in response_lower:
            no_count += 1
            if expected_answer == "no":
                correct += 1
        
        total += 1
    
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "yes_count": yes_count,
        "no_count": no_count,
        "yes_ratio": yes_count / total if total > 0 else 0.0,
    }


def main():
    args = parse_args()
    # TODO: Implement full VQA evaluation pipeline
    print("VQA evaluation script created. Full implementation pending.")


if __name__ == "__main__":
    main()
