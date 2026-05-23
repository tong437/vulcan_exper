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

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LlamaFactory generated_predictions.jsonl for VQA.")
    parser.add_argument("--prediction_file", required=True, help="Path to generated_predictions.jsonl.")
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def token_f1(prediction: str, label: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    label_tokens = normalize_answer(label).split()
    if not pred_tokens and not label_tokens:
        return 1.0
    if not pred_tokens or not label_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(label_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(label_tokens)
    return 2 * precision * recall / (precision + recall)


def to_yesno(text: str, strict: bool = False) -> str | None:
    normalized = normalize_answer(text)
    if normalized in {"yes", "no"}:
        return normalized

    if strict:
        return None

    tokens = normalized.split()
    if tokens and tokens[0] in {"yes", "no"}:
        return tokens[0]

    return None


def main() -> None:
    args = parse_args()
    total = 0
    exact = 0
    normalized_exact = 0
    f1_sum = 0.0
    yesno_total = 0
    yesno_correct = 0
    yesno_predicted = 0
    yesno_label_counts: Counter[str] = Counter()
    yesno_prediction_counts: Counter[str] = Counter()
    yesno_confusion: Counter[str] = Counter()
    with Path(args.prediction_file).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            prediction = record["predict"].strip()
            label = record["label"].strip()
            total += 1
            exact += int(prediction == label)
            normalized_exact += int(normalize_answer(prediction) == normalize_answer(label))
            f1_sum += token_f1(prediction, label)
            label_yesno = to_yesno(label, strict=True)
            if label_yesno is not None:
                yesno_total += 1
                yesno_label_counts[label_yesno] += 1
                prediction_yesno = to_yesno(prediction)
                yesno_predicted += int(prediction_yesno is not None)
                yesno_correct += int(prediction_yesno == label_yesno)
                yesno_prediction_counts[prediction_yesno or "other"] += 1
                yesno_confusion[f"{label_yesno}->{prediction_yesno or 'other'}"] += 1

    if total == 0:
        raise ValueError(f"No predictions found in {args.prediction_file}.")

    metrics = {
        "num_examples": total,
        "exact_match": exact / total,
        "normalized_exact_match": normalized_exact / total,
        "token_f1": f1_sum / total,
    }
    if yesno_total > 0:
        metrics.update(
            {
                "yesno_examples": yesno_total,
                "yesno_accuracy": yesno_correct / yesno_total,
                "yesno_prediction_coverage": yesno_predicted / yesno_total,
                "yesno_label_counts": dict(sorted(yesno_label_counts.items())),
                "yesno_prediction_counts": dict(sorted(yesno_prediction_counts.items())),
                "yesno_confusion": dict(sorted(yesno_confusion.items())),
            }
        )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
