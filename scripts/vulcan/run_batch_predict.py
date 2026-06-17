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

"""Batch predict for multiple checkpoints on val set.

Usage:
    python scripts/vulcan/run_batch_predict.py
"""

import json
import re
import string
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


VAL_DATA = "datasets/vqa_med/val_cls.jsonl"
IMAGE_DIR = "datasets/vqa_med/images"
NUM_SAMPLES = 200  # subset for speed

CHECKPOINTS = {
    "no-align": "saves/qwen35-0_8b-vqa-med-cls/full/sft-continuation-noalign-lr3e6-200steps/checkpoint-200",
    "soft-iou-lam005": "saves/qwen35-0_8b-vqa-med-cls/full/align-top20-q80-temp002-lam005-lr3e6/checkpoint-200",
    "soft-iou-lam02": "saves/qwen35-0_8b-vqa-med-cls/full/align-top20-q80-temp002-lam02-lr3e6-200steps/checkpoint-200",
    "rank-hardneg": "saves/qwen35-0_8b-vqa-med-cls/full/align-rank-hardneg-m02-lam005-lr3e6-200steps/checkpoint-200",
}

MAX_NEW_TOKENS = 32


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def to_yesno(text: str) -> str | None:
    normalized = normalize_answer(text)
    if normalized in {"yes", "no"}:
        return normalized
    tokens = normalized.split()
    if tokens and tokens[0] in {"yes", "no"}:
        return tokens[0]
    return None


def compute_metrics(predictions: list[dict]) -> dict:
    total = 0
    exact = 0
    norm_exact = 0
    f1_sum = 0.0
    yesno_total = 0
    yesno_correct = 0
    yesno_pred_yes = 0
    yesno_pred_no = 0
    yesno_pred_other = 0
    non_yesno_total = 0
    non_yesno_correct = 0

    for rec in predictions:
        pred = rec["predict"].strip()
        label = rec["label"].strip()
        total += 1
        exact += int(pred == label)
        norm_exact += int(normalize_answer(pred) == normalize_answer(label))

        # token F1
        pt = normalize_answer(pred).split()
        lt = normalize_answer(label).split()
        if not pt and not lt:
            f1 = 1.0
        elif not pt or not lt:
            f1 = 0.0
        else:
            common = Counter(pt) & Counter(lt)
            num_same = sum(common.values())
            if num_same == 0:
                f1 = 0.0
            else:
                p = num_same / len(pt)
                r = num_same / len(lt)
                f1 = 2 * p * r / (p + r)
        f1_sum += f1

        label_yn = to_yesno(label)
        if label_yn is not None:
            yesno_total += 1
            pred_yn = to_yesno(pred)
            yesno_correct += int(pred_yn == label_yn)
            if pred_yn == "yes":
                yesno_pred_yes += 1
            elif pred_yn == "no":
                yesno_pred_no += 1
            else:
                yesno_pred_other += 1
        else:
            non_yesno_total += 1
            non_yesno_correct += int(normalize_answer(pred) == normalize_answer(label))

    metrics = {
        "total": total,
        "em": exact / total * 100 if total else 0,
        "nem": norm_exact / total * 100 if total else 0,
        "f1": f1_sum / total * 100 if total else 0,
    }
    if yesno_total > 0:
        metrics["yesno_acc"] = yesno_correct / yesno_total * 100
        metrics["yesno_total"] = yesno_total
        metrics["yesno_pred_yes"] = yesno_pred_yes
        metrics["yesno_pred_no"] = yesno_pred_no
        metrics["yesno_pred_other"] = yesno_pred_other
        # answer gap: how much prediction distribution deviates from label distribution
        # label is always yes or no, so gap = |pred_yes_rate - 0.5| if balanced
        metrics["yesno_yes_rate"] = yesno_pred_yes / yesno_total * 100
    if non_yesno_total > 0:
        metrics["non_yesno_acc"] = non_yesno_correct / non_yesno_total * 100
        metrics["non_yesno_total"] = non_yesno_total

    return metrics


def run_predict(checkpoint_path: str, processor, samples: list[dict], device: str) -> list[dict]:
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    predictions = []
    for i, sample in enumerate(samples):
        messages = sample["messages"]
        image_path = sample.get("images", [None])[0]
        answer = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        content_parts = []
        for msg in messages:
            if msg["role"] == "user":
                text = msg["content"].replace("<image>", "").strip()
                if image_path:
                    content_parts.append({"type": "image", "image": f"file://{Path(IMAGE_DIR) / image_path}"})
                content_parts.append({"type": "text", "text": text})

        conversation = [{"role": "user", "content": content_parts}]
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        image = None
        if image_path:
            img_path = Path(IMAGE_DIR) / image_path
            if img_path.exists():
                image = Image.open(img_path).convert("RGB")

        inputs = processor(text=[prompt], images=[image] if image else None, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        generated = processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

        predictions.append({"predict": generated, "label": answer})

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(samples)} done")

    del model
    torch.cuda.empty_cache()
    return predictions


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load val data
    samples = []
    with open(VAL_DATA) as f:
        for line in f:
            if len(samples) >= NUM_SAMPLES:
                break
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} val samples (subset)")

    # Load processor (shared)
    first_ckpt = next(iter(CHECKPOINTS.values()))
    processor = AutoProcessor.from_pretrained(first_ckpt, trust_remote_code=True)

    all_results = {}

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"  Path: {ckpt_path}")

        preds = run_predict(ckpt_path, processor, samples, device)

        # Save predictions
        pred_file = f"predictions_{name}.jsonl"
        with open(pred_file, "w") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        metrics = compute_metrics(preds)
        all_results[name] = metrics

        print(f"  EM: {metrics['em']:.2f}  NEM: {metrics['nem']:.2f}  F1: {metrics['f1']:.2f}")
        if "yesno_acc" in metrics:
            print(f"  Yes/No acc: {metrics['yesno_acc']:.2f}% ({metrics['yesno_total']} samples)")
            print(f"  Yes/No pred distribution: yes={metrics['yesno_pred_yes']} no={metrics['yesno_pred_no']} other={metrics['yesno_pred_other']}")
        if "non_yesno_acc" in metrics:
            print(f"  Non-yes/no acc: {metrics['non_yesno_acc']:.2f}% ({metrics['non_yesno_total']} samples)")

    # Save all results
    with open("batch_predict_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'EM':>6} {'NEM':>6} {'F1':>6} {'Y/N acc':>8} {'Y/N yes%':>8} {'non-Y/N acc':>11}")
    print("-" * 80)
    for name, m in all_results.items():
        yn_acc = f"{m['yesno_acc']:.1f}" if 'yesno_acc' in m else "N/A"
        yn_yes = f"{m['yesno_yes_rate']:.1f}" if 'yesno_yes_rate' in m else "N/A"
        non_yn = f"{m['non_yesno_acc']:.1f}" if 'non_yesno_acc' in m else "N/A"
        print(f"{name:<20} {m['em']:>6.2f} {m['nem']:>6.2f} {m['f1']:>6.2f} {yn_acc:>8} {yn_yes:>8} {non_yn:>11}")


if __name__ == "__main__":
    main()
