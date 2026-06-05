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

"""Constrained decoding inference for yes/no VQA tasks.

Only allows the model to output 'yes' or 'no' tokens during generation.
This is the most effective technique for binary VQA classification.

Usage:
    python scripts/vulcan/predict_vqa_yesno_constrained.py \
        --model_path saves/qwen35-0_8b-vqa-rad/full/sft-strong \
        --dataset_dir datasets/vqa_rad \
        --output_file predictions_constrained.jsonl \
        --system_prompt "You are a medical visual assistant. \
Look at the medical image carefully. Answer ONLY with yes or no."
"""

import argparse
import json
import re
import string
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
from transformers import LogitsProcessor


class YesNoLogitsProcessor(LogitsProcessor):
    """Constrain generation to only output 'yes' or 'no' tokens.

    On the first generated token, masks all logits to -inf except the
    token IDs corresponding to 'yes' and 'no' (case-insensitive variants).
    After the first token, forces EOS to stop generation immediately.
    """

    def __init__(self, allowed_token_ids: list[int], eos_token_id: int):
        self.allowed_token_ids = allowed_token_ids
        self.eos_token_id = eos_token_id
        self._generated_first = False

    def __len__(self) -> int:
        return 1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self._generated_first:
            mask = torch.full_like(scores, float("-inf"))
            for token_id in self.allowed_token_ids:
                mask[:, token_id] = scores[:, token_id]
            self._generated_first = True
            return mask
        else:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.eos_token_id] = 0.0
            return mask


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


def find_yesno_token_ids(tokenizer) -> list[int]:
    """Find token IDs for 'yes' and 'no' (and common variants like ' Yes', ' No')."""
    candidates = ["yes", "no", "Yes", "No", "YES", "NO", " yes", " no", " Yes", " No"]
    token_ids = set()
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.add(ids[0])
    if not token_ids:
        raise RuntimeError(
            f"Cannot find single-token 'yes'/'no' in tokenizer vocabulary. "
            f"Encoded 'yes'={tokenizer.encode('yes', add_special_tokens=False)}, "
            f"encoded 'no'={tokenizer.encode('no', add_special_tokens=False)}"
        )
    return sorted(token_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained yes/no VQA inference using LlamaFactory models.")
    parser.add_argument("--model_path", required=True, help="Path to base model or HF model id.")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA adapter checkpoint (optional).")
    parser.add_argument("--dataset_dir", default="datasets/vqa_rad", help="LlamaFactory dataset directory.")
    parser.add_argument("--data_file", default="test.jsonl", help="JSONL file under dataset_dir.")
    parser.add_argument("--output_file", default="predictions_constrained.jsonl", help="Output predictions file.")
    parser.add_argument(
        "--system_prompt",
        default="You are a medical visual assistant. Look at the medical image carefully. Answer ONLY with yes or no.",
        help="System prompt for the conversation.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=3, help="Max new tokens to generate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps, etc.")
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.extras.constants import IMAGE_PLACEHOLDER
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_model, load_tokenizer

    infer_args = dict(
        model_name_or_path=args.model_path,
        dataset_dir=args.dataset_dir,
        template="qwen3_5_nothink",
        trust_remote_code=True,
        infer_dtype="bfloat16",
    )
    if args.adapter_path:
        infer_args["adapter_name_or_path"] = [args.adapter_path]

    model_args, data_args, finetuning_args, generating_args = get_infer_args(infer_args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    tokenizer.padding_side = "left"
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    model.eval()

    allowed_ids = find_yesno_token_ids(tokenizer)
    eos_id = template.get_stop_token_ids(tokenizer)
    if isinstance(eos_id, list):
        eos_id = eos_id[0]

    print(f"Allowed yes/no token IDs: {allowed_ids} -> {[tokenizer.decode([tid]) for tid in allowed_ids]}")
    print(f"EOS token ID: {eos_id} -> {tokenizer.decode([eos_id])!r}")

    data_path = Path(args.dataset_dir) / args.data_file
    records = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} examples from {data_path}")

    results = []
    correct = 0
    total = 0

    for i, record in enumerate(records):
        messages = record["messages"]
        label = None
        user_content = None
        system_content = None
        for msg in messages:
            if msg["role"] == "assistant":
                label = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "system":
                system_content = msg["content"]

        if label is None or user_content is None:
            continue

        system_text = args.system_prompt or system_content
        images = record.get("images", [])

        mm_input_dict = {"images": [], "videos": [], "audios": [], "imglens": [0], "vidlens": [0], "audlens": [0]}
        image_inputs = None
        if images:
            from PIL import Image

            image_paths = [Path(args.dataset_dir) / img for img in images]
            image_inputs = [Image.open(p).convert("RGB") for p in image_paths]
            mm_input_dict.update({"images": image_inputs, "imglens": [len(image_inputs)]})

        user_msg = user_content
        if image_inputs and IMAGE_PLACEHOLDER not in user_msg:
            user_msg = IMAGE_PLACEHOLDER * len(image_inputs) + user_msg

        conv_messages = [{"role": "user", "content": user_msg}]
        conv_messages = template.mm_plugin.process_messages(
            conv_messages, mm_input_dict["images"], mm_input_dict["videos"], mm_input_dict["audios"], processor
        )
        paired_messages = conv_messages + [{"role": "assistant", "content": ""}]
        prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, system_text)
        prompt_ids, _ = template.mm_plugin.process_token_ids(
            prompt_ids,
            None,
            mm_input_dict["images"],
            mm_input_dict["videos"],
            mm_input_dict["audios"],
            tokenizer,
            processor,
        )

        input_ids = torch.tensor([prompt_ids], device=device)
        attention_mask = torch.ones_like(input_ids)

        logits_processor = [YesNoLogitsProcessor(allowed_ids, eos_id)]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0][input_ids.shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        label_yesno = to_yesno(label)
        pred_yesno = to_yesno(prediction)
        is_correct = pred_yesno == label_yesno

        if is_correct:
            correct += 1
        total += 1

        results.append(
            {
                "predict": prediction,
                "label": label,
                "pred_normalized": pred_yesno,
                "label_normalized": label_yesno,
                "correct": is_correct,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"[{i+1}/{len(records)}] Running accuracy: {correct/total:.4f} ({correct}/{total})")

    output_path = Path(args.output_file)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*50}")
    print("Constrained Decoding Results")
    print(f"{'='*50}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Predictions saved to: {output_path}")

    from collections import Counter

    pred_counts = Counter(r["pred_normalized"] for r in results)
    label_counts = Counter(r["label_normalized"] for r in results)
    print(f"\nPrediction distribution: {dict(pred_counts)}")
    print(f"Label distribution: {dict(label_counts)}")


if __name__ == "__main__":
    main()
