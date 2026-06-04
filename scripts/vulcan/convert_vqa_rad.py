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
import shutil
import string
from io import BytesIO
from pathlib import Path
from typing import Any


DATASET_INFO = {
    "vqa_rad_train": {
        "file_name": "train.jsonl",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    },
    "vqa_rad_test": {
        "file_name": "test.jsonl",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    },
}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert flaviagiammarino/vqa-rad parquet data into LlamaFactory sharegpt multimodal JSONL files."
        )
    )
    parser.add_argument(
        "--dataset_name_or_path",
        default="flaviagiammarino/vqa-rad",
        help=(
            "Hugging Face dataset id, a load_from_disk directory, a directory containing parquet files, "
            "or a single parquet file."
        ),
    )
    parser.add_argument("--output_dir", default="datasets/vqa_rad", help="Output dataset_dir for LlamaFactory.")
    parser.add_argument("--train_split", default="train", help="Training split name in the source dataset.")
    parser.add_argument("--eval_split", default="test", help="Evaluation split name in the source dataset.")
    parser.add_argument("--question_column", default="question", help="Question column in the source dataset.")
    parser.add_argument("--answer_column", default="answer", help="Answer column in the source dataset.")
    parser.add_argument("--image_column", default="image", help="Image column in the source dataset.")
    parser.add_argument(
        "--prompt_template",
        default="<image>{question}",
        help="Prompt text template. Must contain {question}; include exactly one <image> token.",
    )
    parser.add_argument(
        "--system_prompt",
        default=None,
        help="Optional system prompt prepended to each conversation.",
    )
    parser.add_argument(
        "--yesno_only",
        action="store_true",
        help="Only keep samples whose answer normalizes to yes or no. Normalizes answers to lowercase.",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optionally keep only the first N samples.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args()


def find_parquet_files(dataset_dir: Path, split: str) -> list[str]:
    parquet_files = sorted(dataset_dir.rglob("*.parquet"))
    split_files = [
        str(path)
        for path in parquet_files
        if split.lower() in path.stem.lower() or split.lower() in {part.lower() for part in path.parts}
    ]
    if split_files:
        return split_files

    if len(parquet_files) == 1:
        return [str(parquet_files[0])]

    raise FileNotFoundError(
        f"Cannot find parquet files for split '{split}' under {dataset_dir}. "
        f"Found parquet files: {[str(path) for path in parquet_files[:10]]}"
    )


def load_split(dataset_name_or_path: str, split: str):
    from datasets import DatasetDict, load_dataset, load_from_disk

    source = Path(dataset_name_or_path)
    if source.exists() and source.is_dir():
        if (source / "dataset_dict.json").is_file() or (source / "state.json").is_file():
            dataset = load_from_disk(str(source))
            if isinstance(dataset, DatasetDict):
                return dataset[split]

            return dataset

        data_files = find_parquet_files(source, split)
        return load_dataset("parquet", data_files={split: data_files}, split=split)

    if source.exists() and source.is_file():
        return load_dataset("parquet", data_files={split: str(source)}, split=split)

    return load_dataset(dataset_name_or_path, split=split)


def regularize_image(source_image: Any, image_path: Path) -> None:
    from PIL import Image

    if hasattr(source_image, "save"):
        image = source_image
        image.convert("RGB").save(image_path)
        return

    if isinstance(source_image, dict):
        image_bytes = source_image.get("bytes")
        image_source_path = source_image.get("path")
        if image_bytes is not None:
            image = Image.open(BytesIO(image_bytes))
            image.convert("RGB").save(image_path)
            return

        if image_source_path:
            src = Path(image_source_path)
            if src.is_file():
                shutil.copyfile(src, image_path)
                return

    if isinstance(source_image, bytes):
        image = Image.open(BytesIO(source_image))
        image.convert("RGB").save(image_path)
        return

    if isinstance(source_image, str):
        src = Path(source_image)
        if src.is_file():
            shutil.copyfile(src, image_path)
            return

    raise ValueError(f"Unsupported image value: {type(source_image)}")


def convert_split(
    dataset,
    split_name: str,
    output_file: Path,
    image_dir: Path,
    question_column: str,
    answer_column: str,
    image_column: str,
    prompt_template: str,
    max_samples: int | None,
    system_prompt: str | None = None,
    yesno_only: bool = False,
) -> tuple[int, int]:
    count = 0
    skipped = 0
    limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    with output_file.open("w", encoding="utf-8") as f:
        for index in range(limit):
            example = dataset[index]
            question = str(example[question_column]).strip()
            answer = str(example[answer_column]).strip()
            if not question or not answer:
                continue

            if yesno_only:
                normalized = normalize_answer(answer)
                if normalized not in {"yes", "no"}:
                    skipped += 1
                    continue
                answer = normalized

            image_relpath = f"images/{split_name}_{index:06d}.png"
            regularize_image(example[image_column], image_dir / f"{split_name}_{index:06d}.png")
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_template.format(question=question)})
            messages.append({"role": "assistant", "content": answer})
            record = {
                "messages": messages,
                "images": [image_relpath],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count, skipped


def main() -> None:
    args = parse_args()
    if "{question}" not in args.prompt_template:
        raise ValueError("`--prompt_template` must contain '{question}'.")

    if args.prompt_template.count("<image>") != 1:
        raise ValueError("`--prompt_template` must contain exactly one '<image>' token.")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_split(args.dataset_name_or_path, args.train_split)
    eval_dataset = load_split(args.dataset_name_or_path, args.eval_split)

    train_count, train_skipped = convert_split(
        train_dataset,
        "train",
        output_dir / "train.jsonl",
        image_dir,
        args.question_column,
        args.answer_column,
        args.image_column,
        args.prompt_template,
        args.max_samples,
        system_prompt=args.system_prompt,
        yesno_only=args.yesno_only,
    )
    eval_count, eval_skipped = convert_split(
        eval_dataset,
        "test",
        output_dir / "test.jsonl",
        image_dir,
        args.question_column,
        args.answer_column,
        args.image_column,
        args.prompt_template,
        args.max_samples,
        system_prompt=args.system_prompt,
        yesno_only=args.yesno_only,
    )

    dataset_info = dict(DATASET_INFO)
    if args.yesno_only:
        dataset_info["vqa_rad_train_yesno"] = dataset_info.pop("vqa_rad_train")
        dataset_info["vqa_rad_test_yesno"] = dataset_info.pop("vqa_rad_test")
        dataset_info["vqa_rad_train_yesno"]["file_name"] = "train.jsonl"
        dataset_info["vqa_rad_test_yesno"]["file_name"] = "test.jsonl"

    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Saved {train_count} train examples and {eval_count} test examples to {output_dir}.")
    if args.yesno_only:
        print(f"  Filtered out {train_skipped} non-yes/no train samples, {eval_skipped} non-yes/no eval samples.")
        print(f"LlamaFactory dataset_dir: {output_dir}")
        print("Use dataset: vqa_rad_train_yesno and eval_dataset: vqa_rad_test_yesno")
    else:
        print(f"LlamaFactory dataset_dir: {output_dir}")
        print("Use dataset: vqa_rad_train and eval_dataset: vqa_rad_test")


if __name__ == "__main__":
    main()
