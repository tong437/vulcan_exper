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
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


DATASET_INFO = {
    "vqa_train_cls": {
        "file_name": "train_cls.jsonl",
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
    "vqa_val_cls": {
        "file_name": "val_cls.jsonl",
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
    "vqa_train_modality": {
        "file_name": "train_modality.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    },
    "vqa_train_plane": {
        "file_name": "train_plane.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    },
    "vqa_train_organ": {
        "file_name": "train_organ.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    },
    "vqa_val_modality": {
        "file_name": "val_modality.jsonl",
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
    "vqa_val_plane": {
        "file_name": "val_plane.jsonl",
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
    "vqa_val_organ": {
        "file_name": "val_organ.jsonl",
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


MODALITY_KEYWORDS = [
    "modality",
    "kind of image",
    "type of imaging",
    "what imaging method",
    "is this a t1",
    "is this a t2",
    "is this a ct",
    "is this an mri",
    "is this a normal",
    "is this a contrast",
    "was gi contrast",
    "was iv contrast",
    "was the ct",
    "was the mri",
    "what type of contrast",
    "what was this image taken with",
    "how is the image taken",
    "how was the image taken",
    "is this image normal",
    "does this image look normal",
    "is the ct scan normal",
    "is the gastrointestinal",
    "is the nuclear medicine",
    "is the x-ray normal",
    "is there something wrong",
    "what is the mr weighting",
    "t1 weighted",
    "t2 weighted",
    "flair",
    "what kind of scan",
    "what part of the body",
    "noncontrast",
    "is the mri normal",
    "is the ultrasound normal",
]

PLANE_KEYWORDS = ["plane"]

ORGAN_KEYWORDS = ["organ", "what part of the body"]


def classify_question(question: str) -> str:
    q = question.lower()
    if any(k in q for k in MODALITY_KEYWORDS):
        return "modality"
    if any(k in q for k in PLANE_KEYWORDS):
        return "plane"
    if any(k in q for k in ORGAN_KEYWORDS):
        return "organ system"
    return "unknown"


def normalize_answer(answer: str) -> str:
    text = answer.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("x-ray", "xr").replace("x ray", "xr")
    text = text.replace("ct - noncontrast", "ct - non-contrast")
    text = text.replace("ct - contrast", "ct - contrast")
    text = text.replace("gi & iv contrast", "gi and iv contrast")
    text = text.replace("&", "and")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VQA-Med 2019 dataset into LlamaFactory sharegpt multimodal JSONL files."
    )
    parser.add_argument(
        "--dataset_dir",
        default="/root/autodl-pub-RTX4090-hdd-1/datasets/vqa_med",
        help="Directory containing VQA-Med parquet files.",
    )
    parser.add_argument("--output_dir", default="datasets/vqa_med", help="Output dataset directory.")
    parser.add_argument(
        "--prompt_template",
        default="<image>{question}\nAnswer with the exact label only.",
        help="Prompt text template. Must contain {question}; include exactly one <image> token.",
    )
    parser.add_argument(
        "--system_prompt",
        default=None,
        help="Optional system prompt prepended to each conversation.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers for image processing.")
    return parser.parse_args()


def regularize_image(source_image: Any, image_path: Path) -> None:
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
            from shutil import copyfile

            src = Path(image_source_path)
            if src.is_file():
                copyfile(src, image_path)
                return

    if isinstance(source_image, bytes):
        image = Image.open(BytesIO(source_image))
        image.convert("RGB").save(image_path)
        return

    raise ValueError(f"Unsupported image value: {type(source_image)}")


def process_row(
    row_data: tuple, split_name: str, image_dir: Path, prompt_template: str, system_prompt: str | None
) -> dict | None:
    index, row = row_data
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    if not question or not answer:
        return None

    category = classify_question(question)
    if category == "unknown":
        return None

    image_relpath = f"images/{split_name}_{index:06d}.png"
    try:
        regularize_image(row["image"], image_dir / f"{split_name}_{index:06d}.png")
    except Exception:
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_template.format(question=question)})
    messages.append({"role": "assistant", "content": answer})

    return {
        "category": category,
        "raw_answer": answer,
        "normalized_answer": normalize_answer(answer),
        "record": {
            "messages": messages,
            "images": [image_relpath],
        },
    }


def convert_parquet(
    parquet_path: str,
    split_name: str,
    output_dir: Path,
    prompt_template: str,
    system_prompt: str | None = None,
    num_workers: int = 8,
) -> dict[str, list]:
    df = pd.read_parquet(parquet_path)
    records_by_category: dict[str, list] = {cat: [] for cat in ["modality", "plane", "organ system"]}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_row, (index, row), split_name, output_dir / "images", prompt_template, system_prompt
            ): index
            for index, row in df.iterrows()
        }

        for i, future in enumerate(as_completed(futures)):
            if i % 2000 == 0:
                print(f"  Processed {i}/{len(df)} rows...")
            try:
                result = future.result()
                if result is not None:
                    cat = result["category"]
                    records_by_category[cat].append(result)
            except Exception:
                pass

    return records_by_category


def write_jsonl(file_path: Path, records: list, include_metadata: bool = False) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            if isinstance(record, dict) and "record" in record:
                if include_metadata:
                    output = dict(record["record"])
                    output["category"] = record["category"]
                else:
                    output = record["record"]
            else:
                output = record
            f.write(json.dumps(output, ensure_ascii=False) + "\n")


def write_label_vocab(file_path: Path, answers: list[str]) -> None:
    unique_answers = sorted(set(answers))
    with file_path.open("w", encoding="utf-8") as f:
        for ans in unique_answers:
            f.write(ans + "\n")


def main() -> None:
    args = parse_args()
    if "{question}" not in args.prompt_template:
        raise ValueError("`--prompt_template` must contain '{question}'.")

    if args.prompt_template.count("<image>") != 1:
        raise ValueError("`--prompt_template` must contain exactly one '<image>' token.")

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    train_parquet = dataset_dir / "data" / "train-00000-of-00001-576f60ca22cf40e6.parquet"
    val_parquet = dataset_dir / "data" / "validation-00000-of-00001-28d19417849c34cf.parquet"

    print("Converting training set...")
    train_records = convert_parquet(
        str(train_parquet),
        "train",
        output_dir,
        args.prompt_template,
        system_prompt=args.system_prompt,
        num_workers=args.num_workers,
    )
    for cat, recs in train_records.items():
        print(f"  {cat}: {len(recs)} examples")

    print("Converting validation set...")
    val_records = convert_parquet(
        str(val_parquet),
        "val",
        output_dir,
        args.prompt_template,
        system_prompt=args.system_prompt,
        num_workers=args.num_workers,
    )
    for cat, recs in val_records.items():
        print(f"  {cat}: {len(recs)} examples")

    print("Writing output files...")
    train_cls = train_records["modality"] + train_records["plane"] + train_records["organ system"]
    write_jsonl(output_dir / "train_cls.jsonl", train_cls, include_metadata=False)
    print(f"  train_cls.jsonl: {len(train_cls)} examples")

    write_jsonl(output_dir / "train_modality.jsonl", train_records["modality"], include_metadata=False)
    write_jsonl(output_dir / "train_plane.jsonl", train_records["plane"], include_metadata=False)
    write_jsonl(output_dir / "train_organ.jsonl", train_records["organ system"], include_metadata=False)

    val_cls = val_records["modality"] + val_records["plane"] + val_records["organ system"]
    write_jsonl(output_dir / "val_cls.jsonl", val_cls, include_metadata=True)
    print(f"  val_cls.jsonl: {len(val_cls)} examples")

    write_jsonl(output_dir / "val_modality.jsonl", val_records["modality"], include_metadata=True)
    print(f"  val_modality.jsonl: {len(val_records['modality'])} examples")

    write_jsonl(output_dir / "val_plane.jsonl", val_records["plane"], include_metadata=True)
    print(f"  val_plane.jsonl: {len(val_records['plane'])} examples")

    write_jsonl(output_dir / "val_organ.jsonl", val_records["organ system"], include_metadata=True)
    print(f"  val_organ.jsonl: {len(val_records['organ system'])} examples")

    print("Writing label vocabularies...")
    write_label_vocab(
        output_dir / "labels_modality.txt",
        [r["normalized_answer"] for r in train_records["modality"] + val_records["modality"]],
    )
    write_label_vocab(
        output_dir / "labels_plane.txt",
        [r["normalized_answer"] for r in train_records["plane"] + val_records["plane"]],
    )
    write_label_vocab(
        output_dir / "labels_organ.txt",
        [r["normalized_answer"] for r in train_records["organ system"] + val_records["organ system"]],
    )

    print(
        f"  labels_modality.txt: "
        f"{len({r['normalized_answer'] for r in train_records['modality'] + val_records['modality']})} labels"
    )
    print(
        f"  labels_plane.txt: "
        f"{len({r['normalized_answer'] for r in train_records['plane'] + val_records['plane']})} labels"
    )
    print(
        f"  labels_organ.txt: "
        f"{len({r['normalized_answer'] for r in train_records['organ system'] + val_records['organ system']})} labels"
    )

    dataset_info = dict(DATASET_INFO)
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nDataset saved to {output_dir}")
    print("LlamaFactory dataset_dir:", output_dir)
    print("Use dataset: vqa_train_cls and eval_dataset: vqa_val_cls")
    print("For category-specific evaluation: vqa_val_modality, vqa_val_plane, vqa_val_organ")


if __name__ == "__main__":
    main()
