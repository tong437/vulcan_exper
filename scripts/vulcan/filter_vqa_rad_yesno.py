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
from pathlib import Path


YESNO_DATASET_INFO = {
    "file_name": "test_yesno.jsonl",
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
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a yes/no-only VQA-RAD eval split after conversion.")
    parser.add_argument("--dataset_dir", default="datasets/vqa_rad", help="Converted LlamaFactory dataset directory.")
    parser.add_argument("--source_file", default="test.jsonl", help="Source JSONL file under dataset_dir.")
    parser.add_argument("--output_file", default="test_yesno.jsonl", help="Output JSONL file under dataset_dir.")
    parser.add_argument("--dataset_name", default="vqa_rad_test_yesno", help="dataset_info.json entry to write.")
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def get_label(record: dict) -> str:
    for message in reversed(record["messages"]):
        if message["role"] == "assistant":
            return message["content"]

    raise ValueError(f"Cannot find assistant label in record: {record}")


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    source_path = dataset_dir / args.source_file
    output_path = dataset_dir / args.output_file
    dataset_info_path = dataset_dir / "dataset_info.json"

    total = 0
    kept = 0
    with source_path.open(encoding="utf-8") as reader, output_path.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue

            total += 1
            record = json.loads(line)
            if normalize_answer(get_label(record)) not in {"yes", "no"}:
                continue

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    with dataset_info_path.open(encoding="utf-8") as f:
        dataset_info = json.load(f)

    yesno_info = dict(YESNO_DATASET_INFO)
    yesno_info["file_name"] = args.output_file
    dataset_info[args.dataset_name] = yesno_info
    with dataset_info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Saved {kept} yes/no examples out of {total} records to {output_path}.")
    print(f"Added dataset_info entry: {args.dataset_name}")


if __name__ == "__main__":
    main()
