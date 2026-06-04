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

"""Create a training set with oversampled yes cases for better yes/no balance."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment VQA-RAD training data with oversampled yes cases.")
    parser.add_argument("--input_file", default="datasets/vqa_rad/train.jsonl", help="Input training file.")
    parser.add_argument("--output_file", default="datasets/vqa_rad/train_yesno_augmented.jsonl", help="Output augmented file.")
    parser.add_argument("--yes_weight", type=float, default=3.0, help="Repeat yes cases this many times.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    with Path(args.input_file).open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    yes_records = [r for r in records if r["messages"][-1]["content"].strip().lower() == "yes"]
    no_records = [r for r in records if r["messages"][-1]["content"].strip().lower() == "no"]
    other_records = [r for r in records if r["messages"][-1]["content"].strip().lower() not in ("yes", "no")]

    augmented = []
    for r in yes_records:
        augmented.extend([r] * int(args.yes_weight))
    augmented.extend(no_records)
    augmented.extend(other_records)

    import random
    random.seed(args.seed)
    random.shuffle(augmented)

    with Path(args.output_file).open("w", encoding="utf-8") as f:
        for r in augmented:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Original: {len(records)} ({len(yes_records)} yes, {len(no_records)} no, {len(other_records)} other)")
    print(f"Augmented: {len(augmented)} ({len(yes_records) * int(args.yes_weight)} yes, {len(no_records)} no, {len(other_records)} other)")


if __name__ == "__main__":
    main()
