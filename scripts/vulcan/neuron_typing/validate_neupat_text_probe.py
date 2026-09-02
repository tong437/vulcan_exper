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

"""Audit a prepared NeuPAT text probe before using it for neuron allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_tokenizer  # noqa: E402


DEFAULT_EVAL_FILES = (
    "datasets/vqa_rad/test.jsonl",
    "saves/neuron_typing/pope_data/coco_pope_random.json",
    "saves/neuron_typing/pope_data/coco_pope_popular.json",
    "saves/neuron_typing/pope_data/coco_pope_adversarial.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a prepared NeuPAT text probing set.")
    parser.add_argument("--probe_file", default="data/neupat_text_probe_2048.jsonl")
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/neupat_text_probe.formal.yaml")
    parser.add_argument("--output_file", default="data/neupat_text_probe_2048.validation.json")
    parser.add_argument("--eval_file", action="append", default=[])
    parser.add_argument("--c4_file", default="data/c4_demo.jsonl")
    parser.add_argument("--expected_samples", type=int, default=2048)
    parser.add_argument("--expected_per_source", type=int, default=512)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("<image>", " ").split())


def prompt_text(row: dict[str, Any]) -> str:
    return "\n".join(part for part in (str(row.get("instruction", "")), str(row.get("input", ""))) if part)


def eval_prompts(path: Path) -> list[str]:
    prompts = []
    for row in read_jsonl(path):
        if isinstance(row.get("text"), str):
            prompts.append(row["text"])
        for message in row.get("messages", []):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                prompts.append(message["content"])
    return prompts


def describe(values: list[int]) -> dict[str, float | int]:
    sorted_values = sorted(values)
    percentile_index = min(len(values) - 1, round(0.95 * (len(values) - 1)))
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": sorted_values[percentile_index],
    }


def main() -> None:
    args = parse_args()
    probe_path = Path(args.probe_file)
    output_path = Path(args.output_file)
    rows = read_jsonl(probe_path)
    if len(rows) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} probe rows, found {len(rows)}.")

    required = {"instruction", "input", "output", "probe_source", "source_index", "prompt_sha256"}
    missing = [(index, sorted(required - row.keys())) for index, row in enumerate(rows) if required - row.keys()]
    if missing:
        raise ValueError(f"Probe rows are missing required fields: {missing[:5]}")

    sources = Counter(str(row["probe_source"]) for row in rows)
    if (
        any(count != args.expected_per_source for count in sources.values())
        or sum(sources.values()) != args.expected_samples
    ):
        raise ValueError(f"Unbalanced probe sources: {dict(sources)}")

    prompts = [normalize_text(prompt_text(row)) for row in rows]
    stored_hashes = [str(row["prompt_sha256"]) for row in rows]
    computed_hashes = [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts]
    if stored_hashes != computed_hashes:
        mismatch = next(index for index, pair in enumerate(zip(stored_hashes, computed_hashes)) if pair[0] != pair[1])
        raise ValueError(f"Stored prompt hash mismatch at row {mismatch}.")
    if len(set(stored_hashes)) != len(rows):
        raise ValueError("Probe contains duplicate normalized prompts.")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["do_train"] = False
    config["do_eval"] = False
    config["do_predict"] = False
    model_args, data_args, _, _, _ = get_train_args(config)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    cutoff_len = int(data_args.cutoff_len)

    token_lengths: list[int] = []
    token_lengths_by_source: dict[str, list[int]] = defaultdict(list)
    truncated_by_source: Counter[str] = Counter()
    for row in rows:
        messages = [
            {"role": "user", "content": prompt_text(row)},
            {"role": "assistant", "content": str(row["output"])},
        ]
        prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages)
        length = len(prompt_ids) + len(response_ids)
        source = str(row["probe_source"])
        token_lengths.append(length)
        token_lengths_by_source[source].append(length)
        if length > cutoff_len:
            truncated_by_source[source] += 1

    eval_paths = [Path(path) for path in (args.eval_file or DEFAULT_EVAL_FILES)]
    missing_eval_files = [str(path) for path in eval_paths if not path.is_file()]
    if missing_eval_files:
        raise FileNotFoundError(f"Missing evaluation files: {missing_eval_files}")
    normalized_eval = {
        normalize_text(prompt) for path in eval_paths for prompt in eval_prompts(path) if normalize_text(prompt)
    }
    exact_eval_overlap = sorted(set(prompts) & normalized_eval)

    c4_path = Path(args.c4_file)
    c4_documents = [normalize_text(row["text"]) for row in read_jsonl(c4_path)]
    c4_overlap = sorted(prompt for prompt in set(prompts) if any(prompt in document for document in c4_documents))

    report = {
        "artifact_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "probe_file": str(probe_path.resolve()),
        "probe_sha256": sha256_file(probe_path),
        "num_examples": len(rows),
        "source_counts": dict(sorted(sources.items())),
        "unique_prompt_hashes": len(set(stored_hashes)),
        "template": data_args.template,
        "cutoff_len": cutoff_len,
        "token_length_before_truncation": {
            "all": describe(token_lengths),
            "by_source": {source: describe(values) for source, values in sorted(token_lengths_by_source.items())},
        },
        "truncation": {
            "count": sum(truncated_by_source.values()),
            "rate": sum(truncated_by_source.values()) / len(rows),
            "by_source": dict(sorted(truncated_by_source.items())),
        },
        "evaluation_isolation": {
            "eval_files": [str(path.resolve()) for path in eval_paths],
            "unique_eval_prompts": len(normalized_eval),
            "exact_prompt_overlap_count": len(exact_eval_overlap),
            "exact_prompt_overlaps": exact_eval_overlap,
            "c4_file": str(c4_path.resolve()),
            "c4_document_count": len(c4_documents),
            "probe_prompt_substring_overlap_count": len(c4_overlap),
            "probe_prompt_substring_overlaps": c4_overlap,
        },
        "passed": not exact_eval_overlap and not c4_overlap,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise ValueError("Probe/evaluation isolation check failed.")


if __name__ == "__main__":
    main()
