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

"""Create image-content-disjoint VQA train/dev splits for forgetting-stress discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image-hash-disjoint VQA-RAD stress splits.")
    parser.add_argument("--input_file", default="datasets/vqa_rad/train.jsonl")
    parser.add_argument("--image_root", default="datasets/vqa_rad")
    parser.add_argument("--train_output", default="datasets/vqa_rad/stress_train.jsonl")
    parser.add_argument("--dev_output", default="datasets/vqa_rad/stress_dev.jsonl")
    parser.add_argument("--manifest_file", default="datasets/vqa_rad/stress_split_manifest.json")
    parser.add_argument("--dataset_info_file", default="datasets/vqa_rad/dataset_info.json")
    parser.add_argument("--legacy_test_file", default="datasets/vqa_rad/test.jsonl")
    parser.add_argument("--dev_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--balance_trials", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def answer_of(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"Row has no terminal assistant message: {row}")
    answer = str(messages[-1].get("content") or "").strip().lower()
    if answer not in {"yes", "no"}:
        raise ValueError(f"Stress discovery requires binary VQA rows, got {answer!r}.")
    return answer


def image_path_of(row: dict[str, Any], image_root: Path) -> Path:
    images = row.get("images") or []
    if len(images) != 1:
        raise ValueError(f"Expected exactly one image per VQA row: {row}")
    path = Path(images[0])
    if not path.is_absolute():
        path = image_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def image_hashes(rows: list[dict[str, Any]], image_root: Path) -> list[str]:
    cache: dict[Path, str] = {}
    hashes = []
    for row in rows:
        path = image_path_of(row, image_root).resolve()
        if path not in cache:
            cache[path] = sha256_file(path)
        hashes.append(cache[path])
    return hashes


def select_dev_hashes(
    groups: dict[str, list[int]],
    answers: list[str],
    *,
    dev_fraction: float,
    seed: int,
    balance_trials: int,
) -> set[str]:
    """Choose a deterministic group split close to the target row and label counts."""
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be strictly between zero and one.")
    if balance_trials < 1:
        raise ValueError("balance_trials must be positive.")
    hashes = sorted(groups)
    dev_groups = max(1, min(len(hashes) - 1, round(len(hashes) * dev_fraction)))
    target_rows = len(answers) * dev_fraction
    totals = Counter(answers)
    target_yes = totals["yes"] * dev_fraction
    target_no = totals["no"] * dev_fraction
    rng = random.Random(seed)
    best_key: tuple[float, tuple[str, ...]] | None = None
    best: set[str] | None = None
    for _ in range(balance_trials):
        candidate = hashes.copy()
        rng.shuffle(candidate)
        selected_tuple = tuple(sorted(candidate[:dev_groups]))
        selected = set(selected_tuple)
        indices = [index for image_hash in selected for index in groups[image_hash]]
        counts = Counter(answers[index] for index in indices)
        score = (
            abs(len(indices) - target_rows) / max(target_rows, 1.0)
            + abs(counts["yes"] - target_yes) / max(target_yes, 1.0)
            + abs(counts["no"] - target_no) / max(target_no, 1.0)
        )
        key = (score, selected_tuple)
        if best_key is None or key < best_key:
            best_key = key
            best = selected
    if best is None:
        raise RuntimeError("Failed to select a dev split.")
    return best


def canonicalize_rows(
    rows: list[dict[str, Any]],
    hashes: list[str],
    selected_hashes: set[str],
) -> list[dict[str, Any]]:
    canonical_images: dict[str, str] = {}
    for row, image_hash in zip(rows, hashes):
        canonical_images.setdefault(image_hash, str(row["images"][0]))
    output = []
    for row, image_hash in zip(rows, hashes):
        if image_hash not in selected_hashes:
            continue
        normalized = dict(row)
        normalized["images"] = [canonical_images[image_hash]]
        output.append(normalized)
    return output


def summarize(rows: list[dict[str, Any]], hashes: list[str]) -> dict[str, Any]:
    answers = [answer_of(row) for row in rows]
    return {
        "rows": len(rows),
        "unique_image_hashes": len(set(hashes)),
        "answer_counts": dict(sorted(Counter(answers).items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def register_splits(dataset_info_path: Path, train_path: Path, dev_path: Path) -> None:
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.is_file() else {}
    tags = {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    }
    for name, path in (("vqa_rad_stress_train", train_path), ("vqa_rad_stress_dev", dev_path)):
        try:
            file_name = str(path.resolve().relative_to(dataset_info_path.parent.resolve()))
        except ValueError:
            file_name = str(path.resolve())
        dataset_info[name] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": tags,
        }
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info_path.write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_splits(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_file)
    image_root = Path(args.image_root)
    train_path = Path(args.train_output)
    dev_path = Path(args.dev_output)
    manifest_path = Path(args.manifest_file)
    outputs = (train_path, dev_path, manifest_path)
    if not args.force and any(path.exists() for path in outputs):
        raise FileExistsError("Stress split output exists; pass --force only after reviewing it.")

    rows = read_jsonl(input_path)
    hashes = image_hashes(rows, image_root)
    answers = [answer_of(row) for row in rows]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, image_hash in enumerate(hashes):
        groups[image_hash].append(index)
    dev_hashes = select_dev_hashes(
        groups,
        answers,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
        balance_trials=args.balance_trials,
    )
    train_hashes = set(groups) - dev_hashes
    train_rows = canonicalize_rows(rows, hashes, train_hashes)
    dev_rows = canonicalize_rows(rows, hashes, dev_hashes)
    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)
    dataset_info_path = Path(args.dataset_info_file)
    register_splits(dataset_info_path, train_path, dev_path)

    legacy_audit: dict[str, Any] = {"available": False}
    legacy_path = Path(args.legacy_test_file)
    if legacy_path.is_file():
        legacy_rows = read_jsonl(legacy_path)
        legacy_hashes = set(image_hashes(legacy_rows, image_root))
        legacy_audit = {
            "available": True,
            "file": str(legacy_path.resolve()),
            "rows": len(legacy_rows),
            "unique_image_hashes": len(legacy_hashes),
            "overlap_with_original_input": len(legacy_hashes & set(groups)),
            "overlap_with_stress_train": len(legacy_hashes & train_hashes),
            "overlap_with_stress_dev": len(legacy_hashes & dev_hashes),
        }

    train_output_hashes = image_hashes(train_rows, image_root)
    dev_output_hashes = image_hashes(dev_rows, image_root)
    if set(train_output_hashes) & set(dev_output_hashes):
        raise RuntimeError("Image-content leakage remains between stress train and dev.")
    manifest = {
        "artifact_version": 1,
        "method": "seeded_label_balanced_image_sha256_group_split",
        "created_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "dev_fraction_by_unique_image": args.dev_fraction,
        "balance_trials": args.balance_trials,
        "input_file": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "input": summarize(rows, hashes),
        "train": {
            **summarize(train_rows, train_output_hashes),
            "file": str(train_path.resolve()),
            "sha256": sha256_file(train_path),
        },
        "dev": {
            **summarize(dev_rows, dev_output_hashes),
            "file": str(dev_path.resolve()),
            "sha256": sha256_file(dev_path),
        },
        "train_dev_image_hash_overlap": 0,
        "canonicalized_duplicate_image_references": True,
        "dataset_info_file": str(dataset_info_path.resolve()),
        "legacy_test_audit": legacy_audit,
        "notes": [
            "Only the historical train.jsonl is split; the legacy test split is not used for stress discovery.",
            "Grouping uses decoded-file bytes SHA-256, not generated image filenames.",
            "All questions for identical image content remain in one split and reference one canonical image path.",
            "The dev split may be used for stress selection; it is not a formal final test set.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build_splits(parse_args())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
