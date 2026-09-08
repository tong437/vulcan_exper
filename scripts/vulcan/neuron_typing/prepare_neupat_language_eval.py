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

"""Prepare a reproducible held-out C4 corpus for the formal NeuPAT language gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.parse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prepare_neupat_text_probe import DATASET_SERVER, HUGGING_FACE_API, request_json, sha256_file, write_json


REPOSITORY = "allenai/c4"
CONFIG = "en"
SPLIT = "validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the held-out C4 corpus for NeuPAT validation.")
    parser.add_argument("--output_file", default="data/neupat_language_eval_c4_700.jsonl")
    parser.add_argument("--manifest_file", default="data/neupat_language_eval_c4_700.metadata.json")
    parser.add_argument("--probe_file", default="data/neupat_text_probe_2048.jsonl")
    parser.add_argument("--exclude_file", action="append", default=["data/c4_demo.jsonl"])
    parser.add_argument("--documents", type=int, default=700)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--purpose",
        choices=("formal_evaluation", "stress_development", "stress_lockbox"),
        default="formal_evaluation",
    )
    parser.add_argument("--min_chars", type=int, default=200)
    parser.add_argument("--max_chars", type=int, default=50000)
    parser.add_argument("--dataset_server", default=DATASET_SERVER)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--request_interval", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def rows_url(server: str, *, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {"dataset": REPOSITORY, "config": CONFIG, "split": SPLIT, "offset": offset, "length": length}
    )
    return f"{server.rstrip('/')}/rows?{query}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    args = parse_args()
    if args.documents < 500:
        raise ValueError("Formal language evaluation preparation requires at least 500 documents.")
    output_path = Path(args.output_file)
    manifest_path = Path(args.manifest_file)
    if not args.force and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError("Language evaluation output exists; pass --force only after reviewing it.")

    probe_rows = read_jsonl(Path(args.probe_file))
    probe_prompts = {
        normalize("\n".join(part for part in (row.get("instruction", ""), row.get("input", "")) if part))
        for row in probe_rows
    }
    exclude_hashes = set()
    for path_text in args.exclude_file:
        path = Path(path_text)
        if path.is_file():
            exclude_hashes.update(
                hashlib.sha256(normalize(row["text"]).encode()).hexdigest() for row in read_jsonl(path)
            )

    first = request_json(rows_url(args.dataset_server, offset=0, length=1), timeout=args.timeout, retries=args.retries)
    total_rows = int(first["num_rows_total"])
    block_starts = list(range(0, total_rows, 100))
    random.Random(args.seed).shuffle(block_starts)
    selected = []
    selected_hashes = set()
    counters = Counter()
    used_blocks = []
    for block_start in block_starts:
        payload = request_json(
            rows_url(args.dataset_server, offset=block_start, length=min(100, total_rows - block_start)),
            timeout=args.timeout,
            retries=args.retries,
        )
        used_blocks.append(block_start)
        for item in payload.get("rows", []):
            if item.get("truncated_cells"):
                counters["truncated"] += 1
                continue
            text = str(item["row"].get("text") or "").strip()
            if not args.min_chars <= len(text) <= args.max_chars:
                counters["length"] += 1
                continue
            normalized = normalize(text)
            text_hash = hashlib.sha256(normalized.encode()).hexdigest()
            if text_hash in selected_hashes or text_hash in exclude_hashes:
                counters["duplicate_or_excluded"] += 1
                continue
            if any(prompt and prompt in normalized for prompt in probe_prompts):
                counters["probe_overlap"] += 1
                continue
            selected_hashes.add(text_hash)
            selected.append(
                {
                    "text": text,
                    "source_index": int(item["row_idx"]),
                    "source_url": str(item["row"].get("url") or ""),
                    "text_sha256": text_hash,
                }
            )
            if len(selected) == args.documents:
                break
        print(f"Fetched block {len(used_blocks)}: selected {len(selected)}/{args.documents}", flush=True)
        if len(selected) == args.documents:
            break
        if args.request_interval > 0:
            time.sleep(args.request_interval)
    if len(selected) != args.documents:
        raise ValueError(f"Only selected {len(selected)} valid documents from {len(used_blocks)} blocks.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    repo = request_json(f"{HUGGING_FACE_API}/{REPOSITORY}", timeout=args.timeout, retries=args.retries)
    lengths = [len(row["text"]) for row in selected]
    manifest = {
        "artifact_version": 1,
        "method": "neupat_language_corpus_preparation",
        "purpose": args.purpose,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": REPOSITORY,
        "revision": repo.get("sha"),
        "license": (repo.get("cardData") or {}).get("license"),
        "config": CONFIG,
        "split": SPLIT,
        "seed": args.seed,
        "documents": len(selected),
        "source_indices": sorted(row["source_index"] for row in selected),
        "used_block_starts": used_blocks,
        "unique_text_hashes": len(selected_hashes),
        "excluded_files": [str(Path(path).resolve()) for path in args.exclude_file if Path(path).is_file()],
        "probe_file": str(Path(args.probe_file).resolve()),
        "probe_prompt_substring_overlap": 0,
        "rejection_counts": dict(counters),
        "character_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "output_file": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "notes": [
            "The source is the C4 English validation split, not its train split.",
            "Documents used by the earlier 300-document screening corpus are excluded by normalized SHA-256.",
            "Any document containing a normalized NeuPAT probe prompt is rejected.",
            "The LlamaFactory config packs these documents into fixed-length, all-token PT evaluation blocks.",
            (
                "Stress-development data may select training strength and must not be reported as a final test."
                if args.purpose == "stress_development"
                else "Stress-lockbox data must remain unopened until a stress setting has been frozen."
                if args.purpose == "stress_lockbox"
                else "This corpus is intended for formal evaluation."
            ),
        ],
    }
    write_json(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in ("documents", "output_sha256", "rejection_counts")}, indent=2))


if __name__ == "__main__":
    main()
