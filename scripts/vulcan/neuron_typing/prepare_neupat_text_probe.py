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

"""Prepare a reproducible, macro-balanced NeuPAT text probing set.

The script uses the Hugging Face Dataset Server rows API instead of cloning
whole repositories. This is particularly important for MetaMathQA, whose raw
JSON file is hundreds of megabytes while the NeuPAT probe needs only 512 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DATASET_SERVER = "https://datasets-server.huggingface.co"
HUGGING_FACE_API = "https://huggingface.co/api/datasets"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repository: str
    config: str
    split: str
    converter: Callable[[dict[str, Any]], dict[str, str] | None]
    candidate_multiplier: float = 1.5


def _clean(value: Any) -> str:
    return str(value or "").strip()


def convert_codealpaca(row: dict[str, Any]) -> dict[str, str] | None:
    return {
        "instruction": _clean(row.get("instruction")),
        "input": _clean(row.get("input")),
        "output": _clean(row.get("output")),
    }


def convert_metamathqa(row: dict[str, Any]) -> dict[str, str] | None:
    return {
        "instruction": _clean(row.get("query")),
        "input": "",
        "output": _clean(row.get("response")),
    }


def convert_dolly(row: dict[str, Any]) -> dict[str, str] | None:
    return {
        "instruction": _clean(row.get("instruction")),
        "input": _clean(row.get("context")),
        "output": _clean(row.get("response")),
    }


def convert_halueval(row: dict[str, Any]) -> dict[str, str] | None:
    # We need a response to pass through the standard SFT chat pipeline. Keep
    # only human-verified non-hallucinated responses so probing does not inject
    # deliberately incorrect assistant text into activation statistics.
    if _clean(row.get("hallucination")).lower() != "no":
        return None
    return {
        "instruction": _clean(row.get("user_query")),
        "input": "",
        "output": _clean(row.get("chatgpt_response")),
    }


SOURCES = (
    SourceSpec("codealpaca", "sahil2801/CodeAlpaca-20k", "default", "train", convert_codealpaca),
    SourceSpec("metamathqa", "meta-math/MetaMathQA", "default", "train", convert_metamathqa),
    SourceSpec("dolly", "databricks/databricks-dolly-15k", "default", "train", convert_dolly),
    SourceSpec("halueval", "pminervini/HaluEval", "general", "data", convert_halueval, 3.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the four-source NeuPAT text probe set.")
    parser.add_argument("--output_file", default="data/neupat_text_probe_2048.jsonl")
    parser.add_argument("--manifest_file", default="data/neupat_text_probe_2048.metadata.json")
    parser.add_argument("--samples_per_source", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--block_size", type=int, default=100)
    parser.add_argument("--max_chars", type=int, default=16000)
    parser.add_argument("--dataset_server", default=DATASET_SERVER)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--request_interval", type=float, default=1.5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def request_json(url: str, *, timeout: float, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LlamaFactory-NeuPAT-Probe/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt + 1 < retries:
                retry_after = None
                if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                    retry_after = error.headers.get("Retry-After")
                if retry_after is not None:
                    delay = float(retry_after)
                elif isinstance(error, urllib.error.HTTPError) and error.code == 429:
                    delay = min(15 * (2**attempt), 120)
                else:
                    delay = min(2**attempt, 8)
                print(f"  request failed ({error}); retrying in {delay:g}s", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"Failed after {retries} requests: {url}") from last_error


def rows_url(server: str, source: SourceSpec, *, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": source.repository,
            "config": source.config,
            "split": source.split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{server.rstrip('/')}/rows?{query}"


def source_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def normalize_prompt(example: dict[str, str]) -> str:
    return " ".join(f"{example['instruction']}\n{example['input']}".lower().split())


def select_source_examples(
    source: SourceSpec,
    raw_rows: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
    max_chars: int,
    global_prompt_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(source_seed(seed, source.name))
    rng.shuffle(raw_rows)
    selected: list[dict[str, Any]] = []
    counters = {"converter_rejected": 0, "empty_rejected": 0, "length_rejected": 0, "duplicate_rejected": 0}
    local_hashes: set[str] = set()
    for raw in raw_rows:
        example = source.converter(raw["row"])
        if example is None:
            counters["converter_rejected"] += 1
            continue
        if not example["instruction"] or not example["output"]:
            counters["empty_rejected"] += 1
            continue
        if sum(len(example[key]) for key in ("instruction", "input", "output")) > max_chars:
            counters["length_rejected"] += 1
            continue
        prompt_hash = hashlib.sha256(normalize_prompt(example).encode()).hexdigest()
        if prompt_hash in local_hashes or prompt_hash in global_prompt_hashes:
            counters["duplicate_rejected"] += 1
            continue
        local_hashes.add(prompt_hash)
        global_prompt_hashes.add(prompt_hash)
        selected.append(
            {
                **example,
                "probe_source": source.name,
                "source_repository": source.repository,
                "source_index": int(raw["row_idx"]),
                "prompt_sha256": prompt_hash,
            }
        )
        if len(selected) == target:
            break
    if len(selected) != target:
        raise ValueError(
            f"Source {source.name} produced {len(selected)} valid unique examples, expected {target}. "
            f"Fetched {len(raw_rows)} candidates; counters={counters}."
        )
    return selected, counters


def fetch_source_rows(
    source: SourceSpec,
    *,
    target: int,
    seed: int,
    block_size: int,
    server: str,
    timeout: float,
    retries: int,
    request_interval: float,
) -> tuple[list[dict[str, Any]], int, list[int]]:
    first = request_json(rows_url(server, source, offset=0, length=1), timeout=timeout, retries=retries)
    total = int(first["num_rows_total"])
    candidate_target = math.ceil(target * source.candidate_multiplier)
    block_count = math.ceil(candidate_target / block_size)
    starts = list(range(0, total, block_size))
    if block_count > len(starts):
        raise ValueError(f"Source {source.name} is too small for {block_count} disjoint blocks.")
    rng = random.Random(source_seed(seed, f"{source.name}:blocks"))
    selected_starts = sorted(rng.sample(starts, block_count))
    rows: list[dict[str, Any]] = []
    for block_index, offset in enumerate(selected_starts, start=1):
        length = min(block_size, total - offset)
        payload = request_json(
            rows_url(server, source, offset=offset, length=length), timeout=timeout, retries=retries
        )
        for item in payload.get("rows", []):
            if item.get("truncated_cells"):
                raise ValueError(
                    f"Dataset Server truncated source {source.name} row {item.get('row_idx')}; "
                    "refusing a non-reproducible partial example."
                )
            rows.append({"row_idx": item["row_idx"], "row": item["row"]})
        print(
            f"  {source.name}: fetched block {block_index}/{block_count} ({len(rows)} candidate rows)",
            flush=True,
        )
        if block_index < len(selected_starts) and request_interval > 0:
            time.sleep(request_interval)
    return rows, total, selected_starts


def repository_metadata(source: SourceSpec, *, timeout: float, retries: int) -> dict[str, Any]:
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in source.repository.split("/"))
    payload = request_json(f"{HUGGING_FACE_API}/{quoted_repo}", timeout=timeout, retries=retries)
    card = payload.get("cardData") or {}
    return {
        "repository": source.repository,
        "revision": payload.get("sha"),
        "license": card.get("license"),
        "config": source.config,
        "split": source.split,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.samples_per_source <= 0:
        raise ValueError("samples_per_source must be positive.")
    if not 1 <= args.block_size <= 100:
        raise ValueError("block_size must be in [1, 100], the Dataset Server page limit.")
    output_path = Path(args.output_file)
    manifest_path = Path(args.manifest_file)
    if not args.force and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError("Probe output already exists; use --force only after reviewing the current manifest.")

    global_prompt_hashes: set[str] = set()
    all_examples: list[dict[str, Any]] = []
    source_reports: dict[str, Any] = {}
    for source in SOURCES:
        print(f"Fetching {source.name} from {source.repository}", flush=True)
        raw_rows, total_rows, block_starts = fetch_source_rows(
            source,
            target=args.samples_per_source,
            seed=args.seed,
            block_size=args.block_size,
            server=args.dataset_server,
            timeout=args.timeout,
            retries=args.retries,
            request_interval=args.request_interval,
        )
        selected, rejection_counts = select_source_examples(
            source,
            raw_rows,
            target=args.samples_per_source,
            seed=args.seed,
            max_chars=args.max_chars,
            global_prompt_hashes=global_prompt_hashes,
        )
        all_examples.extend(selected)
        source_reports[source.name] = {
            **repository_metadata(source, timeout=args.timeout, retries=args.retries),
            "total_source_rows": total_rows,
            "candidate_rows_fetched": len(raw_rows),
            "selected_rows": len(selected),
            "selected_source_indices": sorted(example["source_index"] for example in selected),
            "block_starts": block_starts,
            "rejection_counts": rejection_counts,
        }

    random.Random(args.seed).shuffle(all_examples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for example in all_examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    char_lengths = [sum(len(row[key]) for key in ("instruction", "input", "output")) for row in all_examples]
    manifest = {
        "artifact_version": 1,
        "method": "neupat_macro_balanced_text_probe",
        "created_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "samples_per_source": args.samples_per_source,
        "num_examples": len(all_examples),
        "dataset_server": args.dataset_server,
        "output_file": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "prompt_hash_algorithm": "sha256(normalized lowercase instruction + input)",
        "unique_prompt_hashes": len(global_prompt_hashes),
        "character_length": {
            "min": min(char_lengths),
            "max": max(char_lengths),
            "mean": sum(char_lengths) / len(char_lengths),
        },
        "sources": source_reports,
        "notes": [
            "Each source contributes exactly the same number of examples.",
            "HaluEval retains only rows annotated hallucination=no.",
            "Rows with Dataset Server truncation are rejected.",
            "This forward-only probe must remain disjoint from all evaluation sets.",
        ],
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output_file": str(output_path),
                "manifest_file": str(manifest_path),
                "examples": len(all_examples),
                "sha256": manifest["output_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
