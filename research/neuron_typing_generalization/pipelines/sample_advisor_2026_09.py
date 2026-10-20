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

"""Download, sample, and audit the four datasets in the September 2026 advisor plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import SOURCE_ROOT  # noqa: F401
from datasets import Image, load_dataset
from huggingface_hub import HfApi
from neuron_typing_generalization.advisor_sampling import (
    deterministic_take,
    largest_remainder_quotas,
    stable_digest,
)


DATASETS = {
    "mme": {
        "hub_id": "lmms-lab-encoder/MME",
        "revision": "d6c9023f017b564f7b3ccccf5348166bce8fdbcd",
        "config": "default",
        "split": "test",
    },
    "mmbench": {
        "hub_id": "lmms-lab-encoder/MMBench",
        "revision": "56ba1af8954932c4804bd3f522e05ed96e63b654",
        "config": "en",
        "split": "dev",
    },
    "textvqa": {
        "hub_id": "lmms-lab-encoder/textvqa",
        "revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "config": "default",
        "split": "validation",
    },
    "chartqa": {
        "hub_id": "lmms-lab-encoder/ChartQA",
        "revision": "9e63b7df1592a1c2158e735cc1725454aef0d6d9",
        "config": "default",
        "split": "test",
    },
}


def _image_digest(image: dict[str, Any]) -> str:
    data = image.get("bytes")
    if data is not None:
        return hashlib.sha256(data).hexdigest()
    path = image.get("path")
    if not path:
        raise ValueError("Image has neither embedded bytes nor a path.")
    with open(path, "rb") as image_file:
        return hashlib.sha256(image_file.read()).hexdigest()


def _manifest_digest(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def _base_row(
    name: str, spec: dict[str, str], revision: str, source_index: int, row: dict[str, Any]
) -> dict[str, Any]:
    image = row["image"]
    return {
        "dataset": name,
        "dataset_id": spec["hub_id"],
        "revision": revision,
        "config": spec["config"],
        "split": spec["split"],
        "source_index": source_index,
        "image_path": image.get("path"),
        "image_sha256": _image_digest(image),
        "question": row["question"],
    }


def _sample_paired(
    name: str,
    spec: dict[str, str],
    revision: str,
    dataset: Any,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    strata: dict[str, str] = {}
    for source_index, row in enumerate(dataset):
        if name == "mme":
            stratum = str(row["category"])
            group_key = f"{stratum}\x1f{row['question_id']}"
        else:
            stratum = str(row["type"])
            group_key = f"{stratum}\x1f{row['image']['path']}"
        groups[group_key].append(source_index)
        strata[group_key] = stratum

    valid_groups = {key: indices for key, indices in groups.items() if len(indices) == 2}
    capacities = Counter(strata[key] for key in valid_groups)
    quotas = largest_remainder_quotas(capacities, 500)
    selected_group_ids: list[str] = []
    for stratum in sorted(capacities):
        candidates = [key for key in valid_groups if strata[key] == stratum]
        selected_group_ids.extend(
            deterministic_take(candidates, quotas[stratum], seed=seed, namespace=f"{name}:{stratum}")
        )

    rows: list[dict[str, Any]] = []
    for group_id in sorted(selected_group_ids):
        pair_id = stable_digest(name, group_id)
        for pair_position, source_index in enumerate(valid_groups[group_id]):
            source = dataset[source_index]
            record = _base_row(name, spec, revision, source_index, source)
            record.update(
                {
                    "sample_id": f"{name}:{source_index}",
                    "pair_id": pair_id,
                    "pair_position": pair_position,
                    "stratum": strata[group_id],
                    "answer": source["answer"],
                }
            )
            if name == "mme":
                record["question_id"] = source["question_id"]
            rows.append(record)

    audit = {
        "candidate_rows": len(dataset),
        "candidate_groups": len(groups),
        "group_size_distribution": dict(sorted(Counter(map(len, groups.values())).items())),
        "strict_two_question_groups": len(valid_groups),
        "excluded_non_pair_groups": len(groups) - len(valid_groups),
        "stratum_pair_capacities": dict(sorted(capacities.items())),
        "stratum_pair_quotas": dict(sorted(quotas.items())),
        "selected_groups": len(selected_group_ids),
        "selected_rows": len(rows),
    }
    return rows, audit


def _sample_mmbench(
    spec: dict[str, str], revision: str, dataset: Any, *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata: dict[str, list[int]] = defaultdict(list)
    for source_index, source in enumerate(dataset["source"]):
        strata[str(source)].append(source_index)
    capacities = {source: len(indices) for source, indices in strata.items()}
    quotas = largest_remainder_quotas(capacities, 1000)
    selected: list[tuple[str, int]] = []
    for source in sorted(strata):
        by_id = {str(index): index for index in strata[source]}
        chosen = deterministic_take(
            list(by_id), quotas[source], seed=seed, namespace=f"mmbench:{stable_digest(source)}"
        )
        selected.extend((source, by_id[index]) for index in chosen)

    rows = []
    for source, source_index in sorted(selected, key=lambda item: item[1]):
        item = dataset[source_index]
        record = _base_row("mmbench", spec, revision, source_index, item)
        record.update(
            {
                "sample_id": f"mmbench:{item['index']}",
                "question_id": item["index"],
                "stratum_sha256": stable_digest(source),
                "stratum_preview": source if len(source) <= 160 else source[:157] + "...",
                "answer": item["answer"],
                "choices": {key: item[key] for key in ("A", "B", "C", "D")},
                "hint": item["hint"],
                "category": item["category"],
            }
        )
        rows.append(record)

    quota_histogram = Counter(quotas.values())
    audit = {
        "candidate_rows": len(dataset),
        "source_strata": len(strata),
        "source_strata_over_200_characters": sum(len(source) > 200 for source in strata),
        "rows_in_source_strata_over_200_characters": sum(
            len(indices) for source, indices in strata.items() if len(source) > 200
        ),
        "quota_histogram": dict(sorted(quota_histogram.items())),
        "stratum_audit": [
            {
                "stratum_sha256": stable_digest(source),
                "stratum_preview": source if len(source) <= 160 else source[:157] + "...",
                "capacity": capacities[source],
                "ideal_quota": 1000 * capacities[source] / len(dataset),
                "quota": quotas[source],
            }
            for source in sorted(strata, key=lambda value: stable_digest(value))
        ],
        "selected_rows": len(rows),
    }
    return rows, audit


def _sample_textvqa(
    spec: dict[str, str], revision: str, dataset: Any, *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(row["question_id"]): index for index, row in enumerate(dataset)}
    selected_ids = deterministic_take(list(by_id), 1000, seed=seed, namespace="textvqa")
    rows = []
    for question_id in selected_ids:
        source_index = by_id[question_id]
        item = dataset[source_index]
        record = _base_row("textvqa", spec, revision, source_index, item)
        record.update(
            {
                "sample_id": f"textvqa:{item['question_id']}",
                "question_id": item["question_id"],
                "image_id": item["image_id"],
                "answers": item["answers"],
                "ocr_tokens": item["ocr_tokens"],
            }
        )
        rows.append(record)
    return rows, {
        "candidate_rows": len(dataset),
        "unique_question_ids": len(by_id),
        "sampling": "deterministic SHA-256 rank without replacement",
        "selected_rows": len(rows),
    }


def run(output_dir: Path, *, seed: int, offline: bool) -> dict[str, Any]:
    """Create all four manifests and return their combined audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    combined = {"plan": "advisor_2026_09", "seed": seed, "datasets": {}}
    for name, spec in DATASETS.items():
        revision = spec["revision"]
        if not offline and api.dataset_info(spec["hub_id"]).sha != revision:
            raise RuntimeError(f"Pinned revision for {spec['hub_id']} no longer matches the Hub head.")
        dataset = load_dataset(spec["hub_id"], spec["config"], split=spec["split"], revision=revision)
        dataset = dataset.cast_column("image", Image(decode=False))
        if name in {"mme", "chartqa"}:
            rows, audit = _sample_paired(name, spec, revision, dataset, seed=seed)
        elif name == "mmbench":
            rows, audit = _sample_mmbench(spec, revision, dataset, seed=seed)
        else:
            rows, audit = _sample_textvqa(spec, revision, dataset, seed=seed)
        if len(rows) != 1000:
            raise AssertionError(f"{name} produced {len(rows)} rows instead of 1,000.")
        rows.sort(key=lambda row: (row["source_index"], row["sample_id"]))
        manifest_path = output_dir / f"{name}_1000.jsonl"
        with manifest_path.open("w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        audit.update(
            {
                "hub_id": spec["hub_id"],
                "revision": revision,
                "config": spec["config"],
                "split": spec["split"],
                "manifest": manifest_path.name,
                "manifest_sha256": _manifest_digest(rows),
            }
        )
        combined["datasets"][name] = audit
    audit_path = output_dir / "sampling_audit.json"
    audit_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="Use only already cached dataset assets.")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    result = run(args.output_dir, seed=args.seed, offline=args.offline)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
