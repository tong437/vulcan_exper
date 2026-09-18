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

"""Deterministic sampling and immutable sample/pair manifest construction."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jsonschema import Draft202012Validator


SAMPLE_COLUMNS = (
    "dataset",
    "dataset_revision",
    "split",
    "tier",
    "sample_id",
    "image_id",
    "question",
    "answer",
    "pair_group_id",
    "stratum",
    "selection_seed",
    "selection_rank",
    "image_locator",
    "image_sha256",
    "metadata",
)
PAIR_COLUMNS = (
    "dataset",
    "tier",
    "left_sample_id",
    "right_sample_id",
    "pair_group_id",
    "pairing_policy",
    "changed_factor",
    "directional",
    "causal_pair",
    "audit_status",
    "eligible_for_max_search",
)


def normalize_question(question: str) -> str:
    """Normalize question text only for grouping; preserve the original prompt separately."""
    normalized = unicodedata.normalize("NFKC", question).casefold().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split())


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _canonical_json(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (json.dumps(list(rows), ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _select_groups(
    records: Iterable[dict[str, Any]],
    group_key: Callable[[dict[str, Any]], str],
    distinct_key: Callable[[dict[str, Any]], str],
    group_count: int,
    samples_per_group: int,
    seed: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)
    eligible = []
    for key, members in grouped.items():
        distinct_values = {distinct_key(member) for member in members}
        if key and len(distinct_values) >= samples_per_group:
            eligible.append((key, members))
    eligible.sort(key=lambda item: (_stable_rank(seed, item[0]), item[0]))
    if len(eligible) < group_count:
        raise RuntimeError(f"Only {len(eligible)} eligible groups exist; {group_count} requested.")

    selected = []
    for key, members in eligible[:group_count]:
        ordered = sorted(
            members, key=lambda member: (_stable_rank(seed, str(member["sample_id"])), member["sample_id"])
        )
        chosen = []
        seen = set()
        for member in ordered:
            distinct_value = distinct_key(member)
            if distinct_value in seen:
                continue
            seen.add(distinct_value)
            chosen.append(member)
            if len(chosen) == samples_per_group:
                break
        selected.append((key, chosen))
    return selected


def _select_stratified_records(
    records: Iterable[dict[str, Any]],
    stratum_key: Callable[[dict[str, Any]], str],
    sample_count: int,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Deterministically sample strata in proportion to their source-pool frequencies."""
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        stratum = stratum_key(record)
        if not stratum:
            raise ValueError("Unpaired sampling strata must be non-empty.")
        strata[stratum].append(record)
    for stratum, members in strata.items():
        members.sort(key=lambda item: (_stable_rank(seed, f"{stratum}:{item['sample_id']}"), item["sample_id"]))
    total_records = sum(len(members) for members in strata.values())
    if total_records < sample_count:
        raise RuntimeError(f"Only {total_records} stratified records exist; {sample_count} requested.")
    exact = {stratum: sample_count * len(members) / total_records for stratum, members in strata.items()}
    allocated = {stratum: int(value) for stratum, value in exact.items()}
    remainder = sample_count - sum(allocated.values())
    priority = sorted(
        strata,
        key=lambda stratum: (-(exact[stratum] - allocated[stratum]), _stable_rank(seed, stratum), stratum),
    )
    for stratum in priority[:remainder]:
        allocated[stratum] += 1
    selected = [(stratum, record) for stratum, members in strata.items() for record in members[: allocated[stratum]]]
    return sorted(
        selected,
        key=lambda item: (_stable_rank(seed, f"selected:{item[1]['sample_id']}"), item[1]["sample_id"]),
    )


def _sample_row(
    config: Mapping[str, Any],
    tier: str,
    record: Mapping[str, Any],
    group_id: str | None,
    stratum: str | None,
    rank: int,
) -> dict[str, Any]:
    answer = record.get("answer")
    if isinstance(answer, tuple):
        answer = list(answer)
    return {
        "dataset": config["name"],
        "dataset_revision": config["revision"],
        "split": config["split"],
        "tier": tier,
        "sample_id": str(record["sample_id"]),
        "image_id": str(record["image_id"]),
        "question": str(record["question"]),
        "answer": answer,
        "pair_group_id": group_id,
        "stratum": stratum,
        "selection_seed": int(config["sampling"]["seed"]),
        "selection_rank": rank,
        "image_locator": str(record["image_locator"]),
        "image_sha256": None,
        "metadata": dict(record.get("metadata", {})),
    }


def _pair_rows(
    config: Mapping[str, Any],
    tier: str,
    groups: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    changed_factor: str,
    audit_status: str = "passed",
    eligible: bool = True,
) -> list[dict[str, Any]]:
    policy = str(config["pairing"]["controlled"]["kind"])
    rows = []
    for group_id, members in groups:
        for left, right in itertools.combinations(members, 2):
            rows.append(
                {
                    "dataset": config["name"],
                    "tier": tier,
                    "left_sample_id": str(left["sample_id"]),
                    "right_sample_id": str(right["sample_id"]),
                    "pair_group_id": group_id,
                    "pairing_policy": policy,
                    "changed_factor": changed_factor,
                    "directional": False,
                    "causal_pair": False,
                    "audit_status": audit_status,
                    "eligible_for_max_search": eligible,
                }
            )
    return rows


def _vqav2_records(source_dir: Path) -> list[dict[str, Any]]:
    questions_path = next(source_dir.glob("*questions.json"))
    annotations_path = next(source_dir.glob("*annotations.json"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    annotations = {
        int(item["question_id"]): item
        for item in json.loads(annotations_path.read_text(encoding="utf-8"))["annotations"]
    }
    records = []
    for question in questions:
        annotation = annotations[int(question["question_id"])]
        image_id = int(question["image_id"])
        records.append(
            {
                "sample_id": str(question["question_id"]),
                "image_id": str(image_id),
                "question": question["question"],
                "normalized_question": normalize_question(question["question"]),
                "answer": annotation["multiple_choice_answer"],
                "image_locator": f"https://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg",
                "metadata": {
                    "question_type": annotation["question_type"],
                    "answer_type": annotation["answer_type"],
                },
            }
        )
    return records


def _gqa_records(source_dir: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(source_dir / "instructions.parquet")
    records = []
    for item in frame.to_dict(orient="records"):
        groups = item.get("groups") or {}
        records.append(
            {
                "sample_id": str(item["id"]),
                "image_id": str(item["imageId"]),
                "question": item["question"],
                "normalized_question": normalize_question(item["question"]),
                "answer": item.get("answer"),
                "image_locator": f"hf://lmms-lab-encoder/GQA/testdev_balanced_images/{item['imageId']}",
                "metadata": {"groups": groups, "types": item.get("types") or {}},
            }
        )
    return records


def _clevr_records(source_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((source_dir / "CLEVR_val_questions.json").read_text(encoding="utf-8"))
    records = []
    for item in payload["questions"]:
        image_index = int(item["image_index"])
        records.append(
            {
                "sample_id": str(item["question_index"]),
                "image_id": str(image_index),
                "question": item["question"],
                "answer": item.get("answer"),
                "question_family": str(item["question_family_index"]),
                "image_locator": f"clevr://validation/CLEVR_val_{image_index:06d}.png",
                "metadata": {"program": item.get("program", [])},
            }
        )
    return records


def _vizwiz_records(source_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((source_dir / "val.json").read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("annotations", payload.get("questions", []))
    records = []
    for item in items:
        filename = str(item.get("image", item.get("question_id")))
        sample_id = Path(filename).stem
        answers = item.get("answers") or []
        answer_values = [answer["answer"] if isinstance(answer, dict) else str(answer) for answer in answers]
        answerable = item.get("answerable")
        answer_type = item.get("answer_type", "unknown")
        records.append(
            {
                "sample_id": sample_id,
                "image_id": sample_id,
                "question": item["question"],
                "answer": answer_values,
                "stratum": f"answerable={answerable}|answer_type={answer_type}",
                "image_locator": f"https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip#{filename}",
                "metadata": {"answerable": answerable, "answer_type": answer_type},
            }
        )
    return records


def build_dataset_manifests(config_path: Path, source_root: Path) -> dict[str, tuple[list[dict], list[dict]]]:
    """Build CPU fixture, pilot, and optional GPU-smoke rows for one dataset."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = str(config["name"])
    source_dir = source_root / name
    seed = int(config["sampling"]["seed"])
    unpaired_count = int(config["sampling"].get("unpaired_samples", 0))
    unpaired_strategy = config["sampling"].get("unpaired_strategy")
    if unpaired_count and unpaired_strategy != "deterministic_proportional_stratified_hash":
        raise ValueError(f"Unsupported unpaired sampling strategy: {unpaired_strategy!r}")
    outputs: dict[str, tuple[list[dict], list[dict]]] = {}

    if name == "vqav2":
        records = _vqav2_records(source_dir)
        pilot_groups = _select_groups(
            records,
            lambda item: item["normalized_question"],
            lambda item: item["image_id"],
            int(config["sampling"]["pilot_groups"]),
            int(config["sampling"]["samples_per_group"]),
            seed,
        )
        tier_counts = {"cpu_fixture": 8, "gpu_smoke": 8, "pilot": len(pilot_groups)}
        for tier, group_count in tier_counts.items():
            groups = pilot_groups[:group_count]
            rows = [
                _sample_row(config, tier, record, group_id, None, rank)
                for rank, (group_id, members) in enumerate(groups)
                for record in members
            ]
            outputs[tier] = (rows, _pair_rows(config, tier, groups, "image"))
        if unpaired_count:
            selected = _select_stratified_records(
                records,
                lambda item: (
                    f"question_type={item['metadata']['question_type']}|answer_type={item['metadata']['answer_type']}"
                ),
                unpaired_count,
                seed,
            )
            outputs["unpaired_v1_1000"] = (
                [
                    _sample_row(config, "unpaired_v1_1000", record, None, stratum, rank)
                    for rank, (stratum, record) in enumerate(selected)
                ],
                [],
            )
    elif name == "gqa_balanced":
        records = _gqa_records(source_dir)
        pilot_groups = _select_groups(
            records,
            lambda item: item["image_id"],
            lambda item: item["normalized_question"],
            int(config["sampling"]["pilot_groups"]),
            int(config["sampling"]["samples_per_group"]),
            seed,
        )
        for tier, groups in (("cpu_fixture", pilot_groups[:8]), ("pilot", pilot_groups)):
            rows = [
                _sample_row(config, tier, record, group_id, None, rank)
                for rank, (group_id, members) in enumerate(groups)
                for record in members
            ]
            outputs[tier] = (rows, _pair_rows(config, tier, groups, "question"))
        if unpaired_count:
            selected = _select_stratified_records(
                records,
                lambda item: (
                    f"structural={item['metadata']['types'].get('structural', 'unknown')}|"
                    f"semantic={item['metadata']['types'].get('semantic', 'unknown')}"
                ),
                unpaired_count,
                seed,
            )
            outputs["unpaired_v1_1000"] = (
                [
                    _sample_row(config, "unpaired_v1_1000", record, None, stratum, rank)
                    for rank, (stratum, record) in enumerate(selected)
                ],
                [],
            )
    elif name == "clevr":
        records = _clevr_records(source_dir)
        pilot_groups = _select_groups(
            records,
            lambda item: item["question_family"],
            lambda item: item["image_id"],
            int(config["sampling"]["pilot_groups"]),
            int(config["sampling"]["samples_per_group"]),
            seed,
        )
        for tier, groups in (("cpu_fixture", pilot_groups[:8]), ("pilot", pilot_groups)):
            rows = [
                _sample_row(config, tier, record, group_id, None, rank)
                for rank, (group_id, members) in enumerate(groups)
                for record in members
            ]
            pairs = _pair_rows(config, tier, groups, "program_or_scene", audit_status="pending", eligible=False)
            outputs[tier] = (rows, pairs)
        if unpaired_count:
            selected = _select_stratified_records(
                records,
                lambda item: f"question_family={item['question_family']}",
                unpaired_count,
                seed,
            )
            outputs["unpaired_v1_1000"] = (
                [
                    _sample_row(config, "unpaired_v1_1000", record, None, stratum, rank)
                    for rank, (stratum, record) in enumerate(selected)
                ],
                [],
            )
    elif name == "vizwiz_vqa":
        records = _vizwiz_records(source_dir)
        ordered = sorted(records, key=lambda item: (_stable_rank(seed, item["sample_id"]), item["sample_id"]))
        strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in ordered:
            strata[record["stratum"]].append(record)
        interleaved = []
        for offset in itertools.count():
            added = False
            for stratum in sorted(strata):
                if offset < len(strata[stratum]):
                    interleaved.append(strata[stratum][offset])
                    added = True
            if not added:
                break
        for tier, sample_count in (("cpu_fixture", 16), ("pilot", int(config["sampling"]["pilot_samples"]))):
            chosen = interleaved[:sample_count]
            rows = [
                _sample_row(config, tier, record, None, record["stratum"], rank) for rank, record in enumerate(chosen)
            ]
            outputs[tier] = (rows, [])
        if unpaired_count:
            selected = _select_stratified_records(records, lambda item: str(item["stratum"]), unpaired_count, seed)
            outputs["unpaired_v1_1000"] = (
                [
                    _sample_row(config, "unpaired_v1_1000", record, None, stratum, rank)
                    for rank, (stratum, record) in enumerate(selected)
                ],
                [],
            )
    else:
        raise KeyError(f"No manifest builder for {name}.")
    return outputs


def validate_manifest_rows(rows: Sequence[Mapping[str, Any]], schema_path: Path) -> None:
    """Validate every row and report its stable position on failure."""
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for index, row in enumerate(rows):
        errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        if errors:
            raise ValueError(f"Manifest row {index} violates {schema_path.name}: {errors[0].message}")


def write_frozen_manifest(rows: Sequence[Mapping[str, Any]], destination: Path, columns: Sequence[str]) -> str:
    """Write Parquet plus a canonical-content digest, refusing silent replacement."""
    ordered_rows = [{column: row.get(column) for column in columns} for row in rows]
    digest = hashlib.sha256(_canonical_json(ordered_rows)).hexdigest()
    digest_path = destination.with_suffix(destination.suffix + ".sha256")
    if destination.exists():
        existing_digest = digest_path.read_text(encoding="utf-8").strip() if digest_path.exists() else None
        if existing_digest != digest:
            raise RuntimeError(f"Frozen manifest differs from generated content: {destination}")
        return digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    for row in ordered_rows:
        serialized = dict(row)
        if "metadata" in serialized:
            serialized["metadata"] = json.dumps(serialized["metadata"], ensure_ascii=False, sort_keys=True)
        if isinstance(serialized.get("answer"), (list, dict)):
            serialized["answer"] = json.dumps(serialized["answer"], ensure_ascii=False, sort_keys=True)
        frame_rows.append(serialized)
    frame = pd.DataFrame(frame_rows, columns=columns)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".parquet", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    digest_path.write_text(digest + "\n", encoding="utf-8")
    return digest


def verify_frozen_manifest(path: Path, columns: Sequence[str]) -> str:
    """Recompute the canonical row digest from a persisted Parquet manifest."""
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(columns):
        raise ValueError(f"Unexpected columns in frozen manifest {path}.")
    rows = []
    for raw_row in frame.to_dict(orient="records"):
        row = {}
        for column, value in raw_row.items():
            is_missing = not isinstance(value, (list, dict)) and bool(pd.isna(value))
            if is_missing:
                value = None
            if column in {"answer", "metadata"} and isinstance(value, str):
                if column == "metadata" or value.startswith(("[", "{")):
                    value = json.loads(value)
            if hasattr(value, "item"):
                value = value.item()
            row[column] = value
        rows.append(row)
    actual = hashlib.sha256(_canonical_json(rows)).hexdigest()
    digest_path = path.with_suffix(path.suffix + ".sha256")
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(f"Frozen manifest digest mismatch: {path}")
    return actual


def build_and_freeze_all(
    config_dir: Path,
    source_root: Path,
    manifest_root: Path,
    schema_root: Path,
) -> list[Path]:
    """Build and freeze all configured tiers after source metadata has been acquired."""
    written = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        dataset = yaml.safe_load(config_path.read_text(encoding="utf-8"))["name"]
        for tier, (samples, pairs) in build_dataset_manifests(config_path, source_root).items():
            validate_manifest_rows(samples, schema_root / "sample_manifest.schema.json")
            validate_manifest_rows(pairs, schema_root / "pair_manifest.schema.json")
            tier_root = manifest_root / dataset / tier
            sample_path = tier_root / "sample_manifest.parquet"
            pair_path = tier_root / "pair_manifest.parquet"
            write_frozen_manifest(samples, sample_path, SAMPLE_COLUMNS)
            write_frozen_manifest(pairs, pair_path, PAIR_COLUMNS)
            written.extend((sample_path, pair_path))
    return written
