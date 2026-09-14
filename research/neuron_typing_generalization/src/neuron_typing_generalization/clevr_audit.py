# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auditable construction of minimal-program-edit CLEVR sample pairs."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .manifests import PAIR_COLUMNS, SAMPLE_COLUMNS, validate_manifest_rows, write_frozen_manifest


def _program_tokens(
    program: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...]:
    return tuple(
        (
            str(node["function"]),
            tuple(str(value) for value in node.get("value_inputs", ())),
            tuple(int(input_index) - node_index for input_index in node.get("inputs", ())),
        )
        for node_index, node in enumerate(program)
    )


def program_edit_distance(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> int:
    """Return Levenshtein distance over functional-program operations and literal arguments."""
    left_tokens = _program_tokens(left)
    right_tokens = _program_tokens(right)
    previous = list(range(len(right_tokens) + 1))
    for left_index, left_token in enumerate(left_tokens, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right_tokens, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _stable_rank(seed: int, *values: object) -> str:
    return hashlib.sha256(":".join([str(seed), *(str(value) for value in values)]).encode()).hexdigest()


def _load_questions(source_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((source_dir / "CLEVR_val_questions.json").read_text(encoding="utf-8"))
    return list(payload["questions"])


def audit_pending_pairs(sample_path: Path, pair_path: Path) -> dict[str, Any]:
    """Measure whether the original candidate pairs satisfy a minimal edit contract."""
    samples = pd.read_parquet(sample_path)
    pairs = pd.read_parquet(pair_path)
    programs = {str(row.sample_id): json.loads(row.metadata)["program"] for row in samples.itertuples()}
    sample_images = {str(row.sample_id): str(row.image_id) for row in samples.itertuples()}
    distances = []
    same_scene = 0
    for pair in pairs.itertuples():
        left_id = str(pair.left_sample_id)
        right_id = str(pair.right_sample_id)
        distances.append(program_edit_distance(programs[left_id], programs[right_id]))
        same_scene += int(sample_images[left_id] == sample_images[right_id])
    passed = sum(distance == 1 for distance in distances)
    return {
        "tier": str(samples["tier"].iloc[0]),
        "pairs": len(pairs),
        "same_scene_pairs": same_scene,
        "program_edit_distance_one": passed,
        "pass_rate": passed / len(pairs) if len(pairs) else None,
        "distance_histogram": {str(distance): distances.count(distance) for distance in sorted(set(distances))},
        "status": "failed" if passed != len(pairs) or same_scene != len(pairs) else "passed",
        "failure_reason": (
            None
            if passed == len(pairs) and same_scene == len(pairs)
            else "Candidate pairs do not hold the scene fixed while applying exactly one functional-program edit."
        ),
    }


def build_audited_rows(
    config: Mapping[str, Any],
    source_dir: Path,
    pair_count: int = 64,
    tier: str = "audited_v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select disjoint, same-scene CLEVR pairs with program edit distance exactly one."""
    seed = int(config["sampling"]["seed"])
    questions = _load_questions(source_dir)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_image[str(question["image_index"])].append(question)

    candidates = []
    for image_id, members in by_image.items():
        for left, right in itertools.combinations(members, 2):
            if program_edit_distance(left["program"], right["program"]) != 1:
                continue
            candidates.append(
                (
                    _stable_rank(seed, image_id, left["question_index"], right["question_index"]),
                    image_id,
                    left,
                    right,
                )
            )
    candidates.sort(key=lambda item: item[0])

    chosen = []
    used_samples: set[str] = set()
    used_images: set[str] = set()
    for _, image_id, left, right in candidates:
        sample_ids = {str(left["question_index"]), str(right["question_index"])}
        if image_id in used_images or sample_ids & used_samples:
            continue
        chosen.append((image_id, left, right))
        used_images.add(image_id)
        used_samples.update(sample_ids)
        if len(chosen) == pair_count:
            break
    if len(chosen) != pair_count:
        raise RuntimeError(
            f"Only {len(chosen)} disjoint minimal-program-edit CLEVR pairs exist; {pair_count} requested."
        )

    sample_rows = []
    pair_rows = []
    for rank, (image_id, left, right) in enumerate(chosen):
        left_id = str(left["question_index"])
        right_id = str(right["question_index"])
        group_id = f"program-edit:{image_id}:{min(left_id, right_id)}:{max(left_id, right_id)}"
        for question in (left, right):
            question_id = str(question["question_index"])
            sample_rows.append(
                {
                    "dataset": "clevr",
                    "dataset_revision": str(config["revision"]),
                    "split": str(config["split"]),
                    "tier": tier,
                    "sample_id": question_id,
                    "image_id": image_id,
                    "question": str(question["question"]),
                    "answer": question.get("answer"),
                    "pair_group_id": group_id,
                    "stratum": "program_edit_distance=1|same_scene=true",
                    "selection_seed": seed,
                    "selection_rank": rank,
                    "image_locator": f"clevr://validation/CLEVR_val_{int(image_id):06d}.png",
                    "image_sha256": None,
                    "metadata": {
                        "program": question["program"],
                        "question_family": question["question_family_index"],
                        "audit_contract": "same_scene_and_program_edit_distance_1",
                    },
                }
            )
        pair_rows.append(
            {
                "dataset": "clevr",
                "tier": tier,
                "left_sample_id": left_id,
                "right_sample_id": right_id,
                "pair_group_id": group_id,
                "pairing_policy": "same_scene_minimal_program_edit_v1",
                "changed_factor": "program",
                "directional": False,
                "causal_pair": False,
                "audit_status": "passed",
                "eligible_for_max_search": True,
            }
        )

    evidence = {
        "status": "passed",
        "contract": (
            "same image_id and functional-program Levenshtein distance exactly 1 over function, literal arguments, "
            "and relative dependency edges"
        ),
        "candidate_pairs": len(candidates),
        "selected_pairs": len(pair_rows),
        "selected_samples": len(sample_rows),
        "unique_images": len(used_images),
        "disjoint_samples": len(used_samples) == len(sample_rows),
        "program_edit_distance_histogram": {"1": len(pair_rows)},
    }
    return sample_rows, pair_rows, evidence


def build_and_freeze_clevr_audit(
    experiment_root: Path,
    artifact_root: Path,
    pair_count: int = 64,
) -> dict[str, Any]:
    """Audit the pending pilot and freeze a separately named passing tier."""
    config = yaml.safe_load((experiment_root / "configs/datasets/clevr.yaml").read_text(encoding="utf-8"))
    pending = audit_pending_pairs(
        experiment_root / "manifests/clevr/pilot/sample_manifest.parquet",
        experiment_root / "manifests/clevr/pilot/pair_manifest.parquet",
    )
    samples, pairs, replacement = build_audited_rows(config, artifact_root / "sources/clevr", pair_count)
    validate_manifest_rows(samples, experiment_root / "schemas/sample_manifest.schema.json")
    validate_manifest_rows(pairs, experiment_root / "schemas/pair_manifest.schema.json")
    destination = experiment_root / "manifests/clevr/audited_v1"
    sample_digest = write_frozen_manifest(samples, destination / "sample_manifest.parquet", SAMPLE_COLUMNS)
    pair_digest = write_frozen_manifest(pairs, destination / "pair_manifest.parquet", PAIR_COLUMNS)
    return {
        "pending_pilot_audit": pending,
        "audited_v1": replacement,
        "sample_manifest_sha256": sample_digest,
        "pair_manifest_sha256": pair_digest,
    }
