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

from pathlib import Path

import pandas as pd
import pytest
from neuron_typing_generalization.manifests import (
    SAMPLE_COLUMNS,
    _select_stratified_records,
    normalize_question,
    verify_frozen_manifest,
    write_frozen_manifest,
)


def _row(sample_id: str) -> dict:
    return {
        "dataset": "example",
        "dataset_revision": "1234567",
        "split": "validation",
        "tier": "cpu_fixture",
        "sample_id": sample_id,
        "image_id": sample_id,
        "question": "What is this?",
        "answer": ["thing"],
        "pair_group_id": "group",
        "stratum": None,
        "selection_seed": 42,
        "selection_rank": 0,
        "image_locator": "https://example.test/image.jpg",
        "image_sha256": None,
        "metadata": {"source": "synthetic"},
    }


def test_question_normalization_is_stable_for_grouping():
    assert normalize_question("  What's THIS? ") == normalize_question("what s this")


def test_unpaired_stratified_selection_is_deterministic_and_proportional():
    records = [
        *({"sample_id": f"a{index}", "stratum": "a"} for index in range(6)),
        *({"sample_id": f"b{index}", "stratum": "b"} for index in range(2)),
    ]

    first = _select_stratified_records(records, lambda row: row["stratum"], 4, seed=42)
    second = _select_stratified_records(reversed(records), lambda row: row["stratum"], 4, seed=42)

    assert first == second
    assert [stratum for stratum, _ in first].count("a") == 3
    assert [stratum for stratum, _ in first].count("b") == 1


def test_frozen_manifest_can_be_regenerated_but_not_silently_changed(tmp_path: Path):
    destination = tmp_path / "sample_manifest.parquet"
    digest = write_frozen_manifest([_row("one")], destination, SAMPLE_COLUMNS)
    assert write_frozen_manifest([_row("one")], destination, SAMPLE_COLUMNS) == digest
    assert verify_frozen_manifest(destination, SAMPLE_COLUMNS) == digest
    assert len(pd.read_parquet(destination)) == 1

    with pytest.raises(RuntimeError, match="differs"):
        write_frozen_manifest([_row("two")], destination, SAMPLE_COLUMNS)
