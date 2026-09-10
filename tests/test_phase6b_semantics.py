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

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts" / "vulcan" / "neuron_typing"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase6b_semantics import build_semantic_evaluator, evaluate_contract_semantics, validate_contract  # noqa: E402


def _contracts() -> dict[str, dict]:
    artifact = json.loads((ROOT_DIR / "data" / "phase6b_single_samples" / "frozen_samples.json").read_text())
    return {sample["sample_id"]: sample["contract"] for sample in artifact["samples"]}


def test_all_frozen_contracts_validate_and_accept_base_captions() -> None:
    artifact = json.loads((ROOT_DIR / "data" / "phase6b_single_samples" / "frozen_samples.json").read_text())
    for sample in artifact["samples"]:
        validate_contract(sample["contract"])
        result = build_semantic_evaluator(sample["contract"])(sample["base_caption"])
        assert result["automatic_pass"], sample["sample_id"]


def test_contract_rejects_missing_concept_and_generation_garbage() -> None:
    contract = _contracts()["dogs_car"]
    missing = evaluate_contract_semantics("Two dogs sit together in a park.", contract=contract)
    garbage = evaluate_contract_semantics(
        "Two dogs sit in a car looking through a window. User: User: User:", contract=contract
    )
    assert not missing["automatic_pass"]
    assert "missing:car" in missing["reasons"]
    assert not garbage["automatic_pass"]
    assert "output_hygiene_failed" in garbage["reasons"]


def test_contract_rejected_pattern_is_enforced() -> None:
    contract = _contracts()["bicycles_train"]
    result = evaluate_contract_semantics(
        "Several bicycles are inside a subway train, not a bus.",
        contract=contract,
    )
    assert not result["automatic_pass"]
    assert result["rejected_matches"] == ["bus_scene"]
