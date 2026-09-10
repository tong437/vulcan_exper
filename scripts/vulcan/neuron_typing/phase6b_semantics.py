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

"""Generic frozen semantic contracts for Phase 6B single-sample frontiers."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from phase5e_semantics import extract_final_answer
from phase6a_robustness import output_hygiene


CONTRACT_SCHEMA_VERSION = "phase6b_caption_contract_v1"


def validate_contract(contract: dict[str, Any]) -> None:
    required = contract.get("required_concepts")
    if not contract.get("version") or not contract.get("sample_id") or not contract.get("reference_caption"):
        raise ValueError("A Phase 6B contract requires version, sample_id, and reference_caption.")
    if not isinstance(required, list) or not required:
        raise ValueError("A Phase 6B contract requires a non-empty required_concepts list.")
    names = []
    for concept in required:
        name = concept.get("name")
        patterns = concept.get("patterns")
        if not name or not isinstance(patterns, list) or not patterns:
            raise ValueError(f"Invalid required concept: {concept!r}.")
        names.append(name)
        for pattern in patterns:
            re.compile(pattern, re.IGNORECASE)
    if len(names) != len(set(names)):
        raise ValueError(f"Required concept names must be unique: {names}.")
    for pattern in contract.get("rejected_patterns", []):
        re.compile(pattern["pattern"], re.IGNORECASE)


def evaluate_contract_semantics(
    raw_text: str,
    *,
    contract: dict[str, Any],
    terminated_normally: bool = True,
    hit_token_limit: bool = False,
) -> dict[str, Any]:
    """Apply one frozen regex concept contract plus shared output-hygiene gates."""
    validate_contract(contract)
    extraction = extract_final_answer(raw_text)
    answer = extraction["final_answer"]
    concept_matches = {}
    for concept in contract["required_concepts"]:
        matched_patterns = [pattern for pattern in concept["patterns"] if re.search(pattern, answer, re.IGNORECASE)]
        concept_matches[concept["name"]] = {
            "passed": bool(matched_patterns),
            "matched_patterns": matched_patterns,
        }
    rejected_matches = [
        item["name"]
        for item in contract.get("rejected_patterns", [])
        if re.search(item["pattern"], answer, re.IGNORECASE)
    ]
    hygiene = output_hygiene(
        raw_text,
        terminated_normally=terminated_normally,
        hit_token_limit=hit_token_limit,
    )
    missing = [name for name, row in concept_matches.items() if not row["passed"]]
    passed = bool(not missing and not rejected_matches and hygiene["passed"])
    reasons = []
    reasons.extend(f"missing:{name}" for name in missing)
    reasons.extend(f"rejected:{name}" for name in rejected_matches)
    if not hygiene["passed"]:
        reasons.append("output_hygiene_failed")
    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_version": contract["version"],
        "sample_id": contract["sample_id"],
        "verdict": "pass" if passed else "fail",
        "automatic_pass": passed,
        "requires_human_review": False,
        "reasons": reasons or ["all_required_concepts_present"],
        "concept_matches": concept_matches,
        "rejected_matches": rejected_matches,
        "hygiene": hygiene,
        "extraction": extraction,
        "human_confirmation": None,
    }


def build_semantic_evaluator(contract: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    """Bind a validated JSON contract for use by deterministic generation."""
    validate_contract(contract)

    def evaluator(
        raw_text: str,
        *,
        terminated_normally: bool = True,
        hit_token_limit: bool = False,
    ) -> dict[str, Any]:
        return evaluate_contract_semantics(
            raw_text,
            contract=contract,
            terminated_normally=terminated_normally,
            hit_token_limit=hit_token_limit,
        )

    return evaluator
