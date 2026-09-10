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

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase6a_robustness import (  # noqa: E402
    evaluate_robustness_case,
    output_hygiene,
    summarize_model_cases,
)


def test_target_semantics_require_clean_output() -> None:
    clean = evaluate_robustness_case(
        "Several bicycles are parked inside a subway train.", expected_profile="target_semantics"
    )
    repeated = evaluate_robustness_case(
        "Several bicycles are parked inside a subway train. Several bicycles are parked inside a subway train.",
        expected_profile="target_semantics",
    )
    assert clean["passed"]
    assert not repeated["passed"]
    assert not repeated["hygiene"]["checks"]["no_repeated_sentence"]


def test_output_hygiene_rejects_near_duplicate_second_sentence() -> None:
    result = output_hygiene(
        "Three young people point to a pizza. The three young people point to the pizza.",
        terminated_normally=True,
        hit_token_limit=False,
    )
    assert not result["passed"]
    assert not result["checks"]["single_caption_sentence"]


def test_counterfactual_profiles_reject_original_scene_reproduction() -> None:
    train = evaluate_robustness_case("A yellow train waits at a station.", expected_profile="train_without_bicycle")
    bike = evaluate_robustness_case("A decorative bicycle contains a clock.", expected_profile="bicycle_without_train")
    unrelated = evaluate_robustness_case(
        "A white sink stands against a blue wall.", expected_profile="neither_bicycle_nor_train"
    )
    leaked = evaluate_robustness_case(
        "A group of bicycles is parked inside a subway train.", expected_profile="train_without_bicycle"
    )
    assert train["passed"]
    assert bike["passed"]
    assert unrelated["passed"]
    assert not leaked["passed"]
    assert leaked["original_contract_reproduced"]


def test_output_hygiene_detects_role_garbage_and_truncation() -> None:
    role = output_hygiene("assistant: A train is shown.", terminated_normally=True, hit_token_limit=False)
    truncated = output_hygiene("A train is shown.", terminated_normally=False, hit_token_limit=True)
    assert not role["passed"]
    assert not role["checks"]["no_role_garbage"]
    assert not truncated["passed"]


def test_summary_uses_frozen_group_gates() -> None:
    rows = []
    for index in range(5):
        rows.append(
            {
                "case_id": "prompt_canonical" if index == 0 else f"prompt_{index}",
                "group": "prompt_primary",
                "evaluation": {
                    "passed": index < 4,
                    "fact_checks": {"target": index < 4},
                    "original_contract_reproduced": True,
                    "hygiene": {"passed": True, "final_answer": "A group of bicycles on a subway train."},
                },
            }
        )
    for index in range(3):
        rows.append(
            {
                "case_id": f"benign_{index}",
                "group": "benign_image",
                "evaluation": {
                    "passed": index < 2,
                    "fact_checks": {"target": index < 2},
                    "original_contract_reproduced": index < 2,
                    "hygiene": {"passed": True, "final_answer": "A group of bicycles on a subway train."},
                },
            }
        )
    for index in range(6):
        rows.append(
            {
                "case_id": f"counterfactual_{index}",
                "group": "counterfactual",
                "evaluation": {
                    "passed": index < 5,
                    "fact_checks": {"counterfactual": index < 5},
                    "original_contract_reproduced": False,
                    "hygiene": {"passed": True, "final_answer": "A train is visible at a platform."},
                },
            }
        )
    rows.append(
        {
            "case_id": "diagnostic",
            "group": "prompt_diagnostic",
            "evaluation": {
                "passed": True,
                "fact_checks": {"target": True},
                "original_contract_reproduced": True,
                "hygiene": {"passed": True, "final_answer": "The bicycles are inside the train."},
            },
        }
    )
    summary = summarize_model_cases(rows)
    assert summary["all_gates_passed"]
