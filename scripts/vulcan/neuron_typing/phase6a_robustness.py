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

"""Frozen Phase 6A prompt-robustness and counterfactual-sensitivity contract."""

from __future__ import annotations

import re
from typing import Any

from phase5e_semantics import REFERENCE_CAPTION, evaluate_caption_semantics


AUDIT_VERSION = "phase6a_prompt_counterfactual_v2"

_ROLE_GARBAGE_RE = re.compile(r"\b(?:assistant|system|user)\s*:", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def _normalized_words(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text)]


def _normalized_text(text: str) -> str:
    return " ".join(_normalized_words(text))


def _has_repeated_sentence(text: str) -> bool:
    sentences = [" ".join(_normalized_words(sentence)) for sentence in _SENTENCE_RE.findall(text)]
    sentences = [sentence for sentence in sentences if len(sentence.split()) >= 3]
    return len(sentences) != len(set(sentences))


def _substantive_sentence_count(text: str) -> int:
    return sum(len(_normalized_words(sentence)) >= 3 for sentence in _SENTENCE_RE.findall(text))


def _has_repeated_ngram(text: str, n: int = 4, maximum_occurrences: int = 2) -> bool:
    words = _normalized_words(text)
    if len(words) < n:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for offset in range(len(words) - n + 1):
        ngram = tuple(words[offset : offset + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return max(counts.values(), default=0) > maximum_occurrences


def output_hygiene(raw_text: str, *, terminated_normally: bool, hit_token_limit: bool) -> dict[str, Any]:
    """Reject malformed outputs that passed the Phase 5E keyword contract."""
    semantic = evaluate_caption_semantics(
        raw_text,
        terminated_normally=terminated_normally,
        hit_token_limit=hit_token_limit,
    )
    answer = semantic["extraction"]["final_answer"]
    words = _normalized_words(answer)
    checks = {
        "has_final_answer": bool(answer),
        "terminated_normally": bool(terminated_normally and not hit_token_limit),
        "no_unclosed_thinking": not semantic["features"]["unclosed_thinking"],
        "no_role_garbage": not bool(_ROLE_GARBAGE_RE.search(answer)),
        "single_caption_sentence": _substantive_sentence_count(answer) == 1,
        "no_repeated_sentence": not _has_repeated_sentence(answer),
        "no_repeated_fourgram": not _has_repeated_ngram(answer),
        "reasonable_word_count": 4 <= len(words) <= 80,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "word_count": len(words),
        "final_answer": answer,
    }


def evaluate_robustness_case(
    raw_text: str,
    *,
    expected_profile: str,
    terminated_normally: bool = True,
    hit_token_limit: bool = False,
) -> dict[str, Any]:
    """Evaluate one positive or counterfactual Phase 6A output."""
    semantic = evaluate_caption_semantics(
        raw_text,
        terminated_normally=terminated_normally,
        hit_token_limit=hit_token_limit,
    )
    hygiene = output_hygiene(
        raw_text,
        terminated_normally=terminated_normally,
        hit_token_limit=hit_token_limit,
    )
    features = semantic["features"]
    has_bicycle = bool(features["has_bicycle"])
    has_train = bool(features["has_train_scene"])
    has_original_relation = bool(has_bicycle and has_train and features["has_spatial_relation"])

    if expected_profile == "target_semantics":
        fact_checks = {"target_semantic_contract": semantic["automatic_pass"]}
    elif expected_profile == "train_without_bicycle":
        fact_checks = {"mentions_train": has_train, "does_not_mention_bicycle": not has_bicycle}
    elif expected_profile == "bicycle_without_train":
        fact_checks = {"mentions_bicycle": has_bicycle, "does_not_mention_train": not has_train}
    elif expected_profile == "neither_bicycle_nor_train":
        fact_checks = {"does_not_mention_bicycle": not has_bicycle, "does_not_mention_train": not has_train}
    elif expected_profile == "no_original_scene":
        fact_checks = {"does_not_reproduce_original_scene": not has_original_relation}
    else:
        raise ValueError(f"Unknown Phase 6A expected profile: {expected_profile!r}.")

    return {
        "audit_version": AUDIT_VERSION,
        "expected_profile": expected_profile,
        "passed": bool(hygiene["passed"] and all(fact_checks.values())),
        "fact_checks": fact_checks,
        "original_contract_reproduced": has_original_relation,
        "hygiene": hygiene,
        "phase5e_semantics": semantic,
    }


def summarize_model_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate frozen Phase 6A gates for one model."""
    by_group: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_group.setdefault(case["group"], []).append(case)

    def group_summary(name: str) -> dict[str, Any]:
        rows = by_group.get(name, [])
        passed = sum(bool(row["evaluation"]["passed"]) for row in rows)
        return {
            "passed": passed,
            "total": len(rows),
            "pass_rate": passed / len(rows) if rows else None,
            "case_ids": [row["case_id"] for row in rows],
        }

    prompt = group_summary("prompt_primary")
    benign = group_summary("benign_image")
    counterfactual = group_summary("counterfactual")
    diagnostics = group_summary("prompt_diagnostic")
    canonical = next((row for row in cases if row["case_id"] == "prompt_canonical"), None)
    target_rows = [row for row in cases if row["group"] in {"prompt_primary", "benign_image"}]
    target_answers = [
        row.get("generation", {}).get("final_caption", row["evaluation"]["hygiene"]["final_answer"])
        for row in target_rows
    ]
    normalized_target_answers = [_normalized_text(answer) for answer in target_answers]
    normalized_reference = _normalized_text(REFERENCE_CAPTION)
    counterfactual_leaks = [
        row["case_id"]
        for row in by_group.get("counterfactual", [])
        if row["evaluation"]["original_contract_reproduced"]
    ]
    unclean = [row["case_id"] for row in cases if not row["evaluation"]["hygiene"]["passed"]]
    counterfactual_rows = by_group.get("counterfactual", [])
    counterfactual_fact_only_passed = sum(
        all(row["evaluation"]["fact_checks"].values()) for row in counterfactual_rows
    )
    gates = {
        "canonical_target_pass": bool(canonical and canonical["evaluation"]["passed"]),
        "prompt_primary_at_least_4_of_5": prompt["passed"] >= 4 and prompt["total"] == 5,
        "benign_image_at_least_2_of_3": benign["passed"] >= 2 and benign["total"] == 3,
        "zero_original_scene_counterfactual_leaks": not counterfactual_leaks and counterfactual["total"] == 6,
        "counterfactual_fact_checks_at_least_5_of_6": (counterfactual["passed"] >= 5 and counterfactual["total"] == 6),
        "all_outputs_clean": not unclean,
    }
    return {
        "audit_version": AUDIT_VERSION,
        "groups": {
            "prompt_primary": prompt,
            "benign_image": benign,
            "counterfactual": counterfactual,
            "prompt_diagnostic": diagnostics,
        },
        "counterfactual_original_scene_leaks": counterfactual_leaks,
        "unclean_outputs": unclean,
        "diagnostics": {
            "target_variant_unique_answers": len(set(normalized_target_answers)),
            "target_variant_total": len(target_answers),
            "target_variant_exact_reference_answers": sum(
                answer == normalized_reference for answer in normalized_target_answers
            ),
            "target_variant_reference_prefix_answers": sum(
                answer.startswith(normalized_reference) for answer in normalized_target_answers
            ),
            "counterfactual_fact_only_passed": counterfactual_fact_only_passed,
            "counterfactual_total": len(counterfactual_rows),
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
