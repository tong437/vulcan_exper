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

"""Frozen semantic contract and deterministic generation helpers for Phase 5E."""

from __future__ import annotations

import re
import time
from typing import Any

import torch


REFERENCE_CAPTION = "A group of bicycles on a subway train."
SEMANTIC_CONTRACT_VERSION = "phase5e_bicycles_in_train_v2"

_THINK_OPEN_RE = re.compile(r"<(?:think|thinking)>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</(?:think|thinking)>", re.IGNORECASE)
_LABEL_PREFIX_RE = re.compile(r"^(?:final answer|answer|caption|description)\s*:\s*", re.IGNORECASE)
_BICYCLE_RE = re.compile(r"\b(?:bicycle|bike)s?\b", re.IGNORECASE)
_BICYCLE_PLURAL_RE = re.compile(r"\b(?:bicycles|bikes)\b", re.IGNORECASE)
_AMBIGUOUS_CYCLE_RE = re.compile(r"\bcycles?\b", re.IGNORECASE)
_TRAIN_RE = re.compile(
    r"\b(?:train|subway|metro|rail(?:way|road)?\s+car|railcar|railway\s+carriage|carriage)s?\b",
    re.IGNORECASE,
)
_PLURAL_QUANTIFIER_RE = re.compile(
    r"\b(?:several|multiple|many|numerous|various|two|three|four|five|six|seven|eight|nine|ten|"
    r"a\s+group\s+of|a\s+row\s+of|a\s+number\s+of)\b.{0,40}\b(?:bicycle|bike)s?\b",
    re.IGNORECASE,
)
_NEGATED_BICYCLE_RE = re.compile(
    r"\b(?:no|zero|without|not\s+any)\b.{0,30}\b(?:bicycle|bike)s?\b|"
    r"\b(?:bicycle|bike)s?\b.{0,25}\b(?:are|is|were|was)?\s*not\b",
    re.IGNORECASE,
)
_EXPLICIT_SINGULAR_RE = re.compile(
    r"\b(?:only\s+one|exactly\s+one|a\s+single|one)\s+(?:bicycle|bike)\b",
    re.IGNORECASE,
)
_WRONG_SCENE_RE = re.compile(r"\b(?:bus|on\s+the\s+street|outdoors?)\b", re.IGNORECASE)
_OUTSIDE_TRAIN_RE = re.compile(
    r"\b(?:bicycle|bike)s?\b.{0,50}\b(?:outside|beside|next\s+to)\b.{0,35}"
    r"\b(?:train|subway|metro|carriage)\b",
    re.IGNORECASE,
)


def semantic_contract() -> dict[str, Any]:
    """Return the immutable, JSON-ready Phase 5E contract."""
    return {
        "version": SEMANTIC_CONTRACT_VERSION,
        "reference_caption": REFERENCE_CAPTION,
        "required": [
            "bicycle concept",
            "multiple bicycles",
            "train/subway/rail-car/carriage scene",
            "bicycles located in or on the train",
            "complete coherent caption",
        ],
        "rejected": [
            "missing bicycle",
            "wrong outdoor/street/bus scene",
            "explicitly one bicycle",
            "negated bicycle claim",
            "unfinished, malformed, thinking-only, or truncated output",
        ],
        "automatic_verdicts": ["pass", "fail", "review"],
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_final_answer(raw_text: str) -> dict[str, Any]:
    """Extract text after the last closed thinking block and audit malformed blocks."""
    raw_text = raw_text or ""
    opens = list(_THINK_OPEN_RE.finditer(raw_text))
    closes = list(_THINK_CLOSE_RE.finditer(raw_text))
    had_thinking = bool(opens or closes)
    unclosed_thinking = len(opens) > len(closes)
    if unclosed_thinking:
        answer = ""
    elif closes:
        answer = raw_text[closes[-1].end() :]
    else:
        answer = raw_text
    answer = _LABEL_PREFIX_RE.sub("", _normalize_text(answer))
    return {
        "raw_text": raw_text,
        "final_answer": answer,
        "had_thinking": had_thinking,
        "thinking_blocks_opened": len(opens),
        "thinking_blocks_closed": len(closes),
        "unclosed_thinking": unclosed_thinking,
        "thinking_only": had_thinking and not bool(answer),
    }


def _has_spatial_relation(text: str) -> bool:
    bicycle = r"(?:bicycle|bike)s?"
    train = r"(?:train|subway|metro|rail(?:way|road)?\s+car|railcar|railway\s+carriage|carriage)s?"
    located = r"(?:in|inside|within|on|aboard)"
    contains = r"(?:with|contains?|holding|carrying|has|have|features?)"
    patterns = (
        rf"\b{bicycle}\b.{{0,80}}\b{located}\b.{{0,80}}\b{train}\b",
        rf"\b{located}\b.{{0,50}}\b{train}\b.{{0,80}}\b{bicycle}\b",
        rf"\b{train}\b.{{0,60}}\b{contains}\b.{{0,80}}\b{bicycle}\b",
        rf"\b{bicycle}\b.{{0,60}}\b(?:parked|stored|lined\s+up|transported)\b.{{0,80}}\b{train}\b",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def evaluate_caption_semantics(
    raw_text: str,
    *,
    terminated_normally: bool = True,
    hit_token_limit: bool = False,
) -> dict[str, Any]:
    """Apply the conservative single-sample contract and return pass/fail/review."""
    extraction = extract_final_answer(raw_text)
    answer = extraction["final_answer"]
    has_bicycle = bool(_BICYCLE_RE.search(answer))
    has_ambiguous_cycle = bool(_AMBIGUOUS_CYCLE_RE.search(answer)) and not has_bicycle
    has_plural = bool(_BICYCLE_PLURAL_RE.search(answer) or _PLURAL_QUANTIFIER_RE.search(answer))
    has_train = bool(_TRAIN_RE.search(answer))
    has_relation = _has_spatial_relation(answer) if has_bicycle and has_train else False
    has_negation = bool(_NEGATED_BICYCLE_RE.search(answer))
    explicit_singular = bool(_EXPLICIT_SINGULAR_RE.search(answer))
    wrong_scene = bool(_WRONG_SCENE_RE.search(answer) or _OUTSIDE_TRAIN_RE.search(answer))
    word_count = len(re.findall(r"[A-Za-z]+", answer))
    coherent_length = word_count >= 4
    complete = bool(answer and coherent_length and terminated_normally and not hit_token_limit)

    features = {
        "has_bicycle": has_bicycle,
        "has_ambiguous_cycle": has_ambiguous_cycle,
        "has_plural_bicycles": has_plural,
        "has_train_scene": has_train,
        "has_spatial_relation": has_relation,
        "has_bicycle_negation": has_negation,
        "explicitly_singular": explicit_singular,
        "wrong_or_contradictory_scene": wrong_scene,
        "coherent_length": coherent_length,
        "terminated_normally": terminated_normally,
        "hit_token_limit": hit_token_limit,
        "unclosed_thinking": extraction["unclosed_thinking"],
        "thinking_only": extraction["thinking_only"],
    }
    hard_failures = []
    if not answer:
        hard_failures.append("no_final_answer")
    if extraction["unclosed_thinking"]:
        hard_failures.append("unclosed_thinking")
    if not terminated_normally or hit_token_limit:
        hard_failures.append("generation_not_normally_terminated")
    if has_negation:
        hard_failures.append("negated_bicycle_claim")
    if explicit_singular:
        hard_failures.append("explicitly_one_bicycle")
    if wrong_scene:
        hard_failures.append("wrong_or_contradictory_scene")
    if answer and not coherent_length:
        hard_failures.append("incomplete_or_incoherent_caption")

    required = has_bicycle and has_plural and has_train and has_relation and complete
    if hard_failures:
        verdict = "fail"
        reasons = hard_failures
    elif required:
        verdict = "pass"
        reasons = ["all_required_semantics_present"]
    else:
        verdict = "review"
        reasons = []
        if not has_bicycle:
            reasons.append("bicycle_concept_unclear" if has_ambiguous_cycle else "missing_bicycle")
        if not has_plural:
            reasons.append("multiple_bicycles_not_explicit")
        if not has_train:
            reasons.append("missing_train_scene")
        if not has_relation:
            reasons.append("bicycle_train_relation_unclear")
        if not complete:
            reasons.append("caption_completeness_unclear")

    return {
        "contract_version": SEMANTIC_CONTRACT_VERSION,
        "verdict": verdict,
        "automatic_pass": verdict == "pass",
        "requires_human_review": verdict == "review",
        "reasons": reasons,
        "features": features,
        "extraction": extraction,
        "human_confirmation": None,
    }


def _as_token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


@torch.no_grad()
def generate_short_caption(
    model: torch.nn.Module,
    tokenizer,
    prompt_inputs: dict[str, Any],
    *,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Generate one deterministic caption and attach stopping/semantic diagnostics."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    generation_inputs = dict(prompt_inputs)
    generation_inputs.update(
        {
            "do_sample": False,
            "temperature": None,
            "use_cache": True,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sequences = model.generate(**generation_inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prompt_length = int(prompt_inputs["input_ids"].shape[-1])
    generated = sequences[0, prompt_length:].detach().cpu()
    token_ids = generated.tolist()
    eos_ids = _as_token_id_set(getattr(tokenizer, "eos_token_id", None))
    generation_config = getattr(model, "generation_config", None)
    eos_ids.update(_as_token_id_set(getattr(generation_config, "eos_token_id", None)))
    ended_with_eos = bool(token_ids and token_ids[-1] in eos_ids)
    hit_token_limit = len(token_ids) >= max_new_tokens and not ended_with_eos
    terminated_normally = ended_with_eos or len(token_ids) < max_new_tokens
    raw_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    semantic = evaluate_caption_semantics(
        raw_text,
        terminated_normally=terminated_normally,
        hit_token_limit=hit_token_limit,
    )
    return {
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "raw_text": raw_text,
        "final_caption": semantic["extraction"]["final_answer"],
        "generation_config": {
            "do_sample": False,
            "temperature": None,
            "max_new_tokens": max_new_tokens,
            "thinking_requested": False,
        },
        "stop": {
            "eos_token_ids": sorted(eos_ids),
            "ended_with_eos": ended_with_eos,
            "hit_token_limit": hit_token_limit,
            "terminated_normally": terminated_normally,
        },
        "timing": {
            "elapsed_seconds": elapsed,
            "tokens_per_second": len(token_ids) / elapsed if elapsed else None,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        },
        "semantic": semantic,
    }
