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

"""Selection, masking, and official-style metrics for response-state ablation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .interfaces import FFNBinding


STATE_ORDER = ("V", "T", "M", "W")
STATE_TO_CLASS_ID = {"W": 0, "V": 1, "T": 2, "M": 3}
ABLATION_METHODS = ("random", "max", "min")


@dataclass(frozen=True)
class AblationCondition:
    """One executable member of the 14 conceptual ablation conditions."""

    condition: str
    state: str | None
    method: str
    repeat: int
    random: bool


def build_ablation_conditions(random_repeats: int = 10) -> tuple[AblationCondition, ...]:
    """Expand 14 conceptual conditions into baseline, deterministic, and repeated random runs."""
    if random_repeats < 2:
        raise ValueError("random_repeats must be at least two so a sample standard deviation is defined.")
    conditions = [AblationCondition("baseline", None, "baseline", 0, False)]
    for state in STATE_ORDER:
        conditions.extend(
            AblationCondition(f"{state}_random", state, "random", repeat, True)
            for repeat in range(random_repeats)
        )
    for method in ("max", "min"):
        conditions.extend(
            AblationCondition(f"{state}_{method}", state, method, 0, False) for state in STATE_ORDER
        )
    conditions.extend(
        AblationCondition("global_random", None, "random", repeat, True) for repeat in range(random_repeats)
    )
    return tuple(conditions)


def per_layer_deletion_count(d_ff: int, fraction: float) -> int:
    """Return a fixed per-layer count shared by every condition (floor, with a minimum of one)."""
    if d_ff < 1 or not 0.0 < fraction <= 1.0:
        raise ValueError("d_ff must be positive and fraction must lie in (0, 1].")
    return max(1, int(np.floor(d_ff * fraction)))


def response_strength(q: NDArray[np.floating], state: str) -> NDArray[np.float64]:
    """Compute the advisor-defined per-neuron response strength for one layer."""
    values = np.asarray(q, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("q must have shape [2, d_ff] in visual, text order.")
    if state == "V":
        return values[0]
    if state == "T":
        return values[1]
    if state in {"M", "W"}:
        return values[0] + values[1]
    raise ValueError(f"Unknown response state: {state}")


def select_ablation_indices(
    q: NDArray[np.floating],
    classes: NDArray[np.integer],
    d_ff_by_layer: Sequence[int],
    *,
    fraction: float,
    state: str | None,
    method: str,
    seed: int,
    insufficient_pool: str = "error",
) -> dict[int, NDArray[np.int64]]:
    """Select up to the requested fraction of all neurons per layer for one sample.

    State-specific groups draw ``floor(fraction * d_ff_l)`` neurons from the
    sample-conditional state pool. With ``insufficient_pool="all"``, a smaller
    state pool is deleted in full. Global random always draws the target count.
    Max/min ties are resolved by increasing neuron index.
    """
    q_values = np.asarray(q, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.int64)
    widths = np.asarray(d_ff_by_layer, dtype=np.int64)
    if q_values.ndim != 3 or q_values.shape[1] != 2:
        raise ValueError("q must have shape [layers, 2, maximum_d_ff].")
    if class_values.shape != (q_values.shape[0], q_values.shape[2]):
        raise ValueError("classes must align with q as [layers, maximum_d_ff].")
    if widths.shape != (q_values.shape[0],) or np.any(widths < 1) or np.any(widths > q_values.shape[2]):
        raise ValueError("d_ff_by_layer must provide one valid width per layer.")
    if method not in ABLATION_METHODS:
        raise ValueError(f"Unknown ablation method: {method}")
    if insufficient_pool not in {"error", "all"}:
        raise ValueError(f"Unknown insufficient-pool policy: {insufficient_pool}")
    if state is None and method != "random":
        raise ValueError("Only the global random condition may omit a response state.")
    if state is not None and state not in STATE_TO_CLASS_ID:
        raise ValueError(f"Unknown response state: {state}")

    generator = np.random.default_rng(seed)
    selected: dict[int, NDArray[np.int64]] = {}
    for layer, width in enumerate(widths.tolist()):
        count = per_layer_deletion_count(width, fraction)
        if state is None:
            candidates = np.arange(width, dtype=np.int64)
        else:
            candidates = np.flatnonzero(class_values[layer, :width] == STATE_TO_CLASS_ID[state]).astype(np.int64)
        if len(candidates) < count:
            if insufficient_pool == "error" or state is None:
                raise ValueError(
                    f"Layer {layer} has only {len(candidates)} {state or 'global'} candidates; {count} are required."
                )
            count = len(candidates)
        if method == "random":
            indices = np.sort(generator.choice(candidates, size=count, replace=False).astype(np.int64))
        else:
            strengths = response_strength(q_values[layer, :, :width], state)[candidates]
            if not np.isfinite(strengths).all():
                raise ValueError(f"Layer {layer} contains non-finite response strengths.")
            primary = -strengths if method == "max" else strengths
            order = np.lexsort((candidates, primary))
            indices = np.sort(candidates[order[:count]])
        selected[layer] = indices
    return selected


class DownProjectionAblation(AbstractContextManager["DownProjectionAblation"]):
    """Zero selected post-gating coordinates, exactly matching zeroed down-projection columns."""

    def __init__(self, bindings: Iterable[FFNBinding], selected: Mapping[int, NDArray[np.integer]]) -> None:
        self._bindings = tuple(bindings)
        self._selected = {int(layer): np.asarray(indices, dtype=np.int64) for layer, indices in selected.items()}
        self._handles: list[Any] = []

    def __enter__(self) -> DownProjectionAblation:
        if self._handles:
            raise RuntimeError("Ablation hooks are already installed.")
        if set(self._selected) != {binding.layer for binding in self._bindings}:
            raise ValueError("Ablation indices must be supplied for every discovered decoder layer.")
        for binding in self._bindings:
            indices = self._selected[binding.layer]
            if indices.ndim != 1 or len(indices) != len(np.unique(indices)):
                raise ValueError(f"Layer {binding.layer} indices must be unique and one-dimensional.")
            if np.any(indices < 0) or np.any(indices >= binding.d_ff):
                raise IndexError(f"Layer {binding.layer} ablation index is outside d_ff={binding.d_ff}.")

            def hook(
                _module: nn.Module,
                inputs: tuple[Any, ...],
                *,
                layer: int = binding.layer,
            ) -> tuple[Any, ...]:
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise TypeError(f"Layer {layer} down projection did not receive a tensor input.")
                masked = inputs[0].clone()
                layer_indices = torch.as_tensor(self._selected[layer], device=masked.device, dtype=torch.long)
                masked.index_fill_(-1, layer_indices, 0)
                return (masked, *inputs[1:])

            self._handles.append(binding.down_projection.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()


_ARTICLES = {"a", "an", "the"}
_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hes": "he's",
    "im": "i'm",
    "isnt": "isn't",
    "itll": "it'll",
    "its": "it's",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "wheres": "where's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
}
_PUNCTUATION = (";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!")
_COMMA_BETWEEN_DIGITS = re.compile(r"(?<=\d),(?=\d)")
_PERIOD_NOT_DECIMAL = re.compile(r"(?<!\d)\.(?!\d)")


def normalize_vqa_answer(answer: object) -> str:
    """Apply the official VQA lowercase, punctuation, digit, article, and contraction normalization."""
    value = str(answer).replace("\n", " ").replace("\t", " ").strip().lower()
    value = _COMMA_BETWEEN_DIGITS.sub("", value)
    value = _PERIOD_NOT_DECIMAL.sub("", value)
    for punctuation in _PUNCTUATION:
        if punctuation == ",":
            value = value.replace(punctuation, " ")
        elif f"{punctuation} " in value or f" {punctuation}" in value:
            value = value.replace(punctuation, "")
        else:
            value = value.replace(punctuation, " ")
    words = []
    for word in value.split():
        word = _NUMBER_MAP.get(word, word)
        if word not in _ARTICLES:
            words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def textvqa_soft_accuracy(prediction: object, answers: Sequence[object]) -> float:
    """Compute the official leave-one-annotator-out VQA soft accuracy."""
    normalized_prediction = normalize_vqa_answer(prediction)
    normalized_answers = [normalize_vqa_answer(answer) for answer in answers]
    if not normalized_answers:
        raise ValueError("TextVQA scoring requires at least one human answer.")
    accuracies = []
    for held_out_index in range(len(normalized_answers)):
        matches = sum(
            answer == normalized_prediction
            for index, answer in enumerate(normalized_answers)
            if index != held_out_index
        )
        accuracies.append(min(1.0, matches / 3.0))
    return float(np.mean(accuracies))


def _chartqa_number(value: object) -> float | None:
    text = str(value).strip()
    percentage = text.endswith("%")
    if percentage:
        text = text[:-1]
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return None
    return number / 100.0 if percentage else number


def chartqa_relaxed_accuracy(prediction: object, answer: object, tolerance: float = 0.05) -> float:
    """Score ChartQA numeric answers within 5% and non-numeric answers by exact match."""
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("tolerance must lie in [0, 1).")
    predicted_number = _chartqa_number(prediction)
    answer_number = _chartqa_number(answer)
    if predicted_number is not None and answer_number is not None:
        if answer_number == 0.0:
            return float(predicted_number == 0.0)
        return float(abs(predicted_number - answer_number) / abs(answer_number) <= tolerance)
    return float(str(prediction).strip().lower() == str(answer).strip().lower())


def extract_choice(prediction: object) -> str:
    """Extract one A-D option label from a concise or sentence-form model answer."""
    text = str(prediction).strip().upper()
    direct = re.fullmatch(r"[\s\(\[]*([A-D])[\s\)\].,:;!-]*", text)
    if direct:
        return direct.group(1)
    patterns = (
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*[\(\[]?([A-D])\b",
        r"^\s*[\(\[]?([A-D])[\)\].,:;!-]",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def extract_yes_no(prediction: object) -> str:
    """Extract a leading or explicitly stated yes/no answer."""
    text = normalize_vqa_answer(prediction)
    match = re.search(r"(?:^|\banswer is\s+)(yes|no)\b", text)
    return match.group(1) if match else ""


def score_prediction(dataset: str, prediction: object, answer: object) -> float:
    """Dispatch the frozen-dataset evaluation metric and return a score in [0, 1]."""
    if dataset == "mme":
        return float(extract_yes_no(prediction) == str(answer).strip().lower())
    if dataset == "mmbench":
        return float(extract_choice(prediction) == str(answer).strip().upper())
    if dataset == "textvqa":
        if isinstance(answer, (str, bytes)) or not isinstance(answer, Iterable):
            raise TypeError("TextVQA ground truth must be a sequence of human answers.")
        return textvqa_soft_accuracy(prediction, list(answer))
    if dataset == "chartqa":
        return chartqa_relaxed_accuracy(prediction, answer)
    raise ValueError(f"Unsupported ablation dataset: {dataset}")
