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

"""Weight-free processor and decoder-layout adapters for supported VLM families."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .hooks import masks_from_spans
from .interfaces import PreparedInputs, TokenRoleMasks


SUPPORTED_FAMILIES = ("qwen3_5", "qwen3_vl", "llava", "gemma4")


@dataclass(frozen=True)
class ModelLayout:
    """Config-only facts needed before any model weights are loaded."""

    family: str
    architecture: str
    model_path: Path
    decoder_layers: int
    d_ff: int
    image_token_id: int
    visual_tokens_per_image: int | None
    d_ff_by_layer: tuple[int, ...] | None = None


def load_model_layout(model_values: dict[str, Any]) -> ModelLayout:
    """Validate a study model config against its local config.json without loading weights."""
    family = str(model_values["family"])
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported VLM family: {family}")
    model_path = Path(model_values["model_id_or_path"]).resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Local model config is missing: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    expected_architecture = str(model_values["architecture"])
    if expected_architecture not in raw.get("architectures", []):
        raise ValueError(f"Expected architecture {expected_architecture} is absent from {config_path}.")

    decoder = model_values["decoder"]
    raw_text = raw.get("text_config", {})
    raw_layers = raw_text.get("num_hidden_layers")
    raw_d_ff = raw_text.get("intermediate_size")
    expected_layers = int(decoder["expected_layers"])
    expected_d_ff = int(decoder["expected_d_ff"])
    if raw_layers is not None and int(raw_layers) != expected_layers:
        raise ValueError(f"Decoder-layer mismatch: config={raw_layers}, expected={expected_layers}.")
    if raw_d_ff is not None and int(raw_d_ff) != expected_d_ff:
        raise ValueError(f"d_ff mismatch: config={raw_d_ff}, expected={expected_d_ff}.")

    expected_widths = [expected_d_ff] * expected_layers
    for block in decoder.get("expected_d_ff_by_layer", []):
        start, stop, width = int(block["start"]), int(block["stop"]), int(block["d_ff"])
        if not 0 <= start < stop <= expected_layers:
            raise ValueError(f"Invalid layer-width block [{start}, {stop}).")
        expected_widths[start:stop] = [width] * (stop - start)

    image_token_id = raw.get("image_token_id", raw.get("image_token_index"))
    if image_token_id is None:
        raise ValueError(f"Image token ID is absent from {config_path}.")
    visual_tokens = None
    if family == "llava":
        vision = raw["vision_config"]
        visual_tokens = (int(vision["image_size"]) // int(vision["patch_size"])) ** 2
    return ModelLayout(
        family=family,
        architecture=expected_architecture,
        model_path=model_path,
        decoder_layers=expected_layers,
        d_ff=expected_d_ff,
        image_token_id=int(image_token_id),
        visual_tokens_per_image=int(visual_tokens) if visual_tokens is not None else None,
        d_ff_by_layer=tuple(expected_widths),
    )


def find_unique_token_span(input_ids: Tensor, question_token_ids: Tensor) -> tuple[int, int]:
    """Locate one exact question-token subsequence and reject ambiguous templates."""
    if input_ids.ndim != 1 or question_token_ids.ndim != 1 or not len(question_token_ids):
        raise ValueError("input_ids and nonempty question_token_ids must be one-dimensional.")
    width = len(question_token_ids)
    matches = [
        start
        for start in range(len(input_ids) - width + 1)
        if torch.equal(input_ids[start : start + width], question_token_ids)
    ]
    if len(matches) != 1:
        raise ValueError(f"Question token sequence must occur exactly once; found {len(matches)} matches.")
    return matches[0], matches[0] + width


def find_question_token_span(input_ids: Tensor, tokenizer: Any, question: str) -> tuple[int, int]:
    """Locate question tokens while preserving tokenizer context at chat-template boundaries.

    SentencePiece tokenizers can encode the first question token differently after a
    newline than in an isolated string.  Matching a small, explicit template prefix
    and then excluding its tokens keeps the mask on question content only.
    """
    candidates: list[tuple[Tensor, int]] = []
    # Hugging Face chat templates commonly trim surrounding whitespace from text
    # content before tokenization.  Keep the original form first, then admit only
    # that template-equivalent normalization as a fallback.
    question_variants = [question]
    stripped_question = question.strip()
    if stripped_question != question:
        question_variants.append(stripped_question)
    for question_variant in question_variants:
        for prefix in ("", "\n", " ", "\n\n"):
            candidate = tokenizer(prefix + question_variant, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0]
            prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            candidates.append((candidate, len(prefix_ids)))

    matches: list[tuple[int, tuple[int, int]]] = []
    for candidate, prefix_width in candidates:
        if len(candidate) <= prefix_width:
            continue
        question_ids = candidate[prefix_width:]
        width = len(question_ids)
        for start in range(len(input_ids) - width + 1):
            if torch.equal(input_ids[start : start + width], question_ids):
                matches.append((width, (start, start + width)))
    if not matches:
        raise ValueError("Question token span must resolve exactly once; found 0 matches.")
    maximum_width = max(width for width, _ in matches)
    longest = {span for width, span in matches if width == maximum_width}
    if len(longest) != 1:
        raise ValueError(f"Question token span must resolve exactly once; found {len(longest)} longest matches.")
    return next(iter(longest))


def resolve_expanded_roles(
    layout: ModelLayout,
    input_ids: Tensor,
    attention_mask: Tensor,
    question_token_ids: Tensor,
) -> TokenRoleMasks:
    """Resolve visual and question positions in processor-expanded decoder input IDs."""
    if input_ids.ndim != 1 or attention_mask.ndim != 1 or input_ids.shape != attention_mask.shape:
        raise ValueError("Single-sample input_ids and attention_mask must be aligned one-dimensional tensors.")
    visual_positions = torch.nonzero(input_ids == layout.image_token_id, as_tuple=False).flatten()
    if not len(visual_positions):
        raise ValueError(f"No image tokens ({layout.image_token_id}) were found for {layout.family}.")
    if layout.visual_tokens_per_image is not None and len(visual_positions) < layout.visual_tokens_per_image:
        raise ValueError(
            f"Processor output has {len(visual_positions)} visual positions; expected at least "
            f"{layout.visual_tokens_per_image} for {layout.family}."
        )
    question_span = find_unique_token_span(input_ids, question_token_ids)
    visual_spans = [(int(position), int(position) + 1) for position in visual_positions.tolist()]
    return masks_from_spans(attention_mask, visual_spans, [question_span])


def resolve_processor_roles(
    layout: ModelLayout,
    input_ids: Tensor,
    attention_mask: Tensor,
    tokenizer: Any,
    question: str,
) -> TokenRoleMasks:
    """Resolve processor-expanded visual positions and context-aware question positions."""
    if input_ids.ndim != 1 or attention_mask.ndim != 1 or input_ids.shape != attention_mask.shape:
        raise ValueError("Single-sample input_ids and attention_mask must be aligned one-dimensional tensors.")
    visual_positions = torch.nonzero(input_ids == layout.image_token_id, as_tuple=False).flatten()
    if not len(visual_positions):
        raise ValueError(f"No image tokens ({layout.image_token_id}) were found for {layout.family}.")
    if layout.visual_tokens_per_image is not None and len(visual_positions) < layout.visual_tokens_per_image:
        raise ValueError(
            f"Processor output has {len(visual_positions)} visual positions; expected at least "
            f"{layout.visual_tokens_per_image} for {layout.family}."
        )
    question_span = find_question_token_span(input_ids, tokenizer, question)
    visual_spans = [(int(position), int(position) + 1) for position in visual_positions.tolist()]
    return masks_from_spans(attention_mask, visual_spans, [question_span])


class ProcessorInputAdapter:
    """Prepare one multimodal sample using a local processor but no model weights."""

    def __init__(self, layout: ModelLayout, processor: Any) -> None:
        self.family = layout.family
        self.layout = layout
        self.processor = processor

    def prepare_inputs(self, sample: dict[str, Any], *, add_generation_prompt: bool = False) -> PreparedInputs:
        question = str(sample["question"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prepared = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = prepared["input_ids"]
        attention_mask = prepared["attention_mask"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("The preparation adapter accepts exactly one sample at a time.")
        roles = resolve_processor_roles(
            self.layout, input_ids[0], attention_mask[0], self.processor.tokenizer, question
        )
        return PreparedInputs(model_inputs=prepared, token_roles=roles)


def load_local_processor_adapter(model_values: dict[str, Any], use_fast: bool = False) -> ProcessorInputAdapter:
    """Load processor/tokenizer assets locally while never constructing a model."""
    import transformers
    from packaging.version import Version
    from transformers import AutoProcessor

    layout = load_model_layout(model_values)
    backend = {True: "torchvision", False: "pil"}[use_fast]
    backend_options = (
        {"backend": backend}
        if layout.family == "llava" and Version(transformers.__version__) >= Version("5.6.0")
        else {"use_fast": use_fast}
    )
    processor = AutoProcessor.from_pretrained(layout.model_path, local_files_only=True, **backend_options)
    if not hasattr(processor, "apply_chat_template") or not hasattr(processor, "tokenizer"):
        raise RuntimeError(
            f"Installed Transformers cannot construct the multimodal processor for {layout.family}; "
            "upgrade to the model configuration's minimum version."
        )
    return ProcessorInputAdapter(layout, processor)
