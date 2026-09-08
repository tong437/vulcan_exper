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

"""Cached-decode utilities for Phase-5 single-sample gate optimization."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn.functional as F


def generation_inputs(prompt_inputs: dict[str, Any], tokenizer, max_new_tokens: int) -> dict[str, Any]:
    result = dict(prompt_inputs)
    result.update(
        {
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_logits": True,
        }
    )
    return result


@torch.no_grad()
def generate_cached_teacher_trace(
    model: torch.nn.Module,
    tokenizer,
    prompt_inputs: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    output = model.generate(**generation_inputs(prompt_inputs, tokenizer, max_new_tokens))
    prompt_length = prompt_inputs["input_ids"].shape[-1]
    token_ids = output.sequences[0, prompt_length:].detach().cpu()
    raw_logits = torch.stack([row[0].detach().float().cpu() for row in output.logits])
    if raw_logits.shape[0] != token_ids.numel():
        raise RuntimeError(f"Generation returned {token_ids.numel()} tokens but {raw_logits.shape[0]} logit rows.")
    return {
        "token_ids": token_ids,
        "raw_logits": raw_logits,
        "text": tokenizer.decode(token_ids.tolist(), skip_special_tokens=True),
    }


def detach_qwen_cache(cache: Any) -> Any:
    """Detach Qwen3.5 hybrid-cache state in place for one-token truncated BPTT."""
    if cache is None:
        return None
    found = False
    for attribute in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        values = getattr(cache, attribute, None)
        if values is None:
            continue
        found = True
        setattr(cache, attribute, [value.detach() if torch.is_tensor(value) else value for value in values])
    if not found:
        raise TypeError(f"Unsupported cache type for truncated BPTT: {type(cache).__name__}.")
    return cache


def cached_teacher_forcing_steps(
    model: torch.nn.Module,
    prompt_inputs: dict[str, Any],
    teacher_token_ids: torch.Tensor,
    *,
    before_forward: Callable[[int], None] | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield raw next-token logits while forcing the frozen teacher trajectory."""
    if prompt_inputs["input_ids"].shape[0] != 1 or teacher_token_ids.ndim != 1:
        raise ValueError("Cached teacher forcing requires batch size one and one-dimensional teacher tokens.")
    input_ids = prompt_inputs["input_ids"]
    model_kwargs = {key: value for key, value in prompt_inputs.items() if key != "input_ids"}
    model_kwargs["use_cache"] = True
    model_kwargs = model._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)
    for step, teacher_token in enumerate(teacher_token_ids.to(input_ids.device)):
        if before_forward is not None:
            before_forward(step)
        prepare_kwargs = dict(model_kwargs)
        prepare_kwargs["is_first_iteration"] = step == 0
        if step:
            prepare_kwargs["next_sequence_length"] = 1
        model_inputs = model.prepare_inputs_for_generation(input_ids, **prepare_kwargs)
        outputs = model(**model_inputs, return_dict=True)
        logits = outputs.logits[:, -1, :].float()
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        model_kwargs["past_key_values"] = detach_qwen_cache(model_kwargs.get("past_key_values"))
        yield step, logits
        input_ids = torch.cat([input_ids, teacher_token.reshape(1, 1).to(dtype=input_ids.dtype)], dim=-1)


@torch.no_grad()
def collect_cached_teacher_forced_logits(
    model: torch.nn.Module,
    prompt_inputs: dict[str, Any],
    teacher_token_ids: torch.Tensor,
) -> torch.Tensor:
    rows = [
        logits.detach().cpu() for _, logits in cached_teacher_forcing_steps(model, prompt_inputs, teacher_token_ids)
    ]
    return torch.cat(rows, dim=0)


def generated_token_margins(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Return each generated token's logit margin over its strongest competing token."""
    if logits.ndim != 2 or token_ids.ndim != 1 or logits.shape[0] != token_ids.numel():
        raise ValueError(
            f"Expected [tokens, vocab] logits aligned with [tokens] ids, got {logits.shape} and {token_ids.shape}."
        )
    values = logits.float().cpu()
    labels = token_ids.to(torch.long).cpu()
    label_logits = values.gather(1, labels.unsqueeze(1)).squeeze(1)
    top_values, top_indices = values.topk(k=2, dim=-1)
    strongest_competitor = torch.where(top_indices[:, 0].eq(labels), top_values[:, 1], top_values[:, 0])
    return label_logits - strongest_competitor


def cached_fidelity(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_token_ids: torch.Tensor,
) -> dict[str, Any]:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(f"Cached logit shapes differ: {teacher_logits.shape} vs {student_logits.shape}.")
    labels = teacher_token_ids.to(torch.long).cpu()
    teacher = teacher_logits.float().cpu()
    student = student_logits.float().cpu()
    teacher_log_probs = F.log_softmax(teacher, dim=-1)
    student_log_probs = F.log_softmax(student, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_log_probs, log_target=True, reduction="none").sum(dim=-1)
    teacher_top = teacher.argmax(dim=-1)
    student_top = student.argmax(dim=-1)
    agreement = teacher_top.eq(student_top)
    label_agreement = student_top.eq(labels)
    generated_margins = generated_token_margins(student, labels)
    disagreement = (~agreement).nonzero(as_tuple=False)
    label_disagreement = (~label_agreement).nonzero(as_tuple=False)
    low_margin_indices = (generated_margins < 0.25).nonzero(as_tuple=False).flatten()
    minimum_margin = generated_margins.min()
    minimum_margin_indices = generated_margins.eq(minimum_margin).nonzero(as_tuple=False).flatten()
    return {
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "token_agreement": float(agreement.float().mean()),
        "generated_token_agreement": float(label_agreement.float().mean()),
        "first_disagreement": int(disagreement[0]) if disagreement.numel() else None,
        "first_generated_token_disagreement": (int(label_disagreement[0]) if label_disagreement.numel() else None),
        "teacher_greedy_consistency": float(teacher_top.eq(labels).float().mean()),
        "min_generated_token_margin": float(minimum_margin),
        "min_generated_token_margin_indices": minimum_margin_indices.tolist(),
        "p01_generated_token_margin": float(torch.quantile(generated_margins, 0.01)),
        "mean_generated_token_margin": float(generated_margins.mean()),
        "tokens_below_margin_0_25": int(low_margin_indices.numel()),
        "tokens_below_margin_0_25_indices": low_margin_indices.tolist(),
    }
