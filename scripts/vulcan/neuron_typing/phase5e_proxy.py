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

"""Gold-caption proxy metrics and exact static masks for Phase 5E."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from run_phase5_single_sample_frontier import normalize_global_scores, valid_next_token_tensors


def parse_deletion_budgets(value: str) -> list[int]:
    budgets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets or any(item <= 0 for item in budgets):
        raise ValueError(f"Deletion budgets must be positive integers, got {budgets}.")
    return list(dict.fromkeys(budgets))


def build_exact_deletion_masks(
    scores: dict[int, torch.Tensor],
    deletion_budget: int,
    *,
    global_normalization: str = "layer_mean",
) -> dict[int, torch.Tensor]:
    """Delete the globally lowest-scoring neurons with a stable exact budget."""
    if not scores:
        raise ValueError("Saliency scores are empty.")
    normalized = normalize_global_scores(scores, global_normalization)
    layers = sorted(normalized)
    flat = torch.cat([normalized[layer].float().cpu() for layer in layers])
    if not 0 < deletion_budget < flat.numel():
        raise ValueError(f"deletion_budget must lie in [1, {flat.numel() - 1}], got {deletion_budget}.")
    deleted = torch.argsort(flat, descending=False, stable=True)[:deletion_budget]
    flat_mask = torch.zeros(flat.numel(), dtype=torch.bool)
    flat_mask[deleted] = True
    masks = {}
    offset = 0
    for layer in layers:
        width = normalized[layer].numel()
        masks[layer] = flat_mask[offset : offset + width].clone()
        offset += width
    return masks


def decode_gold_caption(tokenizer, valid_labels: torch.Tensor) -> str:
    """Decode only supervised response tokens, excluding template stop tokens."""
    return tokenizer.decode(valid_labels.detach().cpu().tolist(), skip_special_tokens=True).strip()


@torch.no_grad()
def evaluate_gold_proxy(
    model: torch.nn.Module,
    batch: dict[str, Any],
    reference_trace: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate NLL, reference KL, and gold-token margins on gold prefixes."""
    labels = batch["labels"]
    model_inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = model(**model_inputs, use_cache=False)
    logits, valid_labels = valid_next_token_tensors(outputs.logits, labels)
    logits = logits.float()
    reference_logits = reference_trace["valid_logits"].to(logits.device).float()
    expected_labels = reference_trace["valid_labels"].to(valid_labels.device)
    if not torch.equal(valid_labels, expected_labels):
        raise RuntimeError("Gold-caption token positions differ from the frozen reference trace.")

    log_probs = F.log_softmax(logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    per_token_nll = -log_probs.gather(1, valid_labels[:, None]).squeeze(1)
    per_token_kl = F.kl_div(log_probs, reference_log_probs, log_target=True, reduction="none").sum(dim=-1)
    gold_logits = logits.gather(1, valid_labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, valid_labels[:, None], float("-inf"))
    margins = gold_logits - competitors.max(dim=-1).values
    top1 = logits.argmax(dim=-1).eq(valid_labels)
    nll = float(per_token_nll.mean())
    return {
        "num_gold_tokens": int(valid_labels.numel()),
        "mean_nll": nll,
        "perplexity": math.exp(min(nll, 100.0)),
        "delta_nll": nll - float(reference_trace["nll"]),
        "mean_kl": float(per_token_kl.mean()),
        "max_kl": float(per_token_kl.max()),
        "per_token_kl": per_token_kl.detach().cpu().tolist(),
        "gold_top1_accuracy": float(top1.float().mean()),
        "mean_gold_logit_margin": float(margins.mean()),
        "min_gold_logit_margin": float(margins.min()),
        "per_token_gold_logit_margin": margins.detach().cpu().tolist(),
    }
