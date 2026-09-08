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

"""Phase 5A: empirical single-sample FFN compression frontier.

The model weights remain frozen.  A sample-static binary mask is applied to
every token and decoding step, so a successful mask can later be converted to
a structurally narrower checkpoint.  This script searches deterministic
saliency baselines; learned gate refinement is intentionally a separate stage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_guard import save_manifest  # noqa: E402
from phase3_structural_utils import canonical_json_sha256, masks_to_neuron_ids  # noqa: E402
from run_phase2_ablation import (  # noqa: E402
    MLPNeuronAblator,
    build_dataloader,
    find_down_proj_modules,
    load_yaml,
    move_batch_to_device,
    parse_config_override,
)

from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402


SALIENCY_METHODS = ("activation", "contribution", "taylor", "random")
SELECTION_MODES = ("per_layer", "global")


def parse_csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one keep ratio is required.")
    if any(not 0 <= item <= 1 for item in values):
        raise ValueError(f"Keep ratios must lie in [0, 1], got {values}.")
    return list(dict.fromkeys(values))


def parse_csv_choices(value: str, choices: tuple[str, ...], field: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in choices]
    if not values or invalid:
        raise ValueError(f"{field} must contain values from {choices}; got {values}.")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find an empirical single-sample FFN compression frontier.")
    parser.add_argument("--config", required=True, help="LlamaFactory YAML used to load the model and sample.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--eval_dataset", default=None)
    parser.add_argument("--dataset_stage", choices=["sft", "pt"], default="sft")
    parser.add_argument("--sample_offset", type=int, default=2500)
    parser.add_argument("--allow_short_dataset", action="store_true")
    parser.add_argument("--max_image_repeat", type=int, default=5)
    parser.add_argument("--allow_excessive_image_repeats", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument(
        "--keep_ratios",
        default="0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0",
        help="Comma-separated fractions of FFN neurons to keep.",
    )
    parser.add_argument(
        "--methods",
        default="activation,contribution,taylor",
        help=f"Comma-separated saliency methods from {SALIENCY_METHODS}.",
    )
    parser.add_argument(
        "--selection",
        choices=["per_layer", "global", "both"],
        default="per_layer",
        help="Uniform per-layer budgets, a non-uniform global budget, or both.",
    )
    parser.add_argument(
        "--global_normalization",
        choices=["none", "layer_mean", "layer_rank"],
        default="layer_mean",
        help="Normalize layer scales before global selection; layer_rank is approximately uniform by layer.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--trace_target",
        choices=["teacher_generation", "dataset_labels"],
        default="teacher_generation",
        help="Optimize the generated teacher trajectory (primary) or the dataset response labels (control).",
    )
    parser.add_argument("--kl_tolerance", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2045)
    parser.add_argument("--resume", action="store_true", help="Append new conditions to an existing frontier.")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    for override in args.overrides:
        key, value = parse_config_override(override)
        config[key] = value
    config.update({"do_train": False, "do_eval": False, "do_predict": False})
    config.setdefault("output_dir", "saves/neuron_typing/phase5_tmp")
    if args.model_name_or_path is not None:
        config["model_name_or_path"] = args.model_name_or_path
    if args.dataset is not None:
        config["dataset"] = args.dataset
    if args.eval_dataset is not None:
        config["eval_dataset"] = args.eval_dataset
    if args.preprocessing_num_workers is not None:
        config["preprocessing_num_workers"] = args.preprocessing_num_workers
    return config


def valid_next_token_tensors(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(IGNORE_INDEX)
    if not bool(valid.any()):
        raise RuntimeError("The selected sample has no labeled next-token positions.")
    return shift_logits[valid], shift_labels[valid]


class FFNSaliencyCollector:
    """Capture FFN intermediate activations and retain their gradients."""

    def __init__(self, down_proj_modules: list[torch.nn.Module]):
        self.down_proj_modules = down_proj_modules
        self.activations: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> FFNSaliencyCollector:
        for layer_idx, module in enumerate(self.down_proj_modules):
            self.handles.append(module.register_forward_pre_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if not inputs or not torch.is_tensor(inputs[0]):
                return inputs
            activation = inputs[0]
            if not activation.requires_grad:
                activation.requires_grad_(True)
            activation.retain_grad()
            self.activations[layer_idx] = activation
            return inputs

        return hook


def collect_teacher_trace(
    model: torch.nn.Module,
    batch: dict[str, Any],
    down_proj_modules: list[torch.nn.Module],
) -> tuple[dict[str, Any], dict[str, dict[int, torch.Tensor]]]:
    """Run one differentiable teacher pass and compute three saliency baselines."""
    labels = batch["labels"]
    model_inputs = {key: value for key, value in batch.items() if key != "labels"}
    model.zero_grad(set_to_none=True)
    with FFNSaliencyCollector(down_proj_modules) as collector:
        outputs = model(**model_inputs, use_cache=False)
        valid_logits, valid_labels = valid_next_token_tensors(outputs.logits, labels)
        loss = F.cross_entropy(valid_logits, valid_labels)
        loss.backward()

    saliency: dict[str, dict[int, torch.Tensor]] = {name: {} for name in SALIENCY_METHODS[:-1]}
    for layer_idx, module in enumerate(down_proj_modules):
        activation = collector.activations[layer_idx]
        if activation.grad is None:
            raise RuntimeError(f"No activation gradient was retained for layer {layer_idx}.")
        reduce_dims = tuple(range(activation.ndim - 1))
        activation_score = activation.detach().float().abs().sum(dim=reduce_dims).cpu()
        column_norm = module.weight.detach().float().norm(dim=0).cpu()
        saliency["activation"][layer_idx] = activation_score
        saliency["contribution"][layer_idx] = activation_score * column_norm
        saliency["taylor"][layer_idx] = (
            (activation.detach().float() * activation.grad.detach().float()).abs().sum(dim=reduce_dims).cpu()
        )

    teacher_logits = valid_logits.detach().cpu()
    teacher_labels = valid_labels.detach().cpu()
    teacher_nll = float(F.cross_entropy(teacher_logits, teacher_labels))
    trace = {
        "valid_logits": teacher_logits,
        "valid_labels": teacher_labels,
        "nll": teacher_nll,
        "num_label_tokens": int(teacher_labels.numel()),
    }
    model.zero_grad(set_to_none=True)
    return trace, saliency


def layer_rank_normalize(scores: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Map every layer to deterministic [0, 1] percentile ranks."""
    normalized = {}
    for layer_idx, values in scores.items():
        order = torch.argsort(values, descending=False, stable=True)
        ranks = torch.empty_like(values, dtype=torch.float32)
        if values.numel() == 1:
            ranks[order] = 1.0
        else:
            ranks[order] = torch.arange(values.numel(), dtype=torch.float32) / (values.numel() - 1)
        normalized[layer_idx] = ranks
    return normalized


def normalize_global_scores(scores: dict[int, torch.Tensor], mode: str) -> dict[int, torch.Tensor]:
    if mode == "none":
        return scores
    if mode == "layer_rank":
        return layer_rank_normalize(scores)
    if mode == "layer_mean":
        return {
            layer_idx: values / values.float().abs().mean().clamp_min(1e-12) for layer_idx, values in scores.items()
        }
    raise ValueError(f"Unknown global normalization: {mode}.")


def random_scores_like(scores: dict[int, torch.Tensor], seed: int) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {layer_idx: torch.rand(values.shape, generator=generator) for layer_idx, values in scores.items()}


def build_deletion_masks(
    scores: dict[int, torch.Tensor],
    keep_ratio: float,
    selection: str,
    *,
    global_normalization: str = "layer_mean",
) -> dict[int, torch.Tensor]:
    """Keep the highest scores and return True-for-delete masks."""
    if selection not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode: {selection}.")
    if not 0 <= keep_ratio <= 1:
        raise ValueError(f"keep_ratio must lie in [0, 1], got {keep_ratio}.")
    masks = {layer_idx: torch.ones(values.numel(), dtype=torch.bool) for layer_idx, values in scores.items()}
    if selection == "per_layer":
        for layer_idx, values in scores.items():
            keep_count = math.ceil(values.numel() * keep_ratio) if keep_ratio > 0 else 0
            ranked = torch.argsort(values, descending=True, stable=True)
            masks[layer_idx][ranked[:keep_count]] = False
        return masks

    selected_scores = normalize_global_scores(scores, global_normalization)
    layer_indices = sorted(selected_scores)
    flat_scores = torch.cat([selected_scores[layer_idx] for layer_idx in layer_indices])
    keep_count = math.ceil(flat_scores.numel() * keep_ratio) if keep_ratio > 0 else 0
    ranked = torch.argsort(flat_scores, descending=True, stable=True)
    flat_keep = torch.zeros(flat_scores.numel(), dtype=torch.bool)
    flat_keep[ranked[:keep_count]] = True
    offset = 0
    for layer_idx in layer_indices:
        width = selected_scores[layer_idx].numel()
        masks[layer_idx] = ~flat_keep[offset : offset + width]
        offset += width
    return masks


def summarize_masks(masks: dict[int, torch.Tensor]) -> dict[str, Any]:
    total = sum(mask.numel() for mask in masks.values())
    deleted = sum(int(mask.sum()) for mask in masks.values())
    kept_by_layer = {str(layer): int((~mask).sum()) for layer, mask in sorted(masks.items())}
    deleted_by_layer = {str(layer): int(mask.sum()) for layer, mask in sorted(masks.items())}
    return {
        "total_neurons": total,
        "kept_neurons": total - deleted,
        "deleted_neurons": deleted,
        "keep_ratio": (total - deleted) / total,
        "pruning_ratio": deleted / total,
        "kept_by_layer": kept_by_layer,
        "deleted_by_layer": deleted_by_layer,
    }


def parameter_summary(
    model: torch.nn.Module,
    down_proj_modules: list[torch.nn.Module],
    masks: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Report exact parameter capacity addressed by the FFN channel masks."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    removable_by_layer: dict[str, int] = {}
    removed_by_layer: dict[str, int] = {}
    for layer_idx, down_proj in enumerate(down_proj_modules):
        parent_name = None
        for name, module in model.named_modules():
            if module is down_proj:
                parent_name = name.rsplit(".", maxsplit=1)[0]
                break
        if parent_name is None:
            raise RuntimeError(f"Cannot resolve the parent MLP for down_proj layer {layer_idx}.")
        parent = model.get_submodule(parent_name)
        per_neuron = (
            parent.gate_proj.in_features
            + parent.up_proj.in_features
            + parent.down_proj.out_features
            + int(parent.gate_proj.bias is not None)
            + int(parent.up_proj.bias is not None)
        )
        width = parent.down_proj.in_features
        removable_by_layer[str(layer_idx)] = per_neuron * width
        deleted = 0 if masks is None else int(masks[layer_idx].sum())
        removed_by_layer[str(layer_idx)] = per_neuron * deleted
    removable = sum(removable_by_layer.values())
    removed = sum(removed_by_layer.values())
    return {
        "total_model_parameters": total_parameters,
        "mask_addressable_ffn_parameters": removable,
        "maximum_total_parameter_reduction_ratio": removable / total_parameters,
        "removed_parameters": removed,
        "total_parameter_reduction_ratio": removed / total_parameters,
        "removable_by_layer": removable_by_layer,
        "removed_by_layer": removed_by_layer,
    }


@torch.no_grad()
def evaluate_teacher_fidelity(
    model: torch.nn.Module,
    batch: dict[str, Any],
    teacher_trace: dict[str, Any],
) -> dict[str, Any]:
    labels = batch["labels"]
    model_inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = model(**model_inputs, use_cache=False)
    student_logits, student_labels = valid_next_token_tensors(outputs.logits, labels)
    teacher_logits = teacher_trace["valid_logits"].to(student_logits.device).float()
    if not torch.equal(student_labels.cpu(), teacher_trace["valid_labels"]):
        raise RuntimeError("Teacher and student labeled-token positions are not aligned.")
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    per_token_kl = F.kl_div(student_log_probs, teacher_log_probs, log_target=True, reduction="none").sum(dim=-1)
    teacher_top = teacher_logits.argmax(dim=-1)
    student_top = student_logits.argmax(dim=-1)
    agreements = teacher_top.eq(student_top)
    teacher_consistent = teacher_top.cpu().eq(teacher_trace["valid_labels"])
    consistent_agreements = agreements.cpu()[teacher_consistent]
    generated_token_agreements = student_top.cpu().eq(teacher_trace["valid_labels"])
    teacher_token_logits = student_logits.gather(1, teacher_top[:, None]).squeeze(1)
    competitors = student_logits.clone()
    competitors.scatter_(1, teacher_top[:, None], float("-inf"))
    student_teacher_margin = teacher_token_logits - competitors.max(dim=-1).values
    nll = F.cross_entropy(student_logits, student_labels)
    disagreement = (~agreements).nonzero(as_tuple=False)
    return {
        "nll": float(nll),
        "delta_nll": float(nll) - float(teacher_trace["nll"]),
        "mean_kl": float(per_token_kl.mean()),
        "max_kl": float(per_token_kl.max()),
        "token_agreement": float(agreements.float().mean()),
        "teacher_cache_consistency": float(teacher_consistent.float().mean()),
        "consistent_token_agreement": float(consistent_agreements.float().mean()),
        "generated_token_teacher_forced_agreement": float(generated_token_agreements.float().mean()),
        "first_teacher_forced_disagreement": int(disagreement[0]) if disagreement.numel() else None,
        "min_student_margin_for_teacher_token": float(student_teacher_margin.min()),
    }


def build_prompt_inputs(batch: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Remove the SFT response while retaining image tensors and prompt positions."""
    labels = batch["labels"]
    if labels.shape[0] != 1:
        raise ValueError(f"Phase 5A requires batch size 1, got {labels.shape[0]}.")
    valid_positions = labels[0].ne(IGNORE_INDEX).nonzero(as_tuple=False).flatten()
    if not valid_positions.numel():
        raise RuntimeError("The selected sample has no response labels.")
    prompt_length = int(valid_positions[0])
    sequence_keys = {"input_ids", "attention_mask", "position_ids", "token_type_ids", "cache_position"}
    generation_excluded_keys = {"rope_deltas"}
    prompt = {}
    for key, value in batch.items():
        if key == "labels" or key in generation_excluded_keys:
            continue
        if key in sequence_keys and torch.is_tensor(value) and value.shape[-1] >= prompt_length:
            prompt[key] = value[..., :prompt_length]
        else:
            prompt[key] = value
    return prompt, prompt_length


def build_teacher_rollout_batch(prompt_inputs: dict[str, Any], generated_token_ids: torch.Tensor) -> dict[str, Any]:
    """Build a full-forward batch whose labels are the teacher's generated trajectory."""
    input_ids = prompt_inputs["input_ids"]
    if input_ids.shape[0] != 1 or generated_token_ids.ndim != 1:
        raise ValueError("Teacher rollout construction requires one prompt and a one-dimensional generation.")
    generated = generated_token_ids.to(device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
    full_input_ids = torch.cat([input_ids, generated], dim=-1)
    prompt_attention = prompt_inputs.get("attention_mask", torch.ones_like(input_ids))
    full_attention = torch.cat([prompt_attention, torch.ones_like(generated)], dim=-1)
    labels = torch.full_like(full_input_ids, IGNORE_INDEX)
    labels[:, input_ids.shape[-1] :] = generated
    # Let the multimodal model recompute M-RoPE positions for the longer rollout.
    excluded = {"input_ids", "attention_mask", "position_ids", "token_type_ids", "cache_position", "rope_deltas"}
    rollout = {key: value for key, value in prompt_inputs.items() if key not in excluded}
    rollout.update({"input_ids": full_input_ids, "attention_mask": full_attention, "labels": labels})
    return rollout


@torch.no_grad()
def generate_trace(
    model: torch.nn.Module,
    tokenizer,
    prompt_inputs: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    generation_inputs = dict(prompt_inputs)
    generation_inputs.update(
        {
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )
    sequences = model.generate(**generation_inputs)
    prompt_length = prompt_inputs["input_ids"].shape[-1]
    generated = sequences[0, prompt_length:].detach().cpu()
    return {
        "token_ids": generated,
        "text": tokenizer.decode(generated.tolist(), skip_special_tokens=True),
    }


def compare_generation(teacher: dict[str, Any], student: dict[str, Any]) -> dict[str, Any]:
    teacher_ids = teacher["token_ids"].tolist()
    student_ids = student["token_ids"].tolist()
    common_prefix = 0
    for teacher_id, student_id in zip(teacher_ids, student_ids):
        if teacher_id != student_id:
            break
        common_prefix += 1
    exact = teacher_ids == student_ids
    return {
        "exact_match": exact,
        "common_prefix_tokens": common_prefix,
        "first_divergence": None if exact else common_prefix,
        "teacher_token_count": len(teacher_ids),
        "student_token_count": len(student_ids),
        "student_token_ids": student_ids,
        "student_text": student["text"],
    }


def condition_name(method: str, selection: str, keep_ratio: float) -> str:
    ratio = f"{keep_ratio:.8f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{method}__{selection}__keep_{ratio}"


def json_ready_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "nll": trace["nll"],
        "num_label_tokens": trace["num_label_tokens"],
        "valid_labels": trace["valid_labels"].tolist(),
    }


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def merge_resume_result(existing: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """Validate run identity and extend an existing frontier in place."""
    identity_fields = (
        "model_name_or_path",
        "dataset",
        "dataset_stage",
        "sample_offset",
        "global_normalization",
        "max_new_tokens",
        "trace_target",
        "kl_tolerance",
        "seed",
    )
    mismatches = {
        field: (existing["config"].get(field), fresh["config"].get(field))
        for field in identity_fields
        if existing["config"].get(field) != fresh["config"].get(field)
    }
    if mismatches:
        raise ValueError(f"Cannot resume a different Phase 5A run; mismatched fields: {mismatches}.")
    if existing.get("teacher", {}).get("generated_token_ids") != fresh["teacher"]["generated_token_ids"]:
        raise ValueError("Cannot resume because the teacher generation is not reproducible.")
    for field in ("methods", "selections", "keep_ratios"):
        existing["config"][field] = list(
            dict.fromkeys([*existing["config"].get(field, []), *fresh["config"].get(field, [])])
        )
    existing["complete"] = False
    existing["parameter_scope"] = fresh["parameter_scope"]
    existing["layer_widths"] = fresh["layer_widths"]
    return existing


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be positive.")
    if args.kl_tolerance < 0:
        raise ValueError("--kl_tolerance must be non-negative.")
    keep_ratios = parse_csv_floats(args.keep_ratios)
    methods = parse_csv_choices(args.methods, SALIENCY_METHODS, "methods")
    selections = list(SELECTION_MODES) if args.selection == "both" else [args.selection]
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "frontier.json"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Phase 5A output already exists: {result_path}. Pass --resume to append conditions.")
    config = prepare_config(args)
    model_args, data_args, _, finetuning_args, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    dataloader, manifest = build_dataloader(
        config,
        model,
        tokenizer_module,
        template,
        batch_size=1,
        num_workers=args.num_workers,
        sample_offset=args.sample_offset,
        max_samples=1,
        allow_short_dataset=args.allow_short_dataset,
        max_image_repeat=args.max_image_repeat,
        allow_excessive_image_repeats=args.allow_excessive_image_repeats,
        dataset_stage=args.dataset_stage,
    )
    manifest["role"] = "phase5_single_sample_search"
    manifest_path = output_dir / "sample_manifest.json"
    save_manifest(manifest, manifest_path)
    dataset_batch = move_batch_to_device(next(iter(dataloader)), device)
    down_proj_modules = find_down_proj_modules(model)
    prompt_inputs, prompt_length = build_prompt_inputs(dataset_batch)
    teacher_generation = generate_trace(model, tokenizer, prompt_inputs, args.max_new_tokens)
    batch = (
        build_teacher_rollout_batch(prompt_inputs, teacher_generation["token_ids"])
        if args.trace_target == "teacher_generation"
        else dataset_batch
    )
    teacher_trace, saliency = collect_teacher_trace(model, batch, down_proj_modules)

    score_artifact = {
        "scores": saliency,
        "layer_widths": {layer: int(score.numel()) for layer, score in saliency["activation"].items()},
    }
    torch.save(score_artifact, output_dir / "saliency_scores.pt")
    torch.save(
        {
            "valid_logits": teacher_trace["valid_logits"].to(torch.float16),
            "valid_labels": teacher_trace["valid_labels"],
        },
        output_dir / "teacher_logits.pt",
    )

    fresh_result: dict[str, Any] = {
        "complete": False,
        "interpretation": (
            "A feasible condition is a constructive lower bound on achievable pruning, not proof of the global "
            "combinatorial optimum. The mask is sample-static and model weights are frozen."
        ),
        "config": {
            "config_path": args.config,
            "model_name_or_path": config.get("model_name_or_path"),
            "dataset": config.get("eval_dataset") or config.get("dataset"),
            "dataset_stage": args.dataset_stage,
            "sample_offset": args.sample_offset,
            "sample_manifest": str(manifest_path),
            "methods": methods,
            "selections": selections,
            "keep_ratios": keep_ratios,
            "global_normalization": args.global_normalization,
            "max_new_tokens": args.max_new_tokens,
            "trace_target": args.trace_target,
            "kl_tolerance": args.kl_tolerance,
            "seed": args.seed,
        },
        "teacher": {
            **json_ready_trace(teacher_trace),
            "prompt_tokens": prompt_length,
            "generated_token_ids": teacher_generation["token_ids"].tolist(),
            "generated_text": teacher_generation["text"],
        },
        "layer_widths": score_artifact["layer_widths"],
        "parameter_scope": parameter_summary(model, down_proj_modules),
        "conditions": {},
        "best_feasible": {},
    }
    if args.resume and result_path.is_file():
        result = merge_resume_result(json.loads(result_path.read_text(encoding="utf-8")), fresh_result)
    else:
        result = fresh_result
    write_json(result_path, result)

    base_scores = saliency["activation"]
    method_scores = dict(saliency)
    if "random" in methods:
        method_scores["random"] = random_scores_like(base_scores, args.seed)

    seen_masks = {condition["mask_hash"]: name for name, condition in result["conditions"].items()}
    for method in methods:
        for selection in selections:
            for keep_ratio in keep_ratios:
                masks = build_deletion_masks(
                    method_scores[method],
                    keep_ratio,
                    selection,
                    global_normalization=args.global_normalization,
                )
                neuron_ids = masks_to_neuron_ids(masks)
                mask_hash = canonical_json_sha256(neuron_ids)
                name = condition_name(method, selection, keep_ratio)
                if name in result["conditions"]:
                    print(f"Phase 5A skipping completed condition {name}", flush=True)
                    continue
                mask_path = output_dir / "masks" / f"{name}.json"
                write_json(mask_path, neuron_ids)
                print(f"Phase 5A evaluating {name}", flush=True)
                duplicate_of = seen_masks.get(mask_hash)
                seen_masks.setdefault(mask_hash, name)
                summary = summarize_masks(masks)
                if duplicate_of is None:
                    context = (
                        MLPNeuronAblator(model, masks)
                        if any(bool(mask.any()) for mask in masks.values())
                        else nullcontext()
                    )
                    with context:
                        fidelity = evaluate_teacher_fidelity(model, batch, teacher_trace)
                        student_generation = generate_trace(model, tokenizer, prompt_inputs, args.max_new_tokens)
                    generation = compare_generation(teacher_generation, student_generation)
                    feasible = bool(
                        generation["exact_match"]
                        and fidelity["consistent_token_agreement"] == 1.0
                        and fidelity["mean_kl"] <= args.kl_tolerance
                    )
                else:
                    duplicate = result["conditions"][duplicate_of]
                    fidelity = duplicate["teacher_fidelity"]
                    generation = duplicate["generation"]
                    feasible = duplicate["feasible"]
                result["conditions"][name] = {
                    "method": method,
                    "selection": selection,
                    "requested_keep_ratio": keep_ratio,
                    "mask_hash": mask_hash,
                    "mask_file": str(mask_path),
                    "duplicate_of": duplicate_of,
                    "mask_summary": summary,
                    "parameter_summary": parameter_summary(model, down_proj_modules, masks),
                    "teacher_fidelity": fidelity,
                    "generation": generation,
                    "feasible": feasible,
                }
                frontier_key = f"{method}__{selection}"
                current = result["best_feasible"].get(frontier_key)
                if feasible and (current is None or summary["kept_neurons"] < current["kept_neurons"]):
                    result["best_feasible"][frontier_key] = {
                        "condition": name,
                        "kept_neurons": summary["kept_neurons"],
                        "keep_ratio": summary["keep_ratio"],
                        "pruning_ratio": summary["pruning_ratio"],
                    }
                write_json(result_path, result)

    result["complete"] = True
    write_json(result_path, result)
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"complete": result["complete"], "best_feasible": result["best_feasible"]}, indent=2))


if __name__ == "__main__":
    main()
