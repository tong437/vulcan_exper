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

r"""Phase-2 typed neuron ablation evaluation.

This script evaluates whether Phase-1 neuron type scores have causal signal by
zeroing selected FFN intermediate neurons and measuring teacher-forced label
negative log-likelihood/perplexity on an SFT-style dataset.

The metric is deliberately teacher-forced instead of free-form token overlap:
it is more sensitive to small ablations, deterministic under greedy/free-form
generation noise, and works for both VQA answer labels and text-only labels.

Example:
    python scripts/vulcan/neuron_typing/run_phase2_ablation.py \\
        --config scripts/vulcan/neuron_typing/configs/eval_vqa.yaml \\
        --score_file saves/neuron_typing/score_fix_test/scores/neuron_scores.parquet \\
        --output_file saves/neuron_typing/phase2_ablation/vqa_label_nll.json \\
        --max_samples 100 \\
        --batch_size 2 \\
        --num_workers 4 \\
        --ablation none \\
        --ablation multimodal:0.05 \\
        --ablation unknown:0.05 \\
        --ablation layer_random:0.05:multimodal \\
        --ablation random:0.20
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamafactory.data import (  # noqa: E402
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402


TYPE_NAMES = ("visual", "text", "multimodal", "unknown")


@dataclass(frozen=True)
class AblationSpec:
    name: str
    ratio: float = 0.0
    match_type: str | None = None
    seed: int | None = None

    @property
    def result_name(self) -> str:
        parts = [self.name]
        if self.ratio:
            parts.append(f"{self.ratio:g}")
        if self.match_type:
            parts.append(self.match_type)
        if self.seed is not None:
            parts.append(f"seed{self.seed}")
        return ":".join(parts)


class MLPNeuronAblator:
    """Forward pre-hook manager that zeros selected MLP intermediate dimensions."""

    def __init__(self, model: torch.nn.Module, masks_by_layer: dict[int, torch.Tensor]):
        self.model = model
        self.masks_by_layer = masks_by_layer
        self.handles: list[Any] = []
        self.down_proj_modules = find_down_proj_modules(model)

    def __enter__(self) -> MLPNeuronAblator:
        for layer_idx, module in enumerate(self.down_proj_modules):
            mask = self.masks_by_layer.get(layer_idx)
            if mask is None or not bool(mask.any()):
                continue

            self.handles.append(module.register_forward_pre_hook(self._make_hook(layer_idx, mask)))

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()

        self.handles.clear()

    @staticmethod
    def _make_hook(layer_idx: int, mask: torch.Tensor):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if not inputs:
                return inputs

            hidden_states = inputs[0]
            if not torch.is_tensor(hidden_states):
                return inputs

            if hidden_states.shape[-1] != mask.numel():
                raise RuntimeError(
                    f"Ablation mask for layer {layer_idx} has dim {mask.numel()}, "
                    f"but down_proj input has dim {hidden_states.shape[-1]}."
                )

            keep = (~mask).to(device=hidden_states.device, dtype=hidden_states.dtype).view(
                *([1] * (hidden_states.ndim - 1)), -1
            )
            return (hidden_states * keep, *inputs[1:])

        return hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run typed FFN neuron ablations and label-NLL evaluation.")
    parser.add_argument("--config", required=True, help="LlamaFactory YAML config for loading model and eval dataset.")
    parser.add_argument("--score_file", required=True, help="Phase-1 neuron score parquet/csv/jsonl file.")
    parser.add_argument("--output_file", required=True, help="Path to write JSON metrics.")
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        help=(
            "Ablation spec. Repeatable. Formats: none, visual:0.01, text:0.01, multimodal:0.05, "
            "unknown:0.05, unknown_safe:0.05, random:0.20, layer_random:0.05:multimodal."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Override data_args.max_samples before tokenizing.")
    parser.add_argument("--dataset", default=None, help="Override dataset name in the YAML config.")
    parser.add_argument("--eval_dataset", default=None, help="Override eval_dataset name in the YAML config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval dataloader batch size.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override eval dataloader workers.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="Override LlamaFactory dataset preprocessing workers.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--bootstrap_samples", type=int, default=1000, help="Bootstrap samples for NLL/PPL CI.")
    parser.add_argument("--bootstrap_seed", type=int, default=13, help="Bootstrap random seed.")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional dataloader batch limit.")
    parser.add_argument(
        "--selection",
        choices=["per_layer", "global"],
        default="per_layer",
        help="Select top-ratio neurons per layer or globally.",
    )
    parser.add_argument(
        "--score_prefix",
        default=None,
        help="Optional score column prefix. Example: p_ selects p_visual/p_text/...",
    )
    parser.add_argument(
        "--unknown_safe_multimodal_weight",
        type=float,
        default=1.0,
        help="Penalty weight for p_multimodal when selecting unknown_safe neurons.",
    )
    parser.add_argument(
        "--unknown_safe_activation_weight",
        type=float,
        default=0.0,
        help="Penalty weight for normalized mean activation if such a column exists.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices to allow. Default: all layers from score file.",
    )
    parser.add_argument(
        "--dry_run_masks",
        action="store_true",
        help="Only build and summarize masks; do not load model/dataset or evaluate.",
    )
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    if not args.ablation:
        args.ablation = ["none", "multimodal:0.05", "unknown:0.05", "layer_random:0.05:multimodal"]

    return args


def parse_config_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise ValueError(f"Config overrides must use key=value syntax, got: {override}")

    key, value = override.split("=", maxsplit=1)
    key = key.strip()
    if not key:
        raise ValueError(f"Config override key cannot be empty: {override}")

    return key, yaml.safe_load(value)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")

    return data


def parse_ablation_spec(text: str, base_seed: int) -> AblationSpec:
    parts = text.split(":")
    name = parts[0].strip()
    if name == "none":
        if len(parts) != 1:
            raise ValueError(f"`none` ablation does not take extra fields: {text}")
        return AblationSpec(name="none")

    if name not in {*TYPE_NAMES, "unknown_safe", "random", "layer_random"}:
        raise ValueError(f"Unknown ablation type {name!r}.")

    if len(parts) < 2 or not parts[1].strip():
        raise ValueError(f"Ablation {text!r} must provide a ratio, e.g. {name}:0.05.")

    ratio = float(parts[1])
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"Ablation ratio must be in (0, 1], got {ratio}.")

    match_type = None
    seed = None
    if name == "layer_random":
        match_type = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else "multimodal"
        if match_type not in TYPE_NAMES and match_type != "unknown_safe":
            raise ValueError(f"layer_random match type must be one of {TYPE_NAMES} or unknown_safe, got {match_type}.")

    if name in {"random", "layer_random"}:
        seed = base_seed

    return AblationSpec(name=name, ratio=ratio, match_type=match_type, seed=seed)


def read_score_table(path: str | Path):
    import pandas as pd

    score_path = Path(path)
    suffix = score_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        table = pd.read_parquet(score_path)
    elif suffix == ".csv":
        table = pd.read_csv(score_path)
    elif suffix in {".jsonl", ".json"}:
        table = pd.read_json(score_path, lines=(suffix == ".jsonl"))
    else:
        raise ValueError(f"Unsupported score file suffix: {score_path.suffix}")

    if table.empty:
        raise ValueError(f"Score file is empty: {score_path}")

    return table


def find_first_existing(columns, candidates: list[str], field_name: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(f"Cannot find {field_name} column. Tried: {candidates}. Available: {list(columns)}")


def infer_score_columns(table, score_prefix: str | None) -> tuple[str, str, dict[str, str], str | None]:
    columns = set(table.columns)
    layer_col = find_first_existing(columns, ["layer", "layer_idx", "layer_id", "layer_index"], "layer")
    neuron_col = find_first_existing(
        columns, ["neuron", "neuron_idx", "neuron_id", "neuron_index", "index", "ffn_idx"], "neuron"
    )

    score_cols: dict[str, str] = {}
    for type_name in TYPE_NAMES:
        candidates = []
        if score_prefix is not None:
            candidates.append(f"{score_prefix}{type_name}")

        candidates.extend(
            [
                f"p_{type_name}",
                f"{type_name}_prob",
                f"{type_name}_score",
                f"score_{type_name}",
                type_name,
            ]
        )
        score_cols[type_name] = find_first_existing(columns, candidates, f"{type_name} score")

    activation_col = None
    for candidate in ["mean_activation", "activation_mean", "mean_act", "avg_activation", "outlier_score"]:
        if candidate in columns:
            activation_col = candidate
            break

    return layer_col, neuron_col, score_cols, activation_col


def parse_layers(layers: str | None) -> set[int] | None:
    if layers is None or not layers.strip():
        return None

    return {int(item.strip()) for item in layers.split(",") if item.strip()}


def get_layer_dims(table, layer_col: str, neuron_col: str) -> dict[int, int]:
    dims: dict[int, int] = {}
    for layer_idx, group in table.groupby(layer_col):
        max_neuron = int(group[neuron_col].max())
        count = int(group[neuron_col].nunique())
        dims[int(layer_idx)] = max(max_neuron + 1, count)

    return dict(sorted(dims.items()))


def select_top_indices(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    k = max(1, math.ceil(scores.numel() * ratio))
    return scores.topk(k=k, largest=True).indices


def build_score_vector(group, neuron_col: str, score_col: str, dim: int) -> torch.Tensor:
    scores = torch.full((dim,), float("-inf"), dtype=torch.float32)
    neuron_ids = torch.tensor(group[neuron_col].to_numpy(), dtype=torch.long)
    values = torch.tensor(group[score_col].to_numpy(), dtype=torch.float32)
    scores[neuron_ids] = values.nan_to_num(nan=float("-inf"))
    return scores


def build_unknown_safe_vector(
    group,
    neuron_col: str,
    score_cols: dict[str, str],
    activation_col: str | None,
    dim: int,
    multimodal_weight: float,
    activation_weight: float,
) -> torch.Tensor:
    unknown = build_score_vector(group, neuron_col, score_cols["unknown"], dim)
    multimodal = build_score_vector(group, neuron_col, score_cols["multimodal"], dim).nan_to_num(neginf=0.0)
    score = unknown - multimodal_weight * multimodal
    if activation_col is not None and activation_weight:
        activation = build_score_vector(group, neuron_col, activation_col, dim).nan_to_num(neginf=0.0)
        finite = torch.isfinite(activation)
        if bool(finite.any()):
            values = activation[finite]
            denom = (values.max() - values.min()).clamp_min(1e-8)
            activation_norm = torch.zeros_like(activation)
            activation_norm[finite] = (activation[finite] - values.min()) / denom
            score = score - activation_weight * activation_norm

    return score


def build_type_mask(
    table,
    spec: AblationSpec,
    layer_col: str,
    neuron_col: str,
    score_cols: dict[str, str],
    activation_col: str | None,
    layer_dims: dict[int, int],
    allowed_layers: set[int] | None,
    selection: str,
    unknown_safe_multimodal_weight: float,
    unknown_safe_activation_weight: float,
) -> dict[int, torch.Tensor]:
    masks = {layer_idx: torch.zeros(dim, dtype=torch.bool) for layer_idx, dim in layer_dims.items()}
    if spec.name == "none":
        return masks

    if spec.name == "random":
        generator = torch.Generator()
        generator.manual_seed(spec.seed if spec.seed is not None else 0)
        if selection == "global":
            candidates: list[tuple[int, int]] = []
            for layer_idx, dim in layer_dims.items():
                if allowed_layers is not None and layer_idx not in allowed_layers:
                    continue

                candidates.extend((layer_idx, neuron_idx) for neuron_idx in range(dim))

            selected_count = max(1, math.ceil(len(candidates) * spec.ratio))
            perm = torch.randperm(len(candidates), generator=generator)[:selected_count].tolist()
            for index in perm:
                layer_idx, neuron_idx = candidates[index]
                masks[layer_idx][neuron_idx] = True
        else:
            for layer_idx, dim in layer_dims.items():
                if allowed_layers is not None and layer_idx not in allowed_layers:
                    continue

                selected_count = max(1, math.ceil(dim * spec.ratio))
                selected = torch.randperm(dim, generator=generator)[:selected_count]
                masks[layer_idx][selected] = True

        return masks

    score_type = spec.match_type if spec.name == "layer_random" else spec.name
    if score_type is None:
        raise ValueError(f"Cannot build mask for {spec}.")

    grouped = {int(layer_idx): group for layer_idx, group in table.groupby(layer_col)}
    if selection == "global" and spec.name != "layer_random":
        candidates: list[tuple[int, int]] = []
        candidate_scores: list[float] = []
        for layer_idx, dim in layer_dims.items():
            if allowed_layers is not None and layer_idx not in allowed_layers:
                continue

            group = grouped[layer_idx]
            if score_type == "unknown_safe":
                scores = build_unknown_safe_vector(
                    group,
                    neuron_col,
                    score_cols,
                    activation_col,
                    dim,
                    unknown_safe_multimodal_weight,
                    unknown_safe_activation_weight,
                )
            else:
                scores = build_score_vector(group, neuron_col, score_cols[score_type], dim)

            for neuron_idx, value in enumerate(scores.tolist()):
                candidates.append((layer_idx, neuron_idx))
                candidate_scores.append(value)

        selected_count = max(1, math.ceil(len(candidates) * spec.ratio))
        selected = torch.tensor(candidate_scores, dtype=torch.float32).topk(k=selected_count, largest=True).indices
        for index in selected.tolist():
            layer_idx, neuron_idx = candidates[index]
            masks[layer_idx][neuron_idx] = True

        return masks

    target_counts: dict[int, int] = {}
    for layer_idx, dim in layer_dims.items():
        if allowed_layers is not None and layer_idx not in allowed_layers:
            continue

        group = grouped[layer_idx]
        if score_type == "unknown_safe":
            scores = build_unknown_safe_vector(
                group,
                neuron_col,
                score_cols,
                activation_col,
                dim,
                unknown_safe_multimodal_weight,
                unknown_safe_activation_weight,
            )
        else:
            scores = build_score_vector(group, neuron_col, score_cols[score_type], dim)

        selected = select_top_indices(scores, spec.ratio)
        target_counts[layer_idx] = selected.numel()
        if spec.name != "layer_random":
            masks[layer_idx][selected] = True

    if spec.name == "layer_random":
        generator = torch.Generator()
        generator.manual_seed(spec.seed if spec.seed is not None else 0)
        for layer_idx, count in target_counts.items():
            dim = layer_dims[layer_idx]
            selected = torch.randperm(dim, generator=generator)[:count]
            masks[layer_idx][selected] = True

    return masks


def summarize_masks(masks_by_layer: dict[int, torch.Tensor]) -> dict[str, Any]:
    per_layer = {str(layer_idx): int(mask.sum().item()) for layer_idx, mask in sorted(masks_by_layer.items())}
    total = sum(per_layer.values())
    dims = {str(layer_idx): int(mask.numel()) for layer_idx, mask in sorted(masks_by_layer.items())}
    total_dim = sum(dims.values())
    return {
        "selected_neurons": total,
        "total_neurons": total_dim,
        "selected_ratio": total / total_dim if total_dim else 0.0,
        "per_layer_selected": per_layer,
    }


def find_down_proj_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules = [(name, module) for name, module in model.named_modules() if name.endswith("mlp.down_proj")]
    if not modules:
        modules = [(name, module) for name, module in model.named_modules() if name.endswith("down_proj")]

    if not modules:
        raise RuntimeError("Cannot find MLP down_proj modules for ablation hooks.")

    return [module for _, module in modules]


def prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    for override in args.overrides:
        key, value = parse_config_override(override)
        config[key] = value

    config["do_train"] = False
    config["do_eval"] = False
    config["do_predict"] = False
    config.setdefault("output_dir", "saves/vulcan/phase2_ablation_tmp")
    if args.dataset is not None:
        config["dataset"] = args.dataset

    if args.eval_dataset is not None:
        config["eval_dataset"] = args.eval_dataset

    if args.max_samples is not None:
        config["max_samples"] = args.max_samples

    if args.preprocessing_num_workers is not None:
        config["preprocessing_num_workers"] = args.preprocessing_num_workers

    return config


def build_dataloader(
    config: dict[str, Any],
    model: torch.nn.Module,
    tokenizer_module: dict[str, Any],
    template,
    batch_size: int | None,
    num_workers: int | None,
):
    model_args, data_args, training_args, _, _ = get_train_args(config)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    dataset = dataset_module.get("eval_dataset") or dataset_module["train_dataset"]
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer_module["tokenizer"].pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    effective_batch_size = batch_size or training_args.per_device_eval_batch_size or training_args.per_device_train_batch_size
    effective_num_workers = training_args.dataloader_num_workers if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        collate_fn=data_collator,
        num_workers=effective_num_workers,
        shuffle=False,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value

    return moved


@torch.no_grad()
def evaluate_label_nll(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    sample_nll_sums: list[float] = []
    sample_token_counts: list[int] = []
    total_nll = 0.0
    total_tokens = 0
    total_examples = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        labels = batch.pop("labels")
        outputs = model(**batch, use_cache=False)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(IGNORE_INDEX)
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)
        token_nll = -F.log_softmax(shift_logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_nll = token_nll * valid_mask
        batch_nll_sums = token_nll.sum(dim=1)
        batch_token_counts = valid_mask.sum(dim=1)

        for nll_sum, token_count in zip(batch_nll_sums.tolist(), batch_token_counts.tolist()):
            if token_count <= 0:
                continue

            sample_nll_sums.append(float(nll_sum))
            sample_token_counts.append(int(token_count))
            total_nll += float(nll_sum)
            total_tokens += int(token_count)
            total_examples += 1

    if total_examples == 0 or total_tokens == 0:
        raise RuntimeError("No labeled tokens were found. Check that the SFT dataset has assistant labels.")

    mean_nll = total_nll / total_tokens
    return {
        "num_examples": total_examples,
        "num_label_tokens": total_tokens,
        "nll": mean_nll,
        "ppl": math.exp(min(mean_nll, 100.0)),
        "sample_nll_sums": sample_nll_sums,
        "sample_token_counts": sample_token_counts,
    }


def bootstrap_weighted_nll(
    sample_nll_sums: list[float],
    sample_token_counts: list[int],
    num_bootstrap: int,
    seed: int,
) -> dict[str, float] | None:
    if num_bootstrap <= 0 or len(sample_nll_sums) < 2:
        return None

    rng = random.Random(seed)
    n = len(sample_nll_sums)
    values = []
    for _ in range(num_bootstrap):
        nll_sum = 0.0
        token_count = 0
        for _ in range(n):
            index = rng.randrange(n)
            nll_sum += sample_nll_sums[index]
            token_count += sample_token_counts[index]

        values.append(nll_sum / token_count)

    values.sort()
    lo = values[int(0.025 * (len(values) - 1))]
    hi = values[int(0.975 * (len(values) - 1))]
    return {
        "nll_ci_low": lo,
        "nll_ci_high": hi,
        "ppl_ci_low": math.exp(min(lo, 100.0)),
        "ppl_ci_high": math.exp(min(hi, 100.0)),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    score_table = read_score_table(args.score_file)
    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(score_table, args.score_prefix)
    allowed_layers = parse_layers(args.layers)
    if allowed_layers is not None:
        score_table = score_table[score_table[layer_col].isin(allowed_layers)]

    layer_dims = get_layer_dims(score_table, layer_col, neuron_col)
    ablation_specs = [parse_ablation_spec(item, args.seed) for item in args.ablation]
    masks_by_spec = {
        spec.result_name: build_type_mask(
            score_table,
            spec,
            layer_col,
            neuron_col,
            score_cols,
            activation_col,
            layer_dims,
            allowed_layers,
            args.selection,
            args.unknown_safe_multimodal_weight,
            args.unknown_safe_activation_weight,
        )
        for spec in ablation_specs
    }

    mask_summaries = {name: summarize_masks(masks) for name, masks in masks_by_spec.items()}
    if args.dry_run_masks:
        print(json.dumps({"mask_summaries": mask_summaries}, indent=2, ensure_ascii=False))
        return

    config = prepare_config(args)
    model_args, data_args, _, finetuning_args, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    down_proj_count = len(find_down_proj_modules(model))
    if down_proj_count != len(layer_dims):
        print(
            f"Warning: found {down_proj_count} down_proj modules, but score file has {len(layer_dims)} layers. "
            "Layer indices are mapped by down_proj order.",
            file=sys.stderr,
            flush=True,
        )

    dataloader = build_dataloader(config, model, tokenizer_module, template, args.batch_size, args.num_workers)
    results: dict[str, Any] = {
        "config": {
            "config_path": args.config,
            "score_file": args.score_file,
            "max_samples": args.max_samples,
            "dataset": config.get("dataset"),
            "eval_dataset": config.get("eval_dataset"),
            "selection": args.selection,
            "score_columns": score_cols,
            "layer_column": layer_col,
            "neuron_column": neuron_col,
            "activation_column": activation_col,
        },
        "mask_summaries": mask_summaries,
        "metrics": {},
    }

    for spec in ablation_specs:
        result_name = spec.result_name
        print(f"Running ablation: {result_name}", flush=True)
        masks = masks_by_spec[result_name]
        with MLPNeuronAblator(model, masks):
            metrics = evaluate_label_nll(model, dataloader, device, args.max_batches)

        bootstrap = bootstrap_weighted_nll(
            metrics["sample_nll_sums"],
            metrics["sample_token_counts"],
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
        metrics.pop("sample_nll_sums")
        metrics.pop("sample_token_counts")
        if bootstrap is not None:
            metrics.update(bootstrap)

        results["metrics"][result_name] = metrics
        print(
            f"  nll={metrics['nll']:.6f} ppl={metrics['ppl']:.4f} "
            f"examples={metrics['num_examples']} tokens={metrics['num_label_tokens']}",
            flush=True,
        )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to {output_path}", flush=True)


if __name__ == "__main__":
    main()
