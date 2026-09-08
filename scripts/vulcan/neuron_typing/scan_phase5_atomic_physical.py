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

"""Scan layer-restricted physical FFN deletions while reusing one model and teacher trace."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase3_structural_utils import (  # noqa: E402
    canonical_json_sha256,
    count_parameters,
    masks_to_neuron_ids,
    sha256_file,
)
from phase5_cached_utils import cached_teacher_forcing_steps, generate_cached_teacher_trace  # noqa: E402
from phase5_structural_utils import (  # noqa: E402
    build_partial_singleton_cluster_idx,
    validate_partial_singleton_cluster_idx,
)
from run_phase2_ablation import build_dataloader, move_batch_to_device  # noqa: E402
from run_phase5_single_sample_frontier import build_prompt_inputs, compare_generation  # noqa: E402
from verify_phase5_structural_equivalence import load_model_bundle, model_layer_dims, tensor_sha256  # noqa: E402

from llamafactory.train.vulcan.modeling import find_mlp_layers  # noqa: E402
from llamafactory.train.vulcan.pruning import pruning_mlp  # noqa: E402


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise ValueError(f"Expected non-negative comma-separated integers, got {value!r}.")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan single-layer physical deletion candidates on one rollout.")
    parser.add_argument("--static_frontier", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", default=None, help="Comma-separated layers; defaults to every FFN layer.")
    parser.add_argument("--deletion_counts", default="1", help="Nested per-layer deletion counts.")
    parser.add_argument(
        "--combinations",
        default=None,
        help="Explicit comma-separated combinations such as '8:1+19:1,8:1+23:2'.",
    )
    parser.add_argument("--skip_single_layers", action="store_true")
    parser.add_argument("--saliency_method", default="taylor", choices=["activation", "contribution", "taylor"])
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2070)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_lowest_neurons(scores: torch.Tensor, deletion_count: int) -> list[int]:
    if scores.ndim != 1:
        raise ValueError(f"Expected one-dimensional saliency, got {tuple(scores.shape)}.")
    if not 0 < deletion_count < scores.numel():
        raise ValueError(f"deletion_count must lie in [1, {scores.numel() - 1}], got {deletion_count}.")
    return torch.argsort(scores.float().cpu(), descending=False, stable=True)[:deletion_count].tolist()


def parse_combinations(value: str | None) -> list[dict[int, int]]:
    if value is None:
        return []
    combinations = []
    seen = set()
    for raw_group in value.split(","):
        group = {}
        for raw_item in raw_group.strip().split("+"):
            try:
                layer_text, count_text = raw_item.split(":", maxsplit=1)
                layer, count = int(layer_text), int(count_text)
            except ValueError as error:
                raise ValueError(f"Invalid combination item {raw_item!r}; expected layer:count.") from error
            if layer < 0 or count <= 0 or layer in group:
                raise ValueError(f"Invalid or repeated layer in combination {raw_group!r}.")
            group[layer] = count
        key = tuple(sorted(group.items()))
        if len(group) < 2:
            raise ValueError(f"Combination must contain at least two layers, got {raw_group!r}.")
        if key not in seen:
            seen.add(key)
            combinations.append(group)
    return combinations


def combination_name(deletions_by_layer: dict[int, int]) -> str:
    parts = "-".join(f"{layer:02d}x{count}" for layer, count in sorted(deletions_by_layer.items()))
    return f"layers_{parts}"


def build_combined_masks(
    layer_widths: dict[int, int], deleted_ids_by_layer: dict[int, list[int]]
) -> dict[int, torch.Tensor]:
    masks = {layer: torch.zeros(width, dtype=torch.bool) for layer, width in sorted(layer_widths.items())}
    if not deleted_ids_by_layer:
        raise ValueError("Combined deletion mask is empty.")
    for target_layer, deleted_ids in deleted_ids_by_layer.items():
        if target_layer not in layer_widths:
            raise ValueError(f"Unknown target layer {target_layer}; available layers are {sorted(layer_widths)}.")
        if len(deleted_ids) != len(set(deleted_ids)):
            raise ValueError("Atomic deletion IDs contain duplicates.")
        if not deleted_ids or any(neuron < 0 or neuron >= layer_widths[target_layer] for neuron in deleted_ids):
            raise ValueError(f"Invalid deletion IDs for layer {target_layer}: {deleted_ids}.")
        masks[target_layer][torch.tensor(deleted_ids, dtype=torch.long)] = True
    return masks


def build_atomic_masks(
    layer_widths: dict[int, int], target_layer: int, deleted_ids: list[int]
) -> dict[int, torch.Tensor]:
    return build_combined_masks(layer_widths, {target_layer: deleted_ids})


class StopAfterFirstDivergence(StoppingCriteria):
    """Stop greedy generation immediately after its first mismatch with a frozen teacher."""

    def __init__(self, prompt_length: int, teacher_token_ids: torch.Tensor):
        self.prompt_length = prompt_length
        self.teacher_token_ids = teacher_token_ids.to(torch.long).cpu()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        del scores, kwargs
        generated_count = input_ids.shape[-1] - self.prompt_length
        if generated_count <= 0:
            return False
        teacher_index = generated_count - 1
        if teacher_index >= self.teacher_token_ids.numel():
            return True
        return int(input_ids[0, -1]) != int(self.teacher_token_ids[teacher_index])


@torch.no_grad()
def generate_until_divergence(
    model: torch.nn.Module,
    tokenizer,
    prompt_inputs: dict[str, Any],
    teacher_token_ids: torch.Tensor,
) -> dict[str, Any]:
    prompt_length = prompt_inputs["input_ids"].shape[-1]
    generation_inputs = dict(prompt_inputs)
    generation_inputs.update(
        {
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": int(teacher_token_ids.numel()),
            "pad_token_id": tokenizer.pad_token_id,
            "stopping_criteria": StoppingCriteriaList([StopAfterFirstDivergence(prompt_length, teacher_token_ids)]),
        }
    )
    sequences = model.generate(**generation_inputs)
    generated = sequences[0, prompt_length:].detach().cpu()
    return {"token_ids": generated, "text": tokenizer.decode(generated.tolist(), skip_special_tokens=True)}


@torch.no_grad()
def streaming_cached_fidelity(
    model: torch.nn.Module,
    prompt_inputs: dict[str, Any],
    teacher_logits: torch.Tensor,
    teacher_token_ids: torch.Tensor,
) -> dict[str, Any]:
    """Compute cached fidelity without retaining another rollout-by-vocabulary tensor."""
    num_tokens = int(teacher_token_ids.numel())
    if teacher_logits.shape[0] != num_tokens:
        raise ValueError("Teacher logits and generated tokens are not aligned.")
    kl_values = []
    margins = []
    teacher_agreements = []
    label_agreements = []
    teacher_consistencies = []
    for token_step, student_logits in cached_teacher_forcing_steps(model, prompt_inputs, teacher_token_ids):
        teacher_row = teacher_logits[token_step].to(student_logits.device).reshape(1, -1).float()
        student_row = student_logits.float()
        teacher_log_probs = F.log_softmax(teacher_row, dim=-1)
        student_log_probs = F.log_softmax(student_row, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_log_probs, log_target=True, reduction="batchmean")
        label = int(teacher_token_ids[token_step])
        teacher_top = int(teacher_row.argmax(dim=-1))
        student_top = int(student_row.argmax(dim=-1))
        target_logit = student_row[0, label]
        top_values, top_indices = student_row.topk(k=2, dim=-1)
        competitor = top_values[0, 1] if int(top_indices[0, 0]) == label else top_values[0, 0]
        kl_values.append(float(kl))
        margins.append(float(target_logit - competitor))
        teacher_agreements.append(student_top == teacher_top)
        label_agreements.append(student_top == label)
        teacher_consistencies.append(teacher_top == label)

    kl_tensor = torch.tensor(kl_values)
    margin_tensor = torch.tensor(margins)
    teacher_agreement_tensor = torch.tensor(teacher_agreements, dtype=torch.bool)
    label_agreement_tensor = torch.tensor(label_agreements, dtype=torch.bool)
    teacher_consistency_tensor = torch.tensor(teacher_consistencies, dtype=torch.bool)
    disagreement = (~teacher_agreement_tensor).nonzero(as_tuple=False).flatten()
    label_disagreement = (~label_agreement_tensor).nonzero(as_tuple=False).flatten()
    low_margin = (margin_tensor < 0.25).nonzero(as_tuple=False).flatten()
    minimum_margin = margin_tensor.min()
    minimum_indices = margin_tensor.eq(minimum_margin).nonzero(as_tuple=False).flatten()
    return {
        "mean_kl": float(kl_tensor.mean()),
        "max_kl": float(kl_tensor.max()),
        "token_agreement": float(teacher_agreement_tensor.float().mean()),
        "generated_token_agreement": float(label_agreement_tensor.float().mean()),
        "first_disagreement": int(disagreement[0]) if disagreement.numel() else None,
        "first_generated_token_disagreement": int(label_disagreement[0]) if label_disagreement.numel() else None,
        "teacher_greedy_consistency": float(teacher_consistency_tensor.float().mean()),
        "min_generated_token_margin": float(minimum_margin),
        "min_generated_token_margin_indices": minimum_indices.tolist(),
        "p01_generated_token_margin": float(torch.quantile(margin_tensor, 0.01)),
        "mean_generated_token_margin": float(margin_tensor.mean()),
        "tokens_below_margin_0_25": int(low_margin.numel()),
        "tokens_below_margin_0_25_indices": low_margin.tolist(),
    }


def _snapshot_attributes(objects: list[Any], attributes: tuple[str, ...]) -> list[tuple[Any, dict[str, Any]]]:
    snapshots = []
    seen = set()
    for obj in objects:
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        values = {}
        for attribute in attributes:
            values[attribute] = (hasattr(obj, attribute), copy.deepcopy(getattr(obj, attribute, None)))
        snapshots.append((obj, values))
    return snapshots


def _restore_attributes(snapshots: list[tuple[Any, dict[str, Any]]]) -> None:
    for obj, values in snapshots:
        for attribute, (existed, value) in values.items():
            if existed:
                setattr(obj, attribute, value)
            elif hasattr(obj, attribute):
                delattr(obj, attribute)


@contextmanager
def temporary_structural_pruning(model: torch.nn.Module, cluster_idx: list[Any], target_layers: int | list[int]):
    """Physically prune selected layers and fully restore their modules and shared config afterward."""
    target_layers = [target_layers] if isinstance(target_layers, int) else sorted(target_layers)
    layer_refs = find_mlp_layers(model)
    mlps = {layer: layer_refs[layer].mlp for layer in target_layers}
    original_projections = {layer: (mlp.up_proj, mlp.gate_proj, mlp.down_proj) for layer, mlp in mlps.items()}
    root_config = getattr(model, "config", None)
    snapshots = _snapshot_attributes(
        [
            *mlps.values(),
            *(getattr(mlp, "config", None) for mlp in mlps.values()),
            root_config,
            getattr(root_config, "text_config", None),
        ],
        ("intermediate_size", "vulcan_intermediate_sizes"),
    )
    try:
        pruning_mlp(model, cluster_idx)
        yield mlps
    finally:
        for layer, mlp in mlps.items():
            mlp.up_proj, mlp.gate_proj, mlp.down_proj = original_projections[layer]
        _restore_attributes(snapshots)


def expected_projection_hashes(mlp: torch.nn.Module, keep_ids: torch.Tensor) -> dict[str, str]:
    keep_ids = keep_ids.to(mlp.up_proj.weight.device)
    return {
        "gate_proj.weight": tensor_sha256(mlp.gate_proj.weight.index_select(0, keep_ids)),
        "up_proj.weight": tensor_sha256(mlp.up_proj.weight.index_select(0, keep_ids)),
        "down_proj.weight": tensor_sha256(mlp.down_proj.weight.index_select(1, keep_ids)),
    }


def actual_projection_hashes(mlp: torch.nn.Module) -> dict[str, str]:
    return {
        "gate_proj.weight": tensor_sha256(mlp.gate_proj.weight),
        "up_proj.weight": tensor_sha256(mlp.up_proj.weight),
        "down_proj.weight": tensor_sha256(mlp.down_proj.weight),
    }


def refresh_summary(result: dict[str, Any]) -> None:
    candidates = result["candidates"]

    def deletion_count(row: dict[str, Any]) -> int:
        return int(row.get("total_deletion_count", row.get("deletion_count", 0)))

    def feasible_summary(name: str, row: dict[str, Any]) -> dict[str, Any]:
        fidelity = row.get("cached_fidelity")
        return {
            "candidate": name,
            "total_deletion_count": deletion_count(row),
            "mean_kl": None if fidelity is None else fidelity["mean_kl"],
        }

    result["exact_candidates"] = [name for name, row in candidates.items() if row["generation"]["exact_match"]]
    result["strict_candidates"] = [name for name, row in candidates.items() if row.get("strict_feasible")]
    result["best_exact_candidate"] = max(
        (feasible_summary(name, row) for name, row in candidates.items() if row["generation"]["exact_match"]),
        key=lambda row: (row["total_deletion_count"], -row["mean_kl"]),
        default=None,
    )
    result["best_strict_candidate"] = max(
        (feasible_summary(name, row) for name, row in candidates.items() if row.get("strict_feasible")),
        key=lambda row: (row["total_deletion_count"], -row["mean_kl"]),
        default=None,
    )
    result["best_prefix"] = max(
        (
            {
                "candidate": name,
                "common_prefix_tokens": row["generation"]["common_prefix_tokens"],
                "exact_match": row["generation"]["exact_match"],
                "total_deletion_count": deletion_count(row),
            }
            for name, row in candidates.items()
        ),
        key=lambda row: (row["common_prefix_tokens"], row["exact_match"], row["total_deletion_count"]),
        default=None,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    static_path = Path(args.static_frontier).resolve()
    static = load_json(static_path)
    if not static.get("complete") or static.get("config", {}).get("trace_target") != "teacher_generation":
        raise ValueError("Atomic scanning requires a complete teacher-generation frontier.")
    max_new_tokens = args.max_new_tokens or static["config"]["max_new_tokens"]
    if max_new_tokens != static["config"]["max_new_tokens"]:
        raise ValueError("Atomic scanning must use the frozen static-frontier horizon.")
    deletion_counts = parse_csv_ints(args.deletion_counts)
    if 0 in deletion_counts:
        raise ValueError("Atomic deletion counts must be positive.")
    combinations = parse_combinations(args.combinations)
    if args.skip_single_layers and not combinations:
        raise ValueError("--skip_single_layers requires at least one explicit --combinations group.")

    score_path = static_path.parent / "saliency_scores.pt"
    score_artifact = torch.load(score_path, map_location="cpu", weights_only=True)
    if args.saliency_method not in score_artifact["scores"]:
        raise ValueError(f"Missing {args.saliency_method!r} scores in {score_path}.")
    scores = {
        int(layer): values.float().cpu() for layer, values in score_artifact["scores"][args.saliency_method].items()
    }
    layer_widths = {layer: int(values.numel()) for layer, values in scores.items()}
    layers = sorted(scores) if args.layers is None else parse_csv_ints(args.layers)
    if any(layer not in scores for layer in layers):
        raise ValueError(f"Requested layers {layers} are not a subset of {sorted(scores)}.")
    if any(count >= min(layer_widths[layer] for layer in layers) for count in deletion_counts):
        raise ValueError("Every deletion count must leave at least one neuron in each requested layer.")
    for combination in combinations:
        if any(layer not in scores for layer in combination):
            raise ValueError(f"Combination {combination} contains an unknown layer.")
        if any(count >= layer_widths[layer] for layer, count in combination.items()):
            raise ValueError(f"Combination {combination} deletes an invalid number of neurons.")
    combination_names = [combination_name(group) for group in combinations]

    output_dir = Path(args.output_dir)
    result_path = output_dir / "atomic_physical_frontier.json"
    identity = {
        "static_frontier": str(static_path),
        "static_frontier_sha256": sha256_file(static_path),
        "saliency_scores": str(score_path.resolve()),
        "saliency_scores_sha256": sha256_file(score_path),
        "saliency_method": args.saliency_method,
        "max_new_tokens": max_new_tokens,
        "seed": args.seed,
    }
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Atomic output exists: {result_path}. Pass --resume to append candidates.")
    if args.resume and result_path.exists():
        result = load_json(result_path)
        if any(result["config"].get(key) != value for key, value in identity.items()):
            raise ValueError("Cannot resume atomic scanning with changed fixed inputs.")
        result["complete"] = False
        result["config"]["layers"] = list(dict.fromkeys([*result["config"]["layers"], *layers]))
        result["config"]["deletion_counts"] = list(
            dict.fromkeys([*result["config"]["deletion_counts"], *deletion_counts])
        )
        result["config"].setdefault("combinations", [])
        result["config"]["combinations"] = list(dict.fromkeys([*result["config"]["combinations"], *combination_names]))
    else:
        result = {
            "complete": False,
            "interpretation": (
                "Each candidate physically narrows one or more explicitly selected FFN layers. Exact candidates "
                "establish constructive single-sample BF16 lower bounds; failures do not prove that every neuron "
                "in a layer is unsafe."
            ),
            "config": {
                **identity,
                "layers": layers,
                "deletion_counts": deletion_counts,
                "combinations": combination_names,
            },
            "sample_manifest": None,
            "baseline_cached_fidelity": None,
            "candidates": {},
            "exact_candidates": [],
            "strict_candidates": [],
            "best_exact_candidate": None,
            "best_strict_candidate": None,
            "best_prefix": None,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer_module, template, config = load_model_bundle(
        static["config"]["config_path"],
        static["config"]["model_name_or_path"],
        device,
        trust_remote_code=False,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )
    tokenizer = tokenizer_module["tokenizer"]
    dataloader, manifest = build_dataloader(
        config,
        model,
        tokenizer_module,
        template,
        batch_size=1,
        num_workers=args.num_workers,
        sample_offset=static["config"]["sample_offset"],
        max_samples=1,
        allow_short_dataset=False,
        max_image_repeat=5,
        allow_excessive_image_repeats=False,
        dataset_stage=static["config"]["dataset_stage"],
    )
    result["sample_manifest"] = manifest
    dataset_batch = move_batch_to_device(next(iter(dataloader)), device)
    prompt_inputs, _ = build_prompt_inputs(dataset_batch)
    teacher_trace = generate_cached_teacher_trace(model, tokenizer, prompt_inputs, max_new_tokens)
    frozen_ids = static["teacher"]["generated_token_ids"]
    if teacher_trace["token_ids"].tolist() != frozen_ids:
        raise RuntimeError("Original model did not reproduce the frozen atomic-scan teacher trajectory.")
    teacher_generation = {"token_ids": teacher_trace["token_ids"], "text": teacher_trace["text"]}
    if result["baseline_cached_fidelity"] is None:
        result["baseline_cached_fidelity"] = streaming_cached_fidelity(
            model, prompt_inputs, teacher_trace["raw_logits"], teacher_trace["token_ids"]
        )
        baseline = result["baseline_cached_fidelity"]
        if baseline["token_agreement"] != 1.0 or baseline["generated_token_agreement"] != 1.0:
            raise RuntimeError(f"Manual cached baseline does not reproduce the teacher: {baseline}.")
        write_json(result_path, result)

    original_parameters = count_parameters(model)["total"]
    original_dims = model_layer_dims(model)

    def evaluate_candidate(candidate_name: str, deletions_by_layer: dict[int, int]) -> None:
        if candidate_name in result["candidates"]:
            print(f"Atomic scan skipping completed {candidate_name}", flush=True)
            return
        target_layers = sorted(deletions_by_layer)
        deleted_ids_by_layer = {
            layer: select_lowest_neurons(scores[layer], deletion_count)
            for layer, deletion_count in deletions_by_layer.items()
        }
        masks = build_combined_masks(layer_widths, deleted_ids_by_layer)
        cluster_idx = build_partial_singleton_cluster_idx(masks)
        validation = validate_partial_singleton_cluster_idx(cluster_idx, masks)
        neuron_ids = masks_to_neuron_ids(masks)
        mask_path = output_dir / "masks" / f"{candidate_name}.json"
        write_json(mask_path, neuron_ids)
        source_mlps = {layer: find_mlp_layers(model)[layer].mlp for layer in target_layers}
        expected_hashes = {
            layer: expected_projection_hashes(source_mlps[layer], (~masks[layer]).nonzero(as_tuple=False).flatten())
            for layer in target_layers
        }
        expected_parameter_reduction = sum(
            deletions_by_layer[layer]
            * (
                mlp.gate_proj.in_features
                + mlp.up_proj.in_features
                + mlp.down_proj.out_features
                + int(mlp.gate_proj.bias is not None)
                + int(mlp.up_proj.bias is not None)
            )
            for layer, mlp in source_mlps.items()
        )
        print(f"Atomic scan evaluating {candidate_name}", flush=True)
        with temporary_structural_pruning(model, cluster_idx, target_layers) as pruned_mlps:
            structural_checks = {
                "artifact_valid": validation["validated"],
                "only_target_layers_changed": all(
                    width == original_dims[index] - deletions_by_layer.get(index, 0)
                    for index, width in model_layer_dims(model).items()
                ),
                "parameters_reduced_by_expected_count": (
                    original_parameters - count_parameters(model)["total"] == expected_parameter_reduction
                ),
                "singleton_weights_match_expected": all(
                    actual_projection_hashes(pruned_mlps[layer]) == expected_hashes[layer] for layer in target_layers
                ),
            }
            student_trace = generate_until_divergence(model, tokenizer, prompt_inputs, teacher_trace["token_ids"])
            generation = compare_generation(teacher_generation, student_trace)
            fidelity = (
                streaming_cached_fidelity(
                    model, prompt_inputs, teacher_trace["raw_logits"], teacher_trace["token_ids"]
                )
                if generation["exact_match"]
                else None
            )
        if model_layer_dims(model) != original_dims or count_parameters(model)["total"] != original_parameters:
            raise RuntimeError(f"Model restoration failed after {candidate_name}.")
        behavior_preserved = bool(
            generation["exact_match"]
            and fidelity is not None
            and fidelity["token_agreement"] == 1.0
            and fidelity["generated_token_agreement"] == 1.0
        )
        strict_feasible = bool(behavior_preserved and fidelity["mean_kl"] <= static["config"]["kl_tolerance"])
        row = {
            "deletions_by_layer": {str(layer): count for layer, count in deletions_by_layer.items()},
            "total_deletion_count": sum(deletions_by_layer.values()),
            "deleted_neuron_ids_by_layer": {str(layer): ids for layer, ids in deleted_ids_by_layer.items()},
            "deleted_saliency_by_layer": {
                str(layer): [float(scores[layer][neuron]) for neuron in ids]
                for layer, ids in deleted_ids_by_layer.items()
            },
            "mask_file": str(mask_path),
            "mask_sha256": canonical_json_sha256(neuron_ids),
            "structure_checks": structural_checks,
            "generation": generation,
            "cached_fidelity": fidelity,
            "behavior_preserved": behavior_preserved,
            "strict_feasible": strict_feasible,
        }
        if len(target_layers) == 1:
            layer = target_layers[0]
            row.update(
                {
                    "layer": layer,
                    "deletion_count": deletions_by_layer[layer],
                    "deleted_neuron_ids": deleted_ids_by_layer[layer],
                    "deleted_saliency": row["deleted_saliency_by_layer"][str(layer)],
                }
            )
        result["candidates"][candidate_name] = row
        refresh_summary(result)
        write_json(result_path, result)
        print(
            json.dumps(
                {
                    "candidate": candidate_name,
                    "exact": generation["exact_match"],
                    "prefix": generation["common_prefix_tokens"],
                    "strict": strict_feasible,
                }
            ),
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.skip_single_layers:
        for deletion_count in deletion_counts:
            for layer in layers:
                evaluate_candidate(f"layer_{layer:02d}__delete_{deletion_count}", {layer: deletion_count})
    for candidate_name, combination in zip(combination_names, combinations):
        evaluate_candidate(candidate_name, combination)

    result["complete"] = True
    refresh_summary(result)
    write_json(result_path, result)
    return result


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "exact_candidates": result["exact_candidates"],
                "strict_candidates": result["strict_candidates"],
                "best_exact_candidate": result["best_exact_candidate"],
                "best_strict_candidate": result["best_strict_candidate"],
                "best_prefix": result["best_prefix"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
