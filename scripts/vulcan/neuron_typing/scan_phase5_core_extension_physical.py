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

"""Search one-neuron physical extensions of a frozen Phase-5 safety core."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from phase3_structural_utils import (
    canonical_json_sha256,
    count_parameters,
    masks_to_neuron_ids,
    neuron_ids_to_masks,
    resolve_existing_path,
    sha256_file,
)
from phase5_cached_utils import generate_cached_teacher_trace
from phase5_structural_utils import build_partial_singleton_cluster_idx, validate_partial_singleton_cluster_idx
from run_phase2_ablation import build_dataloader, move_batch_to_device
from run_phase5_single_sample_frontier import build_prompt_inputs, compare_generation
from scan_phase5_atomic_physical import (
    actual_projection_hashes,
    expected_projection_hashes,
    generate_until_divergence,
    parse_csv_ints,
    streaming_cached_fidelity,
    temporary_structural_pruning,
)
from verify_phase5_structural_equivalence import load_model_bundle, model_layer_dims

from llamafactory.train.vulcan.modeling import find_mlp_layers


def parse_positive_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"Expected positive comma-separated integers, got {value!r}.")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Physically search a fourth neuron around a frozen safety core.")
    parser.add_argument("--base_artifact_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", default=None, help="Comma-separated layers; default searches every FFN layer.")
    parser.add_argument(
        "--candidate_ranks",
        default="1",
        help="One-indexed Taylor ranks among neurons not already deleted by the core.",
    )
    parser.add_argument("--saliency_method", default="taylor", choices=["activation", "contribution", "taylor"])
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2071)
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


def ranked_extension_neurons(
    scores: torch.Tensor, base_mask: torch.Tensor, candidate_ranks: list[int]
) -> dict[int, int]:
    if scores.ndim != 1 or base_mask.ndim != 1 or scores.numel() != base_mask.numel():
        raise ValueError("Scores and base mask must be aligned one-dimensional tensors.")
    if base_mask.dtype != torch.bool or not bool(torch.isfinite(scores).all()):
        raise ValueError("Base mask must be boolean and saliency scores must be finite.")
    available = (~base_mask).nonzero(as_tuple=False).flatten()
    order = torch.argsort(scores[available].float().cpu(), descending=False, stable=True)
    ranked = available[order].tolist()
    if max(candidate_ranks) > len(ranked):
        raise ValueError(f"Candidate rank exceeds the {len(ranked)} neurons outside the base mask.")
    return {rank: int(ranked[rank - 1]) for rank in candidate_ranks}


def extend_core_mask(base_masks: dict[int, torch.Tensor], layer: int, neuron: int) -> dict[int, torch.Tensor]:
    if layer not in base_masks or not 0 <= neuron < base_masks[layer].numel():
        raise ValueError(f"Invalid extension neuron {layer}:{neuron}.")
    if bool(base_masks[layer][neuron]):
        raise ValueError(f"Extension neuron {layer}:{neuron} is already deleted by the safety core.")
    masks = {index: mask.clone() for index, mask in base_masks.items()}
    masks[layer][neuron] = True
    return masks


def refresh_summary(result: dict[str, Any]) -> None:
    candidates = result["candidates"]
    result["exact_candidates"] = [name for name, row in candidates.items() if row["generation"]["exact_match"]]
    result["strict_candidates"] = [name for name, row in candidates.items() if row["strict_feasible"]]
    result["best_strict_candidate"] = min(
        (
            {
                "candidate": name,
                "added_layer": row["added_layer"],
                "added_neuron": row["added_neuron"],
                "candidate_rank": row["candidate_rank"],
                "mean_kl": row["cached_fidelity"]["mean_kl"],
            }
            for name, row in candidates.items()
            if row["strict_feasible"]
        ),
        key=lambda row: (row["mean_kl"], row["candidate_rank"], row["added_layer"], row["added_neuron"]),
        default=None,
    )
    result["best_prefix"] = max(
        (
            {
                "candidate": name,
                "common_prefix_tokens": row["generation"]["common_prefix_tokens"],
                "exact_match": row["generation"]["exact_match"],
                "added_layer": row["added_layer"],
                "added_neuron": row["added_neuron"],
                "candidate_rank": row["candidate_rank"],
            }
            for name, row in candidates.items()
        ),
        key=lambda row: (row["common_prefix_tokens"], row["exact_match"], -row["candidate_rank"]),
        default=None,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_artifact_dir = Path(args.base_artifact_dir).resolve()
    metadata_path = base_artifact_dir / "metadata.json"
    mask_path = base_artifact_dir / "mask.json"
    metadata = load_json(metadata_path)
    base_neuron_ids = load_json(mask_path)
    if canonical_json_sha256(base_neuron_ids) != metadata["mask_sha256"]:
        raise ValueError("Base artifact mask hash mismatch.")
    if metadata.get("source_gate") != "structural_strict":
        raise ValueError("Core extension search requires a structural-strict base artifact.")

    static_path = resolve_existing_path(metadata["static_frontier"], relative_to=metadata_path)
    static = load_json(static_path)
    frozen_horizon = int(metadata.get("frozen_search_horizon", static["config"]["max_new_tokens"]))
    max_new_tokens = args.max_new_tokens or frozen_horizon
    if max_new_tokens != frozen_horizon:
        raise ValueError("Core extension search must use the frozen base-artifact horizon.")
    kl_tolerance = float(metadata.get("kl_tolerance", static["config"]["kl_tolerance"]))

    score_path = Path(static_path).parent / "saliency_scores.pt"
    score_artifact = torch.load(score_path, map_location="cpu", weights_only=True)
    if args.saliency_method not in score_artifact["scores"]:
        raise ValueError(f"Missing {args.saliency_method!r} scores in {score_path}.")
    scores = {
        int(layer): values.float().cpu() for layer, values in score_artifact["scores"][args.saliency_method].items()
    }
    layer_dims = {int(layer): int(width) for layer, width in metadata["layer_dims"].items()}
    base_masks = neuron_ids_to_masks(base_neuron_ids, layer_dims)
    if set(scores) != set(base_masks):
        raise ValueError("Saliency layers do not match the base artifact.")
    layers = sorted(scores) if args.layers is None else parse_csv_ints(args.layers)
    if any(layer not in scores for layer in layers):
        raise ValueError(f"Requested layers {layers} are not a subset of {sorted(scores)}.")
    candidate_ranks = parse_positive_ints(args.candidate_ranks)
    ranked_by_layer = {
        layer: ranked_extension_neurons(scores[layer], base_masks[layer], candidate_ranks) for layer in layers
    }

    output_dir = Path(args.output_dir)
    result_path = output_dir / "core_extension_physical_frontier.json"
    identity = {
        "base_artifact_dir": str(base_artifact_dir),
        "base_metadata_sha256": sha256_file(metadata_path),
        "base_mask_sha256": canonical_json_sha256(base_neuron_ids),
        "base_deletion_budget": int(metadata["deletion_budget"]),
        "static_frontier": str(static_path),
        "static_frontier_sha256": sha256_file(static_path),
        "saliency_scores": str(score_path.resolve()),
        "saliency_scores_sha256": sha256_file(score_path),
        "saliency_method": args.saliency_method,
        "max_new_tokens": max_new_tokens,
        "kl_tolerance": kl_tolerance,
        "seed": args.seed,
    }
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Core-extension output exists: {result_path}. Pass --resume to append candidates.")
    if args.resume and result_path.exists():
        result = load_json(result_path)
        if any(result["config"].get(key) != value for key, value in identity.items()):
            raise ValueError("Cannot resume core-extension scanning with changed fixed inputs.")
        result["complete"] = False
        result["config"]["layers"] = list(dict.fromkeys([*result["config"]["layers"], *layers]))
        result["config"]["candidate_ranks"] = list(
            dict.fromkeys([*result["config"]["candidate_ranks"], *candidate_ranks])
        )
    else:
        result = {
            "complete": False,
            "interpretation": (
                "Every candidate preserves the frozen structural-strict safety core and physically deletes one "
                "additional neuron. A strict candidate constructively raises the single-sample pruning lower bound."
            ),
            "config": {**identity, "layers": layers, "candidate_ranks": candidate_ranks},
            "sample_manifest": None,
            "teacher_baseline_cached_fidelity": None,
            "candidates": {},
            "exact_candidates": [],
            "strict_candidates": [],
            "best_strict_candidate": None,
            "best_prefix": None,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer_module, template, config = load_model_bundle(
        metadata["config_path"],
        metadata["model_name_or_path"],
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
    if teacher_trace["token_ids"].tolist() != static["teacher"]["generated_token_ids"]:
        raise RuntimeError("Original model did not reproduce the frozen core-extension teacher trajectory.")
    teacher_generation = {"token_ids": teacher_trace["token_ids"], "text": teacher_trace["text"]}
    if result["teacher_baseline_cached_fidelity"] is None:
        result["teacher_baseline_cached_fidelity"] = streaming_cached_fidelity(
            model, prompt_inputs, teacher_trace["raw_logits"], teacher_trace["token_ids"]
        )
        write_json(result_path, result)

    original_parameters = count_parameters(model)["total"]
    original_dims = model_layer_dims(model)
    all_mlps = find_mlp_layers(model)

    for rank in candidate_ranks:
        for layer in layers:
            neuron = ranked_by_layer[layer][rank]
            candidate_name = f"add_layer_{layer:02d}__rank_{rank:03d}__neuron_{neuron:04d}"
            if candidate_name in result["candidates"]:
                print(f"Core extension skipping completed {candidate_name}", flush=True)
                continue
            masks = extend_core_mask(base_masks, layer, neuron)
            cluster_idx = build_partial_singleton_cluster_idx(masks)
            validation = validate_partial_singleton_cluster_idx(cluster_idx, masks)
            neuron_ids = masks_to_neuron_ids(masks)
            candidate_mask_path = output_dir / "masks" / f"{candidate_name}.json"
            write_json(candidate_mask_path, neuron_ids)
            target_layers = [index for index, mask in masks.items() if bool(mask.any())]
            expected_hashes = {
                index: expected_projection_hashes(
                    all_mlps[index].mlp, (~masks[index]).nonzero(as_tuple=False).flatten()
                )
                for index in target_layers
            }
            expected_parameter_reduction = sum(
                int(masks[index].sum())
                * (
                    all_mlps[index].mlp.gate_proj.in_features
                    + all_mlps[index].mlp.up_proj.in_features
                    + all_mlps[index].mlp.down_proj.out_features
                    + int(all_mlps[index].mlp.gate_proj.bias is not None)
                    + int(all_mlps[index].mlp.up_proj.bias is not None)
                )
                for index in target_layers
            )
            print(f"Core extension evaluating {candidate_name}", flush=True)
            with temporary_structural_pruning(model, cluster_idx, target_layers) as pruned_mlps:
                structure_checks = {
                    "artifact_valid": validation["validated"],
                    "only_masked_layers_changed": all(
                        width == original_dims[index] - int(masks[index].sum())
                        for index, width in model_layer_dims(model).items()
                    ),
                    "parameters_reduced_by_expected_count": (
                        original_parameters - count_parameters(model)["total"] == expected_parameter_reduction
                    ),
                    "singleton_weights_match_expected": all(
                        actual_projection_hashes(pruned_mlps[index]) == expected_hashes[index]
                        for index in target_layers
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
            strict_feasible = bool(behavior_preserved and fidelity["mean_kl"] <= kl_tolerance)
            result["candidates"][candidate_name] = {
                "added_layer": layer,
                "added_neuron": neuron,
                "candidate_rank": rank,
                "added_saliency": float(scores[layer][neuron]),
                "total_deletion_count": int(metadata["deletion_budget"]) + 1,
                "deleted_neuron_ids_by_layer": {
                    str(index): mask.nonzero(as_tuple=False).flatten().tolist()
                    for index, mask in masks.items()
                    if bool(mask.any())
                },
                "mask_file": str(candidate_mask_path),
                "mask_sha256": canonical_json_sha256(neuron_ids),
                "structure_checks": structure_checks,
                "generation": generation,
                "cached_fidelity": fidelity,
                "behavior_preserved": behavior_preserved,
                "strict_feasible": strict_feasible,
            }
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
                "num_candidates": len(result["candidates"]),
                "strict_candidates": result["strict_candidates"],
                "best_strict_candidate": result["best_strict_candidate"],
                "best_prefix": result["best_prefix"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
