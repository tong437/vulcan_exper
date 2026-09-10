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

"""Refine Phase 6D bidirectional conflict layers with deterministic delta debugging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase5e_proxy import evaluate_gold_proxy  # noqa: E402
from phase5e_semantics import generate_short_caption  # noqa: E402
from phase6b_semantics import build_semantic_evaluator  # noqa: E402
from phase6c_subnets import (  # noqa: E402
    LayerSets,
    complement_pool,
    deletion_masks,
    derive_seed,
    deterministic_sample,
    kept_from_deleted,
    kept_sha256,
    layerwise_difference,
    layerwise_union,
    summarize_kept,
)
from phase6d_conflicts import ConflictContrast, one_layer  # noqa: E402
from phase6d_delta_debug import choose_reduction, delta_debug_proposals  # noqa: E402
from run_phase2_ablation import MLPNeuronAblator, build_dataloader, load_yaml, move_batch_to_device  # noqa: E402
from run_phase5_single_sample_frontier import build_prompt_inputs, write_json  # noqa: E402
from run_phase5e_retrospective import build_reference_trace  # noqa: E402
from run_phase6b_single_sample_limits import prepare_dataset, sha256_file, validate_inputs  # noqa: E402
from run_phase6d_conflict_localization import append_jsonl, compact_proxy, load_jsonl  # noqa: E402
from verify_phase5_structural_equivalence import load_model_bundle, model_layer_dims  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6D identity-level conflict refinement.")
    parser.add_argument("--localization_analysis", required=True)
    parser.add_argument("--winner_dir", default="saves/neuron_typing/phase6b_frozen_winners_v1")
    parser.add_argument("--sample_file", default="data/phase6b_single_samples/frozen_samples.json")
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/phase5e_coco.yaml")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_candidate_slots", type=int, default=12)
    parser.add_argument("--max_subset_evaluations", type=int, default=96)
    parser.add_argument("--random_seeds", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2066)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def identity_sha256(identities: set[int] | frozenset[int]) -> str:
    payload = ",".join(str(identity) for identity in sorted(identities)).encode()
    return hashlib.sha256(payload).hexdigest()


def build_kept_pair(
    base: LayerSets, donor: LayerSets, layer: int, identities: set[int]
) -> tuple[LayerSets, LayerSets]:
    donor_increment = layerwise_difference(donor, base)
    if not identities <= donor_increment[layer]:
        raise ValueError("Identity refinement subset is not contained in the frozen donor increment.")
    selected = one_layer({candidate: identities if candidate == layer else set() for candidate in base}, layer)
    union = layerwise_union(base, donor)
    return layerwise_union(base, selected), layerwise_difference(union, selected)


def select_candidates(analysis: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    if not analysis.get("complete") or not analysis["integrity"]["all_endpoints_reproduced"]:
        raise ValueError("Identity refinement requires a complete localization with reproduced endpoints.")
    candidates = analysis["identity_refinement_shortlist"]
    return candidates[:limit]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        min(
            args.max_candidate_slots,
            args.max_subset_evaluations,
            args.random_seeds,
            args.max_new_tokens,
        )
        < 1
    ):
        raise ValueError("All Phase 6D identity-refinement budgets must be positive.")
    analysis_path = Path(args.localization_analysis).resolve()
    winner_dir = Path(args.winner_dir).resolve()
    sample_path = Path(args.sample_file).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "phase6d_identity_refinement.json"
    evaluations_path = output_dir / "evaluations.jsonl"
    if output_dir.exists() and not args.resume and (result_path.exists() or evaluations_path.exists()):
        raise FileExistsError(f"Phase 6D identity output already exists: {output_dir}. Pass --resume to continue.")
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    candidates = select_candidates(analysis, args.max_candidate_slots)
    if not candidates:
        raise ValueError("Localization produced no bidirectional layer candidates; identity refinement is gated off.")
    samples_payload = json.loads(sample_path.read_text(encoding="utf-8"))
    samples = validate_inputs(samples_payload, None)
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    target_ids = list(dict.fromkeys(candidate["target"] for candidate in candidates))
    target_samples = [samples_by_id[target] for target in target_ids]
    winners_payload = json.loads((winner_dir / "frozen_winners.json").read_text(encoding="utf-8"))
    winner_rows = {row["sample_id"]: row for row in winners_payload["winners"]}
    layer_dims = {int(layer): int(width) for layer, width in next(iter(winner_rows.values()))["layer_dims"].items()}
    winners = {
        sample_id: kept_from_deleted(
            json.loads((winner_dir / sample_id / "mask.json").read_text(encoding="utf-8")), layer_dims
        )
        for sample_id in winner_rows
    }
    contrast_analysis = {entry["contrast"]["contrast_id"]: entry for entry in analysis["contrasts"]}

    base_config = load_yaml(config_path)
    model_path = Path(args.model_name_or_path or base_config["model_name_or_path"]).resolve()
    run_config = {
        "localization_analysis": str(analysis_path),
        "localization_analysis_sha256": sha256_file(analysis_path),
        "winner_dir": str(winner_dir),
        "winner_manifest_sha256": sha256_file(winner_dir / "frozen_winners.json"),
        "sample_file": str(sample_path),
        "sample_file_sha256": sha256_file(sample_path),
        "config_path": str(config_path),
        "model_name_or_path": str(model_path),
        "max_candidate_slots": args.max_candidate_slots,
        "max_subset_evaluations": args.max_subset_evaluations,
        "random_seeds": args.random_seeds,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "search": "deterministic chunks plus complements; no monotonic binary-search assumption",
        "tie_break": "smaller subset, larger bidirectional NLL margin, lexical subset SHA-256",
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    if args.resume and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing["config"] != run_config:
            raise ValueError("Cannot resume Phase 6D identity refinement with a changed configuration.")
    result = {
        "complete": False,
        "config": run_config,
        "selected_candidates": candidates,
        "evaluations_file": str(evaluations_path),
        "candidate_results": [],
    }
    write_json(result_path, result)
    completed = load_jsonl(evaluations_path)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer_module, template, config = load_model_bundle(
        config_path,
        model_path,
        device,
        trust_remote_code=False,
        preprocessing_num_workers=1,
    )
    if model_layer_dims(model) != layer_dims:
        raise ValueError("Original model layer widths differ from the frozen winners.")
    config.update(
        {
            "dataset_dir": str(prepare_dataset(output_dir, target_samples, samples_payload["canonical_prompt"])),
            "dataset": "phase6b_single_samples",
            "eval_dataset": None,
            "tokenized_path": None,
            "max_samples": len(target_samples),
            "overwrite_cache": True,
            "enable_thinking": False,
        }
    )
    dataloader, dataset_manifest = build_dataloader(
        config,
        model,
        tokenizer_module,
        template,
        batch_size=1,
        num_workers=0,
        sample_offset=0,
        max_samples=len(target_samples),
        allow_short_dataset=False,
        max_image_repeat=1,
        allow_excessive_image_repeats=False,
        dataset_stage="sft",
    )
    result["dataset_manifest"] = dataset_manifest
    tokenizer = tokenizer_module["tokenizer"]
    contexts = {}
    for sample, batch in zip(target_samples, dataloader, strict=True):
        batch = move_batch_to_device(batch, device)
        prompt_inputs, _ = build_prompt_inputs(batch)
        contexts[sample["sample_id"]] = {
            "batch": batch,
            "prompt_inputs": prompt_inputs,
            "reference_trace": build_reference_trace(model, batch),
            "semantic_evaluator": build_semantic_evaluator(sample["contract"]),
        }

    def evaluate(
        evaluation_id: str,
        kept: LayerSets,
        target: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if evaluation_id in completed:
            return completed[evaluation_id]
        context = contexts[target]
        print(f"Phase 6D-I {evaluation_id}", flush=True)
        with MLPNeuronAblator(model, deletion_masks(kept, layer_dims)):
            proxy = compact_proxy(evaluate_gold_proxy(model, context["batch"], context["reference_trace"]))
            generation = generate_short_caption(
                model,
                tokenizer,
                context["prompt_inputs"],
                max_new_tokens=args.max_new_tokens,
                semantic_evaluator=context["semantic_evaluator"],
            )
        row = {
            "evaluation_id": evaluation_id,
            "family": "identity_refinement",
            "target_sample_id": target,
            "kept_sha256": kept_sha256(kept),
            "mask_summary": summarize_kept(kept, layer_dims),
            "variant_metadata": metadata,
            "gold_proxy": proxy,
            "generation": generation,
            "automatic_semantic_pass": generation["semantic"]["automatic_pass"],
        }
        append_jsonl(evaluations_path, row)
        completed[evaluation_id] = row
        return row

    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    for candidate in candidates:
        contrast_payload = contrast_analysis[candidate["contrast_id"]]["contrast"]
        contrast = ConflictContrast(**contrast_payload)
        base = winners[contrast.base]
        donor = winners[contrast.donor]
        union = layerwise_union(base, donor)
        donor_increment = layerwise_difference(donor, base)
        layer = int(candidate["layer"])
        current = set(donor_increment[layer])
        base_nll = contrast_analysis[contrast.contrast_id]["endpoint_reproduction"]["safe_base_nll"]
        union_nll = contrast_analysis[contrast.contrast_id]["endpoint_reproduction"]["failed_union_nll"]
        candidate_id = f"{contrast.contrast_id}__layer_{layer:02d}"
        history = []
        subset_evaluations = 0
        granularity = min(2, len(current))
        stop_reason = "singleton"
        accepted_step = 0

        while len(current) > 1 and subset_evaluations < args.max_subset_evaluations:
            proposal_seed = derive_seed(args.seed, contrast.contrast_id, layer, identity_sha256(current))
            proposals = delta_debug_proposals(current, granularity, seed=proposal_seed)
            remaining = args.max_subset_evaluations - subset_evaluations
            if len(proposals) > remaining:
                stop_reason = "budget_exhausted_before_full_granularity"
                break
            evaluated = []
            for proposal in proposals:
                identities = set(proposal.identities)
                subset_sha = proposal.sha256
                add_kept, loo_kept = build_kept_pair(base, donor, layer, identities)
                metadata = {
                    "candidate_id": candidate_id,
                    "contrast": contrast_payload,
                    "layer": layer,
                    "subset_sha256": subset_sha,
                    "subset_size": len(identities),
                    "granularity": granularity,
                    "proposal_kind": proposal.kind,
                    "partition_index": proposal.partition_index,
                    "control": False,
                }
                prefix = f"refine__{candidate_id}__subset_{subset_sha[:16]}"
                add_row = evaluate(f"{prefix}__add", add_kept, contrast.target, {**metadata, "direction": "add"})
                loo_row = evaluate(f"{prefix}__loo", loo_kept, contrast.target, {**metadata, "direction": "loo"})
                predicate_pass = bool(not add_row["automatic_semantic_pass"] and loo_row["automatic_semantic_pass"])
                evaluated.append(
                    {
                        "identities": identities,
                        "subset_sha256": subset_sha,
                        "subset_size": len(identities),
                        "proposal_kind": proposal.kind,
                        "partition_index": proposal.partition_index,
                        "predicate_pass": predicate_pass,
                        "add_pass": add_row["automatic_semantic_pass"],
                        "loo_pass": loo_row["automatic_semantic_pass"],
                        "add_nll": add_row["gold_proxy"]["mean_nll"],
                        "loo_nll": loo_row["gold_proxy"]["mean_nll"],
                        "joint_nll_margin": min(
                            add_row["gold_proxy"]["mean_nll"] - base_nll,
                            union_nll - loo_row["gold_proxy"]["mean_nll"],
                        ),
                    }
                )
            subset_evaluations += len(proposals)
            passing = [row for row in evaluated if row["predicate_pass"]]
            chosen = choose_reduction(passing)
            history.append(
                {
                    "granularity": granularity,
                    "input_size": len(current),
                    "proposals": [
                        {key: value for key, value in row.items() if key != "identities"} for row in evaluated
                    ],
                    "chosen_subset_sha256": None if chosen is None else chosen["subset_sha256"],
                }
            )
            if chosen is not None:
                current = set(chosen["identities"])
                accepted_step += 1
                counts = {
                    candidate_layer: len(current) if candidate_layer == layer else 0 for candidate_layer in layer_dims
                }
                outside_union = complement_pool(union, layer_dims)
                for replicate in range(args.random_seeds):
                    control_seed = derive_seed(args.seed, candidate_id, accepted_step, replicate)
                    random_add = deterministic_sample(outside_union, counts, seed=control_seed)
                    random_remove = deterministic_sample(base, counts, seed=derive_seed(control_seed, "loo"))
                    control_metadata = {
                        "candidate_id": candidate_id,
                        "contrast": contrast_payload,
                        "layer": layer,
                        "accepted_step": accepted_step,
                        "matched_count": len(current),
                        "control": True,
                        "replicate": replicate,
                    }
                    control_prefix = f"control__{candidate_id}__step_{accepted_step:02d}__r{replicate:02d}"
                    evaluate(
                        f"{control_prefix}__add",
                        layerwise_union(base, random_add),
                        contrast.target,
                        {**control_metadata, "direction": "add", "control_pool": "outside_failed_union"},
                    )
                    evaluate(
                        f"{control_prefix}__loo",
                        layerwise_difference(union, random_remove),
                        contrast.target,
                        {**control_metadata, "direction": "loo", "control_pool": "safe_base_resident"},
                    )
                granularity = min(max(2, granularity - 1), len(current))
            elif granularity < len(current):
                granularity = min(len(current), granularity * 2)
            else:
                stop_reason = "one_minimal_under_tested_partition_order"
                break
        else:
            if subset_evaluations >= args.max_subset_evaluations:
                stop_reason = "subset_evaluation_budget_exhausted"

        candidate_result = {
            "candidate_id": candidate_id,
            "contrast": contrast_payload,
            "layer": layer,
            "initial_size": len(donor_increment[layer]),
            "final_size": len(current),
            "final_identities": sorted(current),
            "final_identity_sha256": identity_sha256(current),
            "accepted_reductions": accepted_step,
            "subset_evaluations": subset_evaluations,
            "stop_reason": stop_reason,
            "history": history,
        }
        write_json(candidate_dir / f"{candidate_id}.json", candidate_result)
        result["candidate_results"].append({key: value for key, value in candidate_result.items() if key != "history"})
        write_json(result_path, result)

    result["complete"] = len(result["candidate_results"]) == len(candidates)
    result["total_model_evaluations"] = len(completed)
    write_json(result_path, result)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "candidates": len(result["candidate_results"]),
                "model_evaluations": len(completed),
            },
            indent=2,
        )
    )
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
