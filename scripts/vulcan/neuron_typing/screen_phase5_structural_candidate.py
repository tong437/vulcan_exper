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

"""Screen hook-feasible masks after in-memory physical Phase-5C pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from phase3_structural_utils import canonical_json_sha256, count_parameters, neuron_ids_to_masks
from phase5_cached_utils import (
    cached_fidelity,
    collect_cached_teacher_forced_logits,
    generate_cached_teacher_trace,
)
from phase5_structural_utils import compare_logits, target_layer_dims, validate_partial_singleton_cluster_idx
from run_phase2_ablation import MLPNeuronAblator, build_dataloader, move_batch_to_device
from run_phase5_single_sample_frontier import (
    build_prompt_inputs,
    build_teacher_rollout_batch,
    compare_generation,
    evaluate_teacher_fidelity,
    generate_trace,
)
from verify_phase5_structural_equivalence import (
    actual_projection_hashes,
    build_teacher_trace,
    collect_valid_logits,
    expected_projection_hashes,
    is_feasible,
    load_json,
    load_model_bundle,
    model_layer_dims,
    write_json,
)

from llamafactory.train.vulcan.pruning import pruning_mlp
from llamafactory.train.vulcan.schema import load_cluster_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen one Phase-5C artifact without writing a checkpoint.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2051)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(args.artifact_dir).resolve()
    metadata = load_json(artifact_dir / "metadata.json")
    neuron_ids = load_json(artifact_dir / "mask.json")
    cluster_idx = load_cluster_idx(artifact_dir / "cluster_idx.json")
    if canonical_json_sha256(neuron_ids) != metadata["mask_sha256"]:
        raise ValueError("Artifact mask hash mismatch.")
    if canonical_json_sha256(cluster_idx) != metadata["cluster_idx_sha256"]:
        raise ValueError("Artifact cluster hash mismatch.")

    learned = load_json(metadata["learned_frontier"])
    static = load_json(metadata["static_frontier"])
    frozen_horizon = int(
        metadata["frozen_search_horizon"]
        if "frozen_search_horizon" in metadata
        else learned["config"]["max_new_tokens"]
    )
    kl_tolerance = float(metadata["kl_tolerance"] if "kl_tolerance" in metadata else learned["config"]["kl_tolerance"])
    max_new_tokens = args.max_new_tokens or frozen_horizon
    if max_new_tokens < frozen_horizon:
        raise ValueError("Structural screening cannot use a horizon shorter than the frozen search horizon.")
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
    dataset_batch = move_batch_to_device(next(iter(dataloader)), device)
    prompt_inputs, _ = build_prompt_inputs(dataset_batch)
    uses_cached_fidelity = metadata.get("source_fidelity_path") == "cached"
    if uses_cached_fidelity:
        teacher_cached_trace = generate_cached_teacher_trace(model, tokenizer, prompt_inputs, max_new_tokens)
        teacher_generation = {
            "token_ids": teacher_cached_trace["token_ids"],
            "text": teacher_cached_trace["text"],
        }
    else:
        teacher_cached_trace = None
        teacher_generation = generate_trace(model, tokenizer, prompt_inputs, max_new_tokens)
    frozen_teacher_ids = static["teacher"]["generated_token_ids"]
    if teacher_generation["token_ids"].tolist()[: len(frozen_teacher_ids)] != frozen_teacher_ids:
        raise RuntimeError("Original model did not reproduce the frozen Phase-5 trajectory prefix.")
    batch = build_teacher_rollout_batch(prompt_inputs, teacher_generation["token_ids"])
    teacher_trace = build_teacher_trace(model, batch)

    original_dims = model_layer_dims(model)
    masks = neuron_ids_to_masks(neuron_ids, original_dims)
    validation = validate_partial_singleton_cluster_idx(cluster_idx, masks)
    expected_dims = target_layer_dims(masks)
    original_parameters = count_parameters(model)
    expected_hashes = expected_projection_hashes(model, cluster_idx)
    with MLPNeuronAblator(model, masks):
        hook_logits = collect_valid_logits(model, batch)
        hook_fidelity = evaluate_teacher_fidelity(model, batch, teacher_trace)
        hook_cached_fidelity = (
            cached_fidelity(
                teacher_cached_trace["raw_logits"],
                collect_cached_teacher_forced_logits(model, prompt_inputs, teacher_generation["token_ids"]),
                teacher_generation["token_ids"],
            )
            if teacher_cached_trace is not None
            else None
        )
        hook_trace = generate_trace(model, tokenizer, prompt_inputs, max_new_tokens)
    hook_generation = compare_generation(teacher_generation, hook_trace)

    pruning_mlp(model, cluster_idx)
    structural_dims = model_layer_dims(model)
    structural_parameters = count_parameters(model)
    structural_logits = collect_valid_logits(model, batch)
    structural_fidelity = evaluate_teacher_fidelity(model, batch, teacher_trace)
    structural_cached_fidelity = (
        cached_fidelity(
            teacher_cached_trace["raw_logits"],
            collect_cached_teacher_forced_logits(model, prompt_inputs, teacher_generation["token_ids"]),
            teacher_generation["token_ids"],
        )
        if teacher_cached_trace is not None
        else None
    )
    structural_trace = generate_trace(model, tokenizer, prompt_inputs, max_new_tokens)
    structural_generation = compare_generation(teacher_generation, structural_trace)
    hook_source_fidelity = hook_cached_fidelity or hook_fidelity
    structural_source_fidelity = structural_cached_fidelity or structural_fidelity
    if uses_cached_fidelity:
        hook_strict_feasible = cached_feasible(hook_source_fidelity, hook_generation, kl_tolerance)
        structural_strict_feasible = cached_feasible(structural_source_fidelity, structural_generation, kl_tolerance)
        hook_behavior_preserved = cached_behavior_preserved(hook_source_fidelity, hook_generation)
        structural_behavior_preserved = cached_behavior_preserved(structural_source_fidelity, structural_generation)
    else:
        hook_strict_feasible = is_feasible(hook_source_fidelity, hook_generation, kl_tolerance)
        structural_strict_feasible = is_feasible(structural_source_fidelity, structural_generation, kl_tolerance)
        hook_behavior_preserved = hook_strict_feasible
        structural_behavior_preserved = structural_strict_feasible
    source_gate = metadata.get("source_gate", "strict")
    physical_probe = source_gate == "physical_probe"
    structural_strict = source_gate == "structural_strict"
    hook_source_feasible = hook_behavior_preserved if physical_probe else hook_strict_feasible
    structural_source_feasible = structural_behavior_preserved if physical_probe else structural_strict_feasible
    checks = {
        "artifact_valid": validation["validated"],
        "dims_match_target": structural_dims == expected_dims,
        "parameters_reduced_by_expected_count": (
            original_parameters["total"] - structural_parameters["total"]
            == metadata["theoretical_reduction"]["removed_parameters"]
        ),
        "singleton_weights_match_expected": actual_projection_hashes(model) == expected_hashes,
        "hook_remains_feasible": hook_source_feasible,
        "structural_feasible": structural_source_feasible,
    }
    required_check_names = [
        "artifact_valid",
        "dims_match_target",
        "parameters_reduced_by_expected_count",
        "singleton_weights_match_expected",
        "structural_feasible",
    ]
    if not structural_strict:
        required_check_names.append("hook_remains_feasible")
    result = {
        "complete": True,
        "passed": all(checks[name] for name in required_check_names),
        "artifact": metadata,
        "sample_manifest": manifest,
        "checks": checks,
        "required_check_names": required_check_names,
        "gate_results": {
            "source_gate": source_gate,
            "hook_behavior_preserved": hook_behavior_preserved,
            "structural_behavior_preserved": structural_behavior_preserved,
            "hook_strict_feasible": hook_strict_feasible,
            "structural_strict_feasible": structural_strict_feasible,
            "kl_tolerance": kl_tolerance,
        },
        "hook": {
            "source_fidelity": hook_source_fidelity,
            "teacher_fidelity": hook_fidelity,
            "cached_fidelity": hook_cached_fidelity,
            "generation": hook_generation,
        },
        "in_memory_structural": {
            "source_fidelity": structural_source_fidelity,
            "teacher_fidelity": structural_fidelity,
            "cached_fidelity": structural_cached_fidelity,
            "generation": structural_generation,
        },
        "hook_vs_structural": {
            "logits": compare_logits(hook_logits, structural_logits),
            "generation": compare_generation(hook_trace, structural_trace),
        },
    }
    write_json(args.output_file, result)
    print(json.dumps({"passed": result["passed"], "output_file": args.output_file}, indent=2))


def cached_feasible(fidelity: dict, generation: dict, kl_tolerance: float) -> bool:
    return bool(cached_behavior_preserved(fidelity, generation) and fidelity["mean_kl"] <= kl_tolerance)


def cached_behavior_preserved(fidelity: dict, generation: dict) -> bool:
    return bool(
        generation["exact_match"]
        and fidelity["token_agreement"] == 1.0
        and fidelity["generated_token_agreement"] == 1.0
    )


if __name__ == "__main__":
    main()
