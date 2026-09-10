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

"""Verify hook, in-memory structural, and reloaded Phase-5C equivalence."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase3_structural_utils import canonical_json_sha256, count_parameters, neuron_ids_to_masks  # noqa: E402
from phase5_cached_utils import (  # noqa: E402
    cached_fidelity,
    collect_cached_teacher_forced_logits,
    generate_cached_teacher_trace,
)
from phase5_structural_utils import (  # noqa: E402
    compare_logits,
    target_layer_dims,
    validate_partial_singleton_cluster_idx,
)
from run_phase2_ablation import (  # noqa: E402
    MLPNeuronAblator,
    build_dataloader,
    load_yaml,
    move_batch_to_device,
)
from run_phase5_single_sample_frontier import (  # noqa: E402
    build_prompt_inputs,
    build_teacher_rollout_batch,
    compare_generation,
    evaluate_teacher_fidelity,
    generate_trace,
    valid_next_token_tensors,
)
from verify_structural_equivalence import actual_projection_hashes, tensor_sha256  # noqa: E402

from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402
from llamafactory.train.vulcan.modeling import find_mlp_layers, get_intermediate_size  # noqa: E402
from llamafactory.train.vulcan.pruning import pruning_mlp  # noqa: E402
from llamafactory.train.vulcan.schema import load_cluster_idx  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase-5C structural equivalence on its frozen sample.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--pruned_model_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override the frozen search horizon; the original model defines the extended teacher rollout.",
    )
    parser.add_argument("--seed", type=int, default=2051)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_model_bundle(
    config_path: str | Path,
    model_path: str | Path,
    device: torch.device,
    *,
    trust_remote_code: bool,
    preprocessing_num_workers: int,
    infer_dtype: str | None = None,
):
    config = load_yaml(config_path)
    config.update(
        {
            "model_name_or_path": str(model_path),
            "trust_remote_code": trust_remote_code,
            "preprocessing_num_workers": preprocessing_num_workers,
            "do_train": False,
            "do_eval": False,
            "do_predict": False,
        }
    )
    if infer_dtype is not None:
        config["infer_dtype"] = infer_dtype
    config.setdefault("output_dir", "saves/neuron_typing/phase5_structural_tmp")
    model_args, data_args, _, finetuning_args, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    target_dtype = None if infer_dtype in (None, "auto") else getattr(torch, infer_dtype)
    model = model.to(device) if target_dtype is None else model.to(device=device, dtype=target_dtype)
    model.eval().requires_grad_(False)
    return model, tokenizer_module, template, config


@torch.no_grad()
def build_teacher_trace(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    labels = batch["labels"]
    outputs = model(**{key: value for key, value in batch.items() if key != "labels"}, use_cache=False)
    valid_logits, valid_labels = valid_next_token_tensors(outputs.logits, labels)
    logits = valid_logits.detach().cpu()
    labels = valid_labels.detach().cpu()
    return {
        "valid_logits": logits,
        "valid_labels": labels,
        "nll": float(F.cross_entropy(logits, labels)),
        "num_label_tokens": int(labels.numel()),
    }


@torch.no_grad()
def collect_valid_logits(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    outputs = model(**{key: value for key, value in batch.items() if key != "labels"}, use_cache=False)
    logits, _ = valid_next_token_tensors(outputs.logits, batch["labels"])
    return logits.detach().cpu()


def expected_projection_hashes(
    model: torch.nn.Module, cluster_idx: list[list[dict[str, Any]] | None]
) -> dict[str, dict[str, str]]:
    result = {}
    for layer_ref, clusters in zip(find_mlp_layers(model), cluster_idx):
        if clusters is None:
            keep_ids = torch.arange(
                get_intermediate_size(layer_ref.mlp), device=layer_ref.mlp.up_proj.weight.device, dtype=torch.long
            )
        else:
            keep_ids = torch.tensor(
                [int(cluster["anchor"]) for cluster in clusters],
                device=layer_ref.mlp.up_proj.weight.device,
                dtype=torch.long,
            )
        result[str(layer_ref.index)] = {
            "gate_proj.weight": tensor_sha256(layer_ref.mlp.gate_proj.weight.index_select(0, keep_ids)),
            "up_proj.weight": tensor_sha256(layer_ref.mlp.up_proj.weight.index_select(0, keep_ids)),
            "down_proj.weight": tensor_sha256(layer_ref.mlp.down_proj.weight.index_select(1, keep_ids)),
        }
    return result


def model_layer_dims(model: torch.nn.Module) -> dict[int, int]:
    return {layer.index: get_intermediate_size(layer.mlp) for layer in find_mlp_layers(model)}


def is_feasible(fidelity: dict[str, Any], generation: dict[str, Any], kl_tolerance: float) -> bool:
    return bool(
        generation["exact_match"]
        and fidelity["consistent_token_agreement"] == 1.0
        and fidelity["mean_kl"] <= kl_tolerance
    )


def is_cached_feasible(fidelity: dict[str, Any], generation: dict[str, Any], kl_tolerance: float) -> bool:
    return bool(is_cached_behavior_preserved(fidelity, generation) and fidelity["mean_kl"] <= kl_tolerance)


def is_cached_behavior_preserved(fidelity: dict[str, Any], generation: dict[str, Any]) -> bool:
    return bool(
        generation["exact_match"]
        and fidelity["token_agreement"] == 1.0
        and fidelity["generated_token_agreement"] == 1.0
    )


def release_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        raise ValueError("Structural verification cannot use a horizon shorter than the frozen Phase-5B search.")
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
        hook_generation_trace = generate_trace(model, tokenizer, prompt_inputs, max_new_tokens)
    hook_generation = compare_generation(teacher_generation, hook_generation_trace)

    pruning_summary = pruning_mlp(model, cluster_idx)
    structural_dims = model_layer_dims(model)
    structural_parameters = count_parameters(model)
    structural_hashes = actual_projection_hashes(model)
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
    structural_generation_trace = generate_trace(model, tokenizer, prompt_inputs, max_new_tokens)
    structural_generation = compare_generation(teacher_generation, structural_generation_trace)
    hook_vs_structural_logits = compare_logits(hook_logits, structural_logits)
    hook_vs_structural_generation = compare_generation(hook_generation_trace, structural_generation_trace)

    del model, tokenizer_module, template
    release_device_cache()

    reloaded, reloaded_tokenizer_module, _, _ = load_model_bundle(
        metadata["config_path"],
        args.pruned_model_path,
        device,
        trust_remote_code=True,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )
    reloaded_tokenizer = reloaded_tokenizer_module["tokenizer"]
    reload_dims = model_layer_dims(reloaded)
    reload_parameters = count_parameters(reloaded)
    reload_hashes = actual_projection_hashes(reloaded)
    reload_logits = collect_valid_logits(reloaded, batch)
    reload_fidelity = evaluate_teacher_fidelity(reloaded, batch, teacher_trace)
    reload_cached_fidelity = (
        cached_fidelity(
            teacher_cached_trace["raw_logits"],
            collect_cached_teacher_forced_logits(reloaded, prompt_inputs, teacher_generation["token_ids"]),
            teacher_generation["token_ids"],
        )
        if teacher_cached_trace is not None
        else None
    )
    reload_generation_trace = generate_trace(reloaded, reloaded_tokenizer, prompt_inputs, max_new_tokens)
    reload_generation = compare_generation(teacher_generation, reload_generation_trace)
    structural_vs_reload_logits = compare_logits(structural_logits, reload_logits)
    structural_vs_reload_generation = compare_generation(structural_generation_trace, reload_generation_trace)

    structure_checks = {
        "artifact_valid": validation["validated"],
        "in_memory_dims_match_target": structural_dims == expected_dims,
        "reload_dims_match_target": reload_dims == expected_dims,
        "parameters_reduced_by_expected_count": (
            original_parameters["total"] - structural_parameters["total"]
            == metadata["theoretical_reduction"]["removed_parameters"]
        ),
        "reload_parameters_match_in_memory": reload_parameters["total"] == structural_parameters["total"],
        "singleton_weights_match_expected": structural_hashes == expected_hashes,
        "reload_weights_match_in_memory": reload_hashes == structural_hashes,
    }
    hook_source_fidelity = hook_cached_fidelity or hook_fidelity
    structural_source_fidelity = structural_cached_fidelity or structural_fidelity
    reload_source_fidelity = reload_cached_fidelity or reload_fidelity
    source_gate = metadata.get("source_gate", "strict")
    physical_probe = source_gate == "physical_probe"
    structural_strict = source_gate == "structural_strict"
    if uses_cached_fidelity:
        hook_strict_feasible = is_cached_feasible(hook_source_fidelity, hook_generation, kl_tolerance)
        structural_strict_feasible = is_cached_feasible(
            structural_source_fidelity, structural_generation, kl_tolerance
        )
        reload_strict_feasible = is_cached_feasible(reload_source_fidelity, reload_generation, kl_tolerance)
        hook_behavior_preserved = is_cached_behavior_preserved(hook_source_fidelity, hook_generation)
        structural_behavior_preserved = is_cached_behavior_preserved(structural_source_fidelity, structural_generation)
        reload_behavior_preserved = is_cached_behavior_preserved(reload_source_fidelity, reload_generation)
    else:
        hook_strict_feasible = is_feasible(hook_source_fidelity, hook_generation, kl_tolerance)
        structural_strict_feasible = is_feasible(structural_source_fidelity, structural_generation, kl_tolerance)
        reload_strict_feasible = is_feasible(reload_source_fidelity, reload_generation, kl_tolerance)
        hook_behavior_preserved = hook_strict_feasible
        structural_behavior_preserved = structural_strict_feasible
        reload_behavior_preserved = reload_strict_feasible
    behavior_checks = {
        "hook_remains_feasible": hook_behavior_preserved if physical_probe else hook_strict_feasible,
        "in_memory_structural_feasible": (
            structural_behavior_preserved if physical_probe else structural_strict_feasible
        ),
        "reloaded_structural_feasible": reload_behavior_preserved if physical_probe else reload_strict_feasible,
        "hook_and_structural_generation_equal": hook_vs_structural_generation["exact_match"],
        "structural_and_reload_generation_equal": structural_vs_reload_generation["exact_match"],
        "structural_and_reload_logits_exact": structural_vs_reload_logits["max_abs"] == 0.0,
    }
    required_behavior_check_names = [
        "in_memory_structural_feasible",
        "reloaded_structural_feasible",
        "structural_and_reload_generation_equal",
        "structural_and_reload_logits_exact",
    ]
    if not structural_strict:
        required_behavior_check_names.extend(["hook_remains_feasible", "hook_and_structural_generation_equal"])
    result = {
        "complete": True,
        "passed": all(structure_checks.values())
        and all(behavior_checks[name] for name in required_behavior_check_names),
        "config": vars(args),
        "sample_manifest": manifest,
        "artifact": metadata,
        "structure": {
            "original_layer_dims": original_dims,
            "target_layer_dims": expected_dims,
            "in_memory_layer_dims": structural_dims,
            "reload_layer_dims": reload_dims,
            "original_parameters": original_parameters,
            "in_memory_parameters": structural_parameters,
            "reload_parameters": reload_parameters,
            "pruning_summary": {
                "original_intermediate_size": pruning_summary.original_intermediate_size,
                "pruned_intermediate_size": pruning_summary.pruned_intermediate_size,
                "num_layers": pruning_summary.num_layers,
            },
            "checks": structure_checks,
        },
        "behavior": {
            "checks": behavior_checks,
            "required_check_names": required_behavior_check_names,
            "gate_results": {
                "source_gate": source_gate,
                "hook_behavior_preserved": hook_behavior_preserved,
                "in_memory_structural_behavior_preserved": structural_behavior_preserved,
                "reloaded_structural_behavior_preserved": reload_behavior_preserved,
                "hook_strict_feasible": hook_strict_feasible,
                "in_memory_structural_strict_feasible": structural_strict_feasible,
                "reloaded_structural_strict_feasible": reload_strict_feasible,
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
            "reloaded_structural": {
                "source_fidelity": reload_source_fidelity,
                "teacher_fidelity": reload_fidelity,
                "cached_fidelity": reload_cached_fidelity,
                "generation": reload_generation,
            },
            "hook_vs_in_memory_logits": hook_vs_structural_logits,
            "in_memory_vs_reload_logits": structural_vs_reload_logits,
        },
    }
    write_json(args.output_file, result)
    print(json.dumps({"passed": result["passed"], "output_file": args.output_file}, indent=2))
    if not result["passed"]:
        raise RuntimeError("Phase-5C structural equivalence failed; see output_file.")


if __name__ == "__main__":
    main()
