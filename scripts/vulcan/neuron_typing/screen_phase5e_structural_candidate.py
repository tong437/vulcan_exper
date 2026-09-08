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

"""Screen a Phase 5E hook mask after true in-memory FFN structural pruning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase3_structural_utils import canonical_json_sha256, count_parameters, neuron_ids_to_masks  # noqa: E402
from phase5_structural_utils import target_layer_dims, validate_partial_singleton_cluster_idx  # noqa: E402
from phase5e_proxy import evaluate_gold_proxy  # noqa: E402
from phase5e_semantics import generate_short_caption  # noqa: E402
from run_phase2_ablation import MLPNeuronAblator, build_dataloader, move_batch_to_device  # noqa: E402
from run_phase5_single_sample_frontier import build_prompt_inputs, write_json  # noqa: E402
from run_phase5e_retrospective import build_reference_trace  # noqa: E402
from verify_phase5_structural_equivalence import (  # noqa: E402
    actual_projection_hashes,
    expected_projection_hashes,
    load_json,
    load_model_bundle,
    model_layer_dims,
)

from llamafactory.train.vulcan.pruning import pruning_mlp  # noqa: E402
from llamafactory.train.vulcan.schema import load_cluster_idx  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Physically screen one Phase 5E structural artifact.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2058)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(args.artifact_dir).resolve()
    metadata = load_json(artifact_dir / "metadata.json")
    if metadata.get("phase") != "5E-C":
        raise ValueError("The artifact is not a Phase 5E-C artifact.")
    neuron_ids = load_json(artifact_dir / "mask.json")
    cluster_idx = load_cluster_idx(artifact_dir / "cluster_idx.json")
    if canonical_json_sha256(neuron_ids) != metadata["mask_sha256"]:
        raise ValueError("Artifact mask hash mismatch.")
    if canonical_json_sha256(cluster_idx) != metadata["cluster_idx_sha256"]:
        raise ValueError("Artifact cluster hash mismatch.")

    static = load_json(metadata["static_frontier"])
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
        dataset_stage="sft",
    )
    batch = move_batch_to_device(next(iter(dataloader)), device)
    prompt_inputs, _ = build_prompt_inputs(batch)
    reference_trace = build_reference_trace(model, batch)
    if reference_trace["valid_labels"].tolist() != static["reference"]["gold_token_ids"]:
        raise RuntimeError("Gold-caption tokens do not reproduce the frozen Phase 5E-A reference.")

    original_dims = model_layer_dims(model)
    masks = neuron_ids_to_masks(neuron_ids, original_dims)
    validation = validate_partial_singleton_cluster_idx(cluster_idx, masks)
    expected_dims = target_layer_dims(masks)
    expected_hashes = expected_projection_hashes(model, cluster_idx)
    original_parameters = count_parameters(model)

    with MLPNeuronAblator(model, masks):
        hook_proxy = evaluate_gold_proxy(model, batch, reference_trace)
        hook_generation = generate_short_caption(model, tokenizer, prompt_inputs, max_new_tokens=args.max_new_tokens)

    pruning_mlp(model, cluster_idx)
    structural_dims = model_layer_dims(model)
    structural_parameters = count_parameters(model)
    structural_hashes = actual_projection_hashes(model)
    structural_proxy = evaluate_gold_proxy(model, batch, reference_trace)
    structural_generation = generate_short_caption(model, tokenizer, prompt_inputs, max_new_tokens=args.max_new_tokens)
    checks = {
        "artifact_valid": validation["validated"],
        "dims_match_target": structural_dims == expected_dims,
        "parameters_reduced_by_expected_count": (
            original_parameters["total"] - structural_parameters["total"]
            == metadata["theoretical_reduction"]["removed_parameters"]
        ),
        "singleton_weights_match_expected": structural_hashes == expected_hashes,
        "hook_automatic_semantic_pass": hook_generation["semantic"]["automatic_pass"],
        "structural_automatic_semantic_pass": structural_generation["semantic"]["automatic_pass"],
    }
    required = [
        "artifact_valid",
        "dims_match_target",
        "parameters_reduced_by_expected_count",
        "singleton_weights_match_expected",
        "structural_automatic_semantic_pass",
    ]
    result = {
        "complete": True,
        "passed": all(checks[name] for name in required),
        "artifact": metadata,
        "sample_manifest": manifest,
        "checks": checks,
        "required_check_names": required,
        "structure": {
            "original_layer_dims": original_dims,
            "target_layer_dims": expected_dims,
            "in_memory_layer_dims": structural_dims,
            "original_parameters": original_parameters,
            "in_memory_parameters": structural_parameters,
        },
        "hook": {"gold_proxy": hook_proxy, "generation": hook_generation},
        "in_memory_structural": {
            "gold_proxy": structural_proxy,
            "generation": structural_generation,
            "human_confirmation": None,
        },
    }
    write_json(args.output_file, result)
    print(json.dumps({"passed": result["passed"], "output_file": args.output_file}, indent=2))
    if not result["passed"]:
        raise RuntimeError("Phase 5E physical structural screen failed; inspect output_file.")


if __name__ == "__main__":
    main()
