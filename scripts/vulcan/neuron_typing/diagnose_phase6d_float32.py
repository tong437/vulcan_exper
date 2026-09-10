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

"""Diagnose whether final Phase 6D BF16 conflict subsets persist in FP32."""

from __future__ import annotations

import argparse
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
from phase6c_subnets import deletion_masks, kept_from_deleted, layerwise_union  # noqa: E402
from run_phase2_ablation import MLPNeuronAblator, build_dataloader, load_yaml, move_batch_to_device  # noqa: E402
from run_phase5_single_sample_frontier import build_prompt_inputs, write_json  # noqa: E402
from run_phase5e_retrospective import build_reference_trace  # noqa: E402
from run_phase6b_single_sample_limits import prepare_dataset, sha256_file, validate_inputs  # noqa: E402
from run_phase6d_conflict_localization import compact_proxy  # noqa: E402
from run_phase6d_identity_refinement import build_kept_pair  # noqa: E402
from verify_phase5_structural_equivalence import load_model_bundle, model_layer_dims  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FP32 diagnostics for final Phase 6D conflict subsets.")
    parser.add_argument("--identity_analysis", required=True)
    parser.add_argument("--winner_dir", default="saves/neuron_typing/phase6b_frozen_winners_v1")
    parser.add_argument("--sample_file", default="data/phase6b_single_samples/frozen_samples.json")
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/phase5e_coco.yaml")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2067)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be positive.")
    analysis_path = Path(args.identity_analysis).resolve()
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if not analysis.get("complete") or not analysis["integrity"]["all_final_subsets_satisfy_bidirectional_predicate"]:
        raise ValueError("FP32 diagnosis requires a complete, validated BF16 identity analysis.")
    winner_dir = Path(args.winner_dir).resolve()
    sample_path = Path(args.sample_file).resolve()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"FP32 diagnostic output already exists: {output_path}.")

    samples_payload = json.loads(sample_path.read_text(encoding="utf-8"))
    samples = validate_inputs(samples_payload, None)
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    target_ids = list(dict.fromkeys(candidate["target"] for candidate in analysis["candidates"]))
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
    base_config = load_yaml(config_path)
    model_path = Path(args.model_name_or_path or base_config["model_name_or_path"]).resolve()
    result = {
        "complete": False,
        "config": {
            "identity_analysis": str(analysis_path),
            "identity_analysis_sha256": sha256_file(analysis_path),
            "winner_dir": str(winner_dir),
            "winner_manifest_sha256": sha256_file(winner_dir / "frozen_winners.json"),
            "sample_file": str(sample_path),
            "sample_file_sha256": sha256_file(sample_path),
            "config_path": str(config_path),
            "model_name_or_path": str(model_path),
            "infer_dtype": "float32",
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "candidates": {},
    }
    write_json(output_path, result)

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
        infer_dtype="float32",
    )
    if model_layer_dims(model) != layer_dims or next(model.parameters()).dtype != torch.float32:
        raise ValueError("FP32 diagnostic model dimensions or dtype are incorrect.")
    dataset_dir = output_path.parent / "float32_dataset"
    config.update(
        {
            "dataset_dir": str(prepare_dataset(dataset_dir, target_samples, samples_payload["canonical_prompt"])),
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

    def evaluate(kept, target: str) -> dict[str, Any]:
        context = contexts[target]
        with MLPNeuronAblator(model, deletion_masks(kept, layer_dims)):
            proxy = compact_proxy(evaluate_gold_proxy(model, context["batch"], context["reference_trace"]))
            generation = generate_short_caption(
                model,
                tokenizer,
                context["prompt_inputs"],
                max_new_tokens=args.max_new_tokens,
                semantic_evaluator=context["semantic_evaluator"],
            )
        return {
            "semantic_pass": generation["semantic"]["automatic_pass"],
            "gold_proxy": proxy,
            "generation": generation,
        }

    for candidate in analysis["candidates"]:
        candidate_payload = json.loads(Path(candidate["final_identity_file"]).read_text(encoding="utf-8"))
        contrast = candidate_payload["contrast"]
        base = winners[contrast["base"]]
        donor = winners[contrast["donor"]]
        union = layerwise_union(base, donor)
        add, loo = build_kept_pair(
            base,
            donor,
            int(candidate["layer"]),
            set(candidate_payload["final_identities"]),
        )
        target = contrast["target"]
        endpoints = {
            "safe_base": evaluate(base, target),
            "failed_union": evaluate(union, target),
            "final_add": evaluate(add, target),
            "final_loo": evaluate(loo, target),
        }
        predicate = bool(
            endpoints["safe_base"]["semantic_pass"]
            and not endpoints["failed_union"]["semantic_pass"]
            and not endpoints["final_add"]["semantic_pass"]
            and endpoints["final_loo"]["semantic_pass"]
        )
        result["candidates"][candidate["candidate_id"]] = {
            "bf16_final_identity_sha256": candidate["final_identity_sha256"],
            "final_size": candidate["final_size"],
            "fp32_full_pass_to_fail_and_bidirectional_predicate": predicate,
            "endpoints": endpoints,
        }
        write_json(output_path, result)
    result["complete"] = len(result["candidates"]) == len(analysis["candidates"])
    result["summary"] = {
        "candidates": len(result["candidates"]),
        "fp32_predicates_reproduced": sum(
            row["fp32_full_pass_to_fail_and_bidirectional_predicate"] for row in result["candidates"].values()
        ),
    }
    result["interpretation_boundary"] = (
        "This is an FP32 hook sensitivity diagnostic on BF16-selected final subsets, not an independent FP32 search "
        "or structural-checkpoint equivalence test."
    )
    write_json(output_path, result)
    print(json.dumps(result["summary"], indent=2))
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
