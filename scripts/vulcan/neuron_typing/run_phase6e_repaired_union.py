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

"""Evaluate frozen Phase 6E repaired unions and matched random repairs."""

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
from phase6c_subnets import deletion_masks, kept_from_deleted  # noqa: E402
from phase6e_repairs import REPAIR_SPECS, assemble_conflicts, build_pair_variants  # noqa: E402
from run_phase2_ablation import MLPNeuronAblator, build_dataloader, load_yaml, move_batch_to_device  # noqa: E402
from run_phase5_single_sample_frontier import build_prompt_inputs, write_json  # noqa: E402
from run_phase5e_retrospective import build_reference_trace  # noqa: E402
from run_phase6b_single_sample_limits import prepare_dataset, sha256_file, validate_inputs  # noqa: E402
from run_phase6d_conflict_localization import append_jsonl, compact_proxy, load_jsonl  # noqa: E402
from verify_phase5_structural_equivalence import load_model_bundle, model_layer_dims  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6E repaired-union validation.")
    parser.add_argument(
        "--identity_analysis",
        default="saves/neuron_typing/phase6d_identity_refinement_v1/analysis.json",
    )
    parser.add_argument("--winner_dir", default="saves/neuron_typing/phase6b_frozen_winners_v1")
    parser.add_argument("--sample_file", default="data/phase6b_single_samples/frozen_samples.json")
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/phase5e_coco.yaml")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--random_seeds", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2068)
    parser.add_argument("--infer_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_frozen_candidates(analysis: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    expected_ids = {candidate.candidate_id for spec in REPAIR_SPECS for candidate in spec.candidates}
    rows = {row["candidate_id"]: row for row in analysis["candidates"] if row["candidate_id"] in expected_ids}
    if set(rows) != expected_ids:
        raise ValueError(f"Phase 6D analysis is missing candidates: {sorted(expected_ids - set(rows))}.")
    payloads = {}
    hashes = {}
    for candidate_id, row in rows.items():
        path = Path(row["final_identity_file"]).resolve()
        payloads[candidate_id] = json.loads(path.read_text(encoding="utf-8"))
        hashes[candidate_id] = sha256_file(path)
    return payloads, hashes


def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.random_seeds, args.max_new_tokens) < 1:
        raise ValueError("--random_seeds and --max_new_tokens must be positive.")
    analysis_path = Path(args.identity_analysis).resolve()
    winner_dir = Path(args.winner_dir).resolve()
    sample_path = Path(args.sample_file).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "phase6e_repaired_union.json"
    evaluations_path = output_dir / "evaluations.jsonl"
    if output_dir.exists() and not args.resume and (result_path.exists() or evaluations_path.exists()):
        raise FileExistsError(f"Phase 6E output already exists: {output_dir}. Pass --resume to continue.")
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if not analysis.get("complete") or not analysis["integrity"]["all_final_subsets_satisfy_bidirectional_predicate"]:
        raise ValueError("Phase 6E requires a complete and validated Phase 6D identity analysis.")
    candidate_payloads, candidate_file_hashes = load_frozen_candidates(analysis)
    samples_payload = json.loads(sample_path.read_text(encoding="utf-8"))
    samples = validate_inputs(samples_payload, None)
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    target_ids = list(dict.fromkeys(target for spec in REPAIR_SPECS for target in spec.targets))
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
    conflicts = {
        spec.edge_id: assemble_conflicts(spec, candidate_payloads, layer_dims) for spec in REPAIR_SPECS
    }
    variants = [
        variant
        for spec in REPAIR_SPECS
        for variant in build_pair_variants(
            spec,
            winners,
            conflicts[spec.edge_id],
            layer_dims,
            random_seeds=args.random_seeds,
            seed=args.seed,
        )
    ]

    model_path = Path(args.model_name_or_path or load_yaml(config_path)["model_name_or_path"]).resolve()
    run_config = {
        "identity_analysis": str(analysis_path),
        "identity_analysis_sha256": sha256_file(analysis_path),
        "candidate_file_sha256": candidate_file_hashes,
        "winner_dir": str(winner_dir),
        "winner_manifest_sha256": sha256_file(winner_dir / "frozen_winners.json"),
        "sample_file": str(sample_path),
        "sample_file_sha256": sha256_file(sample_path),
        "config_path": str(config_path),
        "model_name_or_path": str(model_path),
        "random_seeds": args.random_seeds,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "infer_dtype": args.infer_dtype,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "primary_endpoint": "both_pair_targets_pass_strict_deterministic_free_generation_contract",
        "random_control_pool": "same donor-minus-safe-base increment, same layer and count",
    }
    if args.resume and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing["config"] != run_config:
            raise ValueError("Cannot resume Phase 6E with a changed configuration.")

    manifest = [{key: value for key, value in variant.items() if key != "kept"} for variant in variants]
    expected_evaluations = sum(len(variant["targets"]) for variant in variants)
    result = {
        "complete": False,
        "config": run_config,
        "repair_specs": [
            {
                "edge_id": spec.edge_id,
                "left": spec.left,
                "right": spec.right,
                "safe_base": spec.safe_base,
                "donor": spec.donor,
                "candidates": [candidate.__dict__ for candidate in spec.candidates],
            }
            for spec in REPAIR_SPECS
        ],
        "target_ids": target_ids,
        "expected_variants": len(variants),
        "expected_evaluations": expected_evaluations,
        "evaluations_file": str(evaluations_path),
        "variants": manifest,
        "summary": None,
    }
    write_json(result_path, result)

    structural_dir = output_dir / "structural_masks"
    structural_dir.mkdir(exist_ok=True)
    for variant in variants:
        if variant["condition"] == "repaired_union":
            deleted = {
                str(layer): sorted(set(range(width)) - variant["kept"][layer])
                for layer, width in layer_dims.items()
            }
            write_json(structural_dir / f"{variant['variant_id']}.json", deleted)

    completed = load_jsonl(evaluations_path)
    expected_ids = {
        f"{variant['variant_id']}__target_{target}" for variant in variants for target in variant["targets"]
    }
    if set(completed) - expected_ids:
        raise ValueError(f"Existing rows are not part of this run: {sorted(set(completed) - expected_ids)[:5]}.")

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
        infer_dtype=args.infer_dtype,
    )
    if model_layer_dims(model) != layer_dims:
        raise ValueError("Original model layer widths differ from the frozen winners.")
    if args.infer_dtype == "float32" and next(model.parameters()).dtype != torch.float32:
        raise ValueError("Requested FP32 but the loaded model is not float32.")
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

    evaluation_number = 0
    for variant in variants:
        for target in variant["targets"]:
            evaluation_number += 1
            evaluation_id = f"{variant['variant_id']}__target_{target}"
            if evaluation_id in completed:
                continue
            print(f"Phase 6E [{evaluation_number}/{expected_evaluations}] {evaluation_id}", flush=True)
            context = contexts[target]
            with MLPNeuronAblator(model, deletion_masks(variant["kept"], layer_dims)):
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
                "family": "phase6e_repaired_union",
                "variant_id": variant["variant_id"],
                "edge_id": variant["edge_id"],
                "condition": variant["condition"],
                "target_sample_id": target,
                "kept_sha256": variant["kept_sha256"],
                "mask_summary": variant["mask_summary"],
                "variant_metadata": variant["metadata"],
                "gold_proxy": proxy,
                "generation": generation,
                "automatic_semantic_pass": generation["semantic"]["automatic_pass"],
            }
            append_jsonl(evaluations_path, row)
            completed[evaluation_id] = row

    rows = list(completed.values())
    by_condition = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "evaluations": len(selected),
            "semantic_passes": sum(bool(row["automatic_semantic_pass"]) for row in selected),
        }
    result["summary"] = {"evaluations": len(rows), "by_condition": by_condition}
    result["complete"] = len(rows) == expected_evaluations and set(completed) == expected_ids
    write_json(result_path, result)
    print(json.dumps({"complete": result["complete"], **result["summary"]}, indent=2))
    if not result["complete"]:
        raise RuntimeError(f"Phase 6E produced {len(rows)} of {expected_evaluations} expected evaluations.")
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
