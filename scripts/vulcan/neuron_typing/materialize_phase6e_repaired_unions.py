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

"""Materialize successful Phase 6E repaired unions as permanent checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase3_structural_utils import (  # noqa: E402
    build_singleton_cluster_idx,
    canonical_json_sha256,
    count_parameters,
    neuron_ids_to_masks,
    validate_singleton_cluster_idx,
)
from phase5_structural_utils import compare_logits  # noqa: E402
from phase5e_proxy import evaluate_gold_proxy  # noqa: E402
from phase5e_semantics import generate_short_caption  # noqa: E402
from phase6b_semantics import build_semantic_evaluator  # noqa: E402
from run_phase2_ablation import MLPNeuronAblator, load_yaml  # noqa: E402
from run_phase5_single_sample_frontier import build_prompt_inputs, write_json  # noqa: E402
from run_phase5e_retrospective import build_reference_trace  # noqa: E402
from run_phase6b_single_sample_limits import prepare_dataset, sha256_file  # noqa: E402
from verify_phase5_structural_equivalence import (  # noqa: E402
    actual_projection_hashes,
    collect_valid_logits,
    expected_projection_hashes,
    load_model_bundle,
    model_layer_dims,
)
from verify_phase6b_winners_structural import release, sample_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize successful Phase 6E repaired unions.")
    parser.add_argument("--phase6e_dir", required=True)
    parser.add_argument("--analysis_file", required=True)
    parser.add_argument("--sample_file", default="data/phase6b_single_samples/frozen_samples.json")
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/phase5e_coco.yaml")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2069)
    parser.add_argument("--infer_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--reuse_existing_models", action="store_true")
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def compact_proxy(proxy: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in proxy.items() if not key.startswith("per_token_")}


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be positive.")
    phase6e_dir = Path(args.phase6e_dir).resolve()
    analysis_path = Path(args.analysis_file).resolve()
    output_path = Path(args.output_file).resolve()
    model_output_dir = Path(args.model_output_dir).resolve()
    if output_path.exists() or (model_output_dir.exists() and not args.reuse_existing_models):
        raise FileExistsError("Phase 6E structural output already exists; refusing to overwrite permanent models.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=args.reuse_existing_models)

    run = read_json(phase6e_dir / "phase6e_repaired_union.json")
    analysis = read_json(analysis_path)
    if not run.get("complete") or analysis["summary"]["primary_successes"] != len(analysis["edges"]):
        raise ValueError("Structural materialization requires all BF16 Phase 6E primary criteria to pass.")
    variants = {
        row["variant_id"]: row for row in run["variants"] if row["condition"] == "repaired_union"
    }
    successful_edges = {edge["edge_id"] for edge in analysis["edges"] if edge["primary_success"]}
    selected = {edge: variants[f"{edge}__repaired_union"] for edge in successful_edges}
    recorded = {
        (row["variant_id"], row["target_sample_id"]): row
        for row in load_jsonl(run["evaluations_file"])
        if row["condition"] == "repaired_union"
    }

    samples_payload = read_json(Path(args.sample_file).resolve())
    samples = samples_payload["samples"]
    sample_indices = {sample["sample_id"]: index for index, sample in enumerate(samples)}
    sample_map = {sample["sample_id"]: sample for sample in samples}
    dataset_dir = prepare_dataset(output_path.parent, samples, samples_payload["canonical_prompt"])
    config_path = Path(args.config).resolve()
    model_path = Path(args.model_name_or_path or load_yaml(config_path)["model_name_or_path"]).resolve()
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {
        "complete": False,
        "passed": False,
        "config": {
            "phase6e_dir": str(phase6e_dir),
            "phase6e_result_sha256": sha256_file(phase6e_dir / "phase6e_repaired_union.json"),
            "analysis_file": str(analysis_path),
            "analysis_sha256": sha256_file(analysis_path),
            "sample_file": str(Path(args.sample_file).resolve()),
            "config_path": str(config_path),
            "model_name_or_path": str(model_path),
            "model_output_dir": str(model_output_dir),
            "infer_dtype": args.infer_dtype,
            "reuse_existing_models": args.reuse_existing_models,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "edges": {},
        "interpretation": (
            "A structural checkpoint passes when dimensions, projection hashes, and parameter reduction match and "
            "both reloaded targets pass. Exact hook-to-reload tokens are reported separately as a BF16 stability check."
        ),
    }
    write_json(output_path, result)

    original_model, tokenizer_module, template, config = load_model_bundle(
        config_path,
        model_path,
        device,
        trust_remote_code=False,
        preprocessing_num_workers=1,
        infer_dtype=args.infer_dtype,
    )
    layer_dims = model_layer_dims(original_model)
    original_parameters = count_parameters(original_model)
    config.update(
        {
            "dataset_dir": str(dataset_dir),
            "dataset": "phase6b_single_samples",
            "eval_dataset": None,
            "tokenized_path": None,
            "max_samples": len(samples),
            "overwrite_cache": True,
            "enable_thinking": False,
        }
    )

    prepared = {}
    input_dir = output_path.parent / "structural_inputs"
    input_dir.mkdir(exist_ok=True)
    for edge_id, variant in sorted(selected.items()):
        deleted_ids = read_json(phase6e_dir / "structural_masks" / f"{variant['variant_id']}.json")
        masks = neuron_ids_to_masks(deleted_ids, layer_dims)
        cluster_idx = build_singleton_cluster_idx(masks)
        validation = validate_singleton_cluster_idx(cluster_idx, masks)
        target_dims = {str(layer): int((~mask).sum()) for layer, mask in masks.items()}
        mask_sha256 = canonical_json_sha256(deleted_ids)
        edge_input_dir = input_dir / edge_id
        edge_input_dir.mkdir(exist_ok=True)
        cluster_path = edge_input_dir / "cluster_idx.json"
        metadata_path = edge_input_dir / "metadata.json"
        write_json(cluster_path, cluster_idx)
        write_json(
            metadata_path,
            {
                "phase": "6E-repaired-union-structural",
                "edge_id": edge_id,
                "variant_id": variant["variant_id"],
                "targets": variant["targets"],
                "mask_sha256": mask_sha256,
                "cluster_idx_sha256": canonical_json_sha256(cluster_idx),
                "kept_sha256": variant["kept_sha256"],
                "mask_summary": variant["mask_summary"],
                "target_layer_dims": target_dims,
                "validation": validation,
            },
        )
        hook_targets = {}
        for target in variant["targets"]:
            batch, _ = sample_batch(config, original_model, tokenizer_module, template, sample_indices[target], device)
            prompt, _ = build_prompt_inputs(batch)
            reference = build_reference_trace(original_model, batch)
            with MLPNeuronAblator(original_model, masks):
                logits = collect_valid_logits(original_model, batch)
                proxy = compact_proxy(evaluate_gold_proxy(original_model, batch, reference))
                generation = generate_short_caption(
                    original_model,
                    tokenizer_module["tokenizer"],
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    semantic_evaluator=build_semantic_evaluator(sample_map[target]["contract"]),
                )
            recorded_row = recorded[(variant["variant_id"], target)]
            hook_targets[target] = {
                "logits": logits,
                "reference": reference,
                "gold_proxy": proxy,
                "generation": generation,
                "matches_recorded_tokens": generation["token_ids"] == recorded_row["generation"]["token_ids"],
                "matches_recorded_semantic": generation["semantic"]["automatic_pass"]
                == recorded_row["automatic_semantic_pass"],
            }
        prepared[edge_id] = {
            "variant": variant,
            "masks": masks,
            "target_dims": target_dims,
            "mask_sha256": mask_sha256,
            "cluster_path": cluster_path,
            "metadata_path": metadata_path,
            "expected_hashes": expected_projection_hashes(original_model, cluster_idx),
            "hook_targets": hook_targets,
        }

    for edge_id, item in prepared.items():
        checkpoint_dir = model_output_dir / edge_id
        if args.reuse_existing_models:
            if not (checkpoint_dir / "pruning_summary.json").is_file():
                raise FileNotFoundError(f"Cannot reuse incomplete structural checkpoint: {checkpoint_dir}.")
            item["checkpoint_dir"] = checkpoint_dir
            item["save_stdout"] = "Reused existing frozen structural checkpoint."
            continue
        command = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "vulcan" / "save_pruned_model.py"),
            "--model_name_or_path",
            str(model_path),
            "--cluster_idx_path",
            str(item["cluster_path"]),
            "--output_dir",
            str(checkpoint_dir),
            "--infer_dtype",
            args.infer_dtype,
            "--config",
            str(config_path),
            "--provenance_path",
            str(item["metadata_path"]),
            "--expected_mask_sha256",
            item["mask_sha256"],
        ]
        saved = subprocess.run(command, cwd=ROOT_DIR, check=True, capture_output=True, text=True)
        item["checkpoint_dir"] = checkpoint_dir
        item["save_stdout"] = saved.stdout.strip()

    del original_model, tokenizer_module, template, config, batch, prompt
    release()

    for edge_id, item in sorted(prepared.items()):
        checkpoint_dir = item["checkpoint_dir"]
        model, tokenizer_module, template, config = load_model_bundle(
            config_path,
            checkpoint_dir,
            device,
            trust_remote_code=True,
            preprocessing_num_workers=1,
            infer_dtype=args.infer_dtype,
        )
        config.update(
            {
                "dataset_dir": str(dataset_dir),
                "dataset": "phase6b_single_samples",
                "eval_dataset": None,
                "tokenized_path": None,
                "max_samples": len(samples),
                "overwrite_cache": True,
                "enable_thinking": False,
            }
        )
        structural_parameters = count_parameters(model)
        global_checks = {
            "target_dims_match": model_layer_dims(model)
            == {int(layer): int(width) for layer, width in item["target_dims"].items()},
            "projection_weights_match": actual_projection_hashes(model) == item["expected_hashes"],
            "parameter_reduction_matches": original_parameters["total"] - structural_parameters["total"]
            == item["variant"]["mask_summary"]["deleted_neurons"] * 3 * 1024,
        }
        target_rows = {}
        for target in item["variant"]["targets"]:
            batch, _ = sample_batch(config, model, tokenizer_module, template, sample_indices[target], device)
            prompt, _ = build_prompt_inputs(batch)
            logits = collect_valid_logits(model, batch)
            hook = item["hook_targets"][target]
            proxy = compact_proxy(evaluate_gold_proxy(model, batch, hook["reference"]))
            generation = generate_short_caption(
                model,
                tokenizer_module["tokenizer"],
                prompt,
                max_new_tokens=args.max_new_tokens,
                semantic_evaluator=build_semantic_evaluator(sample_map[target]["contract"]),
            )
            checks = {
                "hook_matches_recorded_tokens": hook["matches_recorded_tokens"],
                "hook_matches_recorded_semantic": hook["matches_recorded_semantic"],
                "reloaded_tokens_match_hook": generation["token_ids"] == hook["generation"]["token_ids"],
                "reloaded_semantic_matches_hook": generation["semantic"]["automatic_pass"]
                == hook["generation"]["semantic"]["automatic_pass"],
                "reloaded_semantic_pass": generation["semantic"]["automatic_pass"],
            }
            target_rows[target] = {
                "functional_pass": checks["reloaded_semantic_pass"],
                "exact_token_equivalence": checks["reloaded_tokens_match_hook"],
                "checks": checks,
                "hook": {"gold_proxy": hook["gold_proxy"], "generation": hook["generation"]},
                "reloaded": {"gold_proxy": proxy, "generation": generation},
                "hook_vs_reloaded_logits": compare_logits(hook["logits"], logits),
            }
        functional_pass = all(global_checks.values()) and all(row["functional_pass"] for row in target_rows.values())
        exact_equivalence = all(global_checks.values()) and all(
            row["exact_token_equivalence"] for row in target_rows.values()
        )
        result["edges"][edge_id] = {
            "passed": functional_pass,
            "functional_dual_pass": all(row["functional_pass"] for row in target_rows.values()),
            "exact_token_equivalence": exact_equivalence,
            "global_checks": global_checks,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_retained": True,
            "target_layer_dims": item["target_dims"],
            "mask_sha256": item["mask_sha256"],
            "original_parameters": original_parameters,
            "reloaded_parameters": structural_parameters,
            "pruning_summary": read_json(checkpoint_dir / "pruning_summary.json"),
            "save_stdout": item["save_stdout"],
            "targets": target_rows,
        }
        write_json(output_path, result)
        del model, tokenizer_module, template, config, batch, prompt
        gc.collect()
        release()

    result["complete"] = set(result["edges"]) == successful_edges
    result["passed"] = result["complete"] and all(row["passed"] for row in result["edges"].values())
    write_json(output_path, result)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "passed": result["passed"],
                "exact_token_equivalence": {
                    edge: row["exact_token_equivalence"] for edge, row in result["edges"].items()
                },
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise RuntimeError("At least one materialized Phase 6E repaired union failed dual-task validation.")


if __name__ == "__main__":
    main()
