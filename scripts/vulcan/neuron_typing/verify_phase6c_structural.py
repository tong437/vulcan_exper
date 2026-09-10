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

"""Physically materialize representative Phase 6C union successes and failures."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
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
from run_phase6b_single_sample_limits import prepare_dataset  # noqa: E402
from verify_phase5_structural_equivalence import (  # noqa: E402
    actual_projection_hashes,
    collect_valid_logits,
    expected_projection_hashes,
    load_model_bundle,
    model_layer_dims,
)
from verify_phase6b_winners_structural import release, sample_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structurally verify selected Phase 6C unions.")
    parser.add_argument("--phase6c_dir", required=True)
    parser.add_argument("--analysis_file", required=True)
    parser.add_argument("--sample_file", default="data/phase6b_single_samples/frozen_samples.json")
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/phase5e_coco.yaml")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--variant_ids", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2065)
    parser.add_argument("--infer_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--skip_recorded_match",
        action="store_true",
        help="Check hook-to-structural equivalence only (useful for a higher-precision sensitivity run).",
    )
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
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase6c_dir = Path(args.phase6c_dir).resolve()
    run = read_json(phase6c_dir / "phase6c_results.json")
    analysis = read_json(Path(args.analysis_file).resolve())
    samples_payload = read_json(Path(args.sample_file).resolve())
    samples = samples_payload["samples"]
    sample_indices = {sample["sample_id"]: index for index, sample in enumerate(samples)}
    sample_map = {sample["sample_id"]: sample for sample in samples}
    variants = {variant["variant_id"]: variant for variant in run["variants"]}
    recorded = {(row["variant_id"], row["target_sample_id"]): row for row in load_jsonl(run["evaluations_file"])}
    selected_ids = (
        [item.strip() for item in args.variant_ids.split(",") if item.strip()]
        if args.variant_ids
        else [row["variant_id"] for row in analysis["structural_candidates"]]
    )
    unknown = set(selected_ids) - set(variants)
    if unknown:
        raise ValueError(f"Unknown Phase 6C variants: {sorted(unknown)}.")

    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_dir = output_path.parent / "structural_inputs"
    input_dir.mkdir(exist_ok=True)
    dataset_dir = prepare_dataset(output_path.parent, samples, samples_payload["canonical_prompt"])
    config_path = Path(args.config).resolve()
    model_path = Path(args.model_name_or_path or load_yaml(config_path)["model_name_or_path"]).resolve()
    result = {
        "complete": False,
        "passed": False,
        "interpretation": (
            "Structural equivalence requires exact dimensions, projection hashes, parameter reduction, and generated "
            "tokens. A selected negative-behavior union passes verification when its hook failure is reproduced exactly."
        ),
        "config": {
            "phase6c_dir": str(phase6c_dir),
            "analysis_file": str(Path(args.analysis_file).resolve()),
            "sample_file": str(Path(args.sample_file).resolve()),
            "config_path": str(config_path),
            "model_name_or_path": str(model_path),
            "variant_ids": selected_ids,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "infer_dtype": args.infer_dtype,
            "skip_recorded_match": args.skip_recorded_match,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "variants": {},
    }
    write_json(output_path, result)

    for variant_id in selected_ids:
        print(f"Phase 6C structural verification: {variant_id}", flush=True)
        variant = variants[variant_id]
        targets = variant["targets"]
        mask_path = phase6c_dir / "structural_masks" / f"{variant_id}.json"
        neuron_ids = read_json(mask_path)
        layer_dims = {
            int(layer): len(neurons) + variant["mask_summary"]["kept_by_layer"][str(layer)]
            for layer, neurons in neuron_ids.items()
        }
        masks = neuron_ids_to_masks(neuron_ids, layer_dims)
        cluster_idx = build_singleton_cluster_idx(masks)
        validation = validate_singleton_cluster_idx(cluster_idx, masks)
        target_dims = {str(layer): int((~mask).sum()) for layer, mask in masks.items()}
        mask_sha256 = canonical_json_sha256(neuron_ids)
        cluster_sha256 = canonical_json_sha256(cluster_idx)
        variant_input_dir = input_dir / variant_id
        variant_input_dir.mkdir(exist_ok=True)
        cluster_path = variant_input_dir / "cluster_idx.json"
        metadata_path = variant_input_dir / "metadata.json"
        write_json(cluster_path, cluster_idx)
        provenance = {
            "phase": "6C-structural-union",
            "variant_id": variant_id,
            "targets": targets,
            "mask_sha256": mask_sha256,
            "cluster_idx_sha256": cluster_sha256,
            "kept_sha256": variant["kept_sha256"],
            "mask_summary": variant["mask_summary"],
            "target_layer_dims": target_dims,
            "validation": validation,
        }
        write_json(metadata_path, provenance)

        original_model, tokenizer_module, template, config = load_model_bundle(
            config_path,
            model_path,
            device,
            trust_remote_code=False,
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
        expected_hashes = expected_projection_hashes(original_model, cluster_idx)
        original_parameters = count_parameters(original_model)
        hook_rows = {}
        references = {}
        for target in targets:
            batch, _ = sample_batch(config, original_model, tokenizer_module, template, sample_indices[target], device)
            prompt, _ = build_prompt_inputs(batch)
            references[target] = build_reference_trace(original_model, batch)
            evaluator = build_semantic_evaluator(sample_map[target]["contract"])
            with MLPNeuronAblator(original_model, masks):
                logits = collect_valid_logits(original_model, batch)
                proxy = compact_proxy(evaluate_gold_proxy(original_model, batch, references[target]))
                generation = generate_short_caption(
                    original_model,
                    tokenizer_module["tokenizer"],
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    semantic_evaluator=evaluator,
                )
            recorded_row = recorded[(variant_id, target)]
            hook_rows[target] = {
                "logits": logits,
                "proxy": proxy,
                "generation": generation,
                "matches_recorded_tokens": generation["token_ids"] == recorded_row["generation"]["token_ids"],
                "matches_recorded_semantic": generation["semantic"]["automatic_pass"]
                == recorded_row["automatic_semantic_pass"],
            }
        del original_model, tokenizer_module, template, config, batch, prompt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with tempfile.TemporaryDirectory(prefix=f"phase6c_{variant_id}_") as temporary:
            pruned_dir = Path(temporary) / "model"
            command = [
                sys.executable,
                str(ROOT_DIR / "scripts" / "vulcan" / "save_pruned_model.py"),
                "--model_name_or_path",
                str(model_path),
                "--cluster_idx_path",
                str(cluster_path),
                "--output_dir",
                str(pruned_dir),
                "--infer_dtype",
                args.infer_dtype,
                "--config",
                str(config_path),
                "--provenance_path",
                str(metadata_path),
                "--expected_mask_sha256",
                mask_sha256,
            ]
            saved = subprocess.run(command, cwd=ROOT_DIR, check=True, capture_output=True, text=True)
            pruning_summary = read_json(pruned_dir / "pruning_summary.json")
            model, tokenizer_module, template, config = load_model_bundle(
                config_path,
                pruned_dir,
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
            structural_dims = model_layer_dims(model)
            structural_hashes = actual_projection_hashes(model)
            structural_parameters = count_parameters(model)
            global_checks = {
                "target_dims_match": structural_dims
                == {int(layer): int(width) for layer, width in target_dims.items()},
                "projection_weights_match": structural_hashes == expected_hashes,
                "parameter_reduction_matches": original_parameters["total"] - structural_parameters["total"]
                == variant["mask_summary"]["deleted_neurons"] * 3 * 1024,
            }
            target_rows = {}
            for target in targets:
                batch, _ = sample_batch(config, model, tokenizer_module, template, sample_indices[target], device)
                prompt, _ = build_prompt_inputs(batch)
                logits = collect_valid_logits(model, batch)
                proxy = compact_proxy(evaluate_gold_proxy(model, batch, references[target]))
                generation = generate_short_caption(
                    model,
                    tokenizer_module["tokenizer"],
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    semantic_evaluator=build_semantic_evaluator(sample_map[target]["contract"]),
                )
                hook = hook_rows[target]
                checks = {
                    "hook_matches_recorded_tokens": hook["matches_recorded_tokens"],
                    "hook_matches_recorded_semantic": hook["matches_recorded_semantic"],
                    "reloaded_tokens_match_hook": generation["token_ids"] == hook["generation"]["token_ids"],
                    "reloaded_semantic_matches_hook": generation["semantic"]["automatic_pass"]
                    == hook["generation"]["semantic"]["automatic_pass"],
                }
                required_checks = (
                    ("reloaded_tokens_match_hook", "reloaded_semantic_matches_hook")
                    if args.skip_recorded_match
                    else tuple(checks)
                )
                target_rows[target] = {
                    "passed": all(checks[name] for name in required_checks),
                    "required_checks": list(required_checks),
                    "checks": checks,
                    "hook": {"gold_proxy": hook["proxy"], "generation": hook["generation"]},
                    "reloaded": {"gold_proxy": proxy, "generation": generation},
                    "hook_vs_reloaded_logits": compare_logits(hook["logits"], logits),
                }
            row = {
                "passed": all(global_checks.values()) and all(value["passed"] for value in target_rows.values()),
                "global_checks": global_checks,
                "mask_sha256": mask_sha256,
                "cluster_idx_sha256": cluster_sha256,
                "target_layer_dims": target_dims,
                "original_parameters": original_parameters,
                "reloaded_parameters": structural_parameters,
                "pruning_summary": pruning_summary,
                "save_stdout": saved.stdout.strip(),
                "targets": target_rows,
                "temporary_checkpoint_retained": False,
            }
            result["variants"][variant_id] = row
            write_json(output_path, result)
            del model, tokenizer_module, template, config, batch, prompt
            release()

    result["complete"] = set(result["variants"]) == set(selected_ids)
    result["passed"] = result["complete"] and all(row["passed"] for row in result["variants"].values())
    write_json(output_path, result)
    print(json.dumps({"complete": result["complete"], "passed": result["passed"]}, indent=2))
    if not result["passed"]:
        raise RuntimeError("At least one Phase 6C structural union failed equivalence verification.")


if __name__ == "__main__":
    main()
