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

"""Materialize, reload, and verify every frozen Phase 6B winner mask."""

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

from phase3_structural_utils import canonical_json_sha256, count_parameters, neuron_ids_to_masks  # noqa: E402
from phase5_structural_utils import compare_logits  # noqa: E402
from phase5e_proxy import evaluate_gold_proxy  # noqa: E402
from phase5e_semantics import generate_short_caption  # noqa: E402
from phase6b_semantics import build_semantic_evaluator  # noqa: E402
from run_phase2_ablation import MLPNeuronAblator, build_dataloader, move_batch_to_device  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Physically verify the frozen Phase 6B winner masks.")
    parser.add_argument("--winner_dir", required=True)
    parser.add_argument("--sample_file", default="data/phase6b_single_samples/frozen_samples.json")
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/phase5e_coco.yaml")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--sample_ids", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2063)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def release(*values: Any) -> None:
    del values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_batch(config, model, tokenizer_module, template, sample_index: int, device: torch.device):
    dataloader, manifest = build_dataloader(
        config,
        model,
        tokenizer_module,
        template,
        batch_size=1,
        num_workers=0,
        sample_offset=sample_index,
        max_samples=1,
        allow_short_dataset=False,
        max_image_repeat=1,
        allow_excessive_image_repeats=False,
        dataset_stage="sft",
    )
    return move_batch_to_device(next(iter(dataloader)), device), manifest


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be positive.")
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    winner_dir = Path(args.winner_dir).resolve()
    winners = load_json(winner_dir / "frozen_winners.json")
    sample_payload = load_json(Path(args.sample_file).resolve())
    samples = sample_payload["samples"]
    sample_indices = {sample["sample_id"]: index for index, sample in enumerate(samples)}
    selected_ids = set(sample_indices if args.sample_ids is None else args.sample_ids.split(","))
    unknown = selected_ids - set(sample_indices)
    if unknown:
        raise ValueError(f"Unknown sample IDs: {sorted(unknown)}.")

    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir = prepare_dataset(output_path.parent, samples, sample_payload["canonical_prompt"])
    config_path = Path(args.config).resolve()
    config_payload = load_json(winner_dir / "frozen_winners.json")
    del config_payload
    from run_phase2_ablation import load_yaml

    model_path = Path(args.model_name_or_path or load_yaml(config_path)["model_name_or_path"]).resolve()
    result = {
        "complete": False,
        "passed": False,
        "config": {
            "winner_dir": str(winner_dir),
            "sample_file": str(Path(args.sample_file).resolve()),
            "config_path": str(config_path),
            "model_name_or_path": str(model_path),
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "samples": {},
    }
    write_json(output_path, result)

    winner_map = {row["sample_id"]: row for row in winners["winners"]}
    for sample_id in sample_indices:
        if sample_id not in selected_ids:
            continue
        print(f"Phase 6B structural verification: {sample_id}", flush=True)
        metadata_path = winner_dir / sample_id / "metadata.json"
        cluster_path = winner_dir / sample_id / "cluster_idx.json"
        mask_path = winner_dir / sample_id / "mask.json"
        metadata = load_json(metadata_path)
        neuron_ids = load_json(mask_path)
        cluster_idx = load_json(cluster_path)
        if metadata != winner_map[sample_id]:
            raise ValueError(f"Winner manifest/metadata mismatch for {sample_id}.")
        if canonical_json_sha256(neuron_ids) != metadata["mask_sha256"]:
            raise ValueError(f"Winner mask hash mismatch for {sample_id}.")
        if canonical_json_sha256(cluster_idx) != metadata["cluster_idx_sha256"]:
            raise ValueError(f"Winner cluster hash mismatch for {sample_id}.")

        original_model, tokenizer_module, template, config = load_model_bundle(
            config_path,
            model_path,
            device,
            trust_remote_code=False,
            preprocessing_num_workers=1,
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
        batch, sample_manifest = sample_batch(
            config, original_model, tokenizer_module, template, sample_indices[sample_id], device
        )
        prompt_inputs, _ = build_prompt_inputs(batch)
        reference_trace = build_reference_trace(original_model, batch)
        original_dims = model_layer_dims(original_model)
        masks = neuron_ids_to_masks(neuron_ids, original_dims)
        expected_hashes = expected_projection_hashes(original_model, cluster_idx)
        original_parameters = count_parameters(original_model)
        evaluator = build_semantic_evaluator(metadata["contract"])
        with MLPNeuronAblator(original_model, masks):
            hook_logits = collect_valid_logits(original_model, batch)
            hook_proxy = evaluate_gold_proxy(original_model, batch, reference_trace)
            hook_generation = generate_short_caption(
                original_model,
                tokenizer_module["tokenizer"],
                prompt_inputs,
                max_new_tokens=args.max_new_tokens,
                semantic_evaluator=evaluator,
            )
        del batch, prompt_inputs, original_model, tokenizer_module, template, config
        release()

        with tempfile.TemporaryDirectory(prefix=f"phase6b_{sample_id}_") as temporary:
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
                "--config",
                str(config_path),
                "--provenance_path",
                str(metadata_path),
                "--expected_mask_sha256",
                metadata["mask_sha256"],
            ]
            saved = subprocess.run(command, cwd=ROOT_DIR, check=True, capture_output=True, text=True)
            pruning_summary = load_json(pruned_dir / "pruning_summary.json")
            structural_model, structural_tokenizer_module, structural_template, structural_config = load_model_bundle(
                config_path,
                pruned_dir,
                device,
                trust_remote_code=True,
                preprocessing_num_workers=1,
            )
            structural_config.update(
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
            structural_batch, _ = sample_batch(
                structural_config,
                structural_model,
                structural_tokenizer_module,
                structural_template,
                sample_indices[sample_id],
                device,
            )
            structural_prompt, _ = build_prompt_inputs(structural_batch)
            structural_logits = collect_valid_logits(structural_model, structural_batch)
            structural_proxy = evaluate_gold_proxy(structural_model, structural_batch, reference_trace)
            structural_generation = generate_short_caption(
                structural_model,
                structural_tokenizer_module["tokenizer"],
                structural_prompt,
                max_new_tokens=args.max_new_tokens,
                semantic_evaluator=evaluator,
            )
            structural_dims = model_layer_dims(structural_model)
            structural_hashes = actual_projection_hashes(structural_model)
            structural_parameters = count_parameters(structural_model)
            logit_difference = compare_logits(hook_logits, structural_logits)
            checks = {
                "target_dims_match": structural_dims
                == {int(layer): int(width) for layer, width in metadata["target_layer_dims"].items()},
                "projection_weights_match": structural_hashes == expected_hashes,
                "parameter_reduction_matches": original_parameters["total"] - structural_parameters["total"]
                == metadata["source_parameter_summary"]["removed_parameters"],
                "hook_semantic_pass": hook_generation["semantic"]["automatic_pass"],
                "reloaded_semantic_pass": structural_generation["semantic"]["automatic_pass"],
                "generation_matches_hook": structural_generation["token_ids"] == hook_generation["token_ids"],
            }
            row = {
                "passed": all(checks.values()),
                "checks": checks,
                "sample_manifest": sample_manifest,
                "mask_sha256": metadata["mask_sha256"],
                "deletion_budget": metadata["deletion_budget"],
                "target_layer_dims": metadata["target_layer_dims"],
                "original_parameters": original_parameters,
                "reloaded_parameters": structural_parameters,
                "pruning_summary": pruning_summary,
                "save_stdout": saved.stdout.strip(),
                "hook": {"gold_proxy": hook_proxy, "generation": hook_generation},
                "reloaded": {"gold_proxy": structural_proxy, "generation": structural_generation},
                "hook_vs_reloaded_logits": logit_difference,
                "temporary_checkpoint_retained": False,
            }
            result["samples"][sample_id] = row
            write_json(output_path, result)
            del structural_batch, structural_prompt, structural_model, structural_tokenizer_module
            del structural_template, structural_config
            release()

    result["complete"] = set(result["samples"]) == selected_ids
    result["passed"] = result["complete"] and all(row["passed"] for row in result["samples"].values())
    write_json(output_path, result)
    print(json.dumps({"complete": result["complete"], "passed": result["passed"]}, indent=2))
    if not result["passed"]:
        raise RuntimeError("At least one frozen Phase 6B winner failed physical reload verification.")


if __name__ == "__main__":
    main()
