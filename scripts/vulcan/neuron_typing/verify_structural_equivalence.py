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

"""Verify hook, in-memory structural, and reloaded structural model equivalence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_vqa import (  # noqa: E402
    load_binary_records,
    prepare_model_batch,
    select_binary_records,
)
from phase3_structural_utils import (  # noqa: E402
    canonical_json_sha256,
    count_parameters,
    neuron_ids_to_masks,
    seed_everything,
    sha256_file,
    validate_singleton_cluster_idx,
)
from run_phase2_ablation import MLPNeuronAblator  # noqa: E402

from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402
from llamafactory.train.vulcan.modeling import find_mlp_layers, get_intermediate_size  # noqa: E402
from llamafactory.train.vulcan.pruning import pruning_mlp  # noqa: E402
from llamafactory.train.vulcan.schema import load_cluster_idx  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase-3 structural pruning equivalence.")
    parser.add_argument("--config", required=True, help="Original model LlamaFactory config.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--pruned_model_path", required=True)
    parser.add_argument("--mask_file", required=True)
    parser.add_argument("--cluster_idx_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--pope_file", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--nll_tolerance",
        type=float,
        default=0.05,
        help="BF16 hook-vs-structural mean label-NLL tolerance.",
    )
    parser.add_argument(
        "--logit_max_abs_tolerance",
        type=float,
        default=1.0,
        help="Diagnostic BF16 full-vocabulary max-absolute-difference reference bound; not a correctness gate.",
    )
    parser.add_argument(
        "--candidate_logprob_tolerance",
        type=float,
        default=0.5,
        help="Diagnostic BF16 yes/no log-prob max-absolute-difference reference bound; not a correctness gate.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_model_bundle(config_path: str | Path, model_path: str | Path, device: torch.device):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config.update(
        {
            "model_name_or_path": str(model_path),
            "do_train": False,
            "do_eval": False,
            "do_predict": False,
        }
    )
    config.setdefault("output_dir", "saves/neuron_typing/phase3_equivalence_tmp")
    model_args, _, _, finetuning_args, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor")
    if processor is None:
        raise RuntimeError("The configured model did not provide a multimodal processor.")
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    model.to(device).eval()
    return model, tokenizer, processor


def model_layer_dims(model: torch.nn.Module) -> dict[int, int]:
    return {layer.index: get_intermediate_size(layer.mlp) for layer in find_mlp_layers(model)}


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().contiguous().view(torch.uint8).cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def expected_singleton_projection_hashes(
    model: torch.nn.Module, cluster_idx: list[list[dict[str, Any]] | None]
) -> dict[str, dict[str, str]]:
    layers = find_mlp_layers(model)
    if len(layers) != len(cluster_idx):
        raise ValueError("Model/cluster layer count mismatch during weight audit.")
    result = {}
    for layer_ref, clusters in zip(layers, cluster_idx):
        if clusters is None:
            raise ValueError("Frozen structural q-band cannot contain null layer clusters.")
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


def actual_projection_hashes(model: torch.nn.Module) -> dict[str, dict[str, str]]:
    return {
        str(layer_ref.index): {
            "gate_proj.weight": tensor_sha256(layer_ref.mlp.gate_proj.weight),
            "up_proj.weight": tensor_sha256(layer_ref.mlp.up_proj.weight),
            "down_proj.weight": tensor_sha256(layer_ref.mlp.down_proj.weight),
        }
        for layer_ref in find_mlp_layers(model)
    }


@torch.inference_mode()
def collect_snapshot(
    model,
    tokenizer,
    processor,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(f"Equivalence check requires single-token yes/no, got yes={yes_ids}, no={no_ids}.")
    yes_id, no_id = yes_ids[0], no_ids[0]
    logits_rows = []
    candidate_rows = []
    predictions = []
    label_nll = []
    source_indices = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        inputs = prepare_model_batch(processor, batch_records, device)
        outputs = model(**inputs, use_cache=False)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            last_positions = torch.full(
                (inputs["input_ids"].shape[0],),
                inputs["input_ids"].shape[1] - 1,
                device=device,
                dtype=torch.long,
            )
        else:
            last_positions = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(len(batch_records), device=device)
        last_logits = outputs.logits[batch_indices, last_positions].float()
        log_probs = F.log_softmax(last_logits, dim=-1)
        candidate_logprobs = log_probs[:, [yes_id, no_id]]
        batch_predictions = torch.where(
            candidate_logprobs[:, 0] > candidate_logprobs[:, 1],
            torch.ones(len(batch_records), device=device, dtype=torch.long),
            torch.zeros(len(batch_records), device=device, dtype=torch.long),
        )
        label_indices = torch.tensor(
            [0 if record["answer"] == "yes" else 1 for record in batch_records],
            device=device,
            dtype=torch.long,
        )
        logits_rows.append(last_logits.cpu())
        candidate_rows.append(candidate_logprobs.cpu())
        predictions.extend(batch_predictions.cpu().tolist())
        label_nll.extend((-candidate_logprobs[batch_indices, label_indices]).cpu().tolist())
        source_indices.extend(record["source_index"] for record in batch_records)
    return {
        "last_logits": torch.cat(logits_rows),
        "candidate_logprobs": torch.cat(candidate_rows),
        "predictions": predictions,
        "label_nll": label_nll,
        "source_indices": source_indices,
    }


def compare_snapshots(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference["source_indices"] != candidate["source_indices"]:
        raise ValueError("Snapshot source indices are not aligned.")
    logits_delta = candidate["last_logits"] - reference["last_logits"]
    candidate_delta = candidate["candidate_logprobs"] - reference["candidate_logprobs"]
    reference_nll = torch.tensor(reference["label_nll"], dtype=torch.float64)
    candidate_nll = torch.tensor(candidate["label_nll"], dtype=torch.float64)
    prediction_matches = [a == b for a, b in zip(reference["predictions"], candidate["predictions"])]
    return {
        "num_samples": len(reference["source_indices"]),
        "logit_max_abs": float(logits_delta.abs().max()),
        "logit_mean_abs": float(logits_delta.abs().mean()),
        "logit_rmse": float(logits_delta.square().mean().sqrt()),
        "candidate_logprob_max_abs": float(candidate_delta.abs().max()),
        "candidate_logprob_mean_abs": float(candidate_delta.abs().mean()),
        "reference_mean_label_nll": float(reference_nll.mean()),
        "candidate_mean_label_nll": float(candidate_nll.mean()),
        "mean_label_nll_delta": float(candidate_nll.mean() - reference_nll.mean()),
        "max_per_sample_label_nll_delta": float((candidate_nll - reference_nll).abs().max()),
        "prediction_matches": int(sum(prediction_matches)),
        "prediction_match_ratio": sum(prediction_matches) / len(prediction_matches),
    }


def gate_bf16_hook_comparison(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Gate low-precision hook/structural agreement without treating GEMM drift as a structural error."""
    checks = {
        "mean_label_nll_delta": abs(metrics["mean_label_nll_delta"]) <= args.nll_tolerance,
        "prediction_match": metrics["prediction_match_ratio"] == 1.0,
    }
    diagnostic_bounds = {
        "logit_max_abs": {
            "within_reference_bound": metrics["logit_max_abs"] <= args.logit_max_abs_tolerance,
            "reference_bound": args.logit_max_abs_tolerance,
        },
        "candidate_logprob_max_abs": {
            "within_reference_bound": metrics["candidate_logprob_max_abs"] <= args.candidate_logprob_tolerance,
            "reference_bound": args.candidate_logprob_tolerance,
        },
    }
    return {"passed": all(checks.values()), "checks": checks, "diagnostic_bounds": diagnostic_bounds}


def gate_exact_reload_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    """Require the saved-and-reloaded structural model to reproduce the in-memory model exactly."""
    checks = {
        "logits_exact": metrics["logit_max_abs"] == 0.0,
        "candidate_logprobs_exact": metrics["candidate_logprob_max_abs"] == 0.0,
        "label_nll_exact": metrics["max_per_sample_label_nll_delta"] == 0.0,
        "prediction_match": metrics["prediction_match_ratio"] == 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def release_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.batch_size <= 0:
        raise ValueError("max_samples and batch_size must be positive.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deletion_ids = load_json(args.mask_file)
    cluster_idx = load_cluster_idx(args.cluster_idx_path)
    metadata = load_json(args.metadata_path)
    if canonical_json_sha256(deletion_ids) != metadata.get("mask_sha256"):
        raise ValueError("mask_file does not match frozen metadata mask_sha256.")
    if canonical_json_sha256(cluster_idx) != metadata.get("cluster_idx_sha256"):
        raise ValueError("cluster_idx_path does not match frozen metadata cluster_idx_sha256.")

    all_records = load_binary_records(args.pope_file, args.image_root)
    records, selection = select_binary_records(all_records, max_samples=args.max_samples)
    if len(records) != args.max_samples:
        raise ValueError(f"Requested {args.max_samples} smoke samples, got {len(records)}.")

    original, tokenizer, processor = load_model_bundle(args.config, args.model_name_or_path, device)
    original_dims = model_layer_dims(original)
    masks = neuron_ids_to_masks(deletion_ids, original_dims)
    validate_singleton_cluster_idx(cluster_idx, masks)
    original_parameters = count_parameters(original)
    expected_projection_hashes = expected_singleton_projection_hashes(original, cluster_idx)

    with MLPNeuronAblator(original, masks):
        hook_snapshot = collect_snapshot(original, tokenizer, processor, records, device, args.batch_size)
    pruning_summary = pruning_mlp(original, cluster_idx)
    in_memory_parameters = count_parameters(original)
    in_memory_dims = model_layer_dims(original)
    in_memory_projection_hashes = actual_projection_hashes(original)
    in_memory_snapshot = collect_snapshot(original, tokenizer, processor, records, device, args.batch_size)
    hook_vs_in_memory = compare_snapshots(hook_snapshot, in_memory_snapshot)
    hook_vs_in_memory_gate = gate_bf16_hook_comparison(hook_vs_in_memory, args)
    del original, tokenizer, processor
    release_device_cache()

    reloaded, reloaded_tokenizer, reloaded_processor = load_model_bundle(args.config, args.pruned_model_path, device)
    reload_parameters = count_parameters(reloaded)
    reload_dims = model_layer_dims(reloaded)
    reload_projection_hashes = actual_projection_hashes(reloaded)
    reload_snapshot = collect_snapshot(
        reloaded, reloaded_tokenizer, reloaded_processor, records, device, args.batch_size
    )
    in_memory_vs_reload = compare_snapshots(in_memory_snapshot, reload_snapshot)
    in_memory_vs_reload_gate = gate_exact_reload_comparison(in_memory_vs_reload)
    del reloaded, reloaded_tokenizer, reloaded_processor
    release_device_cache()

    structure_checks = {
        "in_memory_matches_cluster_target": in_memory_dims
        == {layer: len(clusters) for layer, clusters in enumerate(cluster_idx)},
        "reload_dims_match_in_memory": reload_dims == in_memory_dims,
        "reload_parameters_match_in_memory": reload_parameters["total"] == in_memory_parameters["total"],
        "parameters_reduced": in_memory_parameters["total"] < original_parameters["total"],
        "singleton_weights_match_expected": in_memory_projection_hashes == expected_projection_hashes,
        "reload_weights_match_in_memory": reload_projection_hashes == in_memory_projection_hashes,
    }
    result = {
        "passed": (
            hook_vs_in_memory_gate["passed"] and in_memory_vs_reload_gate["passed"] and all(structure_checks.values())
        ),
        "config": vars(args),
        "device": str(device),
        "selection": selection,
        "inputs": [
            {
                "source_index": record["source_index"],
                "question_id": record["question_id"],
                "image": record["images"][0],
                "answer": record["answer"],
            }
            for record in records
        ],
        "artifacts": {
            "mask_sha256": metadata["mask_sha256"],
            "cluster_idx_sha256": metadata["cluster_idx_sha256"],
            "pruned_model_path": str(Path(args.pruned_model_path).resolve()),
            "pruning_summary_sha256": (
                sha256_file(Path(args.pruned_model_path) / "pruning_summary.json")
                if (Path(args.pruned_model_path) / "pruning_summary.json").is_file()
                else None
            ),
        },
        "structure": {
            "original_layer_dims": original_dims,
            "in_memory_layer_dims": in_memory_dims,
            "reload_layer_dims": reload_dims,
            "original_parameters": original_parameters,
            "in_memory_parameters": in_memory_parameters,
            "reload_parameters": reload_parameters,
            "pruning_summary": {
                "original_intermediate_size": pruning_summary.original_intermediate_size,
                "pruned_intermediate_size": pruning_summary.pruned_intermediate_size,
                "num_layers": pruning_summary.num_layers,
            },
            "checks": structure_checks,
            "projection_hashes": {
                "expected": expected_projection_hashes,
                "in_memory": in_memory_projection_hashes,
                "reload": reload_projection_hashes,
            },
        },
        "hook_vs_in_memory_structural": {
            "metrics": hook_vs_in_memory,
            "gate": hook_vs_in_memory_gate,
        },
        "in_memory_vs_reloaded_structural": {
            "metrics": in_memory_vs_reload,
            "gate": in_memory_vs_reload_gate,
        },
    }
    write_json(args.output_file, result)
    print(json.dumps({"passed": result["passed"], "output_file": args.output_file}, indent=2))
    if not result["passed"]:
        raise RuntimeError("Phase-3 structural equivalence gate failed; see output_file for details.")


if __name__ == "__main__":
    main()
