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

"""Run Phase 6D bidirectional layer localization on frozen conflict edges."""

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
from phase6d_conflicts import CONFLICT_CONTRASTS, build_localization_variants  # noqa: E402
from run_phase2_ablation import MLPNeuronAblator, build_dataloader, load_yaml, move_batch_to_device  # noqa: E402
from run_phase5_single_sample_frontier import build_prompt_inputs, write_json  # noqa: E402
from run_phase5e_retrospective import build_reference_trace  # noqa: E402
from run_phase6b_single_sample_limits import prepare_dataset, sha256_file, validate_inputs  # noqa: E402
from verify_phase5_structural_equivalence import load_model_bundle, model_layer_dims  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6D negative-interference layer localization.")
    parser.add_argument("--winner_dir", default="saves/neuron_typing/phase6b_frozen_winners_v1")
    parser.add_argument("--sample_file", default="data/phase6b_single_samples/frozen_samples.json")
    parser.add_argument(
        "--phase6c_evaluations",
        default="saves/neuron_typing/phase6c_core_shell_composability_v1/evaluations.jsonl",
    )
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/phase5e_coco.yaml")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--random_seeds", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2065)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def compact_proxy(proxy: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in proxy.items() if not key.startswith("per_token_")}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        evaluation_id = row.get("evaluation_id")
        if not evaluation_id or evaluation_id in rows:
            raise ValueError(f"Invalid or duplicate evaluation id at {path}:{line_number}.")
        rows[evaluation_id] = row
    return rows


def validate_phase6c_endpoints(path: Path) -> dict[str, Any]:
    """Verify that every predeclared contrast was pass-to-fail in Phase 6C."""
    rows = load_jsonl(path)
    evidence = {}
    for contrast in CONFLICT_CONTRASTS:
        base_id = f"shell_learned__{contrast.base}__target_{contrast.target}"
        union_id = f"{contrast.union_variant_id}__target_{contrast.target}"
        if base_id not in rows or union_id not in rows:
            raise ValueError(f"Phase 6C lacks an endpoint for {contrast.contrast_id}.")
        base_row = rows[base_id]
        union_row = rows[union_id]
        if not base_row["automatic_semantic_pass"] or union_row["automatic_semantic_pass"]:
            raise ValueError(f"Phase 6C no longer supports the frozen pass-to-fail contrast {contrast.contrast_id}.")
        evidence[contrast.contrast_id] = {
            "safe_base_evaluation_id": base_id,
            "safe_base_nll": base_row["gold_proxy"]["mean_nll"],
            "failed_union_evaluation_id": union_id,
            "failed_union_nll": union_row["gold_proxy"]["mean_nll"],
        }
    return evidence


def result_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "evaluations": len(selected),
            "semantic_passes": sum(bool(row["automatic_semantic_pass"]) for row in selected),
        }
    return {"evaluations": len(rows), "by_family": by_family}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.random_seeds, args.max_new_tokens) < 1:
        raise ValueError("--random_seeds and --max_new_tokens must be positive.")
    winner_dir = Path(args.winner_dir).resolve()
    sample_path = Path(args.sample_file).resolve()
    phase6c_path = Path(args.phase6c_evaluations).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "phase6d_localization.json"
    evaluations_path = output_dir / "evaluations.jsonl"
    if output_dir.exists() and not args.resume and (result_path.exists() or evaluations_path.exists()):
        raise FileExistsError(f"Phase 6D output already exists: {output_dir}. Pass --resume to continue.")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_evidence = validate_phase6c_endpoints(phase6c_path)
    samples_payload = json.loads(sample_path.read_text(encoding="utf-8"))
    samples = validate_inputs(samples_payload, None)
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    target_ids = list(dict.fromkeys(contrast.target for contrast in CONFLICT_CONTRASTS))
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
    variants = build_localization_variants(winners, layer_dims, random_seeds=args.random_seeds, seed=args.seed)

    base_config = load_yaml(config_path)
    model_path = Path(args.model_name_or_path or base_config["model_name_or_path"]).resolve()
    run_config = {
        "winner_dir": str(winner_dir),
        "winner_manifest_sha256": sha256_file(winner_dir / "frozen_winners.json"),
        "sample_file": str(sample_path),
        "sample_file_sha256": sha256_file(sample_path),
        "phase6c_evaluations": str(phase6c_path),
        "phase6c_evaluations_sha256": sha256_file(phase6c_path),
        "config_path": str(config_path),
        "model_name_or_path": str(model_path),
        "random_seeds": args.random_seeds,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "primary_endpoint": "strict_deterministic_free_generation_semantic_pass",
        "shortlist_rule": "same layer causes add-one failure and leave-one-layer-out recovery",
    }
    if args.resume and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing["config"] != run_config:
            raise ValueError("Cannot resume Phase 6D with a changed configuration.")

    variant_manifest = [{key: value for key, value in variant.items() if key != "kept"} for variant in variants]
    result = {
        "complete": False,
        "config": run_config,
        "source_phase6c_evidence": source_evidence,
        "contrasts": [variant["metadata"]["contrast"] for variant in variants if variant["family"] == "endpoint"][::2],
        "target_ids": target_ids,
        "expected_variants": len(variants),
        "expected_evaluations": len(variants),
        "evaluations_file": str(evaluations_path),
        "variants": variant_manifest,
        "summary": None,
    }
    write_json(result_path, result)

    completed = load_jsonl(evaluations_path)
    expected_ids = {variant["variant_id"] for variant in variants}
    if set(completed) - expected_ids:
        raise ValueError(
            f"Existing evaluation rows are not part of this run: {sorted(set(completed) - expected_ids)[:5]}."
        )

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

    for variant_index, variant in enumerate(variants, start=1):
        evaluation_id = variant["variant_id"]
        if evaluation_id in completed:
            continue
        print(f"Phase 6D [{variant_index}/{len(variants)}] {evaluation_id}", flush=True)
        context = contexts[variant["target"]]
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
            "family": variant["family"],
            "variant_id": variant["variant_id"],
            "target_sample_id": variant["target"],
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
    result["summary"] = result_summary(rows)
    result["complete"] = len(rows) == len(variants)
    write_json(result_path, result)
    print(json.dumps({"complete": result["complete"], **result["summary"]}, indent=2))
    if not result["complete"]:
        raise RuntimeError(f"Phase 6D produced {len(rows)} of {len(variants)} expected evaluations.")
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
