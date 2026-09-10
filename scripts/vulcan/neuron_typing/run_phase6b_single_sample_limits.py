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

"""Search exact-budget empirical FFN limits for several frozen single samples."""

from __future__ import annotations

import argparse
import hashlib
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

from phase3_structural_utils import canonical_json_sha256, masks_to_neuron_ids  # noqa: E402
from phase5e_proxy import decode_gold_caption, evaluate_gold_proxy, parse_deletion_budgets  # noqa: E402
from phase5e_semantics import generate_short_caption  # noqa: E402
from phase6b_semantics import build_semantic_evaluator, evaluate_contract_semantics, validate_contract  # noqa: E402
from run_phase2_ablation import (  # noqa: E402
    MLPNeuronAblator,
    build_dataloader,
    find_down_proj_modules,
    load_yaml,
    move_batch_to_device,
)
from run_phase5_learned_gate import ExactBudgetGate, normalized_initial_logits  # noqa: E402
from run_phase5_single_sample_frontier import (  # noqa: E402
    build_prompt_inputs,
    collect_teacher_trace,
    parameter_summary,
    summarize_masks,
    write_json,
)
from run_phase5e_learned_gate import optimize_gold_gate  # noqa: E402
from verify_phase5_structural_equivalence import load_model_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Phase 6B empirical FFN limits across frozen single samples.")
    parser.add_argument("--sample_file", default="data/phase6b_single_samples/frozen_samples.json")
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/phase5e_coco.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deletion_budgets", required=True)
    parser.add_argument("--sample_ids", default=None, help="Optional comma-separated subset of frozen sample IDs.")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--temperature_start", type=float, default=2.0)
    parser.add_argument("--temperature_end", type=float, default=0.5)
    parser.add_argument("--init_noise", type=float, default=0.02)
    parser.add_argument("--gold_ce_weight", type=float, default=1.0)
    parser.add_argument("--reference_kl_weight", type=float, default=1.0)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--history_interval", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2062)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_sample_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    sample_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not sample_ids:
        raise ValueError("--sample_ids must name at least one sample when provided.")
    return list(dict.fromkeys(sample_ids))


def prepare_dataset(output_dir: Path, samples: list[dict[str, Any]], prompt: str) -> Path:
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"<image>\n{prompt}"},
                {"role": "assistant", "content": sample["gold_caption"]},
            ],
            "images": [sample["image"]],
        }
        for sample in samples
    ]
    write_json(dataset_dir / "samples.json", rows)
    write_json(
        dataset_dir / "dataset_info.json",
        {
            "phase6b_single_samples": {
                "file_name": "samples.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
        },
    )
    return dataset_dir


def validate_inputs(payload: dict[str, Any], sample_ids: list[str] | None) -> list[dict[str, Any]]:
    samples = payload.get("samples")
    prompt = payload.get("canonical_prompt")
    if not isinstance(samples, list) or not samples or not prompt:
        raise ValueError("The Phase 6B sample file requires canonical_prompt and a non-empty samples list.")
    ids = [sample.get("sample_id") for sample in samples]
    if len(ids) != len(set(ids)) or any(not sample_id for sample_id in ids):
        raise ValueError(f"Frozen sample IDs must be present and unique: {ids}.")
    missing_images = [sample["image"] for sample in samples if not Path(sample["image"]).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing frozen images: {missing_images}.")
    for sample in samples:
        validate_contract(sample["contract"])
        if sample["contract"]["sample_id"] != sample["sample_id"]:
            raise ValueError(f"Contract/sample ID mismatch for {sample['sample_id']}.")
        frozen_check = evaluate_contract_semantics(sample["base_caption"], contract=sample["contract"])
        if not frozen_check["automatic_pass"]:
            raise ValueError(f"Frozen base caption violates its contract for {sample['sample_id']}.")
    if sample_ids is None:
        return samples
    unknown = sorted(set(sample_ids) - set(ids))
    if unknown:
        raise ValueError(f"Unknown --sample_ids: {unknown}.")
    selected = set(sample_ids)
    return [sample for sample in samples if sample["sample_id"] in selected]


def is_better(current: dict[str, Any] | None, budget: int, mean_nll: float) -> bool:
    return bool(
        current is None
        or budget > current["deletion_budget"]
        or (budget == current["deletion_budget"] and mean_nll < current["mean_nll"])
    )


def initial_result(run_config: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "complete": False,
        "interpretation": (
            "Every reported limit is a best-found feasible construction under the recorded discrete budgets, steps, "
            "and restarts; it is not a proof of a theoretical minimum sufficient subnet."
        ),
        "config": run_config,
        "samples": {
            sample["sample_id"]: {
                "semantic_stratum": sample["semantic_stratum"],
                "image": sample["image"],
                "gold_caption": sample["gold_caption"],
                "frozen_base_caption": sample["base_caption"],
                "contract": sample["contract"],
                "reference": None,
                "baseline_generation": None,
                "runs": {},
                "best_automatic_semantic_pass": None,
            }
            for sample in samples
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    budgets = parse_deletion_budgets(args.deletion_budgets)
    if args.steps < 1 or args.restarts < 1 or args.max_new_tokens < 1:
        raise ValueError("--steps, --restarts, and --max_new_tokens must be positive.")
    if args.temperature_start <= 0 or args.temperature_end <= 0 or args.gradient_clip <= 0:
        raise ValueError("Gate temperatures and --gradient_clip must be positive.")

    sample_path = Path(args.sample_file).resolve()
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    all_samples = validate_inputs(payload, None)
    selected_ids = parse_sample_ids(args.sample_ids)
    active_ids = {sample["sample_id"] for sample in validate_inputs(payload, selected_ids)}
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "single_sample_limits.json"
    dataset_dir = prepare_dataset(output_dir, all_samples, payload["canonical_prompt"])

    config_path = Path(args.config).resolve()
    base_config = load_yaml(config_path)
    model_path = Path(args.model_name_or_path or base_config["model_name_or_path"]).resolve()
    run_config = {
        "sample_file": str(sample_path),
        "sample_file_sha256": sha256_file(sample_path),
        "config_path": str(config_path),
        "model_name_or_path": str(model_path),
        "canonical_prompt": payload["canonical_prompt"],
        "deletion_budgets": budgets,
        "steps": args.steps,
        "restarts": args.restarts,
        "learning_rate": args.learning_rate,
        "temperature_start": args.temperature_start,
        "temperature_end": args.temperature_end,
        "init_noise": args.init_noise,
        "gold_ce_weight": args.gold_ce_weight,
        "reference_kl_weight": args.reference_kl_weight,
        "gradient_clip": args.gradient_clip,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "trace_target": "gold_caption",
        "global_normalization": "layer_mean",
    }
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Phase 6B output exists: {result_path}. Pass --resume to append runs.")
    if args.resume and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        mutable = {"deletion_budgets", "restarts"}
        immutable = {key: value for key, value in run_config.items() if key not in mutable}
        if any(result["config"].get(key) != value for key, value in immutable.items()):
            raise ValueError("Cannot resume Phase 6B with changed samples, model, or optimization settings.")
        result["complete"] = False
        result["config"]["deletion_budgets"] = list(dict.fromkeys([*result["config"]["deletion_budgets"], *budgets]))
        result["config"]["restarts"] = max(result["config"]["restarts"], args.restarts)
    else:
        result = initial_result(run_config, all_samples)
    write_json(result_path, result)

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
    config.update(
        {
            "dataset_dir": str(dataset_dir),
            "dataset": "phase6b_single_samples",
            "eval_dataset": None,
            "tokenized_path": None,
            "max_samples": len(all_samples),
            "overwrite_cache": True,
            "enable_thinking": False,
        }
    )
    dataloader, manifest = build_dataloader(
        config,
        model,
        tokenizer_module,
        template,
        batch_size=1,
        num_workers=0,
        sample_offset=0,
        max_samples=len(all_samples),
        allow_short_dataset=False,
        max_image_repeat=1,
        allow_excessive_image_repeats=False,
        dataset_stage="sft",
    )
    result["dataset_manifest"] = manifest
    tokenizer = tokenizer_module["tokenizer"]
    down_proj_modules = find_down_proj_modules(model)

    for sample_index, (sample, batch) in enumerate(zip(all_samples, dataloader, strict=True)):
        sample_id = sample["sample_id"]
        if sample_id not in active_ids:
            continue
        print(f"Phase 6B preparing {sample_id}", flush=True)
        batch = move_batch_to_device(batch, device)
        prompt_inputs, prompt_tokens = build_prompt_inputs(batch)
        reference_trace, saliency = collect_teacher_trace(model, batch, down_proj_modules)
        decoded_gold = decode_gold_caption(tokenizer, reference_trace["valid_labels"])
        if decoded_gold.rstrip(".") != sample["gold_caption"].strip().rstrip("."):
            raise RuntimeError(f"Decoded gold mismatch for {sample_id}: {decoded_gold!r}.")
        total_neurons = sum(values.numel() for values in saliency["taylor"].values())
        if any(budget >= total_neurons for budget in budgets):
            raise ValueError(f"Deletion budgets must be smaller than {total_neurons}.")

        semantic_evaluator = build_semantic_evaluator(sample["contract"])
        baseline = generate_short_caption(
            model,
            tokenizer,
            prompt_inputs,
            max_new_tokens=args.max_new_tokens,
            semantic_evaluator=semantic_evaluator,
        )
        if not baseline["semantic"]["automatic_pass"]:
            raise RuntimeError(f"Fresh baseline generation failed the frozen contract for {sample_id}.")
        sample_result = result["samples"][sample_id]
        sample_result["reference"] = {
            "prompt_tokens": prompt_tokens,
            "gold_caption": decoded_gold,
            "gold_token_ids": reference_trace["valid_labels"].tolist(),
            "num_gold_tokens": reference_trace["num_label_tokens"],
            "mean_nll": reference_trace["nll"],
        }
        sample_result["baseline_generation"] = baseline
        score_path = output_dir / sample_id / "saliency_scores.pt"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sample_id": sample_id,
                "trace_target": "gold_caption",
                "sample_file_sha256": run_config["sample_file_sha256"],
                "scores": saliency,
            },
            score_path,
        )
        sample_result["saliency_file"] = str(score_path)
        write_json(result_path, result)

        for budget in budgets:
            for restart in range(args.restarts):
                run_name = f"delete_{budget}__restart_{restart}"
                if run_name in sample_result["runs"]:
                    print(f"Phase 6B skipping {sample_id}/{run_name}", flush=True)
                    continue
                run_seed = args.seed + sample_index * 1_000_003 + budget * 1009 + restart
                print(f"Phase 6B optimizing {sample_id}/{run_name}", flush=True)
                initial_logits = normalized_initial_logits(
                    saliency["taylor"],
                    normalization="layer_mean",
                    noise_std=args.init_noise * restart,
                    seed=run_seed,
                    device=device,
                )
                controller = ExactBudgetGate(down_proj_modules, initial_logits, budget)
                masks, training = optimize_gold_gate(
                    model,
                    batch,
                    reference_trace,
                    controller,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    temperature_start=args.temperature_start,
                    temperature_end=args.temperature_end,
                    gold_ce_weight=args.gold_ce_weight,
                    reference_kl_weight=args.reference_kl_weight,
                    gradient_clip=args.gradient_clip,
                    history_interval=args.history_interval,
                )
                neuron_ids = masks_to_neuron_ids(masks)
                mask_path = output_dir / sample_id / "masks" / f"{run_name}.json"
                write_json(mask_path, neuron_ids)
                with MLPNeuronAblator(model, masks):
                    proxy = evaluate_gold_proxy(model, batch, reference_trace)
                    generation = generate_short_caption(
                        model,
                        tokenizer,
                        prompt_inputs,
                        max_new_tokens=args.max_new_tokens,
                        semantic_evaluator=semantic_evaluator,
                    )
                automatic_pass = bool(generation["semantic"]["automatic_pass"])
                row = {
                    "deletion_budget": budget,
                    "restart": restart,
                    "seed": run_seed,
                    "mask_hash": canonical_json_sha256(neuron_ids),
                    "mask_file": str(mask_path),
                    "mask_summary": summarize_masks(masks),
                    "parameter_summary": parameter_summary(model, down_proj_modules, masks),
                    "training": training,
                    "gold_proxy": proxy,
                    "generation": generation,
                    "automatic_semantic_pass": automatic_pass,
                    "human_confirmation": None,
                    "physical_validation": None,
                }
                sample_result["runs"][run_name] = row
                current = sample_result["best_automatic_semantic_pass"]
                if automatic_pass and is_better(current, budget, proxy["mean_nll"]):
                    sample_result["best_automatic_semantic_pass"] = {
                        "run": run_name,
                        "deletion_budget": budget,
                        "restart": restart,
                        "mean_nll": proxy["mean_nll"],
                        "delta_nll": proxy["delta_nll"],
                        "final_caption": generation["final_caption"],
                        "mask_hash": row["mask_hash"],
                        "mask_file": row["mask_file"],
                    }
                write_json(result_path, result)
                del controller

    result["complete"] = all(
        all(
            f"delete_{budget}__restart_{restart}" in sample["runs"]
            for budget in budgets
            for restart in range(args.restarts)
        )
        for sample_id, sample in result["samples"].items()
        if sample_id in active_ids
    )
    write_json(result_path, result)
    return result


def main() -> None:
    result = run(parse_args())
    summary = {
        sample_id: row["best_automatic_semantic_pass"] for sample_id, row in result["samples"].items() if row["runs"]
    }
    print(json.dumps({"complete": result["complete"], "best": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
