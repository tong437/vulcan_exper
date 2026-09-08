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

"""Phase 5E-A: gold-caption static saliency and semantic frontier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_guard import save_manifest  # noqa: E402
from phase3_structural_utils import canonical_json_sha256, masks_to_neuron_ids  # noqa: E402
from phase5e_proxy import (  # noqa: E402
    build_exact_deletion_masks,
    decode_gold_caption,
    evaluate_gold_proxy,
    parse_deletion_budgets,
)
from phase5e_semantics import REFERENCE_CAPTION, generate_short_caption, semantic_contract  # noqa: E402
from run_phase2_ablation import (  # noqa: E402
    MLPNeuronAblator,
    build_dataloader,
    find_down_proj_modules,
    move_batch_to_device,
)
from run_phase5_single_sample_frontier import (  # noqa: E402
    SALIENCY_METHODS,
    collect_teacher_trace,
    parameter_summary,
    parse_csv_choices,
    prepare_config,
    random_scores_like,
    summarize_masks,
    write_json,
)

from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 5E gold-caption static frontier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--eval_dataset", default=None)
    parser.add_argument("--sample_offset", type=int, default=0, help="Index in the frozen one-row Phase 5E dataset.")
    parser.add_argument("--reference_caption", default=REFERENCE_CAPTION)
    parser.add_argument("--deletion_budgets", default="100,250,500,1000,2000,4000,8000")
    parser.add_argument("--methods", default="taylor")
    parser.add_argument(
        "--global_normalization",
        choices=["none", "layer_mean", "layer_rank"],
        default="layer_mean",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2055)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow_baseline_nonpass", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def _normalized_caption(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(" .")


def validate_reference_caption(actual: str, expected: str) -> None:
    if _normalized_caption(actual) != _normalized_caption(expected):
        raise ValueError(f"Frozen gold caption mismatch: expected {expected!r}, decoded dataset labels as {actual!r}.")


def _condition_name(method: str, budget: int) -> str:
    return f"{method}__global__delete_{budget}"


def _resume_identity(result: dict[str, Any], fresh_config: dict[str, Any]) -> None:
    fields = (
        "config_path",
        "model_name_or_path",
        "dataset",
        "sample_offset",
        "reference_caption",
        "global_normalization",
        "max_new_tokens",
        "seed",
    )
    mismatches = {
        field: (result["config"].get(field), fresh_config.get(field))
        for field in fields
        if result["config"].get(field) != fresh_config.get(field)
    }
    if mismatches:
        raise ValueError(f"Cannot resume a different Phase 5E-A run: {mismatches}.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    budgets = parse_deletion_budgets(args.deletion_budgets)
    methods = parse_csv_choices(args.methods, SALIENCY_METHODS, "methods")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be positive.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "static_frontier.json"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Phase 5E-A output exists: {result_path}. Pass --resume to append conditions.")

    config = prepare_config(args)
    config["enable_thinking"] = False
    if config.get("template") != "qwen3_5":
        raise ValueError(
            "Phase 5E requires template=qwen3_5 with enable_thinking=false for an explicit empty think block."
        )
    model_args, data_args, _, finetuning_args, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval().requires_grad_(False)
    dataloader, manifest = build_dataloader(
        config,
        model,
        tokenizer_module,
        template,
        batch_size=1,
        num_workers=args.num_workers,
        sample_offset=args.sample_offset,
        max_samples=1,
        allow_short_dataset=False,
        max_image_repeat=5,
        allow_excessive_image_repeats=False,
        dataset_stage="sft",
    )
    manifest["role"] = "phase5e_gold_caption_search"
    manifest_path = output_dir / "sample_manifest.json"
    save_manifest(manifest, manifest_path)
    write_json(output_dir / "semantic_contract.json", semantic_contract())

    batch = move_batch_to_device(next(iter(dataloader)), device)
    from run_phase5_single_sample_frontier import build_prompt_inputs  # local import keeps reusable helpers grouped

    prompt_inputs, prompt_length = build_prompt_inputs(batch)
    down_proj_modules = find_down_proj_modules(model)
    reference_trace, saliency = collect_teacher_trace(model, batch, down_proj_modules)
    gold_caption = decode_gold_caption(tokenizer, reference_trace["valid_labels"])
    validate_reference_caption(gold_caption, args.reference_caption)
    reference_generation = generate_short_caption(model, tokenizer, prompt_inputs, max_new_tokens=args.max_new_tokens)
    reference_proxy = evaluate_gold_proxy(model, batch, reference_trace)

    score_artifact = {
        "trace_target": "gold_caption",
        "reference_caption": args.reference_caption,
        "scores": saliency,
        "layer_widths": {layer: int(values.numel()) for layer, values in saliency["activation"].items()},
    }
    torch.save(score_artifact, output_dir / "saliency_scores.pt")
    torch.save(
        {
            "trace_target": "gold_caption",
            "reference_caption": args.reference_caption,
            "valid_logits": reference_trace["valid_logits"].to(torch.float16),
            "valid_labels": reference_trace["valid_labels"],
            "nll": reference_trace["nll"],
        },
        output_dir / "reference_logits.pt",
    )
    fresh_config = {
        "config_path": args.config,
        "model_name_or_path": config.get("model_name_or_path"),
        "dataset": config.get("eval_dataset") or config.get("dataset"),
        "sample_offset": args.sample_offset,
        "sample_manifest": str(manifest_path),
        "reference_caption": args.reference_caption,
        "trace_target": "gold_caption",
        "methods": methods,
        "deletion_budgets": budgets,
        "global_normalization": args.global_normalization,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "enable_thinking": False,
        "seed": args.seed,
    }
    if args.resume and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        _resume_identity(result, fresh_config)
        result["complete"] = False
        result["config"]["methods"] = list(dict.fromkeys([*result["config"]["methods"], *methods]))
        result["config"]["deletion_budgets"] = list(dict.fromkeys([*result["config"]["deletion_budgets"], *budgets]))
    else:
        result = {
            "complete": False,
            "interpretation": (
                "Semantic-pass masks are constructive lower bounds for one image and search family. Proxy metrics "
                "rank candidates but do not define semantic success; failed budgets do not prove infeasibility."
            ),
            "config": fresh_config,
            "reference": {
                "gold_caption": gold_caption,
                "num_gold_tokens": reference_trace["num_label_tokens"],
                "gold_token_ids": reference_trace["valid_labels"].tolist(),
                "prompt_tokens": prompt_length,
                "proxy": reference_proxy,
                "generation": reference_generation,
            },
            "layer_widths": score_artifact["layer_widths"],
            "parameter_scope": parameter_summary(model, down_proj_modules),
            "conditions": {},
            "best_semantic_pass": {},
        }
    write_json(result_path, result)
    if not reference_generation["semantic"]["automatic_pass"] and not args.allow_baseline_nonpass:
        raise RuntimeError(
            "The unpruned model did not pass the frozen automatic semantic contract. Inspect static_frontier.json "
            "or pass --allow_baseline_nonpass for diagnostics."
        )

    method_scores = dict(saliency)
    if "random" in methods:
        method_scores["random"] = random_scores_like(saliency["activation"], args.seed)
    total_neurons = sum(values.numel() for values in saliency["activation"].values())
    if any(budget >= total_neurons for budget in budgets):
        raise ValueError(f"Deletion budgets must be smaller than {total_neurons}.")

    for method in methods:
        for budget in budgets:
            name = _condition_name(method, budget)
            if name in result["conditions"]:
                print(f"Phase 5E-A skipping completed condition {name}", flush=True)
                continue
            print(f"Phase 5E-A evaluating {name}", flush=True)
            masks = build_exact_deletion_masks(
                method_scores[method], budget, global_normalization=args.global_normalization
            )
            neuron_ids = masks_to_neuron_ids(masks)
            mask_path = output_dir / "masks" / f"{name}.json"
            write_json(mask_path, neuron_ids)
            with MLPNeuronAblator(model, masks):
                proxy = evaluate_gold_proxy(model, batch, reference_trace)
                generation = generate_short_caption(
                    model, tokenizer, prompt_inputs, max_new_tokens=args.max_new_tokens
                )
            automatic_pass = bool(generation["semantic"]["automatic_pass"])
            row = {
                "method": method,
                "deletion_budget": budget,
                "mask_hash": canonical_json_sha256(neuron_ids),
                "mask_file": str(mask_path),
                "mask_summary": summarize_masks(masks),
                "parameter_summary": parameter_summary(model, down_proj_modules, masks),
                "gold_proxy": proxy,
                "generation": generation,
                "automatic_semantic_pass": automatic_pass,
                "human_confirmation": None,
            }
            result["conditions"][name] = row
            current = result["best_semantic_pass"].get(method)
            if automatic_pass and (current is None or budget > current["deletion_budget"]):
                result["best_semantic_pass"][method] = {
                    "condition": name,
                    "deletion_budget": budget,
                    "mean_nll": proxy["mean_nll"],
                    "delta_nll": proxy["delta_nll"],
                    "final_caption": generation["final_caption"],
                }
            write_json(result_path, result)

    result["complete"] = True
    write_json(result_path, result)
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"complete": result["complete"], "best": result["best_semantic_pass"]}, indent=2))


if __name__ == "__main__":
    main()
