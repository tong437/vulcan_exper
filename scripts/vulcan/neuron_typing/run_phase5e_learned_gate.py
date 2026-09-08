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

"""Phase 5E-B: exact-budget learned FFN gates on the gold caption."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase3_structural_utils import canonical_json_sha256, masks_to_neuron_ids  # noqa: E402
from phase5e_proxy import evaluate_gold_proxy, parse_deletion_budgets  # noqa: E402
from phase5e_semantics import generate_short_caption  # noqa: E402
from run_phase2_ablation import (  # noqa: E402
    MLPNeuronAblator,
    build_dataloader,
    find_down_proj_modules,
    move_batch_to_device,
)
from run_phase5_learned_gate import (  # noqa: E402
    ExactBudgetGate,
    normalized_initial_logits,
    temperature_at_step,
)
from run_phase5_single_sample_frontier import (  # noqa: E402
    build_prompt_inputs,
    collect_teacher_trace,
    parameter_summary,
    prepare_config,
    summarize_masks,
    valid_next_token_tensors,
    write_json,
)

from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Phase 5E exact-budget gold-caption neuron gates.")
    parser.add_argument("--static_frontier", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deletion_budgets", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--temperature_start", type=float, default=2.0)
    parser.add_argument("--temperature_end", type=float, default=0.5)
    parser.add_argument("--init_noise", type=float, default=0.02)
    parser.add_argument("--gold_ce_weight", type=float, default=1.0)
    parser.add_argument("--reference_kl_weight", type=float, default=1.0)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--history_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2056)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    return parser.parse_args()


def optimize_gold_gate(
    model: torch.nn.Module,
    batch: dict[str, Any],
    reference_trace: dict[str, Any],
    controller: ExactBudgetGate,
    *,
    steps: int,
    learning_rate: float,
    temperature_start: float,
    temperature_end: float,
    gold_ce_weight: float,
    reference_kl_weight: float,
    gradient_clip: float,
    history_interval: int,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Optimize an exact hard budget with CE and reference KL on gold prefixes."""
    if steps < 1:
        raise ValueError("steps must be positive.")
    if gold_ce_weight < 0 or reference_kl_weight < 0 or gold_ce_weight + reference_kl_weight == 0:
        raise ValueError("At least one non-negative proxy loss weight must be positive.")
    labels = batch["labels"]
    model_inputs = {key: value for key, value in batch.items() if key != "labels"}
    reference_logits = reference_trace["valid_logits"].to(labels.device).float()
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    expected_labels = reference_trace["valid_labels"].to(labels.device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
    best_objective = float("inf")
    best_masks = controller.hard_deletion_masks()
    best_step = None
    history = []

    with controller:
        for step in range(steps):
            temperature = temperature_at_step(temperature_start, temperature_end, step, steps)
            optimizer.zero_grad(set_to_none=True)
            controller.prepare_ste_gates(temperature)
            outputs = model(**model_inputs, use_cache=False)
            logits, gold_labels = valid_next_token_tensors(outputs.logits, labels)
            logits = logits.float()
            if not torch.equal(gold_labels, expected_labels):
                raise RuntimeError("Learned gate is not aligned with the frozen gold-caption tokens.")
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = F.nll_loss(log_probs, gold_labels)
            kl_loss = F.kl_div(log_probs, reference_log_probs, log_target=True, reduction="batchmean")
            objective = gold_ce_weight * ce_loss + reference_kl_weight * kl_loss
            objective.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(controller.parameters(), gradient_clip))
            objective_value = float(objective.detach())
            if objective_value < best_objective:
                best_objective = objective_value
                best_masks = controller.hard_deletion_masks()
                best_step = step
            optimizer.step()
            if step % history_interval == 0 or step == steps - 1:
                history.append(
                    {
                        "step": step,
                        "objective": objective_value,
                        "gold_ce": float(ce_loss.detach()),
                        "reference_kl": float(kl_loss.detach()),
                        "best_objective": best_objective,
                        "best_step": best_step,
                        "temperature": temperature,
                        "gradient_norm": grad_norm,
                    }
                )
    return best_masks, {"best_objective": best_objective, "best_step": best_step, "history": history}


def _load_static(path: str | Path) -> dict[str, Any]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not result.get("complete") or result.get("config", {}).get("trace_target") != "gold_caption":
        raise ValueError("Phase 5E-B requires a complete gold-caption Phase 5E-A static frontier.")
    return result


def _load_args(static: dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    config = static["config"]
    return SimpleNamespace(
        config=config["config_path"],
        overrides=[],
        model_name_or_path=config.get("model_name_or_path"),
        dataset=None,
        eval_dataset=None,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )


def _better(current: dict[str, Any] | None, budget: int, nll: float) -> bool:
    return bool(
        current is None
        or budget > current["deletion_budget"]
        or (budget == current["deletion_budget"] and nll < current["mean_nll"])
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    budgets = parse_deletion_budgets(args.deletion_budgets)
    if args.steps < 1 or args.restarts < 1:
        raise ValueError("--steps and --restarts must be positive.")
    if args.temperature_start <= 0 or args.temperature_end <= 0:
        raise ValueError("Gate temperatures must be positive.")
    if args.gradient_clip <= 0:
        raise ValueError("--gradient_clip must be positive.")
    static_path = Path(args.static_frontier)
    static = _load_static(static_path)
    static_config = static["config"]
    score_artifact = torch.load(static_path.parent / "saliency_scores.pt", map_location="cpu", weights_only=True)
    if score_artifact.get("trace_target") != "gold_caption":
        raise ValueError("The Phase 5E saliency artifact is not gold-caption aligned.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "learned_frontier.json"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Phase 5E-B output exists: {result_path}. Pass --resume to append runs.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = prepare_config(_load_args(static, args))
    config["enable_thinking"] = False
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
        sample_offset=static_config["sample_offset"],
        max_samples=1,
        allow_short_dataset=False,
        max_image_repeat=5,
        allow_excessive_image_repeats=False,
        dataset_stage="sft",
    )
    batch = move_batch_to_device(next(iter(dataloader)), device)
    prompt_inputs, _ = build_prompt_inputs(batch)
    down_proj_modules = find_down_proj_modules(model)
    reference_trace, fresh_saliency = collect_teacher_trace(model, batch, down_proj_modules)
    if reference_trace["valid_labels"].tolist() != static["reference"]["gold_token_ids"]:
        raise RuntimeError("Gold-caption tokens do not reproduce the frozen Phase 5E-A reference.")
    total_neurons = sum(values.numel() for values in fresh_saliency["taylor"].values())
    if any(budget >= total_neurons for budget in budgets):
        raise ValueError(f"Deletion budgets must be smaller than {total_neurons}.")

    run_config = {
        "static_frontier": str(static_path),
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
        "seed": args.seed,
        "max_new_tokens": static_config["max_new_tokens"],
        "trace_target": "gold_caption",
    }
    if args.resume and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        immutable = {key: value for key, value in run_config.items() if key not in {"deletion_budgets", "restarts"}}
        if any(result["config"].get(key) != value for key, value in immutable.items()):
            raise ValueError("Cannot resume Phase 5E-B with changed optimization settings.")
        result["complete"] = False
        result["config"]["deletion_budgets"] = list(dict.fromkeys([*result["config"]["deletion_budgets"], *budgets]))
        result["config"]["restarts"] = max(result["config"]["restarts"], args.restarts)
    else:
        result = {
            "complete": False,
            "interpretation": (
                "The learned gate uses differentiable gold-caption proxies, while only short free-generation "
                "semantic pass defines automatic feasibility. Human confirmation and physical validation remain pending."
            ),
            "config": run_config,
            "sample_manifest": manifest,
            "reference": static["reference"],
            "runs": {},
            "best_automatic_semantic_pass": None,
        }
    write_json(result_path, result)

    for budget in budgets:
        for restart in range(args.restarts):
            run_name = f"delete_{budget}__restart_{restart}"
            if run_name in result["runs"]:
                print(f"Phase 5E-B skipping completed run {run_name}", flush=True)
                continue
            run_seed = args.seed + budget * 1009 + restart
            print(f"Phase 5E-B optimizing {run_name}", flush=True)
            initial_logits = normalized_initial_logits(
                fresh_saliency["taylor"],
                normalization=static_config["global_normalization"],
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
            mask_path = output_dir / "masks" / f"{run_name}.json"
            write_json(mask_path, neuron_ids)
            with MLPNeuronAblator(model, masks):
                proxy = evaluate_gold_proxy(model, batch, reference_trace)
                generation = generate_short_caption(
                    model,
                    tokenizer,
                    prompt_inputs,
                    max_new_tokens=static_config["max_new_tokens"],
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
            result["runs"][run_name] = row
            current = result["best_automatic_semantic_pass"]
            if automatic_pass and _better(current, budget, proxy["mean_nll"]):
                result["best_automatic_semantic_pass"] = {
                    "run": run_name,
                    "deletion_budget": budget,
                    "restart": restart,
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
    print(json.dumps({"complete": result["complete"], "best": result["best_automatic_semantic_pass"]}, indent=2))


if __name__ == "__main__":
    main()
