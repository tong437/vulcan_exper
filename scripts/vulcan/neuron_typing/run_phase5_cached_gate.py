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

"""Phase 5D: exact-budget gate optimization on the cached decoding path."""

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
from phase5_cached_utils import (  # noqa: E402
    cached_fidelity,
    cached_teacher_forcing_steps,
    collect_cached_teacher_forced_logits,
    generate_cached_teacher_trace,
    generated_token_margins,
)
from run_phase2_ablation import (  # noqa: E402
    MLPNeuronAblator,
    build_dataloader,
    find_down_proj_modules,
    move_batch_to_device,
)
from run_phase5_learned_gate import (  # noqa: E402
    ExactBudgetGate,
    is_better_feasible_candidate,
    normalized_initial_logits,
    parse_csv_ints,
    temperature_at_step,
)
from run_phase5_single_sample_frontier import (  # noqa: E402
    build_prompt_inputs,
    build_teacher_rollout_batch,
    collect_teacher_trace,
    compare_generation,
    evaluate_teacher_fidelity,
    generate_trace,
    parameter_summary,
    prepare_config,
    summarize_masks,
    write_json,
)

from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize exact-budget gates through cached teacher forcing.")
    parser.add_argument("--static_frontier", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deletion_budgets", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--temperature_start", type=float, default=2.0)
    parser.add_argument("--temperature_end", type=float, default=0.5)
    parser.add_argument("--init_noise", type=float, default=0.02)
    parser.add_argument("--margin_weight", type=float, default=0.05)
    parser.add_argument("--margin_target", type=float, default=0.05)
    parser.add_argument(
        "--focus_token_start",
        type=int,
        default=0,
        help="Zero-based inclusive generated-token index whose loss receives extra weight.",
    )
    parser.add_argument(
        "--focus_token_end",
        type=int,
        default=None,
        help="Zero-based exclusive end of the focused token interval; defaults to the rollout length.",
    )
    parser.add_argument(
        "--focus_token_weight",
        type=float,
        default=1.0,
        help="Loss multiplier inside [focus_token_start, focus_token_end).",
    )
    parser.add_argument(
        "--teacher_low_margin_threshold",
        type=float,
        default=None,
        help="Also upweight teacher tokens whose unpruned cached-path margin is below this threshold.",
    )
    parser.add_argument(
        "--teacher_low_margin_weight",
        type=float,
        default=1.0,
        help="Loss multiplier for teacher-low-margin tokens; combined with interval weighting by maximum.",
    )
    parser.add_argument(
        "--robust_margin_threshold",
        type=float,
        default=0.25,
        help="Minimum generated-token logit margin required by the robustness gate.",
    )
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--history_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2060)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    return parser.parse_args()


def load_static_frontier(path: str | Path) -> dict[str, Any]:
    frontier = json.loads(Path(path).read_text(encoding="utf-8"))
    if not frontier.get("complete"):
        raise ValueError("Phase-5 cached optimization requires a complete static frontier.")
    if frontier.get("config", {}).get("trace_target") != "teacher_generation":
        raise ValueError("Phase-5 cached optimization requires a teacher-generation frontier.")
    return frontier


def build_load_args(frontier: dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    config = frontier["config"]
    return SimpleNamespace(
        config=config["config_path"],
        overrides=[],
        model_name_or_path=config.get("model_name_or_path"),
        dataset=None,
        eval_dataset=None,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )


def cached_token_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_token: torch.Tensor,
    *,
    margin_weight: float,
    margin_target: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_logits = teacher_logits.to(student_logits.device).reshape(1, -1).float()
    target = teacher_token.to(student_logits.device).reshape(1, 1).long()
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_log_probs, log_target=True, reduction="batchmean")
    target_logit = student_logits.gather(1, target).squeeze(1)
    competitors = student_logits.clone()
    competitors.scatter_(1, target, float("-inf"))
    margin = target_logit - competitors.max(dim=-1).values
    margin_loss = F.relu(margin_target - margin).mean()
    return kl_loss + margin_weight * margin_loss, kl_loss, margin_loss, margin


def is_better_cached_training_candidate(current: dict[str, Any] | None, candidate: dict[str, Any]) -> bool:
    """Prefer discrete masks that preserve the teacher trajectory before optimizing smooth losses."""

    def rank(metrics: dict[str, Any]) -> tuple[int, int, float, float, float]:
        return (
            metrics["generated_token_agreement_count"],
            metrics["teacher_top_agreement_count"],
            metrics["min_generated_token_margin"],
            -metrics["unweighted_mean_kl"],
            -metrics["objective"],
        )

    return current is None or rank(candidate) > rank(current)


def build_token_loss_weights(
    num_tokens: int,
    *,
    focus_start: int,
    focus_end: int | None,
    focus_weight: float,
    teacher_margins: torch.Tensor | None = None,
    low_margin_threshold: float | None = None,
    low_margin_weight: float = 1.0,
) -> torch.Tensor:
    """Build normalized-loss weights while keeping the reported fidelity metrics unweighted."""
    if num_tokens < 1:
        raise ValueError("Token-loss weighting requires a positive rollout length.")
    resolved_end = num_tokens if focus_end is None else focus_end
    if focus_start < 0 or focus_start >= num_tokens:
        raise ValueError(f"focus_token_start must lie in [0, {num_tokens}), got {focus_start}.")
    if resolved_end <= focus_start or resolved_end > num_tokens:
        raise ValueError(f"focus_token_end must lie in ({focus_start}, {num_tokens}], got {resolved_end}.")
    if focus_weight <= 0:
        raise ValueError(f"focus_token_weight must be positive, got {focus_weight}.")
    if low_margin_weight <= 0:
        raise ValueError(f"teacher_low_margin_weight must be positive, got {low_margin_weight}.")
    if low_margin_threshold is not None and low_margin_threshold < 0:
        raise ValueError(f"teacher_low_margin_threshold must be non-negative, got {low_margin_threshold}.")
    if low_margin_threshold is not None:
        if teacher_margins is None or teacher_margins.ndim != 1 or teacher_margins.numel() != num_tokens:
            shape = None if teacher_margins is None else tuple(teacher_margins.shape)
            raise ValueError(f"teacher_margins must have shape ({num_tokens},), got {shape}.")
    weights = torch.ones(num_tokens, dtype=torch.float32)
    weights[focus_start:resolved_end] = focus_weight
    if low_margin_threshold is not None:
        low_margin = teacher_margins.cpu() < low_margin_threshold
        weights[low_margin] = torch.maximum(weights[low_margin], torch.tensor(low_margin_weight))
    return weights


def optimize_cached_gate(
    model: torch.nn.Module,
    prompt_inputs: dict[str, Any],
    teacher_cached_trace: dict[str, Any],
    controller: ExactBudgetGate,
    *,
    steps: int,
    learning_rate: float,
    temperature_start: float,
    temperature_end: float,
    gradient_clip: float,
    history_interval: int,
    margin_weight: float,
    margin_target: float,
    focus_token_start: int,
    focus_token_end: int | None,
    focus_token_weight: float,
    teacher_low_margin_threshold: float | None,
    teacher_low_margin_weight: float,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Optimize one sample-static gate with one-token truncated BPTT through cache state."""
    teacher_tokens = teacher_cached_trace["token_ids"]
    teacher_logits = teacher_cached_trace["raw_logits"]
    num_tokens = int(teacher_tokens.numel())
    if num_tokens == 0 or steps < 1:
        raise ValueError("Cached optimization requires positive token and optimization counts.")
    teacher_margins = generated_token_margins(teacher_logits, teacher_tokens)
    token_weights = build_token_loss_weights(
        num_tokens,
        focus_start=focus_token_start,
        focus_end=focus_token_end,
        focus_weight=focus_token_weight,
        teacher_margins=teacher_margins,
        low_margin_threshold=teacher_low_margin_threshold,
        low_margin_weight=teacher_low_margin_weight,
    )
    total_token_weight = float(token_weights.sum())
    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
    best_metrics = None
    best_masks = controller.hard_deletion_masks()
    history = []
    with controller:
        for optimization_step in range(steps):
            temperature = temperature_at_step(temperature_start, temperature_end, optimization_step, steps)
            optimizer.zero_grad(set_to_none=True)
            total_objective = 0.0
            total_kl = 0.0
            total_margin = 0.0
            total_unweighted_kl = 0.0
            generated_token_agreement_count = 0
            teacher_top_agreement_count = 0
            minimum_generated_token_margin = float("inf")

            def prepare_gate(_: int) -> None:
                controller.prepare_ste_gates(temperature)

            iterator = cached_teacher_forcing_steps(
                model,
                prompt_inputs,
                teacher_tokens,
                before_forward=prepare_gate,
            )
            for token_step, student_logits in iterator:
                objective, kl_loss, margin_loss, generated_margin = cached_token_loss(
                    student_logits,
                    teacher_logits[token_step],
                    teacher_tokens[token_step],
                    margin_weight=margin_weight,
                    margin_target=margin_target,
                )
                token_weight = float(token_weights[token_step])
                (objective * token_weight / total_token_weight).backward()
                total_objective += float(objective.detach()) * token_weight
                total_kl += float(kl_loss.detach()) * token_weight
                total_margin += float(margin_loss.detach()) * token_weight
                total_unweighted_kl += float(kl_loss.detach())
                student_top = int(student_logits.detach().argmax(dim=-1))
                teacher_top = int(teacher_logits[token_step].argmax())
                teacher_token = int(teacher_tokens[token_step])
                generated_token_agreement_count += int(student_top == teacher_token)
                teacher_top_agreement_count += int(student_top == teacher_top)
                minimum_generated_token_margin = min(minimum_generated_token_margin, float(generated_margin.detach()))

            objective_value = total_objective / total_token_weight
            kl_value = total_kl / total_token_weight
            margin_value = total_margin / total_token_weight
            candidate_metrics = {
                "step": optimization_step,
                "objective": objective_value,
                "weighted_cached_kl": kl_value,
                "margin_loss": margin_value,
                "unweighted_mean_kl": total_unweighted_kl / num_tokens,
                "generated_token_agreement_count": generated_token_agreement_count,
                "teacher_top_agreement_count": teacher_top_agreement_count,
                "min_generated_token_margin": minimum_generated_token_margin,
            }
            grad_norm = float(torch.nn.utils.clip_grad_norm_(controller.parameters(), gradient_clip))
            if is_better_cached_training_candidate(best_metrics, candidate_metrics):
                best_metrics = candidate_metrics
                best_masks = controller.hard_deletion_masks()
            optimizer.step()
            if optimization_step % history_interval == 0 or optimization_step == steps - 1:
                history.append(
                    {
                        "step": optimization_step,
                        "objective": objective_value,
                        "cached_kl_loss": kl_value,
                        "unweighted_mean_kl": candidate_metrics["unweighted_mean_kl"],
                        "margin_loss": margin_value,
                        "generated_token_agreement_count": generated_token_agreement_count,
                        "teacher_top_agreement_count": teacher_top_agreement_count,
                        "min_generated_token_margin": minimum_generated_token_margin,
                        "selected_best_step": best_metrics["step"],
                        "temperature": temperature,
                        "gradient_norm": grad_norm,
                    }
                )
    return best_masks, {
        "best_objective": best_metrics["objective"],
        "best_training_cached_kl": best_metrics["weighted_cached_kl"],
        "selection": best_metrics,
        "selection_rule": (
            "maximize generated-token agreement, teacher-top agreement, and minimum generated-token margin; "
            "then minimize unweighted KL and weighted objective"
        ),
        "history": history,
        "gradient_estimator": "one-token truncated BPTT; cache values detached after every decode step",
        "token_loss_weighting": {
            "focus_start": focus_token_start,
            "focus_end": num_tokens if focus_token_end is None else focus_token_end,
            "focus_weight": focus_token_weight,
            "teacher_low_margin_threshold": teacher_low_margin_threshold,
            "teacher_low_margin_weight": teacher_low_margin_weight,
            "teacher_low_margin_tokens": (
                int((teacher_margins < teacher_low_margin_threshold).sum())
                if teacher_low_margin_threshold is not None
                else 0
            ),
            "normalization_weight": total_token_weight,
        },
    }


def cached_feasibility(fidelity: dict[str, Any], generation: dict[str, Any], kl_tolerance: float) -> tuple[bool, bool]:
    behavioral = bool(generation["exact_match"] and fidelity["mean_kl"] <= kl_tolerance)
    strict = bool(behavioral and fidelity["token_agreement"] == 1.0)
    return strict, behavioral


def candidate_summary(run_name: str, run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": run_name,
        "deletion_budget": run["deletion_budget"],
        "restart": run["restart"],
        "mean_cached_kl": run["cached_fidelity"]["mean_kl"],
        "min_generated_token_margin": run["cached_fidelity"].get("min_generated_token_margin"),
        "pruning_ratio": run["mask_summary"]["pruning_ratio"],
        "total_parameter_reduction_ratio": run["parameter_summary"]["total_parameter_reduction_ratio"],
    }


def rebuild_best(result: dict[str, Any], field: str) -> dict[str, Any] | None:
    best = None
    for run_name, run in result["runs"].items():
        if not run.get(field, False):
            continue
        mean_kl = run["cached_fidelity"]["mean_kl"]
        margin = run["cached_fidelity"].get("min_generated_token_margin", float("-inf"))
        better = is_better_feasible_candidate(best, run["deletion_budget"], mean_kl)
        if best is not None and run["deletion_budget"] == best["deletion_budget"] and field == "robust_feasible":
            best_margin = best.get("min_generated_token_margin")
            better = (
                best_margin is None or margin > best_margin or (margin == best_margin and mean_kl < best["mean_kl"])
            )
        if better:
            best = candidate_summary(run_name, run)
            best["mean_kl"] = best.pop("mean_cached_kl")
    return best


def run(args: argparse.Namespace) -> dict[str, Any]:
    budgets = parse_csv_ints(args.deletion_budgets)
    if args.steps < 1 or args.restarts < 1:
        raise ValueError("--steps and --restarts must be positive.")
    if args.temperature_start <= 0 or args.temperature_end <= 0:
        raise ValueError("Gate temperatures must be positive.")
    if args.margin_weight < 0 or args.margin_target < 0 or args.robust_margin_threshold < 0:
        raise ValueError("Margin settings must be non-negative.")
    if args.focus_token_weight <= 0:
        raise ValueError("--focus_token_weight must be positive.")
    if args.teacher_low_margin_weight <= 0:
        raise ValueError("--teacher_low_margin_weight must be positive.")
    if args.teacher_low_margin_threshold is not None and args.teacher_low_margin_threshold < 0:
        raise ValueError("--teacher_low_margin_threshold must be non-negative.")
    frontier = load_static_frontier(args.static_frontier)
    static_config = frontier["config"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "cached_frontier.json"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Cached-gate output exists: {result_path}. Pass --resume to append work.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = prepare_config(build_load_args(frontier, args))
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
        dataset_stage=static_config["dataset_stage"],
    )
    dataset_batch = move_batch_to_device(next(iter(dataloader)), device)
    prompt_inputs, _ = build_prompt_inputs(dataset_batch)
    teacher_cached_trace = generate_cached_teacher_trace(
        model, tokenizer, prompt_inputs, static_config["max_new_tokens"]
    )
    if teacher_cached_trace["token_ids"].tolist() != frontier["teacher"]["generated_token_ids"]:
        raise RuntimeError("Cached-gate teacher does not reproduce the frozen frontier trajectory.")
    baseline_manual_logits = collect_cached_teacher_forced_logits(
        model, prompt_inputs, teacher_cached_trace["token_ids"]
    )
    baseline_cached_fidelity = cached_fidelity(
        teacher_cached_trace["raw_logits"], baseline_manual_logits, teacher_cached_trace["token_ids"]
    )
    if baseline_cached_fidelity["token_agreement"] != 1.0 or baseline_cached_fidelity["mean_kl"] > 1e-8:
        raise RuntimeError(f"Manual cached path is not aligned with generation: {baseline_cached_fidelity}.")

    batch = build_teacher_rollout_batch(prompt_inputs, teacher_cached_trace["token_ids"])
    down_proj_modules = find_down_proj_modules(model)
    full_teacher_trace, saliency = collect_teacher_trace(model, batch, down_proj_modules)
    total_neurons = sum(values.numel() for values in saliency["taylor"].values())
    if any(budget >= total_neurons for budget in budgets):
        raise ValueError(f"Deletion budgets must be smaller than {total_neurons}.")

    identity = {
        "static_frontier": str(Path(args.static_frontier)),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "temperature_start": args.temperature_start,
        "temperature_end": args.temperature_end,
        "init_noise": args.init_noise,
        "margin_weight": args.margin_weight,
        "margin_target": args.margin_target,
        "focus_token_start": args.focus_token_start,
        "focus_token_end": args.focus_token_end,
        "focus_token_weight": args.focus_token_weight,
        "teacher_low_margin_threshold": args.teacher_low_margin_threshold,
        "teacher_low_margin_weight": args.teacher_low_margin_weight,
        "robust_margin_threshold": args.robust_margin_threshold,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "gradient_estimator": "one_token_truncated_bptt",
    }
    if args.resume and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        backward_compatible_defaults = {
            "focus_token_start": 0,
            "focus_token_end": None,
            "focus_token_weight": 1.0,
            "teacher_low_margin_threshold": None,
            "teacher_low_margin_weight": 1.0,
        }
        stored_identity = result["config"]
        if any(
            stored_identity.get(key, backward_compatible_defaults.get(key)) != value for key, value in identity.items()
        ):
            raise ValueError("Cannot resume cached-gate search with changed optimization settings.")
        for key, default in backward_compatible_defaults.items():
            result["config"].setdefault(key, default)
        result["complete"] = False
        result["config"]["deletion_budgets"] = list(dict.fromkeys([*result["config"]["deletion_budgets"], *budgets]))
        result["config"]["restarts"] = max(result["config"]["restarts"], args.restarts)
    else:
        result = {
            "complete": False,
            "interpretation": (
                "Cached-path feasible masks are constructive single-rollout results. Truncated-BPTT search failures "
                "do not prove infeasibility, and hook masks still require physical BF16 screening."
            ),
            "config": {
                **identity,
                "deletion_budgets": budgets,
                "restarts": args.restarts,
                "kl_tolerance": static_config["kl_tolerance"],
                "max_new_tokens": static_config["max_new_tokens"],
            },
            "sample_manifest": manifest,
            "teacher": {
                "generated_token_ids": teacher_cached_trace["token_ids"].tolist(),
                "generated_text": teacher_cached_trace["text"],
                "baseline_manual_cached_fidelity": baseline_cached_fidelity,
            },
            "runs": {},
            "best_strict_feasible": None,
            "best_robust_feasible": None,
            "best_behavioral_feasible": None,
        }
    result["best_strict_feasible"] = rebuild_best(result, "strict_feasible")
    result["best_robust_feasible"] = rebuild_best(result, "robust_feasible")
    result["best_behavioral_feasible"] = rebuild_best(result, "behavioral_feasible")
    write_json(result_path, result)

    for budget in budgets:
        for restart in range(args.restarts):
            run_name = f"delete_{budget}__restart_{restart}"
            if run_name in result["runs"]:
                print(f"Phase 5D skipping completed run {run_name}", flush=True)
                continue
            run_seed = args.seed + budget * 1009 + restart
            print(f"Phase 5D optimizing {run_name}", flush=True)
            initial_logits = normalized_initial_logits(
                saliency["taylor"],
                normalization=static_config["global_normalization"],
                noise_std=args.init_noise * restart,
                seed=run_seed,
                device=device,
            )
            controller = ExactBudgetGate(down_proj_modules, initial_logits, budget)
            masks, training = optimize_cached_gate(
                model,
                prompt_inputs,
                teacher_cached_trace,
                controller,
                steps=args.steps,
                learning_rate=args.learning_rate,
                temperature_start=args.temperature_start,
                temperature_end=args.temperature_end,
                gradient_clip=args.gradient_clip,
                history_interval=args.history_interval,
                margin_weight=args.margin_weight,
                margin_target=args.margin_target,
                focus_token_start=args.focus_token_start,
                focus_token_end=args.focus_token_end,
                focus_token_weight=args.focus_token_weight,
                teacher_low_margin_threshold=args.teacher_low_margin_threshold,
                teacher_low_margin_weight=args.teacher_low_margin_weight,
            )
            neuron_ids = masks_to_neuron_ids(masks)
            mask_path = output_dir / "masks" / f"{run_name}.json"
            write_json(mask_path, neuron_ids)
            with MLPNeuronAblator(model, masks):
                student_cached_logits = collect_cached_teacher_forced_logits(
                    model, prompt_inputs, teacher_cached_trace["token_ids"]
                )
                cache_metrics = cached_fidelity(
                    teacher_cached_trace["raw_logits"],
                    student_cached_logits,
                    teacher_cached_trace["token_ids"],
                )
                full_forward_metrics = evaluate_teacher_fidelity(model, batch, full_teacher_trace)
                student_generation = generate_trace(model, tokenizer, prompt_inputs, static_config["max_new_tokens"])
            teacher_generation = {
                "token_ids": teacher_cached_trace["token_ids"],
                "text": teacher_cached_trace["text"],
            }
            generation = compare_generation(teacher_generation, student_generation)
            strict, behavioral = cached_feasibility(cache_metrics, generation, static_config["kl_tolerance"])
            robust = bool(strict and cache_metrics["min_generated_token_margin"] >= args.robust_margin_threshold)
            run_result = {
                "deletion_budget": budget,
                "restart": restart,
                "seed": run_seed,
                "mask_hash": canonical_json_sha256(neuron_ids),
                "mask_file": str(mask_path),
                "mask_summary": summarize_masks(masks),
                "parameter_summary": parameter_summary(model, down_proj_modules, masks),
                "training": training,
                "cached_fidelity": cache_metrics,
                "full_forward_diagnostic": full_forward_metrics,
                "generation": generation,
                "strict_feasible": strict,
                "robust_feasible": robust,
                "behavioral_feasible": behavioral,
                "feasible": strict,
            }
            result["runs"][run_name] = run_result
            result["best_strict_feasible"] = rebuild_best(result, "strict_feasible")
            result["best_robust_feasible"] = rebuild_best(result, "robust_feasible")
            result["best_behavioral_feasible"] = rebuild_best(result, "behavioral_feasible")
            write_json(result_path, result)

    result["complete"] = True
    write_json(result_path, result)
    return result


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "best_strict_feasible": result["best_strict_feasible"],
                "best_robust_feasible": result["best_robust_feasible"],
                "best_behavioral_feasible": result["best_behavioral_feasible"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
