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

"""Phase 5B: exact-budget learned gates for one teacher rollout."""

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
from run_phase2_ablation import (  # noqa: E402
    MLPNeuronAblator,
    build_dataloader,
    find_down_proj_modules,
    move_batch_to_device,
)
from run_phase5_single_sample_frontier import (  # noqa: E402
    build_prompt_inputs,
    build_teacher_rollout_batch,
    collect_teacher_trace,
    compare_generation,
    evaluate_teacher_fidelity,
    generate_trace,
    normalize_global_scores,
    parameter_summary,
    prepare_config,
    summarize_masks,
    valid_next_token_tensors,
    write_json,
)

from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"Deletion budgets must be positive integers, got {values}.")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize exact-budget single-sample FFN gates.")
    parser.add_argument("--static_frontier", required=True, help="Phase 5A rollout-aligned frontier.json.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deletion_budgets", required=True, help="Comma-separated exact global deletion counts.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--temperature_start", type=float, default=2.0)
    parser.add_argument("--temperature_end", type=float, default=0.5)
    parser.add_argument("--init_noise", type=float, default=0.02)
    parser.add_argument("--margin_weight", type=float, default=0.1)
    parser.add_argument("--margin_target", type=float, default=0.05)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--history_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2050)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    return parser.parse_args()


def normalized_initial_logits(
    scores: dict[int, torch.Tensor],
    *,
    normalization: str,
    noise_std: float,
    seed: int,
    device: torch.device,
) -> list[torch.Tensor]:
    selected = normalize_global_scores(scores, normalization)
    layer_indices = sorted(selected)
    flat = torch.cat([selected[layer].float() for layer in layer_indices])
    mean = flat.mean()
    std = flat.std().clamp_min(1e-8)
    generator = torch.Generator().manual_seed(seed)
    result = []
    for layer in layer_indices:
        values = (selected[layer].float() - mean) / std
        if noise_std:
            values = values + torch.randn(values.shape, generator=generator) * noise_std
        result.append(values.to(device))
    return result


class ExactBudgetGate(torch.nn.Module):
    """Sample-static global gate with exact hard budget and STE gradients."""

    def __init__(
        self,
        down_proj_modules: list[torch.nn.Module],
        initial_logits: list[torch.Tensor],
        deletion_budget: int,
    ):
        super().__init__()
        if len(down_proj_modules) != len(initial_logits):
            raise ValueError("Gate initialization must provide one tensor per FFN layer.")
        total = sum(values.numel() for values in initial_logits)
        if not 0 < deletion_budget < total:
            raise ValueError(f"deletion_budget must lie in [1, {total - 1}], got {deletion_budget}.")
        self.down_proj_modules = down_proj_modules
        self.logits = torch.nn.ParameterList([torch.nn.Parameter(values.clone()) for values in initial_logits])
        self.deletion_budget = deletion_budget
        self.temperature = 1.0
        self.active_gates: list[torch.Tensor] | None = None
        self.handles: list[Any] = []

    def hard_deletion_masks(self) -> dict[int, torch.Tensor]:
        flat = torch.cat([values.detach() for values in self.logits])
        deleted = torch.argsort(flat, descending=False, stable=True)[: self.deletion_budget]
        flat_mask = torch.zeros(flat.numel(), device=flat.device, dtype=torch.bool)
        flat_mask[deleted] = True
        masks = {}
        offset = 0
        for layer_idx, values in enumerate(self.logits):
            masks[layer_idx] = flat_mask[offset : offset + values.numel()].cpu()
            offset += values.numel()
        return masks

    def prepare_ste_gates(self, temperature: float) -> None:
        self.temperature = temperature
        flat = torch.cat(list(self.logits))
        deleted = torch.argsort(flat.detach(), descending=False, stable=True)[: self.deletion_budget]
        hard = torch.ones_like(flat)
        hard[deleted] = 0.0
        soft = torch.sigmoid(flat / temperature)
        straight_through = soft + (hard - soft).detach()
        gates = []
        offset = 0
        for values in self.logits:
            gates.append(straight_through[offset : offset + values.numel()])
            offset += values.numel()
        self.active_gates = gates

    def __enter__(self) -> ExactBudgetGate:
        for layer_idx, module in enumerate(self.down_proj_modules):
            self.handles.append(module.register_forward_pre_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active_gates = None

    def _make_hook(self, layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if self.active_gates is None:
                raise RuntimeError("Call prepare_ste_gates() before every gated forward pass.")
            hidden_states = inputs[0]
            gate = self.active_gates[layer_idx].to(dtype=hidden_states.dtype)
            gate = gate.view(*([1] * (hidden_states.ndim - 1)), -1)
            return (hidden_states * gate, *inputs[1:])

        return hook


def temperature_at_step(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    fraction = step / (total_steps - 1)
    return start * ((end / start) ** fraction)


def is_better_feasible_candidate(current: dict[str, Any] | None, deletion_budget: int, mean_kl: float) -> bool:
    """Prefer a larger feasible budget, then lower KL within the same budget."""
    return bool(
        current is None
        or deletion_budget > current["deletion_budget"]
        or (deletion_budget == current["deletion_budget"] and mean_kl < current["mean_kl"])
    )


def optimize_gate(
    model: torch.nn.Module,
    batch: dict[str, Any],
    teacher_trace: dict[str, Any],
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
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    if steps < 1:
        raise ValueError("steps must be positive.")
    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
    labels = batch["labels"]
    model_inputs = {key: value for key, value in batch.items() if key != "labels"}
    teacher_logits = teacher_trace["valid_logits"].to(labels.device).float()
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    decision_targets = teacher_trace["valid_labels"].to(labels.device)
    best_objective = float("inf")
    best_kl = float("inf")
    best_masks = controller.hard_deletion_masks()
    history = []
    with controller:
        for step in range(steps):
            temperature = temperature_at_step(temperature_start, temperature_end, step, steps)
            optimizer.zero_grad(set_to_none=True)
            controller.prepare_ste_gates(temperature)
            outputs = model(**model_inputs, use_cache=False)
            student_logits, student_labels = valid_next_token_tensors(outputs.logits, labels)
            if not torch.equal(student_labels.cpu(), teacher_trace["valid_labels"]):
                raise RuntimeError("Learned-gate trajectory is not aligned with the teacher trace.")
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_log_probs, log_target=True, reduction="batchmean")
            teacher_token_logits = student_logits.gather(1, decision_targets[:, None]).squeeze(1)
            competitors = student_logits.clone()
            competitors.scatter_(1, decision_targets[:, None], float("-inf"))
            teacher_margin = teacher_token_logits - competitors.max(dim=-1).values
            margin_loss = F.relu(margin_target - teacher_margin).mean()
            objective = kl_loss + margin_weight * margin_loss
            objective.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(controller.parameters(), gradient_clip))
            objective_value = float(objective.detach())
            kl_value = float(kl_loss.detach())
            margin_value = float(margin_loss.detach())
            if objective_value < best_objective:
                best_objective = objective_value
                best_kl = kl_value
                best_masks = controller.hard_deletion_masks()
            optimizer.step()
            if step % history_interval == 0 or step == steps - 1:
                history.append(
                    {
                        "step": step,
                        "objective": objective_value,
                        "kl_loss": kl_value,
                        "margin_loss": margin_value,
                        "best_objective": best_objective,
                        "best_kl_at_best_objective": best_kl,
                        "temperature": temperature,
                        "gradient_norm": grad_norm,
                    }
                )
    return best_masks, {"best_objective": best_objective, "best_training_kl": best_kl, "history": history}


def load_static_frontier(path: str | Path) -> dict[str, Any]:
    frontier = json.loads(Path(path).read_text(encoding="utf-8"))
    if not frontier.get("complete"):
        raise ValueError("Phase 5A frontier is incomplete.")
    if frontier.get("config", {}).get("trace_target") != "teacher_generation":
        raise ValueError("Phase 5B requires a rollout-aligned Phase 5A frontier.")
    return frontier


def build_phase5a_args(frontier: dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    config = frontier["config"]
    return SimpleNamespace(
        config=config["config_path"],
        overrides=[],
        model_name_or_path=config.get("model_name_or_path"),
        dataset=None,
        eval_dataset=None,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    budgets = parse_csv_ints(args.deletion_budgets)
    if args.steps < 1 or args.restarts < 1:
        raise ValueError("--steps and --restarts must be positive.")
    if args.temperature_start <= 0 or args.temperature_end <= 0:
        raise ValueError("Gate temperatures must be positive.")
    if args.margin_weight < 0 or args.margin_target < 0:
        raise ValueError("Margin weight and target must be non-negative.")
    frontier = load_static_frontier(args.static_frontier)
    static_config = frontier["config"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "learned_frontier.json"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Phase 5B output exists: {result_path}. Pass --resume to add budgets/restarts.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    load_args = build_phase5a_args(frontier, args)
    config = prepare_config(load_args)
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
    teacher_generation = generate_trace(model, tokenizer, prompt_inputs, static_config["max_new_tokens"])
    if teacher_generation["token_ids"].tolist() != frontier["teacher"]["generated_token_ids"]:
        raise RuntimeError("Phase 5B teacher generation does not reproduce the frozen Phase 5A trajectory.")
    batch = build_teacher_rollout_batch(prompt_inputs, teacher_generation["token_ids"])
    down_proj_modules = find_down_proj_modules(model)
    teacher_trace, saliency = collect_teacher_trace(model, batch, down_proj_modules)
    total_neurons = sum(values.numel() for values in saliency["taylor"].values())
    if any(budget >= total_neurons for budget in budgets):
        raise ValueError(f"Deletion budgets must be smaller than {total_neurons}.")

    if args.resume and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        identity = result["config"]
        expected = {
            "static_frontier": str(Path(args.static_frontier)),
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "temperature_start": args.temperature_start,
            "temperature_end": args.temperature_end,
            "init_noise": args.init_noise,
            "margin_weight": args.margin_weight,
            "margin_target": args.margin_target,
            "gradient_clip": args.gradient_clip,
            "seed": args.seed,
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError("Cannot resume Phase 5B with changed optimization settings.")
        result["complete"] = False
        result["config"]["deletion_budgets"] = list(dict.fromkeys([*result["config"]["deletion_budgets"], *budgets]))
        result["config"]["restarts"] = max(result["config"]["restarts"], args.restarts)
    else:
        result = {
            "complete": False,
            "interpretation": (
                "Learned masks are empirical constructive results for one frozen teacher rollout; failed searches "
                "do not prove infeasibility or cross-sample safety."
            ),
            "config": {
                "static_frontier": str(Path(args.static_frontier)),
                "deletion_budgets": budgets,
                "steps": args.steps,
                "restarts": args.restarts,
                "learning_rate": args.learning_rate,
                "temperature_start": args.temperature_start,
                "temperature_end": args.temperature_end,
                "init_noise": args.init_noise,
                "margin_weight": args.margin_weight,
                "margin_target": args.margin_target,
                "gradient_clip": args.gradient_clip,
                "seed": args.seed,
                "kl_tolerance": static_config["kl_tolerance"],
                "max_new_tokens": static_config["max_new_tokens"],
                "trace_target": static_config["trace_target"],
            },
            "sample_manifest": manifest,
            "teacher": frontier["teacher"],
            "runs": {},
            "best_feasible": None,
        }
    result["best_feasible"] = None
    for run_name, prior_run in result["runs"].items():
        if not prior_run["feasible"]:
            continue
        budget = prior_run["deletion_budget"]
        mean_kl = prior_run["teacher_fidelity"]["mean_kl"]
        if is_better_feasible_candidate(result["best_feasible"], budget, mean_kl):
            result["best_feasible"] = {
                "run": run_name,
                "deletion_budget": budget,
                "restart": prior_run["restart"],
                "mean_kl": mean_kl,
                "pruning_ratio": prior_run["mask_summary"]["pruning_ratio"],
                "total_parameter_reduction_ratio": prior_run["parameter_summary"]["total_parameter_reduction_ratio"],
            }
    write_json(result_path, result)

    for budget in budgets:
        for restart in range(args.restarts):
            run_name = f"delete_{budget}__restart_{restart}"
            if run_name in result["runs"]:
                print(f"Phase 5B skipping completed run {run_name}", flush=True)
                continue
            run_seed = args.seed + budget * 1009 + restart
            print(f"Phase 5B optimizing {run_name}", flush=True)
            initial_logits = normalized_initial_logits(
                saliency["taylor"],
                normalization=static_config["global_normalization"],
                noise_std=args.init_noise * restart,
                seed=run_seed,
                device=device,
            )
            controller = ExactBudgetGate(down_proj_modules, initial_logits, budget)
            masks, training = optimize_gate(
                model,
                batch,
                teacher_trace,
                controller,
                steps=args.steps,
                learning_rate=args.learning_rate,
                temperature_start=args.temperature_start,
                temperature_end=args.temperature_end,
                gradient_clip=args.gradient_clip,
                history_interval=args.history_interval,
                margin_weight=args.margin_weight,
                margin_target=args.margin_target,
            )
            neuron_ids = masks_to_neuron_ids(masks)
            mask_path = output_dir / "masks" / f"{run_name}.json"
            write_json(mask_path, neuron_ids)
            with MLPNeuronAblator(model, masks):
                fidelity = evaluate_teacher_fidelity(model, batch, teacher_trace)
                student_generation = generate_trace(model, tokenizer, prompt_inputs, static_config["max_new_tokens"])
            generation = compare_generation(teacher_generation, student_generation)
            feasible = bool(
                generation["exact_match"]
                and fidelity["consistent_token_agreement"] == 1.0
                and fidelity["mean_kl"] <= static_config["kl_tolerance"]
            )
            run_result = {
                "deletion_budget": budget,
                "restart": restart,
                "seed": run_seed,
                "mask_hash": canonical_json_sha256(neuron_ids),
                "mask_file": str(mask_path),
                "mask_summary": summarize_masks(masks),
                "parameter_summary": parameter_summary(model, down_proj_modules, masks),
                "training": training,
                "teacher_fidelity": fidelity,
                "generation": generation,
                "feasible": feasible,
            }
            result["runs"][run_name] = run_result
            current = result["best_feasible"]
            if feasible and is_better_feasible_candidate(current, budget, fidelity["mean_kl"]):
                result["best_feasible"] = {
                    "run": run_name,
                    "deletion_budget": budget,
                    "restart": restart,
                    "mean_kl": fidelity["mean_kl"],
                    "pruning_ratio": run_result["mask_summary"]["pruning_ratio"],
                    "total_parameter_reduction_ratio": run_result["parameter_summary"][
                        "total_parameter_reduction_ratio"
                    ],
                }
            write_json(result_path, result)

    result["complete"] = True
    write_json(result_path, result)
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"complete": result["complete"], "best_feasible": result["best_feasible"]}, indent=2))


if __name__ == "__main__":
    main()
