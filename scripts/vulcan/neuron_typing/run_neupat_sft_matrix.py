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

"""Run the gate-controlled NeuPAT, vanilla full-SFT, and LoRA matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[3]
TRAIN_ENTRY = ROOT_DIR / "src" / "train.py"
DEFAULT_CONFIGS = (
    "vanilla_full=examples/vulcan/qwen35_08b_vqa_rad_vanilla_full_neupat_matrix.yaml",
    "lora=examples/vulcan/qwen35_08b_vqa_rad_lora_neupat_matrix.yaml",
    "neupat=examples/vulcan/qwen35_08b_vqa_rad_neupat_sft.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal NeuPAT SFT comparison matrix.")
    parser.add_argument(
        "--gate_file",
        default="saves/neuron_typing/neupat_protection_formal_500/neupat_causality.json",
    )
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument(
        "--plan_file",
        default="saves/qwen35-0_8b-vqa-rad/neupat_matrix_20260902/matrix_plan.json",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_configs(values: list[str]) -> dict[str, Path]:
    configs = {}
    for value in values or DEFAULT_CONFIGS:
        if "=" not in value:
            raise ValueError(f"Expected NAME=CONFIG, got {value!r}.")
        name, path_text = value.split("=", maxsplit=1)
        if not name or name in configs:
            raise ValueError(f"Invalid or duplicate matrix name: {name!r}.")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        configs[name] = path
    return configs


def validate_matrix(configs: dict[str, Path]) -> dict[str, dict[str, Any]]:
    parsed = {name: yaml.safe_load(path.read_text(encoding="utf-8")) for name, path in configs.items()}
    required_names = {"vanilla_full", "lora", "neupat"}
    if set(parsed) != required_names:
        raise ValueError(f"The formal matrix requires exactly {sorted(required_names)}.")
    shared_keys = (
        "model_name_or_path",
        "dataset_dir",
        "dataset",
        "eval_dataset",
        "template",
        "cutoff_len",
        "image_max_pixels",
        "image_min_pixels",
        "num_train_epochs",
        "gradient_accumulation_steps",
        "seed",
        "data_seed",
    )
    for key in shared_keys:
        values = {name: config.get(key) for name, config in parsed.items()}
        if len(set(values.values())) != 1:
            raise ValueError(f"Matrix configs disagree on {key}: {values}")
    if parsed["vanilla_full"].get("finetuning_type") != "full":
        raise ValueError("vanilla_full must use full fine-tuning.")
    if parsed["lora"].get("finetuning_type") != "lora":
        raise ValueError("lora must use LoRA fine-tuning.")
    if parsed["neupat"].get("finetuning_type") != "full" or not parsed["neupat"].get("use_neupat"):
        raise ValueError("neupat must use NeuPAT-enabled full fine-tuning.")
    if parsed["vanilla_full"].get("learning_rate") != parsed["neupat"].get("learning_rate"):
        raise ValueError("NeuPAT and vanilla full-SFT must use the same learning rate.")
    return parsed


def completed_result(output_dir: Path) -> dict[str, Any] | None:
    result_path = output_dir / "all_results.json"
    state_path = output_dir / "trainer_state.json"
    if not result_path.is_file() or not state_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return {
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "train_loss": result.get("train_loss"),
        "eval_loss": result.get("eval_loss"),
        "train_runtime": result.get("train_runtime"),
    }


def latest_resume_checkpoint(output_dir: Path) -> Path | None:
    """Return the highest structurally complete Trainer/DeepSpeed checkpoint."""
    candidates = []
    for checkpoint in output_dir.glob("checkpoint-*"):
        try:
            step = int(checkpoint.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        state_path = checkpoint / "trainer_state.json"
        optimizer_dirs = list(checkpoint.glob("global_step*"))
        standard_optimizer_state = (checkpoint / "optimizer.pt").is_file() and (checkpoint / "scheduler.pt").is_file()
        model_files = list(checkpoint.glob("*model*.safetensors"))
        if not state_path.is_file() or not (optimizer_dirs or standard_optimizer_state) or not model_files:
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("global_step") != step:
            continue
        candidates.append((step, checkpoint))
    return max(candidates)[1] if candidates else None


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gate_path = Path(args.gate_file)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    primary = gate.get("primary_hypothesis", {})
    if not primary.get("gate_passed"):
        raise ValueError(f"NeuPAT protection-set causal gate has not passed: {gate_path}")

    configs = parse_named_configs(args.config)
    parsed = validate_matrix(configs)
    plan_path = Path(args.plan_file)
    plan = {
        "artifact_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "gate_file": str(gate_path.resolve()),
        "gate_sha256": sha256_file(gate_path),
        "gate_passed": True,
        "run_order": list(configs),
        "runs": {
            name: {
                "config": str(configs[name].resolve()),
                "config_sha256": sha256_file(configs[name]),
                "output_dir": str(Path(config["output_dir"]).resolve()),
                "status": "pending",
            }
            for name, config in parsed.items()
        },
    }
    write_plan(plan_path, plan)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "src")
    env["WANDB_DISABLED"] = "true"
    env["FORCE_TORCHRUN"] = "1"
    for name, config_path in configs.items():
        output_dir = Path(parsed[name]["output_dir"])
        result = completed_result(output_dir)
        if result is not None:
            plan["runs"][name].update({"status": "complete", "result": result})
            write_plan(plan_path, plan)
            print(f"Skipping completed run {name}: {output_dir}", flush=True)
            continue
        resume_checkpoint = None
        if output_dir.exists() and any(output_dir.iterdir()):
            resume_checkpoint = latest_resume_checkpoint(output_dir)
            if resume_checkpoint is None:
                raise FileExistsError(f"Refusing to overwrite incomplete output directory: {output_dir}")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=1",
            str(TRAIN_ENTRY),
            str(config_path),
        ]
        if resume_checkpoint is not None:
            command.append(f"resume_from_checkpoint={resume_checkpoint}")
            plan["runs"][name]["resume_from_checkpoint"] = str(resume_checkpoint.resolve())
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        plan["runs"][name]["status"] = "running"
        write_plan(plan_path, plan)
        subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)
        result = completed_result(output_dir)
        if result is None:
            raise RuntimeError(f"Training command returned without a complete result: {name}")
        plan["runs"][name].update({"status": "complete", "result": result})
        write_plan(plan_path, plan)


if __name__ == "__main__":
    main()
