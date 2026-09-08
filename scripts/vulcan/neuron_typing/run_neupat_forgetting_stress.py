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

"""Run the pre-registered vanilla full-SFT forgetting-stress ladder."""

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
from run_neupat_sft_matrix import completed_result, latest_resume_checkpoint


ROOT_DIR = Path(__file__).resolve().parents[3]
TRAIN_ENTRY = ROOT_DIR / "src" / "train.py"
DEFAULT_CONFIG = "examples/vulcan/qwen35_08b_vqa_rad_forgetting_stress.yaml"
DEFAULT_OUTPUT_ROOT = "saves/qwen35-0_8b-vqa-rad/neupat_forgetting_stress_20260904"
STRESS_LADDER: dict[str, dict[str, float]] = {
    "lr5e-6_ep3": {"learning_rate": 5e-6, "num_train_epochs": 3.0},
    "lr1e-5_ep3": {"learning_rate": 1e-5, "num_train_epochs": 3.0},
    "lr2e-5_ep3": {"learning_rate": 2e-5, "num_train_epochs": 3.0},
    "lr2e-5_ep6": {"learning_rate": 2e-5, "num_train_epochs": 6.0},
    "lr5e-5_ep3": {"learning_rate": 5e-5, "num_train_epochs": 3.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vanilla full-SFT forgetting-stress candidates.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate", action="append", choices=tuple(STRESS_LADDER))
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "stage": "sft",
        "finetuning_type": "full",
        "dataset": "vqa_rad_stress_train",
        "eval_dataset": "vqa_rad_stress_dev",
        "load_best_model_at_end": False,
        "weight_decay": 0.0,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Stress config requires {key}={expected!r}, got {config.get(key)!r}.")
    if config.get("resume_from_checkpoint") is not None:
        raise ValueError("The base stress config must leave resume_from_checkpoint null.")
    return config


def training_command(
    config_path: Path,
    output_dir: Path,
    candidate: dict[str, float],
    *,
    seed: int,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=1",
        str(ROOT_DIR / TRAIN_ENTRY),
        str(config_path),
        f"output_dir={output_dir}",
        f"learning_rate={candidate['learning_rate']}",
        f"num_train_epochs={candidate['num_train_epochs']}",
        f"seed={seed}",
        f"data_seed={seed}",
    ]
    if resume_checkpoint is not None:
        command.append(f"resume_from_checkpoint={resume_checkpoint}")
    return command


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = validate_config(config_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_names = args.candidate or list(STRESS_LADDER)
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("Stress candidates must not be repeated.")

    manifests = {
        "vqa_split": Path(config["dataset_dir"]) / "stress_split_manifest.json",
        "language_dev": Path("data/neupat_language_stress_dev_c4_700.metadata.json"),
        "language_lockbox": Path("data/neupat_language_stress_lockbox_c4_700.metadata.json"),
    }
    missing = [str(path) for path in manifests.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen stress-data manifests: {missing}")
    plan = {
        "artifact_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "Find the least severe vanilla full-SFT setting that causes language forgetting on dev.",
        "selection_order": list(STRESS_LADDER),
        "selected_for_this_invocation": selected_names,
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "seed": args.seed,
        "data_manifests": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in manifests.items()
        },
        "lockbox_policy": "Never evaluate the lockbox while selecting stress strength.",
        "candidates": {},
    }
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "src")
    env["WANDB_DISABLED"] = "true"
    env["FORCE_TORCHRUN"] = "1"
    plan_path = output_root / "stress_plan.json"
    for name, candidate in STRESS_LADDER.items():
        output_dir = output_root / name
        result = completed_result(output_dir)
        plan["candidates"][name] = {
            **candidate,
            "update_exposure": candidate["learning_rate"] * candidate["num_train_epochs"],
            "output_dir": str(output_dir.resolve()),
            "status": "complete" if result is not None else "pending",
            "result": result,
            "command": training_command(config_path, output_dir, candidate, seed=args.seed),
        }
    write_json(plan_path, plan)
    for name in selected_names:
        candidate = STRESS_LADDER[name]
        output_dir = output_root / name
        result = completed_result(output_dir)
        resume_checkpoint = None
        if result is None and output_dir.exists() and any(output_dir.iterdir()):
            resume_checkpoint = latest_resume_checkpoint(output_dir)
            if resume_checkpoint is None:
                raise FileExistsError(f"Refusing to overwrite incomplete output directory: {output_dir}")
        command = training_command(
            config_path,
            output_dir,
            candidate,
            seed=args.seed,
            resume_checkpoint=resume_checkpoint,
        )
        plan["candidates"][name]["command"] = command
        write_json(plan_path, plan)
        if result is not None:
            print(f"Skipping completed candidate {name}: {output_dir}", flush=True)
            continue
        print(" ".join(map(str, command)), flush=True)
        if args.dry_run:
            continue
        plan["candidates"][name]["status"] = "running"
        write_json(plan_path, plan)
        subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)
        result = completed_result(output_dir)
        if result is None:
            raise RuntimeError(f"Training returned without complete artifacts: {name}")
        plan["candidates"][name].update({"status": "complete", "result": result})
        write_json(plan_path, plan)


if __name__ == "__main__":
    main()
