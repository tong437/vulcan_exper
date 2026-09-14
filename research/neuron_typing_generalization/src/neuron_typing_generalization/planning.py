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

"""CLI helpers for inspecting and initializing the experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import expand_study, load_study
from .contracts import STAGE_CONTRACTS
from .runtime import build_run_id, initialize_run


def build_parser(description: str, include_stage: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--study", type=Path, required=True, help="Path to a study YAML file.")
    parser.add_argument(
        "--initialize", action="store_true", help="Create run directories and provenance files; otherwise only print."
    )
    if include_stage:
        parser.add_argument(
            "--check-inputs", action="store_true", help="Report missing inputs required by this stage."
        )
    return parser


def run_matrix_cli() -> None:
    """Print or initialize every run in a study matrix."""
    args = build_parser("Plan a neuron typing generalization study.").parse_args()
    specs = expand_study(load_study(args.study))
    rows = []
    for spec in specs:
        context = initialize_run(spec) if args.initialize else None
        rows.append(
            {
                "run_id": context.run_id if context else build_run_id(spec),
                "model": spec.model.name,
                "dataset": spec.dataset.name,
                "seed": spec.seed,
                "metrics": [metric.name for metric in spec.metrics],
                "run_dir": str(context.run_dir) if context else None,
            }
        )
    print(json.dumps({"study": specs[0].study if specs else None, "run_count": len(rows), "runs": rows}, indent=2))


def run_stage_cli(stage: str) -> None:
    """Inspect a stage across the matrix and optionally initialize its run directories."""
    args = build_parser(f"Prepare the {stage} stage.", include_stage=True).parse_args()
    contract = STAGE_CONTRACTS[stage]
    rows = []
    for spec in expand_study(load_study(args.study)):
        context = initialize_run(spec) if args.initialize or args.check_inputs else None
        missing = []
        if args.check_inputs and context:
            missing = [item for item in contract.requires if not (context.run_dir / item).exists()]
        rows.append(
            {
                "run_id": context.run_id if context else build_run_id(spec),
                "requires": contract.requires,
                "produces": contract.produces,
                "missing_inputs": missing,
                "ready": not missing,
            }
        )
    print(json.dumps({"stage": stage, "runs": rows}, indent=2))
