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

"""Run one model-dataset response-state ablation matrix on a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from neuron_typing_generalization.ablation_runner import run_ablation_generation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--removal-percent", type=float, default=1.0)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--insufficient-pool", choices=("error", "all"), default="error")
    args = parser.parse_args()
    if not 0.0 < args.removal_percent <= 100.0:
        parser.error("--removal-percent must lie in (0, 100].")
    result = run_ablation_generation(
        args.study,
        args.model,
        args.dataset,
        args.output_dir,
        replay_dir=args.replay_dir,
        removal_fraction=args.removal_percent / 100.0,
        random_repeats=args.random_repeats,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
        seed=args.seed,
        insufficient_pool=args.insufficient_pool,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
