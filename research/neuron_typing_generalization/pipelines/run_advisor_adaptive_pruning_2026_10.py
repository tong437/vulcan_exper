#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run one sample-adaptive A1/A2 ablation with frozen multimodal prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from neuron_typing_generalization.adaptive_pruning_runner import run_adaptive_pruning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--statistics-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--a2-only", action="store_true", help="Evaluate the score-budget method without A1.")
    parser.add_argument("--nonuniform", action="store_true", help="Evaluate nonuniform maximum-count budget pruning.")
    parser.add_argument("--verify-clean", action="store_true", help="Verify clean generation before/after each mask.")
    parser.add_argument("--matched-count", action="store_true", help="Match each frozen uniform deletion count.")
    parser.add_argument("--uniform-source", type=Path)
    parser.add_argument("--layer-scan", action="store_true")
    parser.add_argument("--scan-smoke", action="store_true")
    args = parser.parse_args()
    if sum((args.a2_only, args.nonuniform, args.matched_count, args.layer_scan)) > 1:
        parser.error("Choose only one selection method")
    if args.matched_count != (args.uniform_source is not None):
        parser.error("--matched-count and --uniform-source must be supplied together")
    result = run_adaptive_pruning(
        args.study,
        args.model,
        args.dataset,
        args.statistics_root,
        args.baseline_root,
        args.output_dir,
        replay_dir=args.replay_dir,
        epsilon=args.epsilon,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
        methods=("N_match",)
        if args.matched_count
        else (("N",) if args.nonuniform else (("A2",) if args.a2_only else ("A1", "A2"))),
        verify_clean=args.verify_clean,
        uniform_source=args.uniform_source,
        layer_scan=args.layer_scan,
        scan_smoke=args.scan_smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
