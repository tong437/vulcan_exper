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

"""Gate-aware orchestration for Phase-4 cross-modal activation mapping."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete Phase-4 mapping pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--mapping_vqa_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--feature_rank", type=int, default=128)
    parser.add_argument("--target_rank", type=int, default=64)
    parser.add_argument("--alphas", default="0.1,1,10,100")
    parser.add_argument("--null_permutations", type=int, default=20)
    parser.add_argument("--min_mapping_r2", type=float, default=0.01)
    parser.add_argument("--min_incremental_r2", type=float, default=0.002)
    parser.add_argument("--max_null_p", type=float, default=0.05)
    parser.add_argument("--min_positive_layers", type=int, default=4)
    parser.add_argument(
        "--causal_vqa",
        action="append",
        default=[],
        help="Optional repeatable NAME=FILE. Defaults to mapping=--mapping_vqa_file.",
    )
    parser.add_argument("--causal_batch_size", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--stage", choices=["collect", "fit", "group", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _run(command: list[str], stage: str) -> None:
    print(f"\n[{stage}]\n{' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def run_phase4(args: argparse.Namespace) -> dict:
    root = Path(args.output_dir)
    activation_dir = root / "activations"
    fit_dir = root / "mapping"
    group_dir = root / "group_causality"
    root.mkdir(parents=True, exist_ok=True)

    if args.stage in {"collect", "all"}:
        state_path = activation_dir / "collection_state.json"
        collection_complete = False
        if state_path.exists():
            collection_complete = bool(json.loads(state_path.read_text(encoding="utf-8")).get("complete"))
        if collection_complete:
            print(f"Phase-4 collection already complete: {activation_dir}", flush=True)
        else:
            command = [
                sys.executable,
                str(SCRIPT_DIR / "collect_phase4_mapping_activations.py"),
                "--config",
                args.config,
                "--vqa_file",
                args.mapping_vqa_file,
                "--image_root",
                args.image_root,
                "--output_dir",
                str(activation_dir),
                "--batch_size",
                str(args.batch_size),
                "--chunk_size",
                str(args.chunk_size),
                "--seed",
                str(args.seed),
                "--filter_manifest_overlaps",
            ]
            if args.max_images is not None:
                command.extend(("--max_images", str(args.max_images)))
            if args.model_name_or_path:
                command.extend(("--model_name_or_path", args.model_name_or_path))
            if args.calibration_manifest:
                command.extend(("--calibration_manifest", args.calibration_manifest))
            if args.typing_manifest:
                command.extend(("--typing_manifest", args.typing_manifest))
            if args.calibration_manifest and args.typing_manifest:
                command.append("--require_data_isolation")
            if state_path.exists():
                command.append("--resume")
            _run(command, "P4.1 paired activation collection")

    metrics_path = fit_dir / "mapping_metrics.json"
    if args.stage in {"fit", "all"}:
        if metrics_path.exists():
            print(f"Phase-4 mapping fit already complete: {metrics_path}", flush=True)
        else:
            command = [
                sys.executable,
                str(SCRIPT_DIR / "fit_phase4_activation_mapping.py"),
                "--activation_dir",
                str(activation_dir),
                "--score_file",
                args.score_file,
                "--output_dir",
                str(fit_dir),
                "--feature_rank",
                str(args.feature_rank),
                "--target_rank",
                str(args.target_rank),
                "--alphas",
                args.alphas,
                "--null_permutations",
                str(args.null_permutations),
                "--min_mapping_r2",
                str(args.min_mapping_r2),
                "--min_incremental_r2",
                str(args.min_incremental_r2),
                "--max_null_p",
                str(args.max_null_p),
                "--min_positive_layers",
                str(args.min_positive_layers),
                "--seed",
                str(args.seed + 1),
            ]
            _run(command, "P4.2 activation mapping and Gate A/B")

    if args.stage in {"group", "all"}:
        if not metrics_path.exists():
            raise FileNotFoundError(f"Phase-4 mapping metrics are required for group ablation: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not metrics["gates"]["phase4_group_causal_ablation_allowed"]:
            raise RuntimeError(
                "Phase-4 Gate A/B failed. P4.3 group causal ablation and structural pruning are forbidden."
            )
        group_result = group_dir / "phase4_group_causality.json"
        if group_result.exists():
            print(f"Phase-4 group causal evaluation already complete: {group_result}", flush=True)
        else:
            command = [
                sys.executable,
                str(SCRIPT_DIR / "evaluate_phase4_group_causality.py"),
                "--config",
                args.config,
                "--mapping_metrics",
                str(metrics_path),
                "--activation_dir",
                str(activation_dir),
                "--image_root",
                args.image_root,
                "--output_dir",
                str(group_dir),
                "--batch_size",
                str(args.causal_batch_size),
                "--bootstrap_samples",
                str(args.bootstrap_samples),
                "--seed",
                str(args.seed + 2),
            ]
            causal_vqa = args.causal_vqa or [f"mapping={args.mapping_vqa_file}"]
            for value in causal_vqa:
                command.extend(("--vqa", value))
            if args.calibration_manifest:
                command.extend(("--calibration_manifest", args.calibration_manifest))
            if args.typing_manifest:
                command.extend(("--typing_manifest", args.typing_manifest))
            _run(command, "P4.3 held-out group causality and Gate C")

    result = {
        "complete": True,
        "stage": args.stage,
        "output_dir": str(root),
        "activation_state": str(activation_dir / "collection_state.json"),
        "mapping_metrics": str(metrics_path),
        "group_causality": str(group_dir / "phase4_group_causality.json"),
        "structural_pruning_automatic": False,
    }
    (root / "phase4_pipeline.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    result = run_phase4(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
