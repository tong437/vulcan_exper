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

"""Reuse 1000 frozen requests and prepare static calibration-based pruning plans."""

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from neuron_typing_generalization.ablation_runner import _write_json_atomic, evaluation_prompt
from neuron_typing_generalization.structured_throughput import calibration_costs, make_plan
from prepare_advisor_2026_09_assets import SPECS


ROOT = Path("artifacts/neuron_typing_generalization/structured_throughput_1000_2026_10")
MODELS = ("qwen3_vl_2b_instruct", "llava_1_5_7b", "qwen3_5_2b", "gemma_4_e2b_it")
REPLAY = Path("artifacts/neuron_typing_generalization/advisor_2026_09/replay.E84Qcw")
STATS = Path("artifacts/neuron_typing_generalization/control_response_2026_10_frozen")


def prepare_data(name):
    """Reuse the exact previous 1000 requests in frozen order; no new selection."""
    replay = REPLAY / f"{name}_1000.jsonl"
    selected = [json.loads(line) for line in replay.read_text().splitlines()]
    digest = hashlib.sha256(replay.read_bytes()).hexdigest()
    if len(selected) != 1000 or len({r["sample_id"] for r in selected}) != 1000:
        raise ValueError("Expected 1000 unique frozen requests")
    path = ROOT / "datasets" / name / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["count"] != 1000 or old["calibration_replay_sha256"] != digest:
            raise ValueError("Frozen workload changed")
        return
    image_dir = Path("artifacts/neuron_typing_generalization/images") / name
    image_index = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix != ".parquet"}
    rows = []
    for sample in selected:
        target = image_index[sample["image_sha256"]]
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if sha != sample["image_sha256"]:
            raise ValueError("Frozen replay image mismatch")
        choices = sample.get("choices", {})
        if name == "mmbench" and not choices:
            raise ValueError("Frozen MMBench choices missing")
        rows.append(
            {
                **sample,
                "image_path": str(target.resolve()),
                "prompt": evaluation_prompt(name, sample, {"choices": choices}),
                "choices": choices,
                "answer": sample.get("answers", sample.get("answer")),
            }
        )
    _write_json_atomic(
        path,
        {
            "dataset": name,
            "count": 1000,
            "rows": rows,
            "sampling": "existing frozen 1000 requests; original order; drop last incomplete batch",
            "calibration_replay_sha256": digest,
            "calibration_id_overlap": 1000,
        },
    )
    print(f"Reused {name}: 1000 frozen requests", flush=True)


def prepare_plans(model, dataset):
    output = ROOT / "plans" / f"{model}__{dataset}"
    costs, provenance = calibration_costs(STATS / f"{model}__{dataset}")
    for method in ("original", "uniform", "nonuniform"):
        for percent in [0] if method == "original" else range(5, 51, 5):
            plan = make_plan(costs, percent / 100, method)
            plan.update(model=model, dataset=dataset, calibration_sha256=provenance["statistics_sha256"])
            destination = output / f"{method}_{percent:02d}.json"
            if destination.exists() and json.loads(destination.read_text()) != plan:
                raise ValueError(f"Existing plan changed: {destination}")
            _write_json_atomic(destination, plan)
    _write_json_atomic(output / "calibration.json", provenance)
    print(f"Prepared 21 architectures: {model}/{dataset}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="qwen3_vl_2b_instruct")
    parser.add_argument("--dataset", choices=SPECS, default="mmbench")
    parser.add_argument("--plans-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.plans_only and args.data_only:
        parser.error("Choose one preparation mode")
    if not args.data_only:
        prepare_plans(args.model, args.dataset)
    if not args.plans_only:
        prepare_data(args.dataset)


if __name__ == "__main__":
    main()
