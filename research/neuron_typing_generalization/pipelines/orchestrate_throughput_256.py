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

"""Isolated, resumable 256-request sweep on one GPU; historical results are read-only."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import prepare_throughput_2026_10 as prepare
from neuron_typing_generalization.ablation_runner import _write_json_atomic
from run_structured_throughput_2026_10 import fingerprint


OLD = Path("artifacts/neuron_typing_generalization/structured_throughput_1000_2026_10")
ROOT = OLD.with_name("structured_throughput_256_2026_10")
PIPE = Path(__file__).resolve().parent
MODELS = ["llava_1_5_7b", "gemma_4_e2b_it", "qwen3_5_2b", "qwen3_vl_2b_instruct"]
DATASETS = ["mme", "mmbench", "textvqa", "chartqa"]


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen configuration differs: {path}")
    else:
        _write_json_atomic(path, value)


def prepare_dataset(model, dataset):
    source = OLD / "datasets" / dataset / "manifest.json"
    old = json.loads(source.read_text())
    assert old["count"] == len(old["rows"]) == 1000
    rows = old["rows"][:256]
    assert len({r["sample_id"] for r in rows}) == 256
    immutable_json(ROOT / "datasets" / dataset / "manifest.json", {
        "dataset": dataset, "count": 256, "rows": rows,
        "sampling": "first 256 frozen requests, identical order across all models and conditions",
        "source_manifest": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "calibration": "Unchanged 1000-sample scores; only timed workload reduced",
    })
    target = ROOT / "plans" / f"{model}__{dataset}"
    old_plans = OLD / "plans" / target.name
    names = ["original_00.json"] + [f"{m}_{p:02}.json" for m in ("uniform", "nonuniform") for p in range(5, 51, 5)]
    if all((old_plans / name).exists() for name in names):
        for name in names:
            immutable_json(target / name, json.loads((old_plans / name).read_text()))
    elif not all((target / name).exists() for name in names):
        prepare.ROOT = ROOT
        prepare.prepare_plans(model, dataset)


def run(gpu, model, dataset, method, percent, *extra):
    python = "/root/.cache/vulcan-qwen35-fastpath-torch26/bin/python" if model == "qwen3_5_2b" else sys.executable
    script = "run_qwen35_throughput_precision_2026_10.py" if model == "qwen3_5_2b" else "run_structured_throughput_2026_10.py"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "8",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "WANDB_DISABLED": "true", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "THROUGHPUT_OUTPUT_ROOT": str(ROOT), "THROUGHPUT_WORKLOAD_SAMPLES": "256",
           "THROUGHPUT_DIRECT_TIMING": "1",
           "THROUGHPUT_LEGACY_CODE_SHA256": json.loads((ROOT / "experiment.json").read_text())["runner_fingerprint"]}
    command = [python, str(PIPE / script), "--model", model, "--dataset", dataset,
               "--method", method, "--percent", str(percent), *extra]
    with (ROOT / "events.jsonl").open("a") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": "start", "command": command}) + "\n")
    with (ROOT / "logs" / f"{model}.log").open("a") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (ROOT / "events.jsonl").open("a") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": "finished", "command": command}) + "\n")


def correctness(gpu, model, dataset, rates=(10, 30)):
    for method in ("uniform", "nonuniform"):
        for rate in rates:
            run(gpu, model, dataset, method, rate, "--verify-only", "--verify-fp32")
            run(gpu, model, dataset, method, rate, "--verify-only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    args = parser.parse_args()
    if args.model and args.models:
        parser.error("Choose --model or --models")
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with (ROOT / f"worker_gpu{args.gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory, utilization = map(int, subprocess.check_output(
            ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True
        ).strip().split(","))
        if memory > 1024 or utilization > 10:
            raise RuntimeError(f"GPU busy: {memory} MiB, {utilization}%")
        immutable_json(ROOT / "direct_timing_policy_20261025.json", {
            "samples": 256, "calibration_samples": 1000, "models_in_order": MODELS, "datasets": DATASETS,
            "rates": list(range(5, 51, 5)), "batches": [1, 2, 4, 8, 16, 32, 64], "repeats": 3,
            "max_new_tokens": 32, "selection": "frozen first 256", "old_results_read_only": str(OLD),
            "runner_fingerprint": fingerprint(), "verification_policy": "User authorized direct timing, no generation equivalence gate",
            "gpu_groups": [["llava_1_5_7b", "qwen3_vl_2b_instruct"], ["gemma_4_e2b_it", "qwen3_5_2b"]],
        })
        model_locks = []
        selected = [args.model] if args.model else (args.models or MODELS)
        for model in selected:
            model_lock = (ROOT / f"{model}.lock").open("a")
            fcntl.flock(model_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            model_locks.append(model_lock)
        for model in selected:
            for dataset in DATASETS:
                prepare_dataset(model, dataset)
                variants = [("original", 0)] + [(m, p) for p in range(5, 51, 5)
                    for m in (("uniform", "nonuniform") if p % 10 else ("nonuniform", "uniform"))]
                for method, rate in variants:
                    run(args.gpu, model, dataset, method, rate)
            _write_json_atomic(ROOT / f"{model}_complete.json", {"status": "complete", "code_sha256": fingerprint()})


if __name__ == "__main__":
    main()
