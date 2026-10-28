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

"""Sequential pilot -> audit gate -> full sweep, on one exclusive idle GPU."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from neuron_typing_generalization.ablation_runner import _write_json_atomic
from prepare_throughput_2026_10 import ROOT, SPECS
from run_structured_throughput_2026_10 import fingerprint


PIPE = Path(__file__).resolve().parent
ASSIGNMENTS = {
    0: ("qwen3_vl_2b_instruct",),
    1: ("llava_1_5_7b",),
    2: ("qwen3_5_2b",),
    3: ("gemma_4_e2b_it",),
}


def idle(gpu):
    for _ in range(15):
        output = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        memory, utilization = map(int, output.strip().split(","))
        if memory <= 1024 and utilization <= 10:
            return
        time.sleep(2)
    raise RuntimeError(f"GPU{gpu} is busy ({memory} MiB, {utilization}%); retry after it is idle")


def python_for(model):
    return (
        os.environ.get("THROUGHPUT_QWEN35_PYTHON", "/root/.cache/vulcan-qwen35-fastpath-torch26/bin/python")
        if model == "qwen3_5_2b"
        else sys.executable
    )


def prepare(model, dataset):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / f"prepare_{dataset}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(
            [python_for(model), str(PIPE / "prepare_throughput_2026_10.py"), "--model", model, "--dataset", dataset],
            env={**os.environ, "HF_ENDPOINT": "https://huggingface.co", "HF_HUB_DISABLE_XET": "1"},
            check=True,
        )


def run(gpu, model, dataset, method, percent, *extra):
    idle(gpu)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "WANDB_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    subprocess.run(
        [
            python_for(model),
            str(
                PIPE
                / (
                    "run_qwen35_throughput_precision_2026_10.py"
                    if model == "qwen3_5_2b"
                    else "run_structured_throughput_2026_10.py"
                )
            ),
            "--model",
            model,
            "--dataset",
            dataset,
            "--method",
            method,
            "--percent",
            str(percent),
            *extra,
        ],
        env=env,
        check=True,
    )


def correctness(gpu, model, dataset, rates=(10, 30)):
    for method in ("uniform", "nonuniform"):
        for percent in rates:
            run(gpu, model, dataset, method, percent, "--verify-only", "--verify-fp32")
            run(gpu, model, dataset, method, percent, "--verify-only")


def pilot(gpu):
    model, dataset = "qwen3_vl_2b_instruct", "mme"
    prepare(model, dataset)
    correctness(gpu, model, dataset, rates=(30,))
    variants = [("original", 0), ("uniform", 30), ("nonuniform", 30)]
    evidence = {}
    for method, percent in variants:
        run(gpu, model, dataset, method, percent, "--pilot", "--batches", "1", "4")
        for mode, sizes in (("natural", (1, 4)),):
            for size in sizes:
                path = ROOT / "pilot" / f"{model}__{dataset}" / mode / f"{method}_{percent:02d}_b{size:02d}.json"
                row = json.loads(path.read_text())
                if row["status"] != "complete" or len(row["measurements"]) != 1:
                    raise ValueError(f"Pilot incomplete: {path}")
                if any(m["requests"] != 32 or m["req_per_second"] <= 0 for m in row["measurements"]):
                    raise ValueError("Invalid pilot throughput")
                evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json_atomic(
        ROOT / "pilot_passed.json",
        {
            "status": "passed",
            "code_sha256": fingerprint(),
            "model": model,
            "dataset": dataset,
            "evidence_sha256": evidence,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, choices=tuple(ASSIGNMENTS))
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    idle(args.gpu)
    if args.gpu == 0:
        pilot(args.gpu)
    else:
        print(f"GPU{args.gpu} waiting for GPU0 pilot gate; no full benchmark before pilot passes", flush=True)
        while True:
            gate_path = ROOT / "pilot_passed.json"
            if gate_path.exists():
                gate = json.loads(gate_path.read_text())
                if gate["status"] == "passed" and gate["code_sha256"] == fingerprint():
                    break
            time.sleep(30)
    if args.pilot_only:
        return
    print(f"GPU{args.gpu} assigned models: {ASSIGNMENTS[args.gpu]}", flush=True)
    for model in ASSIGNMENTS[args.gpu]:
        for dataset in SPECS:
            prepare(model, dataset)
            correctness(args.gpu, model, dataset)
            variants = [("original", 0)] + [
                (m, p)
                for p in range(5, 51, 5)
                for m in (("uniform", "nonuniform") if p % 10 else ("nonuniform", "uniform"))
            ]
            # Paired U/N conditions alternate order across adjacent rates to reduce order drift.
            for method, percent in variants:
                run(args.gpu, model, dataset, method, percent)
    _write_json_atomic(
        ROOT / f"worker_gpu{args.gpu}_complete.json",
        {
            "code_sha256": fingerprint(),
            "models": list(ASSIGNMENTS[args.gpu]),
            "status": "complete",
        },
    )
    with (ROOT / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for gpu, models in ASSIGNMENTS.items():
            path = ROOT / f"worker_gpu{gpu}_complete.json"
            if not path.exists():
                return
            record = json.loads(path.read_text())
            if record != {"code_sha256": fingerprint(), "models": list(models), "status": "complete"}:
                return
        subprocess.run([sys.executable, str(PIPE / "report_throughput_2026_10.py")], check=True)


if __name__ == "__main__":
    main()
