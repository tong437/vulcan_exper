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

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from neuron_typing_generalization.benchmark import run_scale_benchmark


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-dimension synthetic CPU scale check.")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--d-ff", type=int, default=6144)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    report = run_scale_benchmark(args.samples, args.layers, args.d_ff, args.chunk_size)
    output = EXPERIMENT_ROOT / "reports/cpu_scale_benchmark.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
