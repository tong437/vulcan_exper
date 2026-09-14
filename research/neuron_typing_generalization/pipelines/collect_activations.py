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

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from neuron_typing_generalization.collection import collect_synthetic_fixture
from neuron_typing_generalization.planning import run_stage_cli


def main() -> None:
    if "--run-dir" not in sys.argv:
        run_stage_cli("collect_activations")
        return
    parser = argparse.ArgumentParser(description="Collect reduced activations using an injected model adapter.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help="Exercise the production collection path with a deterministic CPU-only model.",
    )
    args = parser.parse_args()
    if not args.synthetic_fixture:
        parser.error(
            "A real model must be supplied through Python injection; CLI execution requires --synthetic-fixture."
        )
    print(json.dumps(collect_synthetic_fixture(args.run_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
