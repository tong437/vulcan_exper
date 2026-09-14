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
from neuron_typing_generalization.execution import compute_importance_run
from neuron_typing_generalization.planning import run_stage_cli


def main() -> None:
    if "--run-dir" not in sys.argv:
        run_stage_cli("compute_importance")
        return
    parser = argparse.ArgumentParser(description="Compute normalized importance from reduced activation statistics.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compute_importance_run(args.run_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
