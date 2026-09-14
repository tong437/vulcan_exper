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
from pathlib import Path

import _bootstrap  # noqa: F401
from neuron_typing_generalization.sources import fetch_all_sources


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch bounded, non-image dataset metadata.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=EXPERIMENT_ROOT.parents[1] / "artifacts/neuron_typing_generalization",
    )
    args = parser.parse_args()
    for snapshot in fetch_all_sources(EXPERIMENT_ROOT / "configs/datasets", args.artifact_root.resolve()):
        print(snapshot)


if __name__ == "__main__":
    main()
