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
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml
from neuron_typing_generalization.materialize import materialize_selected_images


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize images referenced by frozen pilot manifests only.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/neuron_typing_generalization",
    )
    args = parser.parse_args()
    for config_path in sorted((EXPERIMENT_ROOT / "configs/datasets").glob("*.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        manifest = EXPERIMENT_ROOT / "manifests" / config["name"] / "pilot/sample_manifest.parquet"
        status_path = materialize_selected_images(config_path, manifest, args.artifact_root.resolve())
        print(json.dumps(json.loads(status_path.read_text(encoding="utf-8")), ensure_ascii=False))


if __name__ == "__main__":
    main()
