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
from neuron_typing_generalization.gpu_runner import run_gpu_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one resumable real-model frozen-dataset activation collection.")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--validate-direct-streaming", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--down-norms", type=Path)
    args = parser.parse_args()
    if args.validation_only and not args.validate_direct_streaming:
        parser.error("--validation-only requires --validate-direct-streaming")
    result = run_gpu_collection(
        args.study,
        args.model,
        args.dataset,
        args.max_samples,
        validate_streaming=args.validate_direct_streaming,
        validation_only=args.validation_only,
        replay_dir=args.replay_dir,
        down_norms_path=args.down_norms,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
