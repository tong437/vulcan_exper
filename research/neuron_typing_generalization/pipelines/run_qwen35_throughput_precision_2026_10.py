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

"""Qwen3.5-only precision policy, configured before importing Torch/Triton.

Keep the shared runner unchanged so other live workers retain their code hashes.
FP32 verification uses IEEE dots. BF16 retains default accelerated computation.
Both precision modes use separate versioned caches to avoid stale TF32 kernels.
"""

import hashlib
import os
import sys
from pathlib import Path


def main():
    fp32 = "--verify-fp32" in sys.argv or "--diagnose-fp32" in sys.argv
    if "--model" not in sys.argv or sys.argv[sys.argv.index("--model") + 1] != "qwen3_5_2b":
        raise ValueError("This precision entry point is Qwen3.5 only")
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if fp32:
        os.environ["TRITON_F32_DEFAULT"] = "ieee"
    else:
        os.environ.pop("TRITON_F32_DEFAULT", None)
    mode = "fp32_ieee" if fp32 else "bf16_default"
    cache = Path.home() / ".cache/vulcan-triton-throughput" / code_hash / mode
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)

    import run_structured_throughput_2026_10 as runner

    runtime = runner.runtime
    write = runner._write_json_atomic
    verify = runner.verify

    def approved_verify(model, processor, plan, rows, dtype, *, observe=False):
        if dtype != runner.torch.bfloat16 or observe:
            return verify(model, processor, plan, rows, dtype, observe=observe)
        result = verify(model, processor, plan, rows, dtype, observe=True)
        # "passed" denotes this explicitly amended benchmark gate, not token equivalence.
        result.update(
            status="passed",
            acceptance_policy="user_approved_bf16_sequence_mismatch_2026_10_23",
            bf16_sequence_equivalence_passed=all(c["generated_tokens_equal"] for c in result["comparisons"]),
            approval_note="User requested resuming Qwen3.5 despite known sequence differences; FP32 gate retained",
        )
        return result

    def policy_runtime():
        return {
            **runtime(),
            "qwen35_precision_wrapper_sha256": code_hash,
            "precision_policy": "FP32 IEEE / BF16 default; isolated Triton caches",
            "verification_policy_override": "FP32 required; BF16 differences recorded and user-accepted 2026-10-23",
        }

    def write_with_precision(path, record):
        write(
            path,
            {
                **record,
                "precision_execution": {
                    "mode": mode,
                    "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT", "default"),
                    "triton_cache_dir": str(cache),
                    "wrapper_sha256": code_hash,
                },
            },
        )

    runner.runtime = policy_runtime
    runner.verify = approved_verify
    runner._write_json_atomic = write_with_precision
    if fp32:
        runner.torch.backends.cudnn.allow_tf32 = False
    runner.main()


if __name__ == "__main__":
    main()
