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

"""Independent FP32 backend diagnostic; never writes a verification gate."""

import argparse
import json
import os

import run_structured_throughput_2026_10 as runner
import torch
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation


args = argparse.ArgumentParser()
args.add_argument("--backend", choices=("fast", "reference"), required=True)
options = args.parse_args()
torch.manual_seed(20261022)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if options.backend == "reference":
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    ):
        setattr(implementation, name, None)
    implementation.is_fast_path_available = False
root = runner.ROOT
plan = json.loads((root / "plans/qwen3_5_2b__mme/uniform_10.json").read_text())
rows = json.loads((root / "datasets/mme/manifest.json").read_text())["rows"][:2]
spec = runner._select_spec(runner.STUDY, "qwen3_5_2b", "mme")
layout = runner.load_model_layout(spec.model.values)
processor = runner.load_local_processor_adapter(spec.model.values, use_fast=False).processor
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
model = (
    Qwen3_5ForConditionalGeneration.from_pretrained(
        layout.model_path, local_files_only=True, dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
    )
    .eval()
    .to("cuda:0")
)
proof = runner.verify(model, processor, plan, rows, torch.float32, observe=True)
precision = os.environ.get("TRITON_F32_DEFAULT", "default")
proof.update(
    backend=options.backend,
    runtime=runner.runtime(),
    purpose="diagnosis_only_not_gate",
    triton_f32_default=precision,
    triton_cache_dir=os.environ.get("TRITON_CACHE_DIR"),
)
destination = root / "diagnostics_qwen35_fp32" / f"{options.backend}_{precision}_fresh.json"
runner._write_json_atomic(destination, proof)
print(json.dumps(proof, indent=2), flush=True)
