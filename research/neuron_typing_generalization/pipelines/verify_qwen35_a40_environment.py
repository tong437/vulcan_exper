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

"""Verify actual Qwen3.5 CUDA fast-path calls on a synthetic image request."""

import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation


assert implementation.is_fast_path_available, "Fast path unavailable"
calls = Counter()


def counted(name, function):
    def wrapped(*args, **kwargs):
        calls[name] += 1
        return function(*args, **kwargs)

    return wrapped


for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
    setattr(implementation, name, counted(name, getattr(implementation, name)))

model_path = "/root/autodl-pub-RTX4090-hdd-1/models/Qwen3.5-2B"
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = (
    Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .eval()
    .to("cuda:0")
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": Image.new("RGB", (224, 224), color="red")},
            {"type": "text", "text": "What color is this image? Answer briefly."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
)
inputs = {
    k: v.to(device="cuda:0", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k, v in inputs.items()
}
with torch.inference_mode():
    output = model.generate(**inputs, do_sample=False, min_new_tokens=4, max_new_tokens=4, use_cache=True)
torch.cuda.synchronize()
assert all(
    calls[name] > 0
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    )
), calls
result = {
    "status": "passed",
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "versions": {
        name: version(name) for name in ("transformers", "flash-linear-attention", "fla-core", "causal-conv1d")
    },
    "kernel_calls": dict(calls),
    "generated_ids": output[:, inputs["input_ids"].shape[1] :].tolist(),
    "scope": "Synthetic image, BF16 prefill and cached decoding; not throughput or quality measurement",
}
path = Path(
    "artifacts/neuron_typing_generalization/structured_throughput_1000_2026_10/logs/a40_environment_verified.json"
)
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
