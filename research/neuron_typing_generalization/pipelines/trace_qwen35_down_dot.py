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

"""Trace the first layer-0 differing down-projection element; no benchmark changes."""

import csv
import json
import math

import numpy as np
import run_structured_throughput_2026_10 as runner
import torch
from neuron_typing_generalization.structured_throughput import decoder_mlps
from transformers import Qwen3_5ForConditionalGeneration


torch.manual_seed(20261022)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
root = runner.ROOT
plan = json.loads((root / "plans/qwen3_5_2b__mme/nonuniform_10.json").read_text())
rows = json.loads((root / "datasets/mme/manifest.json").read_text())["rows"][:2]
spec = runner._select_spec(runner.STUDY, "qwen3_5_2b", "mme")
layout = runner.load_model_layout(spec.model.values)
processor = runner.load_local_processor_adapter(spec.model.values, use_fast=False).processor
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    layout.model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa"
).eval().to("cuda:0")
inputs = runner._move_generation_inputs(runner.batch_inputs(processor, rows), torch.device("cuda:0"), torch.bfloat16)
original = decoder_mlps(model)[0].down_proj
weight = original.weight.detach().clone()
assert original.bias is None


def capture():
    down = decoder_mlps(model)[0].down_proj
    captured = {}

    def hook(module, args, output):
        captured["input"] = args[0].detach().clone()
        captured["output"] = output.detach().clone()

    handle = down.register_forward_hook(hook)
    with torch.inference_mode():
        model(**inputs, use_cache=False, logits_to_keep=1)
    handle.remove()
    return captured


with runner.DownProjectionAblation(runner.discover_decoder_ffns(model), runner.removed_indices(plan)):
    masked = capture()
runner.physically_prune(model, plan)
sliced = capture()
keep = torch.tensor(plan["kept"][0], device="cuda:0")
assert torch.equal(masked["input"].index_select(-1, keep), sliced["input"])
assert torch.equal(weight.index_select(1, keep), decoder_mlps(model)[0].down_proj.weight)
coordinates = (masked["output"][0] != sliced["output"][0]).nonzero()
t, o = coordinates[0].tolist()
a = masked["input"][0, t].double().cpu().numpy()
w = weight[o].double().cpu().numpy()
products = a * w
k = keep.cpu().numpy()
reference = math.fsum(products.tolist())
assert reference == math.fsum(products[k].tolist())
ym, ys = [float(v["output"][0, t, o]) for v in (masked, sliced)]
reference_bf16 = float(torch.tensor(reference, dtype=torch.float64).to(torch.bfloat16))
out = root / "diagnostics_qwen35_deep/down_dot"
out.mkdir(parents=True, exist_ok=True)
rank = {int(j): i for i, j in enumerate(k)}
with (out / "products.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["original_channel", "kept_rank", "activation_bf16_exact", "weight_bf16_exact", "product_fp64"])
    writer.writerows((j, rank.get(j, "removed"), a[j], w[j], products[j]) for j in range(len(a)))
np.savez(out / "dot_operands.npz", activation=a, weight=w, products=products, kept=k)
with torch.inference_mode():
    fm = torch.nn.functional.linear(masked["input"].float(), weight.float())
    fs = torch.nn.functional.linear(sliced["input"].float(), weight.index_select(1, keep).float())
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    pm = torch.nn.functional.linear(masked["input"], weight)
    ps = torch.nn.functional.linear(sliced["input"], weight.index_select(1, keep))
result = {
    "sample_id": rows[0]["sample_id"], "batch_sample_ids": [r["sample_id"] for r in rows],
    "coordinate": [0, t, o], "coordinate_order": "sample, input token position, output channel (zero-based)",
    "input_token_id": int(inputs["input_ids"][0, t]),
    "input_token": processor.tokenizer.decode([int(inputs["input_ids"][0, t])]),
    "masked_shape": list(masked["input"].shape), "sliced_shape": list(sliced["input"].shape),
    "retained_inputs_and_weights_bitwise_equal": True, "bias": None,
    "masked_bf16": ym, "sliced_bf16": ys, "signed_difference": ys-ym,
    "fp64_fsum_reference": reference, "fp64_numpy_sum": float(products.sum()),
    "fp64_reference_rounded_to_bf16": reference_bf16,
    "masked_error_vs_fp64": ym-reference, "sliced_error_vs_fp64": ys-reference,
    "midpoint_between_outputs": (ym+ys)/2, "reference_minus_midpoint": reference-(ym+ys)/2,
    "positive_product_sum_fp64": math.fsum(products[products > 0].tolist()),
    "negative_product_sum_fp64": math.fsum(products[products < 0].tolist()),
    "full_shape_fp32_gemm": float(fm[0,t,o]), "sliced_shape_fp32_gemm": float(fs[0,t,o]),
    "full_shape_bf16_no_reduced_reduction": float(pm[0,t,o]),
    "sliced_shape_bf16_no_reduced_reduction": float(ps[0,t,o]),
    "example_products": [{"j": int(j), "activation": float(a[j]), "weight": float(w[j]), "product": float(products[j])} for j in k[:5]],
    "scope": "FP64 reference uses the actual BF16 operands, not an FP64 model; GEMM internal accumulation order is not observed",
}
runner._write_json_atomic(out / "summary.json", result)
print(json.dumps(result, indent=2), flush=True)
