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

"""Audit exact weights, identical-input MLPs, layer propagation and restored widths."""

import copy
import hashlib
import json

import run_structured_throughput_2026_10 as runner
import torch
from neuron_typing_generalization.structured_throughput import decoder_mlps
from transformers import Qwen3_5ForConditionalGeneration


torch.manual_seed(20261022)
torch.backends.cuda.matmul.allow_tf32 = False
root = runner.ROOT
plan = json.loads((root / "plans/qwen3_5_2b__mme/nonuniform_10.json").read_text())
rows = json.loads((root / "datasets/mme/manifest.json").read_text())["rows"][:2]
spec = runner._select_spec(runner.STUDY, "qwen3_5_2b", "mme")
layout = runner.load_model_layout(spec.model.values)
processor = runner.load_local_processor_adapter(spec.model.values, use_fast=False).processor
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
model = (
    Qwen3_5ForConditionalGeneration.from_pretrained(
        layout.model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa"
    )
    .eval()
    .to("cuda:0")
)
originals = [copy.deepcopy(m).cpu() for m in decoder_mlps(model)]
inputs = runner._move_generation_inputs(runner.batch_inputs(processor, rows), torch.device("cuda:0"), torch.bfloat16)
removed = runner.removed_indices(plan)


def stats(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return {
        "equal": torch.equal(a, b),
        "unequal_fraction": float((a != b).float().mean()),
        "max_abs": float((a - b).abs().max()),
        "relative_rms": float((a - b).square().mean().sqrt() / a.square().mean().sqrt().clamp_min(1e-12)),
        "per_sample_max_abs": [(x-y).abs().max().item() for x,y in zip(a,b)],
    }


def capture():
    records = {}
    handles = []

    def before(i):
        def hook(m, args):
            records[(i, "in")] = args[0].detach().cpu().clone()

        return hook

    def after(i):
        def hook(m, args, output):
            records[(i, "out")] = output.detach().cpu().clone()

        return hook

    for i, m in enumerate(decoder_mlps(model)):
        handles.extend([m.register_forward_pre_hook(before(i)), m.register_forward_hook(after(i))])
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits.detach().float().cpu()
    for h in handles:
        h.remove()
    return logits, records


with runner.DownProjectionAblation(runner.discover_decoder_ffns(model), removed):
    ref, ref_records = capture()
    repeat, _ = capture()
runner.physically_prune(model, plan)
actual, actual_records = capture()
layers = []
with torch.inference_mode():
    for i, (original, pruned) in enumerate(zip(originals, decoder_mlps(model))):
        original.to("cuda:0")
        keep = torch.tensor(plan["kept"][i], device="cuda:0")
        delete = torch.tensor(removed[i], device="cuda:0")
        x = ref_records[(i, "in")].to("cuda:0")
        exact = {
            name: torch.equal(getattr(pruned, name).weight, getattr(original, name).weight.index_select(dim, keep))
            for name, dim in [("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)]
        }
        g, u = original.gate_proj(x), original.up_proj(x)
        gp, up = pruned.gate_proj(x), pruned.up_proj(x)
        activation = original.act_fn(g) * u
        masked_activation = activation.clone().index_fill_(-1, delete, 0)
        y = original.down_proj(masked_activation)
        yp = pruned(x)
        shared_down = pruned.down_proj(activation.index_select(-1, keep))
        item = {
            "layer": i,
            "weights_exact": exact,
            "width": plan["original_widths"][i],
            "kept_width": len(keep),
            "input_shape": list(x.shape),
            "same_input_gate": stats(g.index_select(-1, keep), gp),
            "same_input_up": stats(u.index_select(-1, keep), up),
            "same_activation_down": stats(y, shared_down),
            "same_input_mlp": stats(y, yp),
            "end_to_end_input": stats(ref_records[(i, "in")], actual_records[(i, "in")]),
            "end_to_end_output": stats(ref_records[(i, "out")], actual_records[(i, "out")]),
        }
        # Same retained BF16 weights/activations, but FP32 reduction isolates down-projection rounding.
        yf = torch.nn.functional.linear(masked_activation.float(), original.down_proj.weight.float())
        ypf = torch.nn.functional.linear(activation.index_select(-1, keep).float(), pruned.down_proj.weight.float())
        item["same_activation_down_fp32"] = stats(yf, ypf)
        # Zero only removed weight columns; original shape is preserved.
        original.down_proj.weight.index_fill_(1, delete, 0)
        item["zero_weight_vs_zero_activation_same_shape"] = stats(y, original(x))
        layers.append(item)
        original.cpu()
        print("Layer", i, json.dumps(item), flush=True)
    # Restore original GEMM shapes, retaining the same mathematical deleted contributions.
    for original, pruned in zip(originals, decoder_mlps(model)):
        original.to("cuda:0")
        pruned.gate_proj = original.gate_proj
        pruned.up_proj = original.up_proj
        pruned.down_proj = original.down_proj
        pruned.intermediate_size = original.intermediate_size
padded, _ = capture()
result = {
    "status": "diagnostic_only",
    "batch_size": 2,
    "sample_ids": [r["sample_id"] for r in rows],
    "repeat_logits": stats(ref, repeat),
    "sliced_logits": stats(ref, actual),
    "restored_width_zero_weight_logits": stats(ref, padded),
    "layers": layers,
    "masked_argmax": ref.argmax(-1).tolist(),
    "sliced_argmax": actual.argmax(-1).tolist(),
    "restored_width_argmax": padded.argmax(-1).tolist(),
    "input_rows": rows,
    "input_tensors": {
        k: {"shape": list(v.shape), "dtype": str(v.dtype),
            "sha256": hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            "values": v.cpu().tolist() if k != "pixel_values" and v.numel() < 1000 else None}
        for k,v in inputs.items() if isinstance(v, torch.Tensor)
    },
    "decoded_input": processor.tokenizer.batch_decode(inputs["input_ids"].cpu(), skip_special_tokens=False),
    "generation_config": model.generation_config.to_dict(),
    "first_token_candidates": {
        label: {"ids": value[0, -1].topk(5).indices.tolist(), "logits": value[0, -1].topk(5).values.tolist()}
        for label,value in [("masked",ref),("sliced",actual),("restored",padded)]
    },
}
runner._write_json_atomic(root / "diagnostics_qwen35_deep" / "sample_trace.json", result)
print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2), flush=True)
