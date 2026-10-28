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

"""Diagnose nonuniform10 BF16 token divergence without changing benchmark gates."""

import argparse
import json

import run_structured_throughput_2026_10 as runner
import torch
from neuron_typing_generalization.structured_throughput import decoder_mlps
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=(
        "bf16",
        "mlp_fp32",
        "reference",
        "delta_reference",
        "conv_reference",
        "norm_reference",
        "norm_single_round",
        "norm_probe",
    ),
    required=True,
)
args = parser.parse_args()
torch.manual_seed(20261022)
torch.backends.cuda.matmul.allow_tf32 = False
if args.mode == "reference":
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    ):
        setattr(implementation, name, None)
    implementation.is_fast_path_available = False
elif args.mode == "delta_reference":
    implementation.chunk_gated_delta_rule = None
    implementation.fused_recurrent_gated_delta_rule = None
elif args.mode == "conv_reference":
    implementation.causal_conv1d_fn = None
    implementation.causal_conv1d_update = None
elif args.mode == "norm_reference":
    implementation.FusedRMSNormGated = None
elif args.mode == "norm_single_round":
    implementation.FusedRMSNormGated = None

    def single_round(self, x, gate=None):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (y * self.weight.float() * torch.nn.functional.silu(gate.float())).to(x.dtype)

    implementation.Qwen3_5RMSNormGated.forward = single_round
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
if args.mode == "mlp_fp32":
    for mlp in decoder_mlps(model):
        mlp.float()
        mlp.register_forward_pre_hook(lambda module, inputs: (inputs[0].float(),))
        mlp.register_forward_hook(lambda module, inputs, output: output.to(torch.bfloat16))
norm_probes = []
if args.mode == "norm_probe":
    seen = set()

    def probe(name):
        def hook(module, inputs, output):
            if name in seen:
                return
            seen.add(name)
            x, gate = [v.detach().reshape(-1, v.shape[-1])[-64:] for v in inputs[:2]]
            fused = output.detach().reshape(-1, output.shape[-1])[-64:]
            eps = module.eps
            ref = implementation.Qwen3_5RMSNormGated(x.shape[-1], eps=eps).to(x.device, dtype=x.dtype)
            ref.weight.data.copy_(module.weight.data)
            staged = ref(x, gate)
            y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            single = (y * module.weight.float() * torch.nn.functional.silu(gate.float())).to(x.dtype)
            xd, gd = x.double(), gate.double()
            oracle = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + eps)
            oracle = oracle * module.weight.double() * torch.nn.functional.silu(gd)
            norm_probes.append(
                {
                    "module": name,
                    "fused_staged_unequal_fraction": float((fused != staged).float().mean()),
                    "fused_single_unequal_fraction": float((fused != single).float().mean()),
                    "max_abs_error_vs_fp64": {
                        k: float((v.double() - oracle).abs().max())
                        for k, v in (("fused", fused), ("staged", staged), ("single", single))
                    },
                }
            )

        return hook

    for name, module in model.named_modules():
        if module.__class__.__name__ == "FusedRMSNormGated":
            module.register_forward_hook(probe(name))
inputs = [
    runner._move_generation_inputs(runner.batch_inputs(processor, rows[:b]), torch.device("cuda:0"), torch.bfloat16)
    for b in (1, 2)
]
kwargs = runner.generation_kwargs(processor)
kwargs.update(max_new_tokens=32, return_dict_in_generate=True, output_scores=True)


def generate():
    outputs = []
    with torch.inference_mode():
        for batch in inputs:
            out = model.generate(**batch, **kwargs)
            tokens = out.sequences[:, batch["input_ids"].shape[1] :].cpu()
            outputs.append((tokens, [s.float().cpu() for s in out.scores]))
    return outputs


with runner.DownProjectionAblation(runner.discover_decoder_ffns(model), runner.removed_indices(plan)):
    masked = generate()
    masked_repeat = generate()
runner.physically_prune(model, plan)
sliced = generate()
sliced_repeat = generate()
comparisons = []
for b, ((mt, ms), (st, ss)) in enumerate(zip(masked, sliced), 1):
    for j in range(len(mt)):
        a, z = mt[j].tolist(), st[j].tolist()
        divergence = next((k for k in range(min(len(a), len(z))) if a[k] != z[k]), None)
        detail = {}
        if divergence is not None:
            k = divergence
            for label, scores in (("masked", ms), ("sliced", ss)):
                vals, ids = scores[k][j].topk(5)
                detail[label] = {
                    "top5": [
                        {"id": i, "token": processor.tokenizer.decode([i]), "logit": v}
                        for i, v in zip(ids.tolist(), vals.tolist())
                    ],
                    "top1_margin": float(vals[0] - vals[1]),
                    "masked_choice_logit": float(scores[k][j, a[k]]),
                    "sliced_choice_logit": float(scores[k][j, z[k]]),
                }
        comparisons.append(
            {
                "batch_size": b,
                "sample_id": rows[j]["sample_id"],
                "first_difference": divergence,
                "equal": a == z,
                "masked_tokens": a,
                "sliced_tokens": z,
                "masked_text": processor.tokenizer.decode(a, skip_special_tokens=True),
                "sliced_text": processor.tokenizer.decode(z, skip_special_tokens=True),
                "masked_repeat_equal": torch.equal(mt, masked_repeat[b - 1][0]),
                "sliced_repeat_equal": torch.equal(st, sliced_repeat[b - 1][0]),
                "first_difference_scores": detail,
            }
        )
result = {
    "mode": args.mode,
    "status": "diagnostic_only",
    "comparisons": comparisons,
    "same_input_norm_probes": norm_probes,
}
runner._write_json_atomic(root / "diagnostics_qwen35_bf16" / f"{args.mode}.json", result)
print(json.dumps(result, indent=2), flush=True)
