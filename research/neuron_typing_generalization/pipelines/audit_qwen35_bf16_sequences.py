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

"""Paired 128-request BF16 audit; diagnostic only, no benchmark gate changes."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import run_structured_throughput_2026_10 as runner
import torch
import transformers
from eos_sequence_stats import trim_at_eos


torch.manual_seed(20261022)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="qwen3_5_2b", choices=("qwen3_5_2b", "qwen3_vl_2b_instruct", "llava_1_5_7b", "gemma_4_e2b_it")
)
args = parser.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
root = runner.ROOT
outdir = root / ("diagnostics_qwen35_bf16_128" if args.model == "qwen3_5_2b" else f"diagnostics_{args.model}_bf16_128")
plan_path = root / "plans" / f"{args.model}__mme" / "nonuniform_10.json"
data_path = root / "datasets/mme/manifest.json"
plan = json.loads(plan_path.read_text())
rows = json.loads(data_path.read_text())["rows"][:128]
spec = runner._select_spec(runner.STUDY, args.model, "mme")
layout = runner.load_model_layout(spec.model.values)
processor = runner.load_local_processor_adapter(spec.model.values, use_fast=False).processor
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
config = {
    "samples": len(rows),
    "selection": "first128_in_frozen_order",
    "batches": [1, 2],
    "max_new_tokens": 32,
    "runtime": runner.runtime(),
    "triton_f32_default": os.getenv("TRITON_F32_DEFAULT"),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
}
model = (
    getattr(transformers, layout.architecture)
    .from_pretrained(
        layout.model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa"
    )
    .eval()
    .to("cuda:0")
)
kwargs = runner.generation_kwargs(processor)
kwargs.update(return_dict_in_generate=True, output_scores=True)
eos = model.generation_config.eos_token_id
config["effective_eos_token_id"] = eos


def collect(label):
    results = {}
    for size in (1, 2):
        path = outdir / f"{label}_b{size}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["config"] != config:
                raise ValueError("Diagnostic resume config changed")
            results[size] = saved["rows"]
            continue
        output_rows = []
        with torch.inference_mode():
            for offset in range(0, len(rows), size):
                batch = rows[offset : offset + size]
                inputs = runner._move_generation_inputs(
                    runner.batch_inputs(processor, batch), torch.device("cuda:0"), torch.bfloat16
                )
                generated = model.generate(**inputs, **kwargs)
                tokens = generated.sequences[:, inputs["input_ids"].shape[1] :].cpu().tolist()
                top = [s.float().topk(5, dim=-1) for s in generated.scores]
                ids = [t.indices.cpu().tolist() for t in top]
                vals = [t.values.cpu().tolist() for t in top]
                for j, seq in enumerate(tokens):
                    seq = trim_at_eos(seq, eos)
                    text = processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
                    output_rows.append(
                        {
                            "sample_id": batch[j]["sample_id"],
                            "tokens": seq,
                            "text": text,
                            "score": runner.score_prediction("mme", text, batch[j]["answer"]),
                            "steps": [
                                {"ids": i[j], "logits": v[j], "margin": v[j][0] - v[j][1]}
                                for i, v in zip(ids[: len(seq)], vals[: len(seq)])
                            ],
                        }
                    )
                del generated, inputs, top
                if (offset + size) % 16 == 0:
                    print(f"{label} batch={size}: {offset + size}/{len(rows)}", flush=True)
        runner._write_json_atomic(path, {"config": config, "rows": output_rows})
        results[size] = output_rows
    return results


with runner.DownProjectionAblation(runner.discover_decoder_ffns(model), runner.removed_indices(plan)):
    masked = collect("masked")
runner.physically_prune(model, plan)
sliced = collect("sliced")
summaries = []
for size in (1, 2):
    pairs = []
    for a, b in zip(masked[size], sliced[size]):
        assert a["sample_id"] == b["sample_id"]
        first = next((k for k, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None)
        if first is None and len(a["tokens"]) != len(b["tokens"]):
            first = min(len(a["tokens"]), len(b["tokens"]))
        pairs.append(
            {
                "sample_id": a["sample_id"],
                "sequence_equal": a["tokens"] == b["tokens"],
                "text_equal": a["text"] == b["text"],
                "masked_text": a["text"],
                "sliced_text": b["text"],
                "masked_score": a["score"],
                "sliced_score": b["score"],
                "first_difference": first,
                "masked_first_difference": a["steps"][first]
                if first is not None and first < len(a["steps"])
                else None,
                "sliced_first_difference": b["steps"][first]
                if first is not None and first < len(b["steps"])
                else None,
            }
        )
    mismatches = [p for p in pairs if not p["sequence_equal"]]
    margins = {}
    for label in ("masked", "sliced"):
        values = [
            p[f"{label}_first_difference"]["margin"] for p in mismatches if p[f"{label}_first_difference"] is not None
        ]
        margins[label] = {
            "n": len(values),
            "min_median_max": np.quantile(values, [0, 0.5, 1]).tolist() if values else None,
            "exact_ties": sum(v == 0 for v in values),
        }
    summaries.append(
        {
            "batch_size": size,
            "n": len(pairs),
            "sequence_mismatches": len(mismatches),
            "sequence_mismatch_rate": len(mismatches) / len(pairs),
            "text_mismatches": sum(not p["text_equal"] for p in pairs),
            "masked_accuracy": float(np.mean([p["masked_score"] for p in pairs])),
            "sliced_accuracy": float(np.mean([p["sliced_score"] for p in pairs])),
            "score_changed_samples": sum(p["masked_score"] != p["sliced_score"] for p in pairs),
            "divergence_margins": margins,
            "pairs": pairs,
        }
    )
runner._write_json_atomic(
    outdir / "summary.json",
    {"status": "complete", "scope": "diagnostic_only_not_independent_test", "config": config, "summaries": summaries},
)
print(json.dumps([{k: v for k, v in s.items() if k != "pairs"} for s in summaries], indent=2), flush=True)
