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

"""One physical architecture: verification or frozen-1000 full-batch throughput."""

import argparse
import gc
import hashlib
import inspect
import json
import os
import subprocess
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
import transformers
from neuron_typing_generalization.ablation_runner import _move_generation_inputs, _select_spec, _write_json_atomic
from neuron_typing_generalization.adapters import load_local_processor_adapter, load_model_layout
from neuron_typing_generalization.advisor_ablation import DownProjectionAblation, score_prediction
from neuron_typing_generalization.hooks import discover_decoder_ffns
from neuron_typing_generalization.structured_throughput import physically_prune, removed_indices
from PIL import Image
from prepare_throughput_2026_10 import ROOT


STUDY = Path("research/neuron_typing_generalization/configs/studies/advisor_2026_09_main.yaml")
ROOT = Path(os.environ.get("THROUGHPUT_OUTPUT_ROOT", str(ROOT)))
WORKLOAD_SAMPLES = int(os.environ.get("THROUGHPUT_WORKLOAD_SAMPLES", "1000"))


def resume_compatible(saved, current):
    """Explicit legacy migration; preserve workload and numerical settings."""
    if saved == current:
        return True
    legacy = os.environ.get("THROUGHPUT_LEGACY_CODE_SHA256")
    if not legacy or saved.get("code_sha256") not in {legacy, current["code_sha256"]}:
        return False
    excluded = {"code_sha256", "runtime", "correctness_policy"}
    if {k: v for k, v in saved.items() if k not in excluded} != {k: v for k, v in current.items() if k not in excluded}:
        return False
    a, b = saved["runtime"], current["runtime"]
    device_fields = {"device_identity", "visible_devices", "nvidia_smi"}
    return {k: v for k, v in a.items() if k not in device_fields} == {k: v for k, v in b.items() if k not in device_fields}


def fingerprint():
    from neuron_typing_generalization import structured_throughput

    paths = [Path(__file__), Path(structured_throughput.__file__)]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def runtime():
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "device_identity": str(torch.cuda.get_device_properties(0)),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
    }


def batch_inputs(processor, rows):
    images, conversations = [], []
    try:
        for row in rows:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            images.append(image)
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": image}, {"type": "text", "text": row["prompt"]}],
                    }
                ]
            )
        return dict(
            processor.apply_chat_template(
                conversations,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        )
    finally:
        for image in images:
            image.close()


def generation_kwargs(processor, fixed=False):
    tokenizer = processor.tokenizer
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    kwargs = dict(max_new_tokens=32, do_sample=False, use_cache=True, pad_token_id=pad)
    if fixed:
        kwargs["min_new_tokens"] = 32
    return kwargs


def check_logits(reference, actual, *, fp32=False, observe=False):
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise ValueError("Nonfinite verification logits")
    delta = actual.float() - reference.float()
    rms = float(delta.square().mean().sqrt())
    relative = rms / max(float(reference.float().square().mean().sqrt()), 1e-12)
    max_abs = float(delta.abs().max())
    relative_limit, absolute_limit = (1e-5, 1e-3) if fp32 else (0.01, 0.5)
    passed = bool(torch.isfinite(actual).all() and relative <= relative_limit and max_abs <= absolute_limit)
    if not passed and not observe:
        raise ValueError(f"Physical/masked logits mismatch: relative RMS={relative}, max_abs={max_abs}")
    return {
        "relative_rms": relative,
        "max_abs": max_abs,
        "original_threshold_passed": passed,
        "relative_rms_limit": relative_limit,
        "max_abs_limit": absolute_limit,
    }


def verify(model, processor, plan, rows, dtype, *, observe=False):
    device = next(model.parameters()).device
    selected = removed_indices(plan)
    prepared = [batch_inputs(processor, rows[:size]) for size in (1, 2)]
    # Batched tokenization must preserve each individual prompt after removing padding.
    for i in range(2):
        single = batch_inputs(processor, rows[i : i + 1])
        batch = prepared[1]
        if not torch.equal(
            single["input_ids"][0][single["attention_mask"][0].bool()],
            batch["input_ids"][i][batch["attention_mask"][i].bool()],
        ):
            raise ValueError("Batch tokenization changes the prompt")
    references = []
    kwargs = generation_kwargs(processor)
    kwargs["max_new_tokens"] = 4
    forward_kwargs = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    with torch.inference_mode(), DownProjectionAblation(discover_decoder_ffns(model), selected):
        for cpu in prepared:
            inputs = _move_generation_inputs(cpu, device, dtype)
            logits = model(**inputs, use_cache=False, **forward_kwargs).logits[:, -1].float().cpu()
            generated = model.generate(**inputs, **kwargs)[:, inputs["input_ids"].shape[1] :].cpu()
            references.append((logits, generated))
            del inputs
    parameters = physically_prune(model, plan)
    results = []
    with torch.inference_mode():
        for cpu, (ref, tokens) in zip(prepared, references):
            inputs = _move_generation_inputs(cpu, device, dtype)
            logits = model(**inputs, use_cache=False, **forward_kwargs).logits[:, -1].float().cpu()
            result = check_logits(ref, logits, fp32=dtype == torch.float32, observe=observe or dtype == torch.bfloat16)
            generated = model.generate(**inputs, **kwargs)[:, inputs["input_ids"].shape[1] :].cpu()
            equal = torch.equal(tokens, generated)
            if not equal and not observe:
                raise ValueError("Masked and sliced models generate different four-token sequences")
            results.append(
                {
                    "batch_size": len(tokens),
                    **result,
                    "generated_tokens_equal": equal,
                    "masked_tokens": tokens.tolist(),
                    "sliced_tokens": generated.tolist(),
                }
            )
    return {"status": "observed" if observe else "passed", "comparisons": results, "parameters": parameters}


def measure(model, processor, batches, rows, dtype, fixed, repeats):
    kwargs = generation_kwargs(processor, fixed)
    measurements = []
    with torch.inference_mode():
        for warm in range(3):
            inputs = _move_generation_inputs(batches[warm % len(batches)], torch.device("cuda:0"), dtype)
            model.generate(**inputs, **kwargs)
            del inputs
        torch.cuda.synchronize()
        for repeat in range(repeats):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            elapsed, outputs = 0.0, []
            for cpu in batches:
                torch.cuda.synchronize()
                start = time.perf_counter()
                inputs = _move_generation_inputs(cpu, torch.device("cuda:0"), dtype)
                output = model.generate(**inputs, **kwargs)
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                # Decode and scoring excluded from GPU service time, consistently for all methods.
                outputs.extend(output[:, inputs["input_ids"].shape[1] :].cpu().tolist())
                del inputs, output
            if len(outputs) != len(rows):
                raise ValueError("Incomplete benchmark: numerator must match full-batch request count")
            predictions = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            scores = [
                score_prediction(row["dataset"], pred.strip(), row["answer"]) for row, pred in zip(rows, predictions)
            ]
            eos = processor.tokenizer.eos_token_id
            lengths = [len(seq) if fixed or eos not in seq else seq.index(eos) + 1 for seq in outputs]
            if fixed and any(length != 32 for length in lengths):
                raise ValueError("Fixed-length generation did not produce 32 tokens for every request")
            measurements.append(
                {
                    "repeat": repeat,
                    "requests": len(rows),
                    "elapsed_gpu_service_seconds": elapsed,
                    "req_per_second": len(rows) / elapsed,
                    "generated_tokens": sum(lengths),
                    "task_score": float(np.mean(scores)),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "predictions": predictions,
                    "output_token_ids": outputs,
                }
            )
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", choices=("original", "uniform", "nonuniform"), required=True)
    parser.add_argument("--percent", type=int, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-fp32", action="store_true", help="Primary FP32 numerical-equivalence proof")
    parser.add_argument(
        "--diagnose-fp32", action="store_true", help="Independent FP32 diagnostic; never opens the BF16 gate"
    )
    parser.add_argument(
        "--diagnose-bf16", action="store_true", help="Record deviations and tokens; never opens the gate"
    )
    parser.add_argument(
        "--precise-reduction", action="store_true", help="Disable reduced-precision BF16 matmul reductions"
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--fixed-output", action="store_true")
    args = parser.parse_args()
    if args.verify_fp32 and (not args.verify_only or args.diagnose_fp32 or args.diagnose_bf16):
        parser.error("--verify-fp32 requires --verify-only and no diagnostic flags")
    fp32 = args.verify_fp32 or args.diagnose_fp32
    verification_device = "cpu" if fp32 and args.model == "llava_1_5_7b" else "cuda:0"
    if args.diagnose_fp32 and not args.verify_only:
        parser.error("FP32 diagnostic requires --verify-only")
    if args.diagnose_bf16 and (not args.verify_only or args.diagnose_fp32):
        parser.error("BF16 diagnostic requires --verify-only and cannot combine with FP32")
    if args.batches != sorted(set(args.batches)) or set(args.batches) - {1, 2, 4, 8, 16, 32, 64}:
        parser.error("Batches must be unique, increasing powers of two from 1 through 64")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one idle CUDA device")
    torch.manual_seed(20261022)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not args.precise_reduction
    plan_path = ROOT / "plans" / f"{args.model}__{args.dataset}" / f"{args.method}_{args.percent:02d}.json"
    data_path = ROOT / "datasets" / args.dataset / "manifest.json"
    plan, manifest = json.loads(plan_path.read_text()), json.loads(data_path.read_text())
    if manifest["count"] != WORKLOAD_SAMPLES or len(manifest["rows"]) != WORKLOAD_SAMPLES:
        raise ValueError(f"Require exactly {WORKLOAD_SAMPLES} frozen requests")
    rows = [{**row, "dataset": args.dataset} for row in manifest["rows"]]
    if args.pilot:
        rows = rows[:32]
    for row in rows:
        if hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError("Frozen image changed")
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "code_sha256": fingerprint(),
        "runtime": runtime(),
        "repeats": 1 if args.pilot else 3,
        "workload_samples": len(rows),
        "warmup_batches": 3,
        "max_new_tokens": 32,
        "fixed_output": args.fixed_output,
        "dtype": "float32" if fp32 else "bfloat16",
        "verification_device": verification_device,
        "correctness_policy": "user_approved_direct_timing_2026_10_25" if os.environ.get("THROUGHPUT_DIRECT_TIMING") == "1" else "fp32_equivalence_and_bf16_tokens_v1",
        "attention": "sdpa",
        "allow_bf16_reduced_precision_reduction": not args.precise_reduction,
        "timing": "sum of synchronized CPU-tensor transfer + vision/prefill/decode; preprocessing/scoring excluded",
    }
    category = "verification" if args.verify_only else ("pilot" if args.pilot else "results")
    if args.verify_fp32:
        category = "verification_fp32"
    if args.diagnose_fp32:
        category = "diagnostics_fp32"
    if args.diagnose_bf16:
        category = "diagnostics_bf16"
        if args.precise_reduction:
            category += "_precise"
    mode = "fixed32" if args.fixed_output else "natural"
    destination = ROOT / category / f"{args.model}__{args.dataset}" / mode
    stem = f"{args.method}_{args.percent:02d}"
    fp32_proof = ROOT / "verification_fp32" / f"{args.model}__{args.dataset}" / "natural" / f"{stem}.json"
    if args.verify_only and not (fp32 or args.diagnose_bf16):
        numerical = json.loads(fp32_proof.read_text())
        if numerical["status"] != "passed" or any(
            numerical["config"][k] != config[k]
            for k in ("plan_sha256", "data_sha256", "code_sha256", "runtime", "correctness_policy")
        ):
            raise ValueError("Matching FP32 proof required before BF16 verification")
        config["fp32_proof_sha256"] = hashlib.sha256(fp32_proof.read_bytes()).hexdigest()
    if args.verify_only:
        existing = destination / f"{stem}.json"
        if existing.exists():
            saved = json.loads(existing.read_text())
            if saved["config"] != config or saved["status"] != ("observed" if args.diagnose_bf16 else "passed"):
                raise ValueError("Verification configuration changed")
            return
    elif os.environ.get("THROUGHPUT_DIRECT_TIMING") != "1":
        for method in ("uniform", "nonuniform"):
            for percent in (30,) if args.pilot else (10, 30):
                proof = (
                    ROOT
                    / "verification"
                    / f"{args.model}__{args.dataset}"
                    / "natural"
                    / f"{method}_{percent:02d}.json"
                )
                verified = json.loads(proof.read_text())
                pc = verified["config"]
                numerical_path = ROOT / "verification_fp32" / f"{args.model}__{args.dataset}" / "natural" / proof.name
                if pc.get("fp32_proof_sha256") != hashlib.sha256(numerical_path.read_bytes()).hexdigest():
                    raise ValueError("FP32 proof missing or changed")
                verified_plan = ROOT / "plans" / f"{args.model}__{args.dataset}" / f"{method}_{percent:02d}.json"
                if pc["plan_sha256"] != hashlib.sha256(verified_plan.read_bytes()).hexdigest():
                    raise ValueError("Verified architecture plan changed")
                if verified["status"] != "passed" or any(
                    pc[k] != config[k]
                    for k in (
                        "code_sha256",
                        "data_sha256",
                        "runtime",
                        "correctness_policy",
                        "allow_bf16_reduced_precision_reduction",
                        "dtype",
                        "attention",
                    )
                ):
                    raise ValueError("Matching correctness verification required before timing")
        if not args.pilot:
            gate = json.loads((ROOT / "pilot_passed.json").read_text())
            if gate["code_sha256"] != fingerprint() or gate["status"] != "passed":
                raise ValueError("Pilot gate not passed for this code")
    if not args.verify_only:
        finished = True
        for size in args.batches:
            path = destination / f"{stem}_b{size:02d}.json"
            if not path.exists():
                finished = False
                continue
            prior = json.loads(path.read_text())
            if not resume_compatible(prior["config"], config):
                raise ValueError(f"Resume configuration mismatch: {path}")
            if prior["status"] not in {"complete", "oom", "skipped_after_oom"}:
                raise ValueError(f"Unexpected saved status: {path}")
        if finished:
            print(f"Already completed: {stem}; reused without loading model", flush=True)
            return
    spec = _select_spec(STUDY, args.model, args.dataset)
    layout = load_model_layout(spec.model.values)
    processor = load_local_processor_adapter(spec.model.values, use_fast=False).processor
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    dtype = torch.float32 if fp32 else torch.bfloat16
    # CPU construction allows slicing before transfer; no CPU offload in measured inference.
    model = getattr(transformers, layout.architecture).from_pretrained(
        layout.model_path, local_files_only=True, dtype=dtype, device_map="cpu", attn_implementation="sdpa"
    )
    model.eval()
    if args.verify_only:
        model.to(verification_device)
        proof = verify(model, processor, plan, rows[:2], dtype, observe=args.diagnose_bf16)
        _write_json_atomic(existing, {"config": config, **proof})
        print(f"{proof['status'].upper()} {stem}", flush=True)
        return
    parameters = physically_prune(model, plan)
    model.to("cuda:0")
    oom_at = None
    for batch_size in args.batches:
        output_path = destination / f"{stem}_b{batch_size:02d}.json"
        if output_path.exists():
            previous = json.loads(output_path.read_text())
            if not resume_compatible(previous["config"], config):
                raise ValueError("Resume configuration mismatch")
            if previous["status"] == "oom":
                oom_at = batch_size
            continue
        record = {
            "config": config,
            "batch_size": batch_size,
            "parameters": parameters,
            "target_rate": plan["target_rate"],
            "actual_rate": plan["actual_rate"],
            "aligned_widths": plan["aligned_widths"],
        }
        if oom_at is not None:
            _write_json_atomic(output_path, {**record, "status": "skipped_after_oom", "first_oom_batch": oom_at})
            continue
        used = len(rows) // batch_size * batch_size
        measured_rows = rows[:used]
        record.update(requests_available=len(rows), requests_used=used, dropped_requests=len(rows) - used)
        batches = [batch_inputs(processor, measured_rows[i : i + batch_size]) for i in range(0, used, batch_size)]
        try:
            measurements = measure(
                model, processor, batches, measured_rows, dtype, args.fixed_output, config["repeats"]
            )
            record.update(
                status="complete",
                measurements=measurements,
                median_req_per_second=float(np.median([m["req_per_second"] for m in measurements])),
            )
        except torch.cuda.OutOfMemoryError:
            oom_at = batch_size
            record.update(status="oom", measurements=[], first_oom_batch=batch_size)
            gc.collect()
            torch.cuda.empty_cache()
        _write_json_atomic(output_path, record)
        del batches
        print(f"{stem} batch={batch_size}: {record['status']}", flush=True)


if __name__ == "__main__":
    main()
