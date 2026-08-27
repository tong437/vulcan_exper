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

"""Benchmark parameter, memory, prefill, TTFT, and generation effects of structural pruning."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from evaluate_vqa import load_binary_records, prepare_model_batch, select_binary_records
from phase3_structural_utils import (
    count_parameters,
    directory_size_bytes,
    model_weight_size_bytes,
    seed_everything,
    summarize_measurements,
    timed_runs,
)
from verify_structural_equivalence import load_model_bundle, model_layer_dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark original versus Phase-3 structural model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--original_model_path", required=True)
    parser.add_argument("--pruned_model_path", required=True)
    parser.add_argument("--pope_file", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_sizes", default="1,4")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def prepare_text_batch(processor, records: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    prompts = []
    for record in records:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{record['question']} Answer with only yes or no."}],
            }
        ]
        prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    inputs = processor(text=prompts, padding=True, return_tensors="pt")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def benchmark_prefill(
    model,
    inputs: dict[str, Any],
    *,
    batch_size: int,
    warmup: int,
    repeats: int,
    bootstrap_samples: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    input_tokens = int(inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).sum().item())
    reset_peak_memory(device)
    times = timed_runs(
        lambda: model(**inputs, use_cache=False),
        warmup=warmup,
        repeats=repeats,
        synchronize=cuda_sync,
    )
    result = summarize_measurements(times, bootstrap_samples=bootstrap_samples, seed=seed)
    result.update(
        {
            "peak_memory_bytes": peak_memory_bytes(device),
            "batch_size": batch_size,
            "input_tokens": input_tokens,
            "samples_per_second_from_median": batch_size / result["median"],
            "tokens_per_second_from_median": input_tokens / result["median"],
        }
    )
    return result


def forced_generation_kwargs(inputs: dict[str, Any], tokenizer, new_tokens: int) -> dict[str, Any]:
    return {
        **inputs,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "min_new_tokens": new_tokens,
        "max_new_tokens": new_tokens,
    }


def validate_generated_token_count(model, generation_kwargs: dict[str, Any], expected: int) -> int:
    output = model.generate(**generation_kwargs)
    generated = int(output.shape[-1] - generation_kwargs["input_ids"].shape[-1])
    if generated != expected:
        raise RuntimeError(f"Fixed-length generation produced {generated} tokens, expected {expected}.")
    return generated


@torch.inference_mode()
def benchmark_model(
    *,
    name: str,
    config: str,
    model_path: str,
    records: list[dict[str, Any]],
    batch_sizes: list[int],
    warmup: int,
    repeats: int,
    max_new_tokens: int,
    bootstrap_samples: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    cuda_sync()
    load_wall_start = time.perf_counter()
    model, tokenizer, processor = load_model_bundle(config, model_path, device)
    cuda_sync()
    load_seconds = time.perf_counter() - load_wall_start

    first_parameter = next(model.parameters())
    result: dict[str, Any] = {
        "name": name,
        "model_path": str(Path(model_path).resolve()),
        "parameters": count_parameters(model),
        "checkpoint_bytes": directory_size_bytes(model_path),
        "model_weight_bytes": model_weight_size_bytes(model_path),
        "load_seconds": load_seconds,
        "dtype": str(first_parameter.dtype),
        "layer_dims": model_layer_dims(model),
        "prefill": {"multimodal": {}, "text_only": {}},
        "generation": {},
    }
    for batch_size in batch_sizes:
        if batch_size > len(records):
            raise ValueError(f"batch_size={batch_size} exceeds available records={len(records)}.")
        batch_records = records[:batch_size]
        multimodal_inputs = prepare_model_batch(processor, batch_records, device)
        text_inputs = prepare_text_batch(processor, batch_records, device)
        result["prefill"]["multimodal"][str(batch_size)] = benchmark_prefill(
            model,
            multimodal_inputs,
            batch_size=batch_size,
            warmup=warmup,
            repeats=repeats,
            bootstrap_samples=bootstrap_samples,
            seed=seed + batch_size,
            device=device,
        )
        result["prefill"]["text_only"][str(batch_size)] = benchmark_prefill(
            model,
            text_inputs,
            batch_size=batch_size,
            warmup=warmup,
            repeats=repeats,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1000 + batch_size,
            device=device,
        )

        if batch_size == 1:
            ttft_kwargs = forced_generation_kwargs(multimodal_inputs, tokenizer, 1)
            total_kwargs = forced_generation_kwargs(multimodal_inputs, tokenizer, max_new_tokens)
            generated_tokens = validate_generated_token_count(model, total_kwargs, max_new_tokens)
            reset_peak_memory(device)
            ttft_times = timed_runs(
                lambda: model.generate(**ttft_kwargs),
                warmup=warmup,
                repeats=repeats,
                synchronize=cuda_sync,
            )
            total_times = timed_runs(
                lambda: model.generate(**total_kwargs),
                warmup=warmup,
                repeats=repeats,
                synchronize=cuda_sync,
            )
            ttft = summarize_measurements(ttft_times, bootstrap_samples=bootstrap_samples, seed=seed + 101)
            total = summarize_measurements(total_times, bootstrap_samples=bootstrap_samples, seed=seed + 102)
            decode_seconds = total["median"] - ttft["median"]
            result["generation"] = {
                "batch_size": 1,
                "max_new_tokens": max_new_tokens,
                "actual_generated_tokens": generated_tokens,
                "fixed_length_generation": generated_tokens == max_new_tokens,
                "ttft_seconds": ttft,
                "total_seconds": total,
                "decode_tokens_per_second_from_medians": (
                    (generated_tokens - 1) / decode_seconds if generated_tokens > 1 and decode_seconds > 0 else None
                ),
                "peak_memory_bytes": peak_memory_bytes(device),
            }
    return result


def bootstrap_latency_comparison(
    original_values: list[float],
    candidate_values: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    original = np.asarray(original_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    original_median = float(np.median(original))
    candidate_median = float(np.median(candidate))
    result = {
        "latency_speedup": original_median / candidate_median,
        "latency_delta_seconds": candidate_median - original_median,
        "bootstrap_mode": "independent_run_resampling",
    }
    if bootstrap_samples <= 0 or min(len(original), len(candidate)) < 2:
        return result
    rng = np.random.default_rng(seed)
    original_samples = rng.choice(original, size=(bootstrap_samples, len(original)), replace=True)
    candidate_samples = rng.choice(candidate, size=(bootstrap_samples, len(candidate)), replace=True)
    original_medians = np.median(original_samples, axis=1)
    candidate_medians = np.median(candidate_samples, axis=1)
    speedups = original_medians / candidate_medians
    deltas = candidate_medians - original_medians
    result.update(
        {
            "latency_speedup_ci95": np.percentile(speedups, [2.5, 97.5]).tolist(),
            "latency_delta_seconds_ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
        }
    )
    return result


def bootstrap_decode_comparison(
    original_generation: dict[str, Any],
    candidate_generation: dict[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    original_tokens = int(original_generation["actual_generated_tokens"]) - 1
    candidate_tokens = int(candidate_generation["actual_generated_tokens"]) - 1
    if original_tokens != candidate_tokens or original_tokens <= 0:
        raise ValueError("Decode comparison requires the same positive fixed decode-token count.")
    original_total = np.asarray(original_generation["total_seconds"]["raw"], dtype=np.float64)
    original_ttft = np.asarray(original_generation["ttft_seconds"]["raw"], dtype=np.float64)
    candidate_total = np.asarray(candidate_generation["total_seconds"]["raw"], dtype=np.float64)
    candidate_ttft = np.asarray(candidate_generation["ttft_seconds"]["raw"], dtype=np.float64)
    observed_original_seconds = float(np.median(original_total) - np.median(original_ttft))
    observed_candidate_seconds = float(np.median(candidate_total) - np.median(candidate_ttft))
    result = {
        "decode_tokens": original_tokens,
        "decode_throughput_ratio": observed_original_seconds / observed_candidate_seconds,
        "original_decode_tokens_per_second": original_tokens / observed_original_seconds,
        "candidate_decode_tokens_per_second": candidate_tokens / observed_candidate_seconds,
        "bootstrap_mode": "independent_run_resampling",
    }
    if bootstrap_samples <= 0:
        return result
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(bootstrap_samples):
        original_decode = np.median(rng.choice(original_total, len(original_total), replace=True)) - np.median(
            rng.choice(original_ttft, len(original_ttft), replace=True)
        )
        candidate_decode = np.median(rng.choice(candidate_total, len(candidate_total), replace=True)) - np.median(
            rng.choice(candidate_ttft, len(candidate_ttft), replace=True)
        )
        if original_decode > 0 and candidate_decode > 0:
            ratios.append(original_decode / candidate_decode)
    if not ratios:
        raise RuntimeError("All bootstrapped decode durations were non-positive.")
    result["decode_throughput_ratio_ci95"] = np.percentile(ratios, [2.5, 97.5]).tolist()
    return result


def compare_benchmarks(
    original: dict[str, Any],
    pruned: dict[str, Any],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    original_parameters = original["parameters"]["total"]
    pruned_parameters = pruned["parameters"]["total"]
    comparison: dict[str, Any] = {
        "removed_parameters": original_parameters - pruned_parameters,
        "removed_parameter_ratio": (original_parameters - pruned_parameters) / original_parameters,
        "parameter_storage_bytes_delta": (
            pruned["parameters"]["storage_bytes"] - original["parameters"]["storage_bytes"]
        ),
        "checkpoint_bytes_delta": pruned["checkpoint_bytes"] - original["checkpoint_bytes"],
        "checkpoint_size_ratio": pruned["checkpoint_bytes"] / original["checkpoint_bytes"],
        "model_weight_bytes_delta": pruned["model_weight_bytes"] - original["model_weight_bytes"],
        "model_weight_size_ratio": pruned["model_weight_bytes"] / original["model_weight_bytes"],
        "weight_file_delta_minus_parameter_storage_delta": (
            (pruned["model_weight_bytes"] - original["model_weight_bytes"])
            - (pruned["parameters"]["storage_bytes"] - original["parameters"]["storage_bytes"])
        ),
        "prefill": {"multimodal": {}, "text_only": {}},
    }
    primary_rows = []
    for scenario in ("multimodal", "text_only"):
        for batch_size in original["prefill"][scenario]:
            original_row = original["prefill"][scenario][batch_size]
            pruned_row = pruned["prefill"][scenario][batch_size]
            row = bootstrap_latency_comparison(
                original_row["raw"],
                pruned_row["raw"],
                bootstrap_samples=bootstrap_samples,
                seed=seed + (0 if scenario == "multimodal" else 1000) + int(batch_size),
            )
            row["peak_memory_delta_bytes"] = (
                pruned_row["peak_memory_bytes"] - original_row["peak_memory_bytes"]
                if original_row["peak_memory_bytes"] is not None
                else None
            )
            comparison["prefill"][scenario][batch_size] = row
            if scenario == "multimodal":
                primary_rows.append((f"multimodal_prefill_bs{batch_size}", row))
    original_generation = original["generation"]
    pruned_generation = pruned["generation"]
    ttft_comparison = bootstrap_latency_comparison(
        original_generation["ttft_seconds"]["raw"],
        pruned_generation["ttft_seconds"]["raw"],
        bootstrap_samples=bootstrap_samples,
        seed=seed + 2001,
    )
    total_comparison = bootstrap_latency_comparison(
        original_generation["total_seconds"]["raw"],
        pruned_generation["total_seconds"]["raw"],
        bootstrap_samples=bootstrap_samples,
        seed=seed + 2002,
    )
    comparison["generation"] = {
        "fixed_length_generation": bool(
            original_generation["fixed_length_generation"] and pruned_generation["fixed_length_generation"]
        ),
        "ttft": ttft_comparison,
        "total_latency": total_comparison,
        "decode": bootstrap_decode_comparison(
            original_generation,
            pruned_generation,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 2003,
        ),
        "peak_memory_delta_bytes": (
            pruned_generation["peak_memory_bytes"] - original_generation["peak_memory_bytes"]
            if original_generation["peak_memory_bytes"] is not None
            else None
        ),
    }
    primary_rows.extend([("generation_ttft", ttft_comparison), ("generation_total", total_comparison)])
    stable_improvements = [
        name for name, row in primary_rows if row.get("latency_speedup_ci95", [0.0, float("inf")])[0] > 1.0
    ]
    significant_slowdowns = [
        name for name, row in primary_rows if row.get("latency_speedup_ci95", [0.0, float("inf")])[1] < 1.0
    ]
    comparison["engineering_gate"] = {
        "passed": bool(stable_improvements) and not significant_slowdowns,
        "requires_stable_primary_improvement": True,
        "requires_no_significant_primary_slowdown": True,
        "stable_primary_improvements": stable_improvements,
        "significant_primary_slowdowns": significant_slowdowns,
        "primary_scenarios": [name for name, _ in primary_rows],
    }
    return comparison


def main() -> None:
    args = parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if any(value <= 0 for value in batch_sizes):
        raise ValueError("batch_sizes must be positive.")
    if args.max_new_tokens <= 1:
        raise ValueError("max_new_tokens must be at least 2 for TTFT/decode separation.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_records = load_binary_records(args.pope_file, args.image_root)
    records, selection = select_binary_records(all_records, max_samples=max(batch_sizes))

    common = {
        "config": args.config,
        "records": records,
        "batch_sizes": batch_sizes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "device": device,
    }
    original = benchmark_model(name="original", model_path=args.original_model_path, **common)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pruned = benchmark_model(name="structural_qband", model_path=args.pruned_model_path, **common)
    result = {
        "config": vars(args),
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "selection": selection,
        "models": {"original": original, "structural_qband": pruned},
        "comparison": compare_benchmarks(
            original,
            pruned,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    write_json(args.output_file, result)
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
