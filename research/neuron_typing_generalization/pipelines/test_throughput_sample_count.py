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

"""Independent LLaVA/MME sample-count pilot; never resumes the full sweep."""

import argparse
import hashlib
import json
import random
import statistics
import time

import run_structured_throughput_2026_10 as runner
import torch
import transformers
from eos_sequence_stats import trim_at_eos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llava_1_5_7b", "gemma_4_e2b_it"], default="llava_1_5_7b")
    parser.add_argument("--counts", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--same-run-baseline", action="store_true")
    args = parser.parse_args()
    assert all(0 < n <= 1000 for n in args.counts)
    if args.same_run_baseline and 1000 not in args.counts:
        parser.error("Same-run baseline requires --counts to include 1000")
    torch.manual_seed(20261024)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    assert torch.cuda.device_count() == 1
    label = "llava" if args.model == "llava_1_5_7b" else "gemma"
    out = runner.ROOT / f"sample_count_pilot_{label}_mme_20261024"
    if args.same_run_baseline:
        out = out.with_name(out.name + "_paired")
        if args.counts != [128, 1000]:
            out = out.with_name(out.name + "_" + "_".join(map(str, args.counts)))
    if (out / "summary.json").exists():
        raise FileExistsError(f"Preserve completed results: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = runner.ROOT / "datasets/mme/manifest.json"
    rows = [{**r, "dataset": "mme"} for r in json.loads(manifest.read_text())["rows"]]
    assert len(rows) == 1000
    spec = runner._select_spec(runner.STUDY, args.model, "mme")
    layout = runner.load_model_layout(spec.model.values)
    processor = runner.load_local_processor_adapter(spec.model.values, use_fast=False).processor
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    config = {
        "runtime": runner.runtime(), "model": args.model, "dataset": "mme", "architecture": "original",
        "sample_counts": args.counts, "batch_sizes": [1, 16], "repeats": 3,
        "baseline_policy": "Same-process remeasured 1000-target baseline" if args.same_run_baseline else "Historical baseline",
        "sampling": "nested prefixes of frozen order; drop incomplete batch",
        "max_new_tokens": 32, "do_sample": False, "warmup_batches_per_cell": 3,
        "runner_fingerprint": runner.fingerprint(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "practical_difference_reference_percent": 5,
    }
    baselines = {}
    for batch in [1, 16]:
        path = runner.ROOT / f"results/{args.model}__mme/natural" / f"original_00_b{batch:02}.json"
        saved = json.loads(path.read_text())
        assert saved["status"] == "complete" and len(saved["measurements"]) == 3
        baselines[batch] = saved
    config["historical_baselines"] = {str(b): d["config"] for b, d in baselines.items()}
    runner._write_json_atomic(out / "config.json", config)
    model = getattr(transformers, layout.architecture).from_pretrained(
        layout.model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa"
    ).eval().to("cuda:0")
    prepared = {}
    for batch in [1, 16]:
        used = max(args.counts) // batch * batch
        prepared[batch] = [runner.batch_inputs(processor, rows[i:i+batch]) for i in range(0, used, batch)]
    orders = []
    records = []
    for repeat in range(3):
        cells = [(b, n) for b in [1, 16] for n in args.counts]
        random.Random(20261024 + repeat).shuffle(cells)
        orders.append(cells)
        for batch, count in cells:
            used = count // batch * batch
            start = time.time()
            measurement = runner.measure(
                model, processor, prepared[batch][:used//batch], rows[:used], torch.bfloat16, False, 1
            )[0]
            measurement["repeat"] = repeat
            measurement["generated_tokens_eos_corrected"] = sum(
                len(trim_at_eos(seq, model.generation_config.eos_token_id))
                for seq in measurement["output_token_ids"]
            )
            record = {"batch_size": batch, "target_samples": count, "actual_samples": used,
                      "started_unix": start, "finished_unix": time.time(), "measurement": measurement}
            runner._write_json_atomic(out / f"b{batch:02}_n{count:04}_r{repeat}.json", record)
            records.append(record)
            print(f"DONE repeat={repeat} batch={batch} n={count} used={used} req/s={measurement['req_per_second']:.4f}", flush=True)
    summaries = []
    for batch in [1, 16]:
        baseline = [m["req_per_second"] for m in baselines[batch]["measurements"]]
        historical_baseline = baseline
        if args.same_run_baseline:
            baseline = [r["measurement"]["req_per_second"] for r in records
                        if r["batch_size"] == batch and r["target_samples"] == 1000]
        for count in args.counts:
            rates = [r["measurement"]["req_per_second"] for r in records if r["batch_size"] == batch and r["target_samples"] == count]
            median = statistics.median(rates)
            summaries.append({"batch_size": batch, "target_samples": count, "rates": rates,
                              "median_req_s": median, "min_req_s": min(rates), "max_req_s": max(rates),
                              "historical_baseline_rates": historical_baseline,
                              "comparison_baseline_rates": baseline,
                              "paired_difference_percent": [100*(v/b-1) for v,b in zip(rates,baseline)] if args.same_run_baseline else None,
                              "difference_vs_1000_percent": 100*(median/statistics.median(baseline)-1)})
    runner._write_json_atomic(out / "summary.json", {"status": "complete", "config": config, "orders": orders, "summaries": summaries})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
