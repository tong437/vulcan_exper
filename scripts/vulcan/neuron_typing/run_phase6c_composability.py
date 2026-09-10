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

"""Run Phase 6C causal core, conditional-shell, and union-composability interventions."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase5e_proxy import evaluate_gold_proxy  # noqa: E402
from phase5e_semantics import generate_short_caption  # noqa: E402
from phase6b_semantics import build_semantic_evaluator  # noqa: E402
from phase6c_subnets import (  # noqa: E402
    LayerSets,
    canonical_kept_ids,
    complement_pool,
    deletion_masks,
    deterministic_sample,
    dose_counts,
    frequency_core,
    kept_from_deleted,
    kept_sha256,
    layer_counts,
    layerwise_difference,
    layerwise_union,
    summarize_kept,
)
from run_phase2_ablation import (  # noqa: E402
    MLPNeuronAblator,
    build_dataloader,
    load_yaml,
    move_batch_to_device,
)
from run_phase5_single_sample_frontier import build_prompt_inputs, write_json  # noqa: E402
from run_phase5e_retrospective import build_reference_trace  # noqa: E402
from run_phase6b_single_sample_limits import prepare_dataset, sha256_file, validate_inputs  # noqa: E402
from verify_phase5_structural_equivalence import load_model_bundle, model_layer_dims  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6C subnet geometry interventions.")
    parser.add_argument("--winner_dir", default="saves/neuron_typing/phase6b_frozen_winners_v1")
    parser.add_argument("--sample_file", default="data/phase6b_single_samples/frozen_samples.json")
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/phase5e_coco.yaml")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--necessity_doses", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--necessity_seeds", type=int, default=10)
    parser.add_argument("--random_shell_seeds", type=int, default=3)
    parser.add_argument("--union_control_seeds", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2064)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_doses(value: str) -> list[float]:
    doses = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not doses or any(not 0 < dose <= 1 for dose in doses):
        raise ValueError(f"Necessity doses must lie in (0, 1], got {doses}.")
    return list(dict.fromkeys(doses))


def dose_label(dose: float) -> str:
    return f"{round(dose * 100):03d}pct"


def compact_proxy(proxy: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in proxy.items() if not key.startswith("per_token_")}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        evaluation_id = row.get("evaluation_id")
        if not evaluation_id or evaluation_id in rows:
            raise ValueError(f"Invalid or duplicate evaluation id at {path}:{line_number}.")
        rows[evaluation_id] = row
    return rows


def build_variants(
    winners: dict[str, LayerSets],
    layer_dims: dict[int, int],
    *,
    doses: list[float],
    necessity_seeds: int,
    random_shell_seeds: int,
    union_control_seeds: int,
    seed: int,
) -> list[dict[str, Any]]:
    sample_ids = list(winners)
    c7 = frequency_core(winners, 7)
    variants: list[dict[str, Any]] = []

    def add(family: str, variant_id: str, kept: LayerSets, targets: list[str], metadata: dict[str, Any]) -> None:
        summary = summarize_kept(kept, layer_dims)
        variants.append(
            {
                "family": family,
                "variant_id": variant_id,
                "kept": kept,
                "targets": targets,
                "metadata": metadata,
                "kept_sha256": kept_sha256(kept),
                "mask_summary": summary,
            }
        )

    for threshold in (7, 6, 5):
        core = frequency_core(winners, threshold)
        add(
            "core_sufficiency",
            f"core_frequency_ge_{threshold}",
            core,
            sample_ids,
            {"minimum_frequency": threshold},
        )

    for sample_id, winner in winners.items():
        for dose in doses:
            counts = dose_counts(c7, dose)
            label = dose_label(dose)
            core_replicates = 1 if dose == 1 else necessity_seeds
            for replicate in range(core_replicates):
                intervention_seed = seed + replicate
                removed = c7 if dose == 1 else deterministic_sample(c7, counts, seed=intervention_seed)
                add(
                    "core_necessity",
                    f"necessity__{sample_id}__dose_{label}__core__r{replicate:02d}",
                    layerwise_difference(winner, removed),
                    [sample_id],
                    {
                        "sample_id": sample_id,
                        "dose": dose,
                        "intervention": "core",
                        "replicate": replicate,
                        "removed_by_layer": {str(layer): count for layer, count in layer_counts(removed).items()},
                    },
                )
            for replicate in range(necessity_seeds):
                intervention_seed = seed + 10_000 + replicate
                removed = deterministic_sample(winner, counts, seed=intervention_seed)
                add(
                    "core_necessity",
                    f"necessity__{sample_id}__dose_{label}__random__r{replicate:02d}",
                    layerwise_difference(winner, removed),
                    [sample_id],
                    {
                        "sample_id": sample_id,
                        "dose": dose,
                        "intervention": "matched_random",
                        "replicate": replicate,
                        "removed_by_layer": {str(layer): count for layer, count in layer_counts(removed).items()},
                    },
                )

    outside_c7 = complement_pool(c7, layer_dims)
    for source_id, winner in winners.items():
        shell = layerwise_difference(winner, c7)
        reconstructed = layerwise_union(c7, shell)
        if reconstructed != winner:
            raise RuntimeError(f"Cannot reconstruct frozen winner {source_id} from C7 and its shell.")
        add(
            "shell_swap",
            f"shell_learned__{source_id}",
            reconstructed,
            sample_ids,
            {"source_sample_id": source_id, "shell_type": "learned", "replicate": None},
        )
        counts = layer_counts(shell)
        for replicate in range(random_shell_seeds):
            random_shell = deterministic_sample(outside_c7, counts, seed=seed + 20_000 + replicate)
            add(
                "shell_swap",
                f"shell_random__{source_id}__r{replicate:02d}",
                layerwise_union(c7, random_shell),
                sample_ids,
                {"source_sample_id": source_id, "shell_type": "matched_random", "replicate": replicate},
            )

    for left, right in itertools.combinations(sample_ids, 2):
        union = layerwise_union(winners[left], winners[right])
        pair_id = f"{left}__{right}"
        add(
            "pair_union",
            f"union__{pair_id}",
            union,
            [left, right],
            {"left": left, "right": right, "variant_type": "learned_union", "replicate": None},
        )
        for base, donor in ((left, right), (right, left)):
            increments = layerwise_difference(winners[donor], winners[base])
            pool = complement_pool(winners[base], layer_dims)
            counts = layer_counts(increments)
            for replicate in range(union_control_seeds):
                random_increment = deterministic_sample(pool, counts, seed=seed + 30_000 + replicate)
                add(
                    "pair_union",
                    f"union_control__{pair_id}__base_{base}__r{replicate:02d}",
                    layerwise_union(winners[base], random_increment),
                    [left, right],
                    {
                        "left": left,
                        "right": right,
                        "variant_type": "matched_random_expansion",
                        "base_sample_id": base,
                        "donor_sample_id": donor,
                        "replicate": replicate,
                    },
                )

    variant_ids = [variant["variant_id"] for variant in variants]
    if len(variant_ids) != len(set(variant_ids)):
        raise RuntimeError("Phase 6C generated duplicate variant IDs.")
    return variants


def result_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        passed = sum(row["automatic_semantic_pass"] for row in selected)
        by_family[family] = {
            "evaluations": len(selected),
            "semantic_passes": passed,
            "semantic_pass_rate": passed / len(selected),
        }
    return {"evaluations": len(rows), "by_family": by_family}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.necessity_seeds, args.random_shell_seeds, args.union_control_seeds, args.max_new_tokens) < 1:
        raise ValueError("Seed counts and --max_new_tokens must be positive.")
    doses = parse_doses(args.necessity_doses)
    winner_dir = Path(args.winner_dir).resolve()
    sample_path = Path(args.sample_file).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "phase6c_results.json"
    evaluations_path = output_dir / "evaluations.jsonl"
    if output_dir.exists() and not args.resume and (result_path.exists() or evaluations_path.exists()):
        raise FileExistsError(f"Phase 6C output already exists: {output_dir}. Pass --resume to continue.")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_payload = json.loads(sample_path.read_text(encoding="utf-8"))
    samples = validate_inputs(samples_payload, None)
    sample_ids = [sample["sample_id"] for sample in samples]
    winners_payload = json.loads((winner_dir / "frozen_winners.json").read_text(encoding="utf-8"))
    winner_rows = {row["sample_id"]: row for row in winners_payload["winners"]}
    if list(winner_rows) != sample_ids:
        raise ValueError("Frozen winner order does not match the frozen Phase 6B sample order.")
    layer_dims = {int(layer): int(width) for layer, width in next(iter(winner_rows.values()))["layer_dims"].items()}
    winners = {
        sample_id: kept_from_deleted(
            json.loads((winner_dir / sample_id / "mask.json").read_text(encoding="utf-8")), layer_dims
        )
        for sample_id in sample_ids
    }
    variants = build_variants(
        winners,
        layer_dims,
        doses=doses,
        necessity_seeds=args.necessity_seeds,
        random_shell_seeds=args.random_shell_seeds,
        union_control_seeds=args.union_control_seeds,
        seed=args.seed,
    )
    expected_evaluations = sum(len(variant["targets"]) for variant in variants)

    base_config = load_yaml(config_path)
    model_path = Path(args.model_name_or_path or base_config["model_name_or_path"]).resolve()
    run_config = {
        "winner_dir": str(winner_dir),
        "winner_manifest_sha256": sha256_file(winner_dir / "frozen_winners.json"),
        "sample_file": str(sample_path),
        "sample_file_sha256": sha256_file(sample_path),
        "config_path": str(config_path),
        "model_name_or_path": str(model_path),
        "necessity_doses": doses,
        "necessity_seeds": args.necessity_seeds,
        "random_shell_seeds": args.random_shell_seeds,
        "union_control_seeds": args.union_control_seeds,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "primary_endpoint": "strict_deterministic_free_generation_semantic_pass",
    }
    if args.resume and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing["config"] != run_config:
            raise ValueError("Cannot resume Phase 6C with a changed configuration.")

    variant_manifest = [{key: value for key, value in variant.items() if key != "kept"} for variant in variants]
    result = {
        "complete": False,
        "config": run_config,
        "sample_ids": sample_ids,
        "expected_variants": len(variants),
        "expected_evaluations": expected_evaluations,
        "evaluations_file": str(evaluations_path),
        "variants": variant_manifest,
        "summary": None,
    }
    write_json(result_path, result)

    mask_dir = output_dir / "structural_masks"
    mask_dir.mkdir(exist_ok=True)
    for variant in variants:
        if (
            variant["family"] in {"core_sufficiency", "pair_union"}
            and variant["metadata"].get("variant_type", "learned_union") == "learned_union"
        ):
            deleted = complement_pool(variant["kept"], layer_dims)
            write_json(mask_dir / f"{variant['variant_id']}.json", canonical_kept_ids(deleted))

    completed = load_jsonl(evaluations_path)
    unexpected = set(completed) - {
        f"{variant['variant_id']}__target_{target}" for variant in variants for target in variant["targets"]
    }
    if unexpected:
        raise ValueError(f"Existing evaluation rows are not part of this run: {sorted(unexpected)[:5]}.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer_module, template, config = load_model_bundle(
        config_path,
        model_path,
        device,
        trust_remote_code=False,
        preprocessing_num_workers=1,
    )
    if model_layer_dims(model) != layer_dims:
        raise ValueError("Original model layer widths differ from the frozen winners.")
    config.update(
        {
            "dataset_dir": str(prepare_dataset(output_dir, samples, samples_payload["canonical_prompt"])),
            "dataset": "phase6b_single_samples",
            "eval_dataset": None,
            "tokenized_path": None,
            "max_samples": len(samples),
            "overwrite_cache": True,
            "enable_thinking": False,
        }
    )
    dataloader, dataset_manifest = build_dataloader(
        config,
        model,
        tokenizer_module,
        template,
        batch_size=1,
        num_workers=0,
        sample_offset=0,
        max_samples=len(samples),
        allow_short_dataset=False,
        max_image_repeat=1,
        allow_excessive_image_repeats=False,
        dataset_stage="sft",
    )
    result["dataset_manifest"] = dataset_manifest
    tokenizer = tokenizer_module["tokenizer"]
    contexts = {}
    for sample, batch in zip(samples, dataloader, strict=True):
        batch = move_batch_to_device(batch, device)
        prompt_inputs, _ = build_prompt_inputs(batch)
        contexts[sample["sample_id"]] = {
            "batch": batch,
            "prompt_inputs": prompt_inputs,
            "reference_trace": build_reference_trace(model, batch),
            "semantic_evaluator": build_semantic_evaluator(sample["contract"]),
        }

    for variant_index, variant in enumerate(variants, start=1):
        pending_targets = [
            target for target in variant["targets"] if f"{variant['variant_id']}__target_{target}" not in completed
        ]
        if not pending_targets:
            continue
        masks = deletion_masks(variant["kept"], layer_dims)
        print(
            f"Phase 6C [{variant_index}/{len(variants)}] {variant['variant_id']} -> {','.join(pending_targets)}",
            flush=True,
        )
        with MLPNeuronAblator(model, masks):
            for target in pending_targets:
                context = contexts[target]
                proxy = compact_proxy(evaluate_gold_proxy(model, context["batch"], context["reference_trace"]))
                generation = generate_short_caption(
                    model,
                    tokenizer,
                    context["prompt_inputs"],
                    max_new_tokens=args.max_new_tokens,
                    semantic_evaluator=context["semantic_evaluator"],
                )
                evaluation_id = f"{variant['variant_id']}__target_{target}"
                row = {
                    "evaluation_id": evaluation_id,
                    "family": variant["family"],
                    "variant_id": variant["variant_id"],
                    "target_sample_id": target,
                    "kept_sha256": variant["kept_sha256"],
                    "mask_summary": variant["mask_summary"],
                    "variant_metadata": variant["metadata"],
                    "gold_proxy": proxy,
                    "generation": generation,
                    "automatic_semantic_pass": generation["semantic"]["automatic_pass"],
                }
                append_jsonl(evaluations_path, row)
                completed[evaluation_id] = row

    rows = list(completed.values())
    result["summary"] = result_summary(rows)
    result["complete"] = len(rows) == expected_evaluations
    write_json(result_path, result)
    print(json.dumps({"complete": result["complete"], **result["summary"]}, indent=2))
    if not result["complete"]:
        raise RuntimeError(f"Phase 6C produced {len(rows)} of {expected_evaluations} expected evaluations.")
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
