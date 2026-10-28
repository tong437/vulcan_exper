# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EOS-aware comparisons; retain the first terminal token, exclude subsequent padding."""

import hashlib
import json
from pathlib import Path


def trim_at_eos(tokens, eos_token_id):
    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id])
    return tokens[: next((i + 1 for i, token in enumerate(tokens) if token in eos), len(tokens))]


def main():
    root = Path("artifacts/neuron_typing_generalization/structured_throughput_1000_2026_10")
    models = {
        "qwen35": "Qwen3.5-2B",
        "qwen3_vl_2b_instruct": "Qwen3-VL-2B-Instruct",
        "llava_1_5_7b": "llava-1.5-7b-hf",
        "gemma_4_e2b_it": "gemma-4-E2B-it",
    }
    reports = []
    for model, folder in models.items():
        model_root = Path("/root/autodl-pub-RTX4090-hdd-1/models") / folder
        source = model_root / "generation_config.json"
        if source.exists():
            eos = json.loads(source.read_text())["eos_token_id"]
        else:
            source = model_root / "config.json"
            config = json.loads(source.read_text())
            eos = config.get("eos_token_id", config.get("text_config", {}).get("eos_token_id"))
        if eos is None:
            raise ValueError(f"Missing generation EOS for {model}")
        directory = root / f"diagnostics_{model}_bf16_128"
        for batch in (1, 2):
            paths = [directory / f"{label}_b{batch}.json" for label in ("masked", "sliced")]
            a, b = [json.loads(p.read_text())["rows"] for p in paths]
            assert len(a) == len(b) == 128
            pairs = []
            for left, right in zip(a, b):
                assert left["sample_id"] == right["sample_id"]
                lt, rt = [trim_at_eos(row["tokens"], eos) for row in (left, right)]
                first = next((i for i, (x, y) in enumerate(zip(lt, rt)) if x != y), None)
                if first is None and len(lt) != len(rt):
                    first = min(len(lt), len(rt))
                pairs.append({
                    "sample_id": left["sample_id"], "old_equal": left["tokens"] == right["tokens"],
                    "equal": lt == rt, "masked_tokens": lt, "sliced_tokens": rt,
                    "masked_text": left["text"], "sliced_text": right["text"],
                    "first_difference": first,
                    "score_changed": left["score"] != right["score"],
                })
            report = {
                "model": model, "batch_size": batch, "n": len(pairs), "eos_token_id": eos,
                "eos_source": str(source), "eos_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "old_mismatches": sum(not p["old_equal"] for p in pairs),
                "corrected_mismatches": sum(not p["equal"] for p in pairs),
                "first_token_mismatches": sum(p["first_difference"] == 0 for p in pairs),
                "score_changed_samples": sum(p["score_changed"] for p in pairs),
                "pairs": pairs,
            }
            reports.append(report)
            print(json.dumps({k: v for k, v in report.items() if k not in {"pairs", "input_sha256"}}))
    output = root / "eos_corrected_sequence_audit.json"
    output.write_text(json.dumps({
        "scope": "Reanalysis of saved sequences, no new inference; original artifacts preserved",
        "policy": "First effective generation EOS inclusive; ignore subsequent tokens",
        "results": reports,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
