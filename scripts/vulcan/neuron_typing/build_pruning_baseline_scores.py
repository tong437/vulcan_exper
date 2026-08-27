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

"""Add weight-magnitude and activation-frequency pruning baselines to a score table."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors import safe_open


WEIGHT_KEY_TEMPLATE = "model.language_model.layers.{layer}.mlp.{projection}.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FFN pruning-baseline scores.")
    parser.add_argument("--score_file", required=True, help="Phase-1 neuron_type_scores parquet file.")
    parser.add_argument("--model_path", required=True, help="Hugging Face model directory with safetensors weights.")
    parser.add_argument("--output_file", required=True, help="Output parquet; the source file is never overwritten.")
    return parser.parse_args()


def compute_group_weight_magnitude(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute one group-L2 magnitude per SwiGLU intermediate channel."""
    if gate_weight.ndim != 2 or up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("FFN projection weights must all be matrices.")
    if gate_weight.shape != up_weight.shape:
        raise ValueError(f"gate/up shape mismatch: {gate_weight.shape} versus {up_weight.shape}.")
    intermediate_size = gate_weight.shape[0]
    if down_weight.shape[1] != intermediate_size:
        raise ValueError(
            f"down_proj has {down_weight.shape[1]} input channels, expected {intermediate_size}."
        )
    magnitude_sq = gate_weight.float().square().sum(dim=1)
    magnitude_sq += up_weight.float().square().sum(dim=1)
    magnitude_sq += down_weight.float().square().sum(dim=0)
    return magnitude_sq.sqrt()


def _weight_map(model_path: Path) -> tuple[dict[str, str], list[str]]:
    index_path = model_path / "model.safetensors.index.json"
    single_path = model_path / "model.safetensors"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict):
            raise ValueError(f"Invalid safetensors index: {index_path}")
        return mapping, sorted(set(mapping.values()))
    if single_path.exists():
        with safe_open(single_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        return dict.fromkeys(keys, single_path.name), [single_path.name]
    raise FileNotFoundError(f"Cannot find model safetensors in {model_path}.")


def load_weight_magnitudes(model_path: str | Path, layer_dims: dict[int, int]) -> dict[int, torch.Tensor]:
    """Load only decoder FFN projections and return per-layer channel magnitudes."""
    root = Path(model_path)
    mapping, filenames = _weight_map(root)
    magnitudes: dict[int, torch.Tensor] = {}
    with ExitStack() as stack:
        handles = {
            filename: stack.enter_context(safe_open(root / filename, framework="pt", device="cpu"))
            for filename in filenames
        }
        for layer, expected_dim in sorted(layer_dims.items()):
            tensors = {}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                key = WEIGHT_KEY_TEMPLATE.format(layer=layer, projection=projection)
                if key not in mapping:
                    raise KeyError(f"Missing model weight {key!r}.")
                tensors[projection] = handles[mapping[key]].get_tensor(key)
            magnitude = compute_group_weight_magnitude(
                tensors["gate_proj"], tensors["up_proj"], tensors["down_proj"]
            )
            if magnitude.numel() != expected_dim:
                raise ValueError(
                    f"Layer {layer} produced {magnitude.numel()} magnitude values, expected {expected_dim}."
                )
            magnitudes[layer] = magnitude.cpu()
    return magnitudes


def add_pruning_baseline_scores(
    table: pd.DataFrame,
    magnitudes: dict[int, torch.Tensor],
) -> pd.DataFrame:
    """Join exact weight magnitude and threshold-response frequency by layer/neuron."""
    required = {"layer", "neuron_idx", "r_unknown"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Score table is missing required columns: {sorted(missing)}.")
    if table.duplicated(["layer", "neuron_idx"]).any():
        raise ValueError("Score table contains duplicate (layer, neuron_idx) rows.")

    magnitude_rows = []
    for layer, values in sorted(magnitudes.items()):
        magnitude_rows.extend(
            {"layer": layer, "neuron_idx": neuron, "weight_magnitude": float(value)}
            for neuron, value in enumerate(values.tolist())
        )
    magnitude_table = pd.DataFrame(magnitude_rows)
    result = table.merge(magnitude_table, on=["layer", "neuron_idx"], how="left", validate="one_to_one")
    if result["weight_magnitude"].isna().any():
        missing_rows = result.loc[result["weight_magnitude"].isna(), ["layer", "neuron_idx"]].head()
        raise ValueError(f"Missing weight magnitudes after join:\n{missing_rows}")
    result["activation_frequency"] = (1.0 - result["r_unknown"].astype(float)).clip(0.0, 1.0)
    return result


def build_metadata(
    args: argparse.Namespace,
    table: pd.DataFrame,
    layer_dims: dict[int, int],
) -> dict[str, Any]:
    return {
        "source_score_file": str(Path(args.score_file).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "output_file": str(Path(args.output_file).resolve()),
        "num_neurons": len(table),
        "layer_dims": {str(layer): dim for layer, dim in layer_dims.items()},
        "weight_magnitude_definition": (
            "sqrt(sum(gate_proj[j,:]^2) + sum(up_proj[j,:]^2) + sum(down_proj[:,j]^2))"
        ),
        "activation_frequency_definition": "1 - r_unknown",
    }


def main() -> None:
    args = parse_args()
    source = Path(args.score_file)
    output = Path(args.output_file)
    if source.resolve() == output.resolve():
        raise ValueError("output_file must differ from score_file; source scores are immutable.")
    table = pd.read_parquet(source)
    layer_dims = {
        int(layer): max(int(group["neuron_idx"].max()) + 1, int(group["neuron_idx"].nunique()))
        for layer, group in table.groupby("layer")
    }
    magnitudes = load_weight_magnitudes(args.model_path, layer_dims)
    result = add_pruning_baseline_scores(table, magnitudes)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(build_metadata(args, result, layer_dims), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(result)} neuron scores to {output}")
    print(f"Saved metadata to {metadata_path}")
    print(result[["weight_magnitude", "activation_frequency"]].describe().to_string())


if __name__ == "__main__":
    main()
