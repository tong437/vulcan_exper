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

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_phase1_stability import (  # noqa: E402
    per_layer_jaccard,
    per_layer_spearman,
    validate_prefix_experiment,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_prefix_validation_proves_same_calibration_and_rows(tmp_path):
    calibration = tmp_path / "neuron_quantiles.pt"
    calibration.write_bytes(b"same calibration")
    common = {
        "model_name": "model",
        "sample_offset": 500,
        "threshold_mode": "quantile",
        "quantile_path": str(calibration),
        "quantile_idx_visual": 1,
        "quantile_idx_text": 0,
    }
    small_config = tmp_path / "small_config.json"
    large_config = tmp_path / "large_config.json"
    _write_json(small_config, {**common, "actual_samples": 2})
    _write_json(large_config, {**common, "actual_samples": 3})

    manifest_common = {"dataset": "data", "tokenized_path": "cache", "role": "typing"}
    small_manifest = tmp_path / "small_manifest.json"
    large_manifest = tmp_path / "large_manifest.json"
    _write_json(
        small_manifest,
        {
            **manifest_common,
            "num_rows": 2,
            "source_indices": [500, 501],
            "row_image_ids": [["a"], ["b"]],
        },
    )
    _write_json(
        large_manifest,
        {
            **manifest_common,
            "num_rows": 3,
            "source_indices": [500, 501, 502],
            "row_image_ids": [["a"], ["b"], ["c"]],
        },
    )

    result = validate_prefix_experiment(small_config, large_config, small_manifest, large_manifest)
    assert result["validated"]
    assert result["same_calibration_sha256"]
    assert result["strict_typing_prefix"]
    assert result["small_samples"] == 2
    assert result["large_samples"] == 3


def test_prefix_validation_rejects_nonprefix_rows(tmp_path):
    calibration = tmp_path / "neuron_quantiles.pt"
    calibration.write_bytes(b"same calibration")
    config = {
        "model_name": "model",
        "sample_offset": 500,
        "threshold_mode": "quantile",
        "quantile_path": str(calibration),
        "quantile_idx_visual": 1,
        "quantile_idx_text": 0,
    }
    small_config = tmp_path / "small_config.json"
    large_config = tmp_path / "large_config.json"
    _write_json(small_config, {**config, "actual_samples": 1})
    _write_json(large_config, {**config, "actual_samples": 2})
    small_manifest = tmp_path / "small_manifest.json"
    large_manifest = tmp_path / "large_manifest.json"
    manifest = {"dataset": "data", "tokenized_path": "cache", "role": "typing"}
    _write_json(
        small_manifest,
        {**manifest, "num_rows": 1, "source_indices": [500], "row_image_ids": [["wrong"]]},
    )
    _write_json(
        large_manifest,
        {**manifest, "num_rows": 2, "source_indices": [500, 501], "row_image_ids": [["a"], ["b"]]},
    )

    with pytest.raises(ValueError, match="exact prefix"):
        validate_prefix_experiment(small_config, large_config, small_manifest, large_manifest)


def test_layerwise_stability_summaries():
    merged = pd.DataFrame(
        {
            "layer": [0, 0, 0, 1, 1, 1],
            "q_multimodal_small": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "q_multimodal_large": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )
    spearman = per_layer_spearman(merged, "q_multimodal")
    assert spearman["mean"] == pytest.approx(0.0)
    assert spearman["per_layer"] == {"0": 1.0, "1": -1.0}

    jaccard = per_layer_jaccard({(0, 0), (1, 0)}, {(0, 0), (1, 1)}, [0, 1])
    assert jaccard["mean"] == pytest.approx(0.5)
    assert jaccard["per_layer"] == {"0": 1.0, "1": 0.0}
