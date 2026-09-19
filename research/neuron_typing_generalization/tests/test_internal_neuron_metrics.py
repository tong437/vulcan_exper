# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from neuron_typing_generalization.internal_neuron_metrics import collect_activation_rms_profile


def test_collect_activation_rms_profile_streams_exact_per_neuron_moments(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "scores/sample_parts").mkdir(parents=True)
    (run_dir / "distributions").mkdir()
    parts = []
    samples = (
        np.asarray([[[3.0, 4.0], [4.0, 3.0]]], dtype=np.float32),
        np.asarray([[[5.0, 8.0], [12.0, 6.0]]], dtype=np.float32),
    )
    for index, values in enumerate(samples):
        path = run_dir / f"scores/sample_parts/{index:06d}.npz"
        np.savez(
            path,
            activation_rms=values,
            token_count=np.asarray([[1, 1]], dtype=np.int32),
            valid=np.asarray([[True, True]]),
        )
        parts.append({"sample_index": index, "sample_id": str(index), "file": str(path.relative_to(run_dir))})
    (run_dir / "scores/manifest.json").write_text(
        json.dumps({"samples": 2, "layers": 1, "d_ff_by_layer": [2], "parts": parts})
    )
    pd.DataFrame(
        {
            "layer": [0, 0],
            "neuron": [0, 1],
            "change_rate_distinct_pairs": [0.0, 1.0],
            "normalized_entropy": [0.0, 0.5],
            "effective_class_count": [1.0, 2.0],
            "dominant_class": ["W", "V"],
            "dominant_proportion": [1.0, 0.5],
        }
    ).to_parquet(run_dir / "distributions/neuron_profiles.parquet", index=False)

    result = collect_activation_rms_profile(run_dir, tmp_path / "output")
    profile = pd.read_parquet(tmp_path / "output" / result["profile"])

    assert profile["activation_rms_visual_mean"].tolist() == pytest.approx([4.0, 6.0])
    assert profile["activation_rms_visual_std"].tolist() == pytest.approx([1.0, 2.0])
    assert profile["activation_rms_text_mean"].tolist() == pytest.approx([8.0, 4.5])
    assert profile["activation_rms_text_std"].tolist() == pytest.approx([4.0, 1.5])
    expected_first_pooled = np.mean([np.sqrt(12.5), np.sqrt(84.5)])
    assert profile.iloc[0]["activation_rms_pooled_mean"] == pytest.approx(expected_first_pooled)
    assert profile["effective_class_count"].tolist() == [1.0, 2.0]
    assert result["samples"] == 2
