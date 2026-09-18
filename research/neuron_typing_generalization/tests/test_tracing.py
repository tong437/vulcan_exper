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
from neuron_typing_generalization.tracing import export_neuron_trace


def test_export_neuron_trace_aligns_sample_scores_classes_and_metadata(tmp_path: Path):
    for directory in ("scores/sample_parts", "classifications/sample_parts", "samples"):
        (tmp_path / directory).mkdir(parents=True)
    pd.DataFrame(
        [
            {"sample_id": "a", "image_id": "ia", "question": "first?", "answer": "yes", "stratum": "s1"},
            {"sample_id": "b", "image_id": "ib", "question": "second?", "answer": "no", "stratum": "s2"},
        ]
    ).to_parquet(tmp_path / "samples/sample_manifest.parquet", index=False)
    score_parts, class_parts = [], []
    for index, (sample_id, q_values, class_id) in enumerate((("a", (0.2, 0.8), 2), ("b", (0.7, 0.3), 1))):
        score_path = tmp_path / f"scores/sample_parts/{index:06d}.npz"
        class_path = tmp_path / f"classifications/sample_parts/{index:06d}.npz"
        q = np.asarray([[[0.1, q_values[0]], [0.1, q_values[1]]]], dtype=np.float32)
        np.savez(
            score_path,
            sample_id=np.asarray(sample_id),
            q=q,
            importance=q,
            activation_rms=q,
            token_count=np.asarray([[4, 2]], dtype=np.int32),
        )
        np.savez(class_path, sample_id=np.asarray(sample_id), classes=np.asarray([[0, class_id]], dtype=np.int8))
        score_parts.append(
            {"sample_index": index, "sample_id": sample_id, "file": str(score_path.relative_to(tmp_path))}
        )
        class_parts.append(
            {"sample_index": index, "sample_id": sample_id, "file": str(class_path.relative_to(tmp_path))}
        )
    base = {"samples": 2, "layers": 1, "d_ff": 2, "d_ff_by_layer": [2]}
    (tmp_path / "scores/manifest.json").write_text(json.dumps({**base, "parts": score_parts}))
    (tmp_path / "classifications/manifest.json").write_text(json.dumps({**base, "parts": class_parts}))

    result = export_neuron_trace(tmp_path, layer=0, neuron=1)

    trace = pd.read_parquet(result["parquet"])
    assert trace["sample_id"].tolist() == ["a", "b"]
    assert trace["class"].tolist() == ["T", "V"]
    assert trace["q_visual"].tolist() == pytest.approx([0.2, 0.7])
    assert Path(result["csv"]).is_file()
    assert Path(result["figure"]).is_file()
