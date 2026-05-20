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
from pathlib import Path
from typing import Any


Cluster = dict[str, Any]
ClusterIdx = list[list[Cluster] | None]


def validate_cluster_idx(cluster_idx: Any) -> ClusterIdx:
    if not isinstance(cluster_idx, list):
        raise ValueError("cluster_idx must be a list whose length equals the number of MLP layers.")

    for layer_idx, layer_clusters in enumerate(cluster_idx):
        if layer_clusters is None:
            continue

        if not isinstance(layer_clusters, list):
            raise ValueError(f"cluster_idx[{layer_idx}] must be a list or null.")

        for cluster_idx_in_layer, cluster in enumerate(layer_clusters):
            if not isinstance(cluster, dict):
                raise ValueError(f"cluster_idx[{layer_idx}][{cluster_idx_in_layer}] must be an object.")

            if "anchor" not in cluster or "neuron" not in cluster:
                raise ValueError(f"cluster_idx[{layer_idx}][{cluster_idx_in_layer}] must contain anchor and neuron.")

            if not isinstance(cluster["anchor"], int):
                raise ValueError(f"cluster_idx[{layer_idx}][{cluster_idx_in_layer}].anchor must be an integer.")

            if not isinstance(cluster["neuron"], list) or not all(isinstance(i, int) for i in cluster["neuron"]):
                raise ValueError(f"cluster_idx[{layer_idx}][{cluster_idx_in_layer}].neuron must be a list of ints.")

    return cluster_idx


def load_cluster_idx(path: str | Path) -> ClusterIdx:
    with Path(path).open(encoding="utf-8") as f:
        return validate_cluster_idx(json.load(f))


def save_cluster_idx(cluster_idx: ClusterIdx, path: str | Path) -> None:
    validated_cluster_idx = validate_cluster_idx(cluster_idx)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(validated_cluster_idx, f, indent=2)
        f.write("\n")
