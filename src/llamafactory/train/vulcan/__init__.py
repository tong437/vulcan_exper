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

from .activation_align import ActivationAligner
from .clustering import (
    MultimodalActivationStats,
    build_layerwise_cluster_idx,
    build_multimodal_cluster_idx,
    build_third_keep_ratios,
    build_uniform_cluster_idx,
    collect_mlp_activations,
    collect_multimodal_mlp_activations,
    get_cluster_greedy_match,
)
from .collapse_loss import (
    get_collapse_lambdas,
    get_collapse_schedule_factor,
    init_collapse_lambdas,
    weight_collapse_loss,
)
from .modeling import find_mlp_layers
from .pruning import pruning_mlp
from .schema import load_cluster_idx, save_cluster_idx


__all__ = [
    "ActivationAligner",
    "MultimodalActivationStats",
    "build_layerwise_cluster_idx",
    "build_multimodal_cluster_idx",
    "build_third_keep_ratios",
    "build_uniform_cluster_idx",
    "collect_mlp_activations",
    "collect_multimodal_mlp_activations",
    "find_mlp_layers",
    "get_cluster_greedy_match",
    "get_collapse_lambdas",
    "get_collapse_schedule_factor",
    "init_collapse_lambdas",
    "load_cluster_idx",
    "pruning_mlp",
    "save_cluster_idx",
    "weight_collapse_loss",
]
