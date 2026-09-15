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

"""Filesystem contracts between experiment stages."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StageContract:
    """Required inputs and promised outputs, relative to one run directory."""

    requires: tuple[str, ...]
    produces: tuple[str, ...]


STAGE_ORDER = (
    "collect_activations",
    "compute_importance",
    "classify_neurons",
    "compare_distributions",
    "evaluate",
)

STAGE_CONTRACTS = {
    "collect_activations": StageContract(
        requires=("samples/sample_manifest.parquet", "samples/pair_manifest.parquet"),
        produces=("activations/sample_parts", "activations/down_projection_norms.npy", "activations/manifest.json"),
    ),
    "compute_importance": StageContract(
        requires=("activations/manifest.json",),
        produces=("scores/sample_parts", "scores/manifest.json"),
    ),
    "classify_neurons": StageContract(
        requires=("scores/manifest.json",),
        produces=("classifications/sample_parts", "classifications/manifest.json"),
    ),
    "compare_distributions": StageContract(
        requires=(
            "classifications/manifest.json",
            "samples/sample_manifest.parquet",
            "samples/pair_manifest.parquet",
        ),
        produces=(
            "distributions/pairwise_changes.parquet",
            "distributions/neuron_profiles.parquet",
            "distributions/cohort_summary.parquet",
            "distributions/transition_matrices.npz",
            "distributions/transition_manifest.json",
            "distributions/analysis_summary.json",
        ),
    ),
    "evaluate": StageContract(
        requires=(
            "distributions/pairwise_changes.parquet",
            "distributions/neuron_profiles.parquet",
            "distributions/cohort_summary.parquet",
            "distributions/transition_matrices.npz",
            "distributions/transition_manifest.json",
            "distributions/analysis_summary.json",
        ),
        produces=(
            "metrics/summary.json",
            "metrics/pilot_statistics.json",
            "metrics/reproducibility.json",
            "figures/manifest.json",
        ),
    ),
}
