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

"""Extension interfaces that isolate model, dataset, and metric differences."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from torch import Tensor, nn


@dataclass(frozen=True)
class TokenRoleMasks:
    """Mutually exclusive decoder positions used by the experiment."""

    visual: Tensor
    text: Tensor
    control: Tensor
    padding: Tensor


@dataclass(frozen=True)
class PreparedInputs:
    """Model inputs paired with auditable token-role masks."""

    model_inputs: Mapping[str, Any]
    token_roles: TokenRoleMasks


@dataclass(frozen=True)
class FFNBinding:
    """One decoder layer's down projection and stable neuron dimensions."""

    layer: int
    down_projection: nn.Module
    d_model: int
    d_ff: int


class ModelAdapter(Protocol):
    """Expose model-specific loading, tokenization, and neuron hooks."""

    family: str

    def prepare_inputs(self, sample: Mapping[str, Any]) -> PreparedInputs: ...

    def iter_ffn_bindings(self) -> Iterable[FFNBinding]: ...


ActivationCallback = Callable[[int, Tensor, Tensor], None]


class DatasetAdapter(Protocol):
    """Yield stable sample identities and payloads from one frozen split."""

    def iter_samples(self) -> Iterable[Mapping[str, Any]]: ...

    def write_manifest(self, destination: Path) -> None: ...


class ImportanceMetric(Protocol):
    """Accumulate and finalize one neuron-importance definition."""

    name: str

    def update(self, batch: Mapping[str, Any]) -> None: ...

    def finalize(self) -> Mapping[str, Any]: ...
