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

"""Small explicit registries for research adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


Factory = Callable[[dict[str, Any]], Any]


class ComponentRegistry:
    """Map configuration names to factories while rejecting accidental replacement."""

    def __init__(self) -> None:
        self._factories: dict[str, Factory] = {}

    def register(self, name: str, factory: Factory) -> None:
        if name in self._factories:
            raise ValueError(f"Component is already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str, config: dict[str, Any]) -> Any:
        try:
            factory = self._factories[name]
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"Unknown component '{name}'. Available: {available}") from error
        return factory(config)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


MODEL_ADAPTERS = ComponentRegistry()
DATASET_ADAPTERS = ComponentRegistry()
IMPORTANCE_METRICS = ComponentRegistry()
