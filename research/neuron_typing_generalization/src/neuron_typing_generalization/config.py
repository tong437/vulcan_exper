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

"""Typed loading and expansion of study configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ComponentConfig:
    """A named model, dataset, or metric configuration."""

    name: str
    source: Path
    values: dict[str, Any]


@dataclass(frozen=True)
class StudyConfig:
    """A complete, unexpanded study definition."""

    name: str
    source: Path
    artifact_root: str
    models: tuple[ComponentConfig, ...]
    datasets: tuple[ComponentConfig, ...]
    metrics: tuple[ComponentConfig, ...]
    seeds: tuple[int, ...]
    values: dict[str, Any]


@dataclass(frozen=True)
class ExperimentSpec:
    """One model/dataset/seed run with all importance metrics attached."""

    study: str
    model: ComponentConfig
    dataset: ComponentConfig
    metrics: tuple[ComponentConfig, ...]
    seed: int
    artifact_root: str
    study_values: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly resolved configuration."""
        return {
            "study": self.study,
            "seed": self.seed,
            "model": self.model.values,
            "dataset": self.dataset.values,
            "metrics": [metric.values for metric in self.metrics],
            "study_options": self.study_values,
        }


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return value


def _load_components(study_path: Path, references: Any, kind: str) -> tuple[ComponentConfig, ...]:
    if not isinstance(references, list) or not references:
        raise ValueError(f"Study field '{kind}' must be a non-empty list.")

    components = []
    for reference in references:
        component_path = (study_path.parent / str(reference)).resolve()
        values = _read_mapping(component_path)
        name = values.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Component is missing a non-empty name: {component_path}")
        components.append(ComponentConfig(name=name, source=component_path, values=values))
    return tuple(components)


def load_study(path: str | Path) -> StudyConfig:
    """Load a study and all referenced component configurations."""
    study_path = Path(path).resolve()
    values = _read_mapping(study_path)
    name = values.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Study field 'name' must be a non-empty string.")

    seeds = values.get("seeds", [42])
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("Study field 'seeds' must be a non-empty list of integers.")

    reserved = {"name", "artifact_root", "models", "datasets", "metrics", "seeds"}
    return StudyConfig(
        name=name,
        source=study_path,
        artifact_root=str(values.get("artifact_root", "artifacts/neuron_typing_generalization")),
        models=_load_components(study_path, values.get("models"), "models"),
        datasets=_load_components(study_path, values.get("datasets"), "datasets"),
        metrics=_load_components(study_path, values.get("metrics"), "metrics"),
        seeds=tuple(seeds),
        values={key: value for key, value in values.items() if key not in reserved},
    )


def expand_study(study: StudyConfig) -> list[ExperimentSpec]:
    """Expand a study into model/dataset/seed runs without duplicating activation collection per metric."""
    return [
        ExperimentSpec(
            study=study.name,
            model=model,
            dataset=dataset,
            metrics=study.metrics,
            seed=seed,
            artifact_root=study.artifact_root,
            study_values=study.values,
        )
        for model, dataset, seed in product(study.models, study.datasets, study.seeds)
    ]
