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

from pathlib import Path

import yaml
from neuron_typing_generalization.config import expand_study, load_study
from neuron_typing_generalization.registry import ComponentRegistry
from neuron_typing_generalization.runtime import build_run_id, initialize_run


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_main_study_expands_without_recollecting_per_metric():
    study = load_study(EXPERIMENT_ROOT / "configs/studies/main.yaml")
    specs = expand_study(study)

    assert len(specs) == 12
    assert {metric.name for metric in specs[0].metrics} == {"down_projection_weighted_rms"}
    assert {spec.dataset.name for spec in specs} == {"vqav2", "gqa_balanced", "clevr", "vizwiz_vqa"}
    assert len({build_run_id(spec) for spec in specs}) == 12


def test_initialize_run_writes_resolved_config_and_layout(tmp_path, monkeypatch):
    study = load_study(EXPERIMENT_ROOT / "configs/studies/main.yaml")
    spec = expand_study(study)[0]
    monkeypatch.setenv("NEURON_TYPING_ARTIFACT_ROOT", str(tmp_path))

    context = initialize_run(spec)

    resolved = yaml.safe_load((context.run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved == spec.as_dict()
    assert (context.run_dir / "metadata.json").is_file()
    assert (context.run_dir / "activations").is_dir()
    assert (context.run_dir / "distributions").is_dir()
    assert (context.run_dir / "samples/sample_manifest.parquet").is_file()
    assert (context.run_dir / "samples/pair_manifest.parquet").is_file()


def test_registry_rejects_duplicate_names():
    registry = ComponentRegistry()
    registry.register("example", lambda config: config)

    try:
        registry.register("example", lambda config: config)
    except ValueError as error:
        assert "already registered" in str(error)
    else:
        raise AssertionError("Duplicate registration must fail.")
