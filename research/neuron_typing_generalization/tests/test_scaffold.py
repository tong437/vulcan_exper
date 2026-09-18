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

import pandas as pd
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


def test_vqav2_gpu_studies_expand_to_the_same_three_models():
    expected = {"qwen3_5_2b", "llava_1_5_7b", "gemma_4_e2b_it"}
    for name, tier in (("vqav2_three_model_smoke.yaml", "gpu_smoke"), ("vqav2_three_model_pilot.yaml", "pilot")):
        specs = expand_study(load_study(EXPERIMENT_ROOT / "configs/studies" / name))
        assert {spec.model.name for spec in specs} == expected
        assert {spec.dataset.name for spec in specs} == {"vqav2"}
        assert {spec.study_values["sampling"]["manifest_tier"] for spec in specs} == {tier}


def test_remaining_dataset_study_uses_audited_clevr_tier(tmp_path, monkeypatch):
    study = load_study(EXPERIMENT_ROOT / "configs/studies/remaining_three_dataset_pilot.yaml")
    specs = expand_study(study)
    assert len(specs) == 9
    clevr = next(spec for spec in specs if spec.model.name == "qwen3_5_2b" and spec.dataset.name == "clevr")
    monkeypatch.setenv("NEURON_TYPING_ARTIFACT_ROOT", str(tmp_path))

    context = initialize_run(clevr)
    samples = pd.read_parquet(context.run_dir / "samples/sample_manifest.parquet")
    pairs = pd.read_parquet(context.run_dir / "samples/pair_manifest.parquet")

    assert set(samples["tier"]) == {"audited_v1"}
    assert set(pairs["tier"]) == {"audited_v1"}
    assert len(samples) == 128
    assert len(pairs) == 64


def test_clevr_precision_extension_uses_a_new_frozen_tier():
    study = load_study(EXPERIMENT_ROOT / "configs/studies/clevr_audited_v2_111.yaml")
    specs = expand_study(study)

    assert len(specs) == 3
    assert {spec.dataset.name for spec in specs} == {"clevr"}
    assert {spec.study_values["sampling"]["manifest_tier"] for spec in specs} == {"audited_v2_111"}


def test_unpaired_neuron_study_expands_to_twelve_independent_runs():
    study = load_study(EXPERIMENT_ROOT / "configs/studies/unpaired_neuron_v1_1000.yaml")
    specs = expand_study(study)

    assert len(specs) == 12
    assert {spec.dataset.name for spec in specs} == {"vqav2", "gqa_balanced", "clevr", "vizwiz_vqa"}
    assert {spec.study_values["sampling"]["manifest_tier"] for spec in specs} == {"unpaired_v1_1000"}
    assert {spec.study_values["sampling"]["expected_samples"] for spec in specs} == {1000}
    assert {spec.study_values["sampling"]["require_pair_manifest"] for spec in specs} == {False}


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
