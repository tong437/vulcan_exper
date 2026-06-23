# Copyright 2026 the LlamaFactory team.
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

import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "vulcan" / "profile_vision_encoder.py"
SPEC = importlib.util.spec_from_file_location("profile_vision_encoder", SCRIPT_PATH)
profile = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(profile)


class DummyVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Linear(4, 4, bias=False)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False) for _ in range(2)])
        self.merger = torch.nn.Linear(4, 4, bias=False)

    def forward(self, inputs):
        hidden_states = self.patch_embed(inputs)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.merger(hidden_states)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = DummyVisual()
        self.language_model = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 4, bias=False)


def test_resolve_nested_vision_modules():
    model = DummyModel()
    modules = profile._resolve_modules(model, ["visual.blocks"])
    assert [name for name, _ in modules] == ["visual.blocks.0", "visual.blocks.1"]
    assert profile._common_module_path(["visual.patch_embed", "visual.blocks"]) == "visual"


def test_parameter_summary_does_not_double_count_projector():
    model = DummyModel()
    summary = profile._parameter_summary(
        model,
        vision_paths=["visual"],
        projector_paths=["visual.merger"],
        language_paths=["language_model", "lm_head"],
    )
    assert summary["vision_encoder"]["parameters"] == 48
    assert summary["projector"]["parameters"] == 16
    assert summary["language_model"]["parameters"] == 32
    assert summary["other"]["parameters"] == 0
    assert sum(summary[group]["parameters"] for group in ["vision_encoder", "projector", "language_model", "other"]) == (
        summary["total"]["parameters"]
    )


def test_cpu_module_timer_records_pipeline_and_projector():
    model = DummyModel()
    timer = profile.ModuleTimer(use_cuda=False)
    timer.register("vision_pipeline_ms", [("visual", model.visual)])
    timer.register("projector_ms", [("visual.merger", model.visual.merger)])
    model.visual(torch.ones(1, 4))
    elapsed = timer.elapsed_ms()
    timer.close()
    assert elapsed["vision_pipeline_ms"] >= elapsed["projector_ms"] >= 0.0
