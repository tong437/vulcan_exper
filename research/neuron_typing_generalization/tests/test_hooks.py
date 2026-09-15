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

import torch
from neuron_typing_generalization.hooks import (
    DownProjectionCapture,
    StreamingImportanceCollector,
    discover_decoder_ffns,
    masks_from_spans,
)


class FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Linear(3, 2, bias=False)

    def forward(self, value):
        return self.down_proj(value)


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeMLP()

    def forward(self, value):
        return self.mlp(value)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])

    def forward(self, value):
        for layer in self.model.language_model.layers:
            value = torch.nn.functional.pad(layer(value), (0, 1))
        return value


class FakeAuxiliaryDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer()])


def test_discovery_capture_streaming_and_hook_removal():
    model = FakeModel()
    bindings = discover_decoder_ffns(model)
    assert [(item.layer, item.d_model, item.d_ff) for item in bindings] == [(0, 2, 3), (1, 2, 3)]
    roles = masks_from_spans(torch.ones(3), visual_spans=[(0, 1)], text_spans=[(1, 2)])
    collector = StreamingImportanceCollector(roles)
    callbacks = []

    with DownProjectionCapture(bindings, lambda layer, values, weight: callbacks.append((layer, values, weight))):
        model(torch.ones(3, 3))
    assert [item[0] for item in callbacks] == [0, 1]
    for layer, values, weight in callbacks:
        collector.update(layer, values, weight)
    assert list(collector.finalize()) == [0, 1]

    callbacks.clear()
    model(torch.ones(3, 3))
    assert callbacks == []


def test_span_masks_leave_template_positions_as_control_and_padding():
    roles = masks_from_spans(torch.tensor([1, 1, 1, 1, 0]), [(1, 2)], [(2, 4)])
    assert roles.visual.tolist() == [False, True, False, False, False]
    assert roles.text.tolist() == [False, False, True, True, False]
    assert roles.control.tolist() == [True, False, False, False, False]
    assert roles.padding.tolist() == [False, False, False, False, True]


def test_discovery_ignores_short_auxiliary_decoder_root():
    model = FakeModel()
    model.mtp = FakeAuxiliaryDecoder()
    bindings = discover_decoder_ffns(model)
    assert [binding.layer for binding in bindings] == [0, 1]
