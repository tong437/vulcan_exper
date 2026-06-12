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

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5MLP


class VulcanQwen3_5Config(Qwen3_5Config):
    r"""Qwen3.5 config carrying one MLP intermediate width per decoder layer."""

    # Keep the native model type so LlamaFactory's Qwen3.5 patches and
    # multimodal freeze rules continue to apply after reload.
    model_type = "qwen3_5"

    def __init__(self, vulcan_intermediate_sizes=None, **kwargs):
        super().__init__(**kwargs)
        self.vulcan_intermediate_sizes = vulcan_intermediate_sizes


class VulcanQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    r"""Qwen3.5 variant that reconstructs non-uniform Vulcan-pruned MLPs."""

    config_class = VulcanQwen3_5Config

    def __init__(self, config):
        super().__init__(config)
        intermediate_sizes = getattr(config, "vulcan_intermediate_sizes", None)
        if intermediate_sizes is None:
            intermediate_sizes = getattr(config.text_config, "vulcan_intermediate_sizes", None)
        if intermediate_sizes is None:
            return

        layers = self.model.language_model.layers
        if len(intermediate_sizes) != len(layers):
            raise ValueError(
                f"vulcan_intermediate_sizes has {len(intermediate_sizes)} entries, but the model has {len(layers)} layers."
            )

        for layer, intermediate_size in zip(layers, intermediate_sizes):
            layer.mlp = Qwen3_5MLP(config.text_config, int(intermediate_size))


__all__ = ["VulcanQwen3_5Config", "VulcanQwen3_5ForConditionalGeneration"]
