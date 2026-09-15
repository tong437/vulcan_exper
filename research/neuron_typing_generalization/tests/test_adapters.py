# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest
import torch
from neuron_typing_generalization.adapters import (
    ModelLayout,
    find_question_token_span,
    find_unique_token_span,
    resolve_expanded_roles,
)


@pytest.mark.parametrize(
    ("family", "image_token", "visual_count"),
    [("qwen3_5", 90, 4), ("llava", 91, 3), ("gemma4", 92, 2)],
)
def test_supported_family_layouts_resolve_expanded_visual_and_question_tokens(
    family: str, image_token: int, visual_count: int
):
    layout = ModelLayout(family, "Fake", Path("/unused"), 2, 4, image_token, visual_count)
    ids = torch.tensor([1, *([image_token] * visual_count), 10, 11, 2])
    roles = resolve_expanded_roles(layout, ids, torch.ones_like(ids), torch.tensor([10, 11]))
    assert int(roles.visual.sum()) == visual_count
    assert roles.text.tolist()[-3:-1] == [True, True]
    assert not torch.any(roles.visual & roles.text)


def test_question_span_rejects_missing_and_ambiguous_sequences():
    with pytest.raises(ValueError, match="found 0"):
        find_unique_token_span(torch.tensor([1, 2]), torch.tensor([3]))
    with pytest.raises(ValueError, match="found 2"):
        find_unique_token_span(torch.tensor([1, 2, 1]), torch.tensor([1]))


def test_visual_count_contract_rejects_unexpanded_placeholder():
    layout = ModelLayout("llava", "Fake", Path("/unused"), 2, 4, 90, 4)
    with pytest.raises(ValueError, match="expected at least 4"):
        resolve_expanded_roles(layout, torch.tensor([90, 10]), torch.ones(2), torch.tensor([10]))


class _ContextSensitiveTokenizer:
    def __call__(self, value, add_special_tokens=False, return_tensors="pt"):
        del add_special_tokens, return_tensors
        values = {
            "": [],
            "question": [10, 11],
            "\n": [7],
            "\nquestion": [7, 12, 11],
            " ": [8],
            " question": [8, 10, 11],
            "\n\n": [7, 7],
            "\n\nquestion": [7, 7, 12, 11],
        }
        return {"input_ids": torch.tensor([values[value]])}


def test_question_span_uses_chat_boundary_context_without_masking_prefix():
    span = find_question_token_span(torch.tensor([1, 7, 12, 11, 2]), _ContextSensitiveTokenizer(), "question")
    assert span == (2, 4)
