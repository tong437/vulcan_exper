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

import pytest
import torch
from neuron_typing_generalization.importance import (
    INVALID_CLASS,
    class_ids_to_labels,
    classify_neuron_shares,
    compute_modality_importance,
    normalize_importance,
    validate_token_masks,
)


def test_weighted_rms_matches_hand_calculation_and_ignores_sign():
    activations = torch.tensor([[3.0, -4.0], [-3.0, 4.0], [100.0, 100.0]])
    weight = torch.tensor([[3.0, 0.0], [4.0, 2.0]])
    result = compute_modality_importance(activations, weight, torch.tensor([True, True, False]))

    torch.testing.assert_close(result.activation_rms, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(result.down_projection_norm, torch.tensor([5.0, 2.0]))
    torch.testing.assert_close(result.importance, torch.tensor([15.0, 8.0]))
    assert result.token_count.item() == 2
    assert result.valid.item()


def test_batched_masks_and_empty_modality_are_explicitly_invalid():
    activations = torch.ones((2, 3, 2), dtype=torch.bfloat16)
    weight = torch.eye(2)
    masks = torch.tensor([[True, False, False], [False, False, False]])
    result = compute_modality_importance(activations, weight, masks)

    torch.testing.assert_close(result.importance[0], torch.ones(2))
    assert torch.isnan(result.importance[1]).all()
    assert result.valid.tolist() == [True, False]


def test_normalization_and_strict_uniform_share_four_way_classes():
    visual = normalize_importance(torch.tensor([[2.0, 1.0, 1.0, 0.0]])).values
    text = normalize_importance(torch.tensor([[1.0, 2.0, 1.0, 0.0]])).values
    result = classify_neuron_shares(visual, text)

    torch.testing.assert_close(visual.sum(dim=-1), torch.ones(1))
    assert result.threshold == 0.25
    assert result.classes.tolist() == [[1, 2, 0, 0]]
    assert class_ids_to_labels(result.classes[0]) == ["V", "T", "W", "W"]


def test_multimodal_class_and_invalid_zero_denominator():
    visual = normalize_importance(torch.tensor([[2.0, 1.0], [0.0, 0.0]]))
    text = normalize_importance(torch.tensor([[3.0, 1.0], [0.0, 0.0]]))
    result = classify_neuron_shares(visual.values, text.values, visual.valid, text.valid)

    assert result.classes[0].tolist() == [3, 0]
    assert result.classes[1].tolist() == [INVALID_CLASS, INVALID_CLASS]
    assert not result.valid[1].any()


def test_token_roles_must_be_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_token_masks(torch.tensor([True]), torch.tensor([True]))


def test_shapes_are_rejected_before_broadcasting():
    with pytest.raises(ValueError, match="must match"):
        compute_modality_importance(torch.ones(2, 3), torch.ones(4, 3), torch.ones(3, dtype=torch.bool))
