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

from typing import TYPE_CHECKING, Literal

import torch

from .modeling import find_mlp_layers


if TYPE_CHECKING:
    from torch import nn

    from ...hparams import FinetuningArguments


class ActivationAligner:
    r"""Top-k activation mask alignment between visual and text tokens.

    Registers forward hooks on each MLP ``down_proj`` layer to capture the
    intermediate activation  ``A = silu(gate) * up``.  Before each forward
    pass the trainer calls :meth:`set_input_ids` so the hook knows which
    tokens are visual vs text.  After the forward pass
    :meth:`compute_alignment_loss` derives soft top-k masks from pooled
    per-modality activations and returns a differentiable regularisation loss.
    """

    def __init__(
        self,
        model: "nn.Module",
        finetuning_args: "FinetuningArguments",
        image_token_id: int,
    ) -> None:
        self.model = model
        self.lambda_ = finetuning_args.align_lambda
        self.temperature = finetuning_args.align_temperature
        self.quantile = finetuning_args.align_quantile
        self.pool_type: Literal["mean", "max"] = finetuning_args.align_pool_type
        self.loss_type: Literal["l1", "neg_iou"] = finetuning_args.align_loss_type
        self.image_token_id = image_token_id

        self.mlp_layers = find_mlp_layers(model)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._input_ids: torch.Tensor | None = None
        self._act_store: dict[int, torch.Tensor] = {}

        self._register_hooks()

    @staticmethod
    def _make_hook(layer_idx: int, act_store: dict[int, torch.Tensor]):
        def hook_fn(module, args, output):
            act_store[layer_idx] = args[0]

        return hook_fn

    def set_input_ids(self, input_ids: torch.Tensor) -> None:
        self._input_ids = input_ids
        self._act_store.clear()

    def compute_alignment_loss(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        input_ids = self._input_ids

        if not self._act_store or input_ids is None:
            return torch.zeros((), device=device, dtype=torch.float32)

        seq_len = min(next(iter(self._act_store.values())).shape[1], input_ids.shape[1])
        input_ids = input_ids[:, :seq_len]

        visual_mask = input_ids == self.image_token_id
        text_mask = ~visual_mask

        if not visual_mask.any() or not text_mask.any():
            return torch.zeros((), device=device, dtype=torch.float32)

        losses = []
        for layer_idx in sorted(self._act_store.keys()):
            act = self._act_store[layer_idx][:, :seq_len, :]

            a_v = act[visual_mask]
            a_t = act[text_mask]

            if self.pool_type == "mean":
                mean_v = a_v.mean(dim=0)
                mean_t = a_t.mean(dim=0)
            else:
                mean_v = a_v.max(dim=0).values
                mean_t = a_t.max(dim=0).values

            tau = torch.quantile(mean_t.detach().float(), self.quantile)

            soft_v = torch.sigmoid((mean_v - tau) / self.temperature)
            soft_t = torch.sigmoid((mean_t - tau) / self.temperature)

            if self.loss_type == "l1":
                layer_loss = (soft_v - soft_t).abs().mean()
            else:
                intersection = (soft_v * soft_t).sum()
                union = soft_v.sum() + soft_t.sum() - intersection
                layer_loss = -(intersection / (union + 1e-8))

            losses.append(layer_loss)

        total_loss = torch.stack(losses).mean()
        final_loss = self.lambda_ * total_loss

        self._act_store.clear()
        self._input_ids = None

        return final_loss

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._act_store.clear()
