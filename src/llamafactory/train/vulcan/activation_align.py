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

    def _register_hooks(self) -> None:
        act_store = self._act_store
        for idx, layer_ref in enumerate(self.mlp_layers):
            hook = layer_ref.mlp.down_proj.register_forward_hook(self._make_hook(idx, act_store))
            self._hooks.append(hook)

    _debug_printed = False
    _loss_printed = False

    @staticmethod
    def _make_hook(layer_idx: int, act_store: dict[int, torch.Tensor]):
        def hook_fn(module, args, output):
            act_store[layer_idx] = args[0]
            if not ActivationAligner._debug_printed:
                print(f"[ALIGN DEBUG] hook fired layer={layer_idx} act_shape={args[0].shape} dtype={args[0].dtype}")

        return hook_fn

    def set_input_ids(self, input_ids: torch.Tensor) -> None:
        self._input_ids = input_ids
        self._act_store.clear()
        if not type(self)._debug_printed:
            image_count = (input_ids == self.image_token_id).sum().item()
            print(f"[ALIGN DEBUG] set_input_ids shape={input_ids.shape} image_tokens={image_count} image_token_id={self.image_token_id}")

    def compute_alignment_loss(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        input_ids = self._input_ids

        if not type(self)._debug_printed:
            print(
                f"[ALIGN DEBUG] compute_alignment_loss: act_store={len(self._act_store)} keys, "
                f"input_ids={'None' if input_ids is None else input_ids.shape}"
            )
            type(self)._debug_printed = True

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
        if not type(self)._loss_printed:
            print(
                f"[ALIGN DEBUG] total_loss={total_loss.item():.6f} final_loss={final_loss.item():.6f} "
                f"lambda={self.lambda_} n_layers={len(losses)} "
                f"visual_tokens={visual_mask.sum().item()} text_tokens={text_mask.sum().item()}"
            )
            # print first layer stats
            act0 = self._act_store[sorted(self._act_store.keys())[0]][:, :seq_len, :]
            a_v0 = act0[visual_mask]
            a_t0 = act0[~visual_mask]
            mean_v0 = a_v0.mean(dim=0).float()
            mean_t0 = a_t0.mean(dim=0).float()
            tau0 = torch.quantile(mean_t0, self.quantile)
            soft_v0 = torch.sigmoid((mean_v0 - tau0) / self.temperature)
            soft_t0 = torch.sigmoid((mean_t0 - tau0) / self.temperature)
            print(
                f"[ALIGN DEBUG] layer0: tau={tau0.item():.4f} "
                f"soft_v_mean={soft_v0.mean().item():.4f} soft_t_mean={soft_t0.mean().item():.4f} "
                f"|soft_v-soft_t|_mean={(soft_v0-soft_t0).abs().mean().item():.4f}"
            )
            type(self)._loss_printed = True

        self._act_store.clear()
        self._input_ids = None

        return final_loss

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._act_store.clear()
