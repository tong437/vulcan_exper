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

from ...extras.constants import IGNORE_INDEX
from .modeling import find_mlp_layers


if TYPE_CHECKING:
    from torch import nn

    from ...hparams import FinetuningArguments


class ActivationAligner:
    r"""Top-k activation mask alignment between visual and task-relevant text tokens.

    The hook captures each MLP ``down_proj`` input, i.e. the gated intermediate
    activation ``A = silu(gate) * up``. The trainer supplies batch token masks
    before the forward pass. After the forward pass, this class pools visual and
    text activations separately, converts each pooled vector into a soft top-k
    neuron mask, and penalizes mismatch between the two masks.
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
        self.loss_type: Literal["l1", "soft_iou", "neg_iou"] = finetuning_args.align_loss_type
        self.text_mode: Literal["answer", "question", "qa"] = finetuning_args.align_text_mode
        self.image_token_id = image_token_id

        self.mlp_layers = find_mlp_layers(model)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._input_ids: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None
        self._act_store: dict[int, torch.Tensor] = {}
        self._last_log: dict[str, float] = {}

        self._register_hooks()

    @staticmethod
    def _make_hook(layer_idx: int, act_store: dict[int, torch.Tensor]):
        def hook_fn(module, args, output):
            act_store[layer_idx] = args[0]

        return hook_fn

    def _register_hooks(self) -> None:
        for layer_ref in self.mlp_layers:
            self._hooks.append(
                layer_ref.mlp.down_proj.register_forward_hook(self._make_hook(layer_ref.index, self._act_store))
            )

    def set_input_ids(self, input_ids: torch.Tensor) -> None:
        self.set_batch(input_ids=input_ids)

    def set_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        self._input_ids = input_ids
        self._labels = labels
        self._attention_mask = attention_mask
        self._act_store.clear()
        self._last_log = {}

    def get_log(self) -> dict[str, float]:
        return self._last_log

    def _build_token_masks(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = self._input_ids[:, :seq_len].to(device)
        visual_mask = input_ids == self.image_token_id

        if self._attention_mask is None:
            valid_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        else:
            valid_mask = self._attention_mask[:, :seq_len].to(device).bool()

        visual_mask = visual_mask & valid_mask
        non_visual_mask = (~visual_mask) & valid_mask
        if self._labels is None:
            return visual_mask, non_visual_mask

        labels = self._labels[:, :seq_len].to(device)
        answer_mask = (labels != IGNORE_INDEX) & non_visual_mask
        question_mask = (labels == IGNORE_INDEX) & non_visual_mask
        if self.text_mode == "answer":
            text_mask = answer_mask
        elif self.text_mode == "question":
            text_mask = question_mask
        else:
            text_mask = answer_mask | question_mask

        return visual_mask, text_mask

    def _pool_activation(self, act: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        selected = act[token_mask].abs()
        if self.pool_type == "mean":
            return selected.mean(dim=0)

        return selected.max(dim=0).values

    def _soft_topk_mask(self, pooled_activation: torch.Tensor) -> torch.Tensor:
        tau = torch.quantile(pooled_activation.detach().float(), self.quantile)
        return torch.sigmoid((pooled_activation - tau.to(pooled_activation.device)) / self.temperature)

    def _compute_layer_loss(self, soft_v: torch.Tensor, soft_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        intersection = (soft_v * soft_t).sum()
        union = soft_v.sum() + soft_t.sum() - intersection
        soft_iou = intersection / (union + 1e-8)
        if self.loss_type == "l1":
            return (soft_v - soft_t).abs().mean(), soft_iou
        if self.loss_type == "neg_iou":
            return -soft_iou, soft_iou

        return 1.0 - soft_iou, soft_iou

    def compute_alignment_loss(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        input_ids = self._input_ids
        if not self._act_store or input_ids is None:
            return torch.zeros((), device=device, dtype=torch.float32)

        seq_len = min(next(iter(self._act_store.values())).shape[1], input_ids.shape[1])
        if self._labels is not None:
            seq_len = min(seq_len, self._labels.shape[1])
        if self._attention_mask is not None:
            seq_len = min(seq_len, self._attention_mask.shape[1])

        visual_mask, text_mask = self._build_token_masks(seq_len, device)
        visual_tokens = int(visual_mask.sum().item())
        text_tokens = int(text_mask.sum().item())
        if visual_tokens == 0 or text_tokens == 0:
            self._last_log = {
                "align_visual_tokens": float(visual_tokens),
                "align_text_tokens": float(text_tokens),
            }
            self._clear_batch_state()
            return torch.zeros((), device=device, dtype=torch.float32)

        losses = []
        ious = []
        soft_v_means = []
        soft_t_means = []
        for layer_idx in sorted(self._act_store.keys()):
            act = self._act_store[layer_idx][:, :seq_len, :]
            pooled_v = self._pool_activation(act, visual_mask)
            pooled_t = self._pool_activation(act, text_mask)
            soft_v = self._soft_topk_mask(pooled_v)
            soft_t = self._soft_topk_mask(pooled_t)
            layer_loss, soft_iou = self._compute_layer_loss(soft_v, soft_t)
            losses.append(layer_loss)
            ious.append(soft_iou.detach().float())
            soft_v_means.append(soft_v.detach().float().mean())
            soft_t_means.append(soft_t.detach().float().mean())

        raw_loss = torch.stack(losses).mean()
        final_loss = self.lambda_ * raw_loss
        self._last_log = {
            "align_loss": final_loss.detach().float().item(),
            "align_raw_loss": raw_loss.detach().float().item(),
            "align_soft_iou": torch.stack(ious).mean().item(),
            "align_visual_tokens": float(visual_tokens),
            "align_text_tokens": float(text_tokens),
            "align_mask_v_mean": torch.stack(soft_v_means).mean().item(),
            "align_mask_t_mean": torch.stack(soft_t_means).mean().item(),
        }
        self._clear_batch_state()
        return final_loss

    def _clear_batch_state(self) -> None:
        self._act_store.clear()
        self._input_ids = None
        self._labels = None
        self._attention_mask = None

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._clear_batch_state()
        self._last_log.clear()
