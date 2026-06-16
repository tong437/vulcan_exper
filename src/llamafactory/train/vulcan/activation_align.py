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

import math
from typing import TYPE_CHECKING, Literal

import torch

from ...extras.constants import IGNORE_INDEX
from .modeling import find_mlp_layers
from .schema import ClusterIdx


if TYPE_CHECKING:
    from torch import nn

    from ...hparams import FinetuningArguments


class ActivationAligner:
    r"""Neuron- or cluster-level alignment between visual and task-relevant text tokens.

    The hook captures each MLP ``down_proj`` input, i.e. the gated intermediate
    activation ``A = silu(gate) * up``. The trainer supplies batch token masks
    before the forward pass. After the forward pass, this class pools visual and
    text activations separately. Neuron mode converts pooled vectors into soft
    top-k masks. Cluster mode aggregates neuron salience over a precomputed
    Vulcan cluster_idx and minimizes image-question/image-answer JS divergence.
    """

    def __init__(
        self,
        model: "nn.Module",
        finetuning_args: "FinetuningArguments",
        image_token_id: int,
        cluster_idx: ClusterIdx | None = None,
    ) -> None:
        self.model = model
        self.mode: Literal["neuron", "cluster"] = finetuning_args.align_mode
        self.lambda_ = finetuning_args.align_lambda
        self.temperature = finetuning_args.align_temperature
        self.quantile = finetuning_args.align_quantile
        self.pool_type: Literal["mean", "max"] = finetuning_args.align_pool_type
        self.loss_type: Literal["l1", "soft_iou", "neg_iou", "rank_margin"] = finetuning_args.align_loss_type
        self.margin = finetuning_args.align_margin
        self.text_mode: Literal["answer", "question", "qa"] = finetuning_args.align_text_mode
        self.question_weight = finetuning_args.align_question_weight
        self.answer_weight = finetuning_args.align_answer_weight
        self.cluster_idx = cluster_idx
        self.cluster_temperature = finetuning_args.align_cluster_temperature
        self.cluster_question_weight = finetuning_args.align_cluster_question_weight
        self.cluster_answer_weight = finetuning_args.align_cluster_answer_weight
        self.layer_start_ratio = finetuning_args.align_layer_start_ratio
        self.layer_end_ratio = finetuning_args.align_layer_end_ratio
        self.image_token_id = image_token_id
        self.mlp_layers = find_mlp_layers(model)
        self._layer_start_position = int(len(self.mlp_layers) * self.layer_start_ratio)
        self._layer_end_position = math.ceil(len(self.mlp_layers) * self.layer_end_ratio)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._input_ids: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None
        self._act_store: dict[int, torch.Tensor] = {}
        self._last_log: dict[str, float] = {}
        self._cluster_tensor_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, int]] = {}

        self._validate_cluster_idx()

        self._register_hooks()

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, args, output):
            # Non-reentrant checkpointing may execute this hook again during
            # backward recomputation. The batch state has already been cleared
            # by then, so do not retain a stale autograd graph.
            if self._input_ids is not None:
                self._act_store[layer_idx] = args[0]

        return hook_fn

    def _register_hooks(self) -> None:
        for layer_ref in self.mlp_layers:
            self._hooks.append(layer_ref.mlp.down_proj.register_forward_hook(self._make_hook(layer_ref.index)))

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

    def get_captured_activations(self) -> list[tuple[int, torch.Tensor]]:
        return sorted(self._act_store.items())

    def _validate_cluster_idx(self) -> None:
        if self.mode != "cluster":
            return

        if self.cluster_idx is None:
            raise ValueError("cluster_idx is required for cluster-aware activation alignment.")

        if len(self.cluster_idx) != len(self.mlp_layers):
            raise ValueError(
                f"cluster_idx has {len(self.cluster_idx)} layers, but the model has {len(self.mlp_layers)} MLP layers."
            )

        active_cluster_layers = 0
        for layer_position, (layer_ref, layer_clusters) in enumerate(zip(self.mlp_layers, self.cluster_idx)):
            if not self._is_active_layer_position(layer_position) or layer_clusters is None:
                continue

            active_cluster_layers += 1
            intermediate_size = int(layer_ref.mlp.up_proj.weight.shape[0])
            neurons = [int(neuron) for cluster in layer_clusters for neuron in cluster["neuron"]]
            if not neurons:
                raise ValueError(f"cluster_idx[{layer_position}] contains no neurons.")

            if min(neurons) < 0 or max(neurons) >= intermediate_size:
                raise ValueError(
                    f"cluster_idx[{layer_position}] contains an index outside intermediate size {intermediate_size}."
                )

            if len(set(neurons)) != len(neurons):
                raise ValueError(f"cluster_idx[{layer_position}] assigns at least one neuron to multiple clusters.")

            if set(neurons) != set(range(intermediate_size)):
                raise ValueError(
                    f"cluster_idx[{layer_position}] must assign every neuron in intermediate size {intermediate_size}."
                )

        if active_cluster_layers == 0:
            raise ValueError("No cluster_idx layers remain after applying activation alignment layer range.")

    def _is_active_layer_position(self, layer_position: int) -> bool:
        return self._layer_start_position <= layer_position < self._layer_end_position

    def _build_token_masks(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        visual_mask, question_mask, answer_mask = self._build_modality_masks(seq_len, device)
        if self.text_mode == "answer":
            text_mask = answer_mask
        elif self.text_mode == "question":
            text_mask = question_mask
        else:
            text_mask = answer_mask | question_mask

        return visual_mask, text_mask

    def _build_modality_masks(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = self._input_ids[:, :seq_len].to(device)
        visual_mask = input_ids == self.image_token_id

        if self._attention_mask is None:
            valid_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        else:
            valid_mask = self._attention_mask[:, :seq_len].to(device).bool()

        visual_mask = visual_mask & valid_mask
        non_visual_mask = (~visual_mask) & valid_mask
        if self._labels is None:
            return visual_mask, non_visual_mask, torch.zeros_like(non_visual_mask)

        labels = self._labels[:, :seq_len].to(device)
        answer_mask = (labels != IGNORE_INDEX) & non_visual_mask
        question_mask = (labels == IGNORE_INDEX) & non_visual_mask
        return visual_mask, question_mask, answer_mask

    def _pool_activation(self, act: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        selected = act[token_mask].float().abs()
        if self.pool_type == "mean":
            return selected.mean(dim=0)

        return selected.max(dim=0).values

    def _soft_topk_mask(self, pooled_activation: torch.Tensor) -> torch.Tensor:
        pooled_f32 = pooled_activation.float()
        tau = torch.quantile(pooled_f32.detach(), self.quantile)
        return torch.sigmoid((pooled_f32 - tau) / self.temperature)

    def _hard_topk_mask(self, pooled_activation: torch.Tensor) -> torch.Tensor:
        pooled_f32 = pooled_activation.float()
        topk = max(1, math.ceil((1.0 - self.quantile) * pooled_f32.numel()))
        topk = min(topk, pooled_f32.numel())
        topk_indices = torch.topk(pooled_f32.detach(), k=topk, sorted=False).indices
        hard_mask = torch.zeros_like(pooled_f32, dtype=torch.bool)
        hard_mask.scatter_(dim=0, index=topk_indices, value=True)
        return hard_mask

    @staticmethod
    def _compute_hard_iou(hard_v: torch.Tensor, hard_t: torch.Tensor) -> torch.Tensor:
        intersection = (hard_v & hard_t).sum(dtype=torch.float32)
        union = (hard_v | hard_t).sum(dtype=torch.float32)
        return intersection / union.clamp_min(1.0)

    def _compute_layer_loss(self, soft_v: torch.Tensor, soft_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        intersection = (soft_v * soft_t).sum()
        union = soft_v.sum() + soft_t.sum() - intersection
        soft_iou = intersection / (union + 1e-8)
        if self.loss_type == "l1":
            return (soft_v - soft_t).abs().mean(), soft_iou
        if self.loss_type == "neg_iou":
            return -soft_iou, soft_iou

        return 1.0 - soft_iou, soft_iou

    def _rank_margin_loss(
        self, pooled_text: torch.Tensor, visual_topk_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if visual_topk_mask.all():
            zero = pooled_text.sum() * 0.0
            return zero, zero.detach(), zero.detach(), zero.detach()

        normalized_text = pooled_text.float() / pooled_text.detach().float().mean().clamp_min(1e-8)
        top_score = normalized_text[visual_topk_mask].mean()
        other_values = normalized_text[~visual_topk_mask]
        hard_negative_count = min(int(visual_topk_mask.sum().item()), other_values.numel())
        other_score = torch.topk(other_values, k=hard_negative_count, sorted=False).values.mean()
        gap = top_score - other_score
        loss = torch.relu(top_score.new_tensor(self.margin) - gap)
        return loss, gap.detach().float(), top_score.detach().float(), other_score.detach().float()

    def _compute_rank_margin_alignment_loss(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        visual_mask, question_mask, answer_mask = self._build_modality_masks(seq_len, device)
        visual_tokens = int(visual_mask.sum().item())
        question_tokens = int(question_mask.sum().item())
        answer_tokens = int(answer_mask.sum().item())
        if visual_tokens == 0 or (
            (self.question_weight == 0 or question_tokens == 0) and (self.answer_weight == 0 or answer_tokens == 0)
        ):
            return torch.zeros((), device=device, dtype=torch.float32), {
                "align_visual_tokens": float(visual_tokens),
                "align_question_tokens": float(question_tokens),
                "align_answer_tokens": float(answer_tokens),
                "align_rank_layers": 0.0,
            }

        layer_losses = []
        question_losses = []
        answer_losses = []
        question_gaps = []
        answer_gaps = []
        question_top_scores = []
        question_other_scores = []
        answer_top_scores = []
        answer_other_scores = []
        soft_ious = []
        hard_ious = []
        soft_v_means = []
        soft_text_means = []
        active_layers = 0
        combined_text_mask = torch.zeros_like(visual_mask)
        if self.question_weight > 0:
            combined_text_mask = combined_text_mask | question_mask
        if self.answer_weight > 0:
            combined_text_mask = combined_text_mask | answer_mask

        for layer_position, layer_ref in enumerate(self.mlp_layers):
            if not self._is_active_layer_position(layer_position) or layer_ref.index not in self._act_store:
                continue

            act = self._act_store[layer_ref.index][:, :seq_len, :]
            pooled_v = self._pool_activation(act, visual_mask)
            visual_topk_mask = self._hard_topk_mask(pooled_v)
            weighted_losses = []
            active_layers += 1

            if self.question_weight > 0 and question_tokens > 0:
                pooled_question = self._pool_activation(act, question_mask)
                question_loss, question_gap, question_top_score, question_other_score = self._rank_margin_loss(
                    pooled_question, visual_topk_mask
                )
                weighted_losses.append(self.question_weight * question_loss)
                question_losses.append(question_loss.detach().float())
                question_gaps.append(question_gap)
                question_top_scores.append(question_top_score)
                question_other_scores.append(question_other_score)

            if self.answer_weight > 0 and answer_tokens > 0:
                pooled_answer = self._pool_activation(act, answer_mask)
                answer_loss, answer_gap, answer_top_score, answer_other_score = self._rank_margin_loss(
                    pooled_answer, visual_topk_mask
                )
                weighted_losses.append(self.answer_weight * answer_loss)
                answer_losses.append(answer_loss.detach().float())
                answer_gaps.append(answer_gap)
                answer_top_scores.append(answer_top_score)
                answer_other_scores.append(answer_other_score)

            if weighted_losses:
                layer_losses.append(torch.stack(weighted_losses).sum())

            if combined_text_mask.any():
                pooled_text = self._pool_activation(act, combined_text_mask)
                soft_v = self._soft_topk_mask(pooled_v)
                soft_text = self._soft_topk_mask(pooled_text)
                _, soft_iou = self._compute_layer_loss(soft_v, soft_text)
                soft_ious.append(soft_iou.detach().float())
                hard_ious.append(self._compute_hard_iou(visual_topk_mask, self._hard_topk_mask(pooled_text)))
                soft_v_means.append(soft_v.detach().float().mean())
                soft_text_means.append(soft_text.detach().float().mean())

        if not layer_losses:
            return torch.zeros((), device=device, dtype=torch.float32), {
                "align_visual_tokens": float(visual_tokens),
                "align_question_tokens": float(question_tokens),
                "align_answer_tokens": float(answer_tokens),
                "align_rank_layers": 0.0,
            }

        raw_loss = torch.stack(layer_losses).mean()
        log_values = {
            "align_visual_tokens": float(visual_tokens),
            "align_text_tokens": float(question_tokens + answer_tokens),
            "align_question_tokens": float(question_tokens),
            "align_answer_tokens": float(answer_tokens),
            "align_rank_layers": float(active_layers),
            "align_question_weight": float(self.question_weight),
            "align_answer_weight": float(self.answer_weight),
            "align_margin": float(self.margin),
        }
        if question_losses:
            log_values["align_rank_question_loss"] = torch.stack(question_losses).mean().item()
            log_values["align_rank_question_gap"] = torch.stack(question_gaps).mean().item()
            log_values["align_rank_question_top_score"] = torch.stack(question_top_scores).mean().item()
            log_values["align_rank_question_other_score"] = torch.stack(question_other_scores).mean().item()
        if answer_losses:
            log_values["align_rank_answer_loss"] = torch.stack(answer_losses).mean().item()
            log_values["align_rank_answer_gap"] = torch.stack(answer_gaps).mean().item()
            log_values["align_rank_answer_top_score"] = torch.stack(answer_top_scores).mean().item()
            log_values["align_rank_answer_other_score"] = torch.stack(answer_other_scores).mean().item()
        if soft_ious:
            log_values["align_soft_iou"] = torch.stack(soft_ious).mean().item()
            log_values["align_hard_topk_iou"] = torch.stack(hard_ious).mean().item()
            log_values["align_mask_v_mean"] = torch.stack(soft_v_means).mean().item()
            log_values["align_mask_t_mean"] = torch.stack(soft_text_means).mean().item()

        return raw_loss, log_values

    def _pool_absolute_activation_by_sample(
        self, absolute_act: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_samples = token_mask.any(dim=1)
        if self.pool_type == "mean":
            weights = token_mask.unsqueeze(-1).to(absolute_act.dtype)
            pooled = (absolute_act * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            pooled = absolute_act.masked_fill(~token_mask.unsqueeze(-1), float("-inf")).amax(dim=1)
            pooled = torch.where(valid_samples.unsqueeze(-1), pooled, torch.zeros_like(pooled))

        return pooled, valid_samples

    def _get_cluster_tensors(
        self, layer_position: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        cache_key = (layer_position, str(device))
        if cache_key in self._cluster_tensor_cache:
            return self._cluster_tensor_cache[cache_key]

        layer_clusters = self.cluster_idx[layer_position]
        if not layer_clusters:
            raise ValueError(f"cluster_idx[{layer_position}] has no clusters.")

        neuron_indices = []
        cluster_ids = []
        for cluster_id, cluster in enumerate(layer_clusters):
            neurons = [int(neuron) for neuron in cluster["neuron"]]
            neuron_indices.extend(neurons)
            cluster_ids.extend([cluster_id] * len(neurons))

        tensors = (
            torch.tensor(neuron_indices, device=device, dtype=torch.long),
            torch.tensor(cluster_ids, device=device, dtype=torch.long),
            len(layer_clusters),
        )
        self._cluster_tensor_cache[cache_key] = tensors
        return tensors

    def _cluster_distribution(self, pooled: torch.Tensor, layer_position: int) -> torch.Tensor:
        neuron_indices, cluster_ids, num_clusters = self._get_cluster_tensors(layer_position, pooled.device)
        selected = pooled.index_select(dim=1, index=neuron_indices)
        expanded_cluster_ids = cluster_ids.unsqueeze(0).expand(pooled.shape[0], -1)
        cluster_sums = pooled.new_zeros((pooled.shape[0], num_clusters))
        cluster_sums.scatter_add_(dim=1, index=expanded_cluster_ids, src=selected)

        cluster_sizes = torch.bincount(cluster_ids, minlength=num_clusters).to(pooled.dtype).clamp_min(1.0)
        cluster_scores = cluster_sums / cluster_sizes.unsqueeze(0)
        logits = torch.log(cluster_scores.clamp_min(1e-8)) / self.cluster_temperature
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _js_divergence(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        midpoint = 0.5 * (first + second)
        first_kl = (first * (first.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=-1)
        second_kl = (second * (second.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=-1)
        return 0.5 * (first_kl + second_kl)

    def _compute_cluster_alignment_loss(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        visual_mask, question_mask, answer_mask = self._build_modality_masks(seq_len, device)
        layer_losses = []
        question_js_values = []
        answer_js_values = []
        cluster_counts = []

        for layer_position, layer_ref in enumerate(self.mlp_layers):
            if not self._is_active_layer_position(layer_position) or self.cluster_idx[layer_position] is None:
                continue

            act = self._act_store[layer_ref.index][:, :seq_len, :]
            absolute_act = act.float().abs()
            pooled_visual, valid_visual = self._pool_absolute_activation_by_sample(absolute_act, visual_mask)
            pooled_question, valid_question = self._pool_absolute_activation_by_sample(absolute_act, question_mask)
            pooled_answer, valid_answer = self._pool_absolute_activation_by_sample(absolute_act, answer_mask)
            dist_visual = self._cluster_distribution(pooled_visual, layer_position)

            weighted_losses = []
            if self.cluster_question_weight > 0:
                valid = valid_visual & valid_question
                if valid.any():
                    dist_question = self._cluster_distribution(pooled_question, layer_position)
                    question_js = self._js_divergence(dist_visual[valid], dist_question[valid]).mean()
                    weighted_losses.append(self.cluster_question_weight * question_js)
                    question_js_values.append(question_js.detach())

            if self.cluster_answer_weight > 0:
                valid = valid_visual & valid_answer
                if valid.any():
                    dist_answer = self._cluster_distribution(pooled_answer, layer_position)
                    answer_js = self._js_divergence(dist_visual[valid], dist_answer[valid]).mean()
                    weighted_losses.append(self.cluster_answer_weight * answer_js)
                    answer_js_values.append(answer_js.detach())

            if weighted_losses:
                layer_losses.append(torch.stack(weighted_losses).sum())
                cluster_counts.append(len(self.cluster_idx[layer_position]))

        if not layer_losses:
            return torch.zeros((), device=device, dtype=torch.float32), {
                "align_visual_tokens": float(visual_mask.sum().item()),
                "align_question_tokens": float(question_mask.sum().item()),
                "align_answer_tokens": float(answer_mask.sum().item()),
                "align_cluster_layers": 0.0,
            }

        raw_loss = torch.stack(layer_losses).mean()
        log_values = {
            "align_visual_tokens": float(visual_mask.sum().item()),
            "align_question_tokens": float(question_mask.sum().item()),
            "align_answer_tokens": float(answer_mask.sum().item()),
            "align_cluster_layers": float(len(layer_losses)),
            "align_clusters_mean": float(sum(cluster_counts) / len(cluster_counts)),
        }
        if question_js_values:
            question_js = torch.stack(question_js_values).mean().item()
            log_values["align_cluster_js_question"] = question_js
            log_values["align_cluster_similarity_question"] = 1.0 - question_js / math.log(2.0)
        if answer_js_values:
            answer_js = torch.stack(answer_js_values).mean().item()
            log_values["align_cluster_js_answer"] = answer_js
            log_values["align_cluster_similarity_answer"] = 1.0 - answer_js / math.log(2.0)

        return raw_loss, log_values

    def compute_alignment_loss(
        self, return_raw_loss: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        input_ids = self._input_ids
        if not self._act_store or input_ids is None:
            return torch.zeros((), device=device, dtype=torch.float32)

        seq_len = min(next(iter(self._act_store.values())).shape[1], input_ids.shape[1])
        if self._labels is not None:
            seq_len = min(seq_len, self._labels.shape[1])
        if self._attention_mask is not None:
            seq_len = min(seq_len, self._attention_mask.shape[1])

        if self.mode == "cluster":
            raw_loss, log_values = self._compute_cluster_alignment_loss(seq_len, device)
            final_loss = self.lambda_ * raw_loss
            self._check_loss_gradient(final_loss)
            self._last_log = {
                "align_loss": final_loss.detach().float().item(),
                "align_raw_loss": raw_loss.detach().float().item(),
                **log_values,
            }
            self._clear_batch_state()
            if return_raw_loss:
                return final_loss, raw_loss

            return final_loss

        if self.loss_type == "rank_margin":
            raw_loss, log_values = self._compute_rank_margin_alignment_loss(seq_len, device)
            final_loss = self.lambda_ * raw_loss
            self._check_loss_gradient(final_loss)
            self._last_log = {
                "align_loss": final_loss.detach().float().item(),
                "align_raw_loss": raw_loss.detach().float().item(),
                **log_values,
            }
            self._clear_batch_state()
            if return_raw_loss:
                return final_loss, raw_loss

            return final_loss

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
        hard_ious = []
        soft_v_means = []
        soft_t_means = []
        active_layers = 0
        for layer_position, layer_ref in enumerate(self.mlp_layers):
            if not self._is_active_layer_position(layer_position) or layer_ref.index not in self._act_store:
                continue

            active_layers += 1
            act = self._act_store[layer_ref.index][:, :seq_len, :]
            pooled_v = self._pool_activation(act, visual_mask)
            pooled_t = self._pool_activation(act, text_mask)
            soft_v = self._soft_topk_mask(pooled_v)
            soft_t = self._soft_topk_mask(pooled_t)
            hard_v = self._hard_topk_mask(pooled_v)
            hard_t = self._hard_topk_mask(pooled_t)
            layer_loss, soft_iou = self._compute_layer_loss(soft_v, soft_t)
            losses.append(layer_loss)
            ious.append(soft_iou.detach().float())
            hard_ious.append(self._compute_hard_iou(hard_v, hard_t))
            soft_v_means.append(soft_v.detach().float().mean())
            soft_t_means.append(soft_t.detach().float().mean())

        if not losses:
            self._last_log = {
                "align_visual_tokens": float(visual_tokens),
                "align_text_tokens": float(text_tokens),
                "align_active_layers": 0.0,
            }
            self._clear_batch_state()
            return torch.zeros((), device=device, dtype=torch.float32)

        raw_loss = torch.stack(losses).mean()
        final_loss = self.lambda_ * raw_loss
        self._check_loss_gradient(final_loss)
        self._last_log = {
            "align_loss": final_loss.detach().float().item(),
            "align_raw_loss": raw_loss.detach().float().item(),
            "align_soft_iou": torch.stack(ious).mean().item(),
            "align_hard_topk_iou": torch.stack(hard_ious).mean().item(),
            "align_active_layers": float(active_layers),
            "align_visual_tokens": float(visual_tokens),
            "align_text_tokens": float(text_tokens),
            "align_mask_v_mean": torch.stack(soft_v_means).mean().item(),
            "align_mask_t_mean": torch.stack(soft_t_means).mean().item(),
        }
        self._clear_batch_state()
        if return_raw_loss:
            return final_loss, raw_loss

        return final_loss

    def _check_loss_gradient(self, final_loss: torch.Tensor) -> None:
        if (
            self.model.training
            and torch.is_grad_enabled()
            and not final_loss.requires_grad
            and final_loss.detach().float().item() != 0.0
        ):
            raise RuntimeError(
                "Activation alignment loss has requires_grad=False. "
                "This is likely caused by reentrant gradient checkpointing running the first "
                "forward pass under torch.no_grad(), so forward hooks capture detached activations. "
                "Fix: set `use_reentrant_gc: false` in your training config."
            )

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
