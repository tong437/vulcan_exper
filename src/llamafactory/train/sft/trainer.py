# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        vulcan_cluster_idx: Optional[list[Optional[list[dict[str, Any]]]]] = None,
        activation_aligner: Optional[Any] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model
        self.vulcan_cluster_idx = vulcan_cluster_idx
        self.activation_aligner = activation_aligner
        self._vulcan_log_cache: dict[str, float] = {}
        self._align_grad_diagnostic_done = False

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            custom_optim = create_custom_optimizer(self.model, self.args, self.finetuning_args)
            if custom_optim is not None:
                self.optimizer = custom_optim
            else:
                vision_lr = getattr(self.finetuning_args, "vision_tower_lr", None)
                proj_lr = getattr(self.finetuning_args, "projector_lr", None)
                lang_lr = getattr(self.finetuning_args, "language_model_lr", None)
                if any(lr is not None for lr in (vision_lr, proj_lr, lang_lr)):
                    base_lr = self.args.learning_rate
                    vision_params, projector_params, language_params = [], [], []
                    for name, param in self.model.named_parameters():
                        if not param.requires_grad:
                            continue
                        if "visual.blocks" in name or "visual.pos_embed" in name or "visual.patch_embed" in name:
                            vision_params.append(param)
                        elif "visual.merger" in name or "multi_modal_projector" in name:
                            projector_params.append(param)
                        else:
                            language_params.append(param)

                    param_groups = []
                    if vision_params:
                        param_groups.append(
                            {
                                "params": vision_params,
                                "lr": vision_lr or base_lr,
                                "weight_decay": self.args.weight_decay,
                            }
                        )
                    if projector_params:
                        param_groups.append(
                            {
                                "params": projector_params,
                                "lr": proj_lr or base_lr,
                                "weight_decay": self.args.weight_decay,
                            }
                        )
                    if language_params:
                        param_groups.append(
                            {
                                "params": language_params,
                                "lr": lang_lr or base_lr,
                                "weight_decay": self.args.weight_decay,
                            }
                        )

                    optim_class, optim_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
                    self.optimizer = optim_class(param_groups, **optim_kwargs)
                else:
                    self.optimizer = None
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def _add_vulcan_loss(self, model: "torch.nn.Module", loss: "torch.Tensor") -> "torch.Tensor":
        if not self.finetuning_args.use_collapse_loss:
            return loss

        if self.vulcan_cluster_idx is None:
            raise ValueError("Vulcan collapse loss is enabled, but cluster_idx was not loaded.")

        from ..vulcan import get_collapse_lambdas, weight_collapse_loss

        unwrapped_model = self.accelerator.unwrap_model(model)
        lambda1, lambda2 = get_collapse_lambdas(unwrapped_model, self.finetuning_args)
        loss_collapse = weight_collapse_loss(
            unwrapped_model,
            self.vulcan_cluster_idx,
            lambda1,
            lambda2,
            use_weight_proxy=self.finetuning_args.collapse_use_weight_proxy,
        )
        sft_loss = loss.detach().float()
        collapse_loss = loss_collapse.detach().float()
        self._vulcan_log_cache = {
            "sft_loss": sft_loss.item(),
            "collapse_loss": collapse_loss.item(),
            "collapse_loss_ratio": (collapse_loss / sft_loss.clamp_min(1e-8)).item(),
            "collapse_lambda1": lambda1.detach().float().item(),
            "collapse_lambda2": lambda2.detach().float().item(),
            "collapse_lambda1_grad_norm": (
                lambda1.grad.detach().float().norm().item() if lambda1.grad is not None else 0.0
            ),
            "collapse_lambda2_grad_norm": (
                lambda2.grad.detach().float().norm().item() if lambda2.grad is not None else 0.0
            ),
        }
        return loss + loss_collapse.float()

    def _run_align_grad_diagnostic(
        self,
        sft_loss: "torch.Tensor",
        align_raw_loss: "torch.Tensor",
        align_lambda: float,
        sampled_layers: list[int],
        sampled_activations: list["torch.Tensor"],
    ) -> None:
        # Diagnose gradients in activation space instead of parameter space.
        # Calling autograd.grad on ZeRO-managed parameters triggers DeepSpeed's
        # incomplete-backward hooks and corrupts its partition-reduction state.
        sft_grads = torch.autograd.grad(sft_loss.float(), sampled_activations, retain_graph=True, allow_unused=True)
        align_grads = torch.autograd.grad(align_raw_loss, sampled_activations, retain_graph=True, allow_unused=True)

        device = sft_loss.device
        sft_sq = torch.zeros((), device=device, dtype=torch.float32)
        align_sq = torch.zeros((), device=device, dtype=torch.float32)
        dot = torch.zeros((), device=device, dtype=torch.float32)
        for sft_grad, align_grad in zip(sft_grads, align_grads):
            if sft_grad is not None:
                sft_sq = sft_sq + sft_grad.detach().float().square().sum()
            if align_grad is not None:
                align_sq = align_sq + align_grad.detach().float().square().sum()
            if sft_grad is not None and align_grad is not None:
                dot = dot + (sft_grad.detach().float() * align_grad.detach().float()).sum()

        sft_norm = sft_sq.sqrt()
        align_raw_norm = align_sq.sqrt()
        weighted_align_norm = abs(align_lambda) * align_raw_norm
        ratio = weighted_align_norm / sft_norm.clamp_min(1e-12)
        cosine = dot / (sft_norm * align_raw_norm).clamp_min(1e-12)
        diagnostic = {
            "gradient_space": "mlp_down_proj_input",
            "sampled_layers": sampled_layers,
            "sft_activation_grad_norm": sft_norm.item(),
            "align_raw_activation_grad_norm": align_raw_norm.item(),
            "align_weighted_activation_grad_norm": weighted_align_norm.item(),
            "align_to_sft_activation_grad_ratio": ratio.item(),
            "sft_align_activation_grad_cosine": cosine.item(),
        }
        logger.info_rank0(f"Vulcan align gradient diagnostic: {json.dumps(diagnostic)}")

    def _add_align_loss(self, loss: "torch.Tensor") -> "torch.Tensor":
        if self.activation_aligner is None:
            return loss

        run_diagnostic = (
            os.environ.get("VULCAN_ALIGN_GRAD_DIAGNOSTIC", "0") == "1"
            and not self._align_grad_diagnostic_done
            and self.model.training
            and torch.is_grad_enabled()
        )
        if run_diagnostic:
            activation_items = self.activation_aligner.get_captured_activations()
            if not activation_items:
                raise RuntimeError("Cannot run activation gradient diagnostic without captured MLP activations.")

            sample_indices = sorted({0, len(activation_items) // 2, len(activation_items) - 1})
            sampled_layers = [activation_items[idx][0] for idx in sample_indices]
            sampled_activations = [activation_items[idx][1] for idx in sample_indices]
            align_loss, align_raw_loss = self.activation_aligner.compute_alignment_loss(return_raw_loss=True)
            self._run_align_grad_diagnostic(
                loss,
                align_raw_loss,
                self.activation_aligner.lambda_,
                sampled_layers,
                sampled_activations,
            )
            self._align_grad_diagnostic_done = True
        else:
            align_loss = self.activation_aligner.compute_alignment_loss()

        self._vulcan_log_cache.update(self.activation_aligner.get_log())

        return loss + align_loss.to(loss.device, dtype=loss.dtype)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return_outputs = kwargs.get("return_outputs", False)
        if len(args) > 0:
            return_outputs = args[0]

        self._vulcan_log_cache = {}
        if self.activation_aligner is not None:
            self.activation_aligner.set_batch(
                input_ids=inputs["input_ids"],
                labels=inputs.get("labels"),
                attention_mask=inputs.get("attention_mask"),
            )

        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            loss = self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        else:
            loss_outputs = super().compute_loss(model, inputs, *args, **kwargs)
            if isinstance(loss_outputs, tuple):
                loss, outputs = loss_outputs
            else:
                loss, outputs = loss_outputs, None

        # NOTE: sft_loss, align_loss, etc. in _vulcan_log_cache come from the *last*
        # micro-batch within a gradient-accumulation window.  The HuggingFace Trainer's
        # own `loss` key in trainer_log.jsonl is the *average* over the logging interval.
        # Do not treat them as the same statistical quantity.
        self._vulcan_log_cache["sft_loss"] = loss.detach().float().item()
        loss = self._add_vulcan_loss(model, loss)
        loss = self._add_align_loss(loss)
        if return_outputs:
            return loss, outputs

        return loss

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self._vulcan_log_cache:
            logs = {**logs, **self._vulcan_log_cache}

        return super().log(logs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
