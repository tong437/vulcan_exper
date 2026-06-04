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

"""Multi-LR callback: set different learning rates for vision tower, projector, and LoRA params.

Usage (add to training command):
    --callback_script scripts/vulcan/multi_lr_callback.py \
    --vision_lr 1e-5 --projector_lr 5e-5 --lora_lr 2e-4

Or import and register manually in a custom training script.
"""


from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class MultiLRCallback(TrainerCallback):
    """Replace optimizer param groups with component-specific learning rates.

    Identifies parameters by name:
    - Vision tower: names containing 'visual.blocks' or 'visual.patch_embed' or 'visual.pos_embed'
    - Projector: names containing 'visual.merger'
    - LoRA / language model: everything else with requires_grad=True

    Args:
        vision_lr: Learning rate for vision tower parameters.
        projector_lr: Learning rate for multi-modal projector parameters.
        lora_lr: Learning rate for LoRA / language model parameters.
    """

    def __init__(self, vision_lr: float = 1e-5, projector_lr: float = 5e-5, lm_lr: float = 5e-6):
        self.vision_lr = vision_lr
        self.projector_lr = projector_lr
        self.lm_lr = lm_lr
        self._applied = False

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model=None,
        optimizer=None,
        **kwargs,
    ):
        if self._applied or optimizer is None:
            return

        from llamafactory.extras import logging

        logger = logging.get_logger(__name__)

        vision_params = []
        projector_params = []
        lm_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "visual.blocks" in name or "visual.patch_embed" in name or "visual.pos_embed" in name:
                vision_params.append(param)
            elif "visual.merger" in name:
                projector_params.append(param)
            else:
                lm_params.append(param)

        if not vision_params and not projector_params:
            logger.warning_rank0(
                "MultiLRCallback: No vision/projector params found. "
                "Make sure freeze_vision_tower=false and the model has visual components."
            )
            return

        wd = args.weight_decay

        # Copy optimizer defaults (amsgrad, betas, eps, etc.) from existing group
        optim_defaults = {
            k: v
            for k, v in optimizer.param_groups[0].items()
            if k not in ("params", "lr", "weight_decay")
        }

        new_param_groups = []
        if vision_params:
            new_param_groups.append(dict(params=vision_params, lr=self.vision_lr, weight_decay=wd, **optim_defaults))
        if projector_params:
            new_param_groups.append(
                dict(params=projector_params, lr=self.projector_lr, weight_decay=wd, **optim_defaults)
            )
        if lm_params:
            new_param_groups.append(dict(params=lm_params, lr=self.lm_lr, weight_decay=wd, **optim_defaults))

        optimizer.param_groups.clear()
        optimizer.param_groups.extend(new_param_groups)

        self._applied = True
        logger.info_rank0(
            f"MultiLRCallback applied: vision_lr={self.vision_lr}, projector_lr={self.projector_lr}, "
            f"lm_lr={self.lm_lr} | vision_params={len(vision_params)}, "
            f"projector_params={len(projector_params)}, lm_params={len(lm_params)}"
        )
