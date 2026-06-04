#!/usr/bin/env python3
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

"""Training entry point with per-component learning rates (DeepSpeed-compatible).

Patches SFT trainer's create_optimizer to use separate LRs for
vision tower, projector, and language model.

Usage:
    WANDB_DISABLED=true torchrun --nproc_per_node=1 --master_port=29520 \\
        scripts/vulcan/train_with_multi_lr.py \\
        examples/vulcan/qwen35_08b_vqa_rad_full_sft_yesno_multilr.yaml \\
        model_name_or_path=/root/autodl-pub-RTX4090-hdd-1/models/qwen3.5-0.8b \\
        vision_lr=1e-5 \\
        projector_lr=5e-5 \\
        lm_lr=5e-6
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


def _classify_params(model):
    vision_params, projector_params, lm_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "visual.blocks" in name or "visual.patch_embed" in name or "visual.pos_embed" in name:
            vision_params.append(param)
        elif "visual.merger" in name:
            projector_params.append(param)
        else:
            lm_params.append(param)
    return vision_params, projector_params, lm_params


def _make_multilr_create_optimizer(vision_lr, projector_lr, lm_lr):
    """Return a create_optimizer method with per-component LRs."""

    def create_optimizer(self):
        from llamafactory.extras import logging
        from llamafactory.train.trainer_utils import create_custom_optimizer

        logger = logging.get_logger(__name__)

        if self.optimizer is not None:
            return self.optimizer

        vision_params, projector_params, lm_params = _classify_params(self.model)

        if not vision_params and not projector_params:
            logger.warning_rank0("MultiLR: no vision/projector params found, falling back to default.")
            optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
            if optimizer is None:
                self.optimizer = super(type(self), self).create_optimizer()
                return self.optimizer
            self.optimizer = optimizer
            return optimizer

        opt_cls, opt_kwargs = type(self).get_optimizer_cls_and_kwargs(self.args)

        param_groups = []
        if vision_params:
            param_groups.append(dict(params=vision_params, lr=vision_lr, weight_decay=self.args.weight_decay))
        if projector_params:
            param_groups.append(dict(params=projector_params, lr=projector_lr, weight_decay=self.args.weight_decay))
        if lm_params:
            param_groups.append(dict(params=lm_params, lr=lm_lr, weight_decay=self.args.weight_decay))

        self.optimizer = opt_cls(param_groups, **opt_kwargs)

        logger.info_rank0(
            f"MultiLR optimizer: vision_lr={vision_lr} ({len(vision_params)} params), "
            f"projector_lr={projector_lr} ({len(projector_params)} params), "
            f"lm_lr={lm_lr} ({len(lm_params)} params)"
        )
        return self.optimizer

    return create_optimizer


def main() -> None:
    custom_args = {}
    remaining_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("vision_lr="):
            custom_args["vision_lr"] = float(arg.split("=", 1)[1])
        elif arg.startswith("projector_lr="):
            custom_args["projector_lr"] = float(arg.split("=", 1)[1])
        elif arg.startswith("lm_lr="):
            custom_args["lm_lr"] = float(arg.split("=", 1)[1])
        else:
            remaining_args.append(arg)

    sys.argv = [sys.argv[0]] + remaining_args

    vision_lr = custom_args.get("vision_lr", 1e-5)
    projector_lr = custom_args.get("projector_lr", 5e-5)
    lm_lr = custom_args.get("lm_lr", 5e-6)

    # Patch SFT trainer's create_optimizer BEFORE run_exp creates the trainer
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    CustomSeq2SeqTrainer.create_optimizer = _make_multilr_create_optimizer(vision_lr, projector_lr, lm_lr)

    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
