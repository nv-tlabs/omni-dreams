# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List, Optional

import torch
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from omnidreams._src.imaginaire.utils import distributed
from omnidreams._src.imaginaire.utils.callback import Callback


@torch.jit.script
def _fused_nan_to_num(params: List[torch.Tensor]):
    for param in params:
        torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0, out=param)


class GradClip(Callback):
    def __init__(
        self, clip_norm=1.0, force_finite: bool = True, model_key: Optional[str] = None, fsdp_enabled: bool = False
    ):
        self.clip_norm = clip_norm
        self.force_finite = force_finite
        self.model_key = model_key
        self.fsdp_enabled = fsdp_enabled

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler
        if isinstance(model_ddp, distributed.DistributedDataParallel):
            model = model_ddp.module
        else:
            model = model_ddp

        # select sub-network if specified
        if self.model_key is not None:
            items = self.model_key.split(".")
            for item in items:
                model = getattr(model, item)


        if self.force_finite:
            params = []
            for param in model.parameters():
                if param.grad is not None:
                    params.append(param.grad)
                    # torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            _fused_nan_to_num(params)

        # check if FSDP is used
        if isinstance(model, FSDP) and self.fsdp_enabled:
            total_norm = model.clip_grad_norm_(self.clip_norm)
        else:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm, foreach=True)

        # log
        if iteration % self.config.trainer.logging_iter == 0:
            if wandb.run:
                wandb.log({"clip_grad_norm": total_norm.item()}, step=iteration)
