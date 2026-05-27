# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import List, Tuple

import torch
import wandb

from omnidreams._src.imaginaire.utils import distributed
from omnidreams._src.imaginaire.utils.callback import Callback
from omnidreams._src.predict2.models.text2world_model import DiffusionModel


@torch.jit.script
def _fused_nan_to_num(params: List[torch.Tensor]):
    for param in params:
        torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0, out=param)


@dataclass
class _MagnitudeRecord:
    state: float = 0
    iter_count: int = 0

    def reset(self) -> None:
        self.state = 0
        self.iter_count = 0

    def update(self, cur_state: torch.Tensor) -> None:
        # Unwrap DTensor first (full_tensor() materializes the global tensor),
        # then move to CPU for accumulation. .cpu() on a DTensor does NOT
        # unwrap it — division on the resulting DTensor would all_reduce on
        # CPU and fail with "No backend type associated with device type cpu"
        # under NCCL. Accumulating on CPU also avoids device-mismatch errors
        # when consecutive batches return total_norm tensors on different
        # devices, and get_stat() calls .item() anyway.
        if hasattr(cur_state, "full_tensor"):
            cur_state = cur_state.full_tensor()
        self.state += cur_state.detach().cpu().float()
        self.iter_count += 1

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            avg_state = self.state / self.iter_count
            avg_state = avg_state.item()
        else:
            avg_state = 0
        self.reset()
        return avg_state


class GradClip(Callback):
    """
    This callback is used to clip the gradient norm of the model.
    It also logs the average gradient norm of the model to wandb.
    """

    def __init__(self, clip_norm=1.0, force_finite: bool = True):
        self.clip_norm = clip_norm
        self.force_finite = force_finite

        self.img_mag_log = _MagnitudeRecord()
        self.video_mag_log = _MagnitudeRecord()
        self._cur_state = None

    def on_training_step_start(
        self, model: DiffusionModel, data_batch: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        if model.is_image_batch(data_batch):
            self._cur_state = self.img_mag_log
        else:
            self._cur_state = self.video_mag_log

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
        params = []

        if self.force_finite:
            for param in model.parameters():
                if param.grad is not None:
                    params.append(param.grad)
            _fused_nan_to_num(params)

        total_norm = model.clip_grad_norm_(self.clip_norm)

        self._cur_state.update(total_norm)
        if iteration % self.config.trainer.logging_iter == 0:
            avg_img_mag, avg_video_mag = self.img_mag_log.get_stat(), self.video_mag_log.get_stat()
            if wandb.run:
                wandb.log(
                    {
                        "clip_grad_norm/image": avg_img_mag,
                        "clip_grad_norm/video": avg_video_mag,
                        "iteration": iteration,
                    },
                    step=iteration,
                )
