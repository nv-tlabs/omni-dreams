# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch
import wandb

from omnidreams._src.imaginaire.callbacks.every_n import EveryN
from omnidreams._src.imaginaire.model import ImaginaireModel
from omnidreams._src.imaginaire.trainer import ImaginaireTrainer
from omnidreams._src.imaginaire.utils import distributed, log


class LogWeight(EveryN):
    def __init__(
        self,
        every_n: Optional[int] = 100,
        step_size: int = 1,
        barrier_after_run: bool = True,
        run_at_start: bool = False,
    ):
        super().__init__(
            every_n=every_n,
            step_size=step_size,
            barrier_after_run=barrier_after_run,
            run_at_start=run_at_start,
        )

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if "logging_dict" in output_batch:
            logging_dict = output_batch["logging_dict"]

            if distributed.is_rank0():
                info = {}
                for k, v in logging_dict.items():
                    info[f"model_weight/{k}"] = v

                if info and wandb.run:
                    wandb.log(info, step=iteration)

                log.info(f"Log weight at iteration {iteration}: {info}")
