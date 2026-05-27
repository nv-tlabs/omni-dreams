# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch
import torch.distributed

from omnidreams._src.imaginaire.checkpointer.base import AbstractCheckpointer
from omnidreams._src.imaginaire.model import ImaginaireModel


class Checkpointer(AbstractCheckpointer):
    """
    A dummy checkpointer that does not save or load anything. This is useful for debugging jobs or share workload with collobrators.
    """

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        pass

    def load(
        self,
        model: ImaginaireModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        grad_scaler: Optional[torch.amp.GradScaler] = None,
    ) -> int:
        return 0
