# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Any, Dict, Optional

import torch
from megatron.core import parallel_state

from omnidreams._src.imaginaire.checkpointer.tp import Checkpointer as BaseCheckpointer
from omnidreams._src.imaginaire.model import ImaginaireModel
from omnidreams._src.imaginaire.utils import misc


class Checkpointer(BaseCheckpointer):
    KEYS_TO_SAVE = ["model", "optim", "trainer", "scheduler", "ema"]
    KEYS_TO_POSTFIX = {
        "model": "model",
        "optim": "optim",
        "ema": "ema",
        "scheduler": "scheduler",
        "trainer": "",
    }

    @misc.timer("generate saving state dict")
    def generate_save_state_dict(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> Optional[Dict[str, Any]]:
        state_dict = {}
        if parallel_state.get_data_parallel_rank() == 0:
            trainer_state = dict(
                grad_scaler=grad_scaler.state_dict(),
                iteration=iteration,
            )
            model_state = model.state_dict()
            optim_state = optimizer.state_dict()
            scheduler_state = scheduler.state_dict()
            self.callbacks.on_save_checkpoint(model, state_dict=trainer_state)

            trainer_state, model_state, optim_state, scheduler_state = misc.to(
                [trainer_state, model_state, optim_state, scheduler_state], device="cpu"
            )

            state_dict = {
                "trainer": trainer_state,
                "model": model_state,
                "optim": optim_state,
                "scheduler": scheduler_state,
            }

            if parallel_state.get_data_parallel_rank() < 3:
                ema_state = model.ema.state_dict()
                state_dict["ema"] = ema_state

        return state_dict

    def add_type_postfix_to_checkpoint_path(self, key: str, checkpoint_path: str, model: ImaginaireModel) -> str:

        # we need to get which ema should be saved
        assert key in self.KEYS_TO_SAVE
        post_fix = self.KEYS_TO_POSTFIX[key]

        if post_fix:
            checkpoint_path = checkpoint_path.replace(".pt", f"_{post_fix}.pt")
        else:
            checkpoint_path = checkpoint_path

        if key == "ema":
            dp_rank = parallel_state.get_data_parallel_rank()
            checkpoint_path = checkpoint_path.replace(".pt", f"{dp_rank}.pt")

        if key == "trainer":
            return checkpoint_path
        else:
            mp_rank = parallel_state.get_model_parallel_group().rank()
            checkpoint_path = checkpoint_path.replace(".pt", f"_mp_{mp_rank}.pt")

        return checkpoint_path
