# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Dict

from omnidreams._src.imaginaire.model import ImaginaireModel
from omnidreams._src.imaginaire.utils import log
from omnidreams._src.imaginaire.utils.callback import Callback
from omnidreams._src.imaginaire.utils.distributed import rank0_only
from omnidreams._src.imaginaire.utils.easy_io import easy_io


class ModelParamStats(Callback):
    def __init__(
        self,
        save_s3: bool = False,
    ):
        self.save_s3 = save_s3
        self.name = self.__class__.__name__

    @rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        try:
            model_stat: Dict = model.model_param_stats()
        except AttributeError:
            raise AttributeError("Model does not have model_param_stats method. Please implement it.")

        log_str = ""
        for k, v in model_stat.items():
            log_str += f"{k}: {v}\n"
        log.info(f"Model param Stats:\n{log_str}")

        if self.save_s3:
            easy_io.dump(model_stat, f"s3://rundir/{self.name}.yaml")
