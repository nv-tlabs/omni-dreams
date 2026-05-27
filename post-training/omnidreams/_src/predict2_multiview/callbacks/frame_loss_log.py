# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from omnidreams._src.imaginaire.model import ImaginaireModel
from omnidreams._src.imaginaire.utils.callback import Callback

"""
   Dummy FrameLossLog callback used in multiview training with view dropout, where batches don't have the same number of views / frames.
"""


class DummyFrameLossLog(Callback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        pass

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ):
        pass
