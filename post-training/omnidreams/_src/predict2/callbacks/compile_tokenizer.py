# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from omnidreams._src.imaginaire.utils import log
from omnidreams._src.imaginaire.utils.callback import Callback
from omnidreams._src.predict2.models.text2world_model import DiffusionModel


class CompileTokenizer(Callback):
    def __init__(self, enabled: bool = False, compile_after_iterations: int = 4, dynamic: bool = False):
        super().__init__()
        self.enabled = enabled
        self.compiled = False
        self.compile_after_iterations = compile_after_iterations
        self.skip_counter = 0
        self.dynamic = (
            dynamic  # If there are issues with constant recompilations you may set this value to None or True
        )

    def on_training_step_start(
        self, model: DiffusionModel, data_batch: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        if not self.enabled or self.compiled:
            return

        if isinstance(model.tokenizer, torch.jit.ScriptModule):
            log.critical(
                f"The Tokenizer model {type(model.tokenizer)} is a JIT model, which is not compilable. The Tokenizer will not be compiled."
            )

        if self.skip_counter == self.compile_after_iterations:
            try:
                # PyTorch >= 2.7
                torch._dynamo.config.recompile_limit = 32
            except AttributeError:
                try:
                    torch._dynamo.config.cache_size_limit = 32
                except AttributeError:
                    log.warning(
                        "Tokenizer compilation requested, but Torch Dynamo is unavailable – skipping compilation."
                    )
                    self.enabled = False
                    return

            model.tokenizer.encode = torch.compile(model.tokenizer.encode, dynamic=self.dynamic)
            self.compiled = True
        self.skip_counter += 1
