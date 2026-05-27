# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams._src.imaginaire.checkpointer.ddp import Checkpointer as DDPCheckpointer
from omnidreams._src.imaginaire.model import ImaginaireModel


class Checkpointer(DDPCheckpointer):
    """
    Checkpointer class for Tensor Parallelism (TP) in distributed training.

    This implementation supports the combination of Tensor Parallelism (TP) and Data Parallel Processing (DDP), with optional Context Parallelism (CP).

    Note:
    - Fully Sharded Data Parallelism (FSDP) is not supported by this checkpointer.
    - In principle, this implementation is also compatible with Pipeline Parallelism (PP) and Expert Parallelism (EP), which are other forms of model parallelism. However, PP and EP have not been tested yet.
    """

    def add_type_postfix_to_checkpoint_path(self, key: str, checkpoint_path: str, model: ImaginaireModel) -> str:
        """
        Overwrite the `add_type_postfix_to_checkpoint_path` function of the base class (DDP checkpointer)
        to append the TP-rank postfix to the checkpoint path.
        """
        checkpoint_path = super().add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
        if key == "trainer":
            return checkpoint_path
        else:
            checkpoint_path = checkpoint_path.replace(".pt", f"_mp_{self.mp_rank}.pt")

        return checkpoint_path
