# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata

from omnidreams._src.imaginaire.utils import log
from omnidreams._src.reason1.utils.checkpoint import remap_model_state_dict


class RenameLoadPlanner(DefaultLoadPlanner):
    """
    RenameLoadPlanner that renames variables during checkpoint load.
    """

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        super().set_up_planner(
            state_dict=state_dict,
            metadata=metadata,
            is_coordinator=is_coordinator,
        )
        # Do an early check to see if the checkpoint is valid and print the state dict if not
        # The reason is the original defauly planner's error message is not helpful enough when the keys are mismatched
        missing_keys = []
        for fqn, obj in state_dict.items():
            # ignore state_dict keys which do not exist in `state_dict` if strict=False
            if fqn not in metadata.state_dict_metadata:
                missing_keys.append(fqn)
        if missing_keys:
            log.critical(f"Missing keys in checkpoint: {missing_keys}...")
            log.critical(f"Checkpoint keys: {list(metadata.state_dict_metadata)}...")

        if need_remapping(metadata):
            log.critical("Old checkpoint, requires remapping of tensors")
            self.state_dict = remap_model_state_dict(self.state_dict)


def need_remapping(metadata: Metadata) -> bool:
    # Check if there is substring "mlp.down_projs" in any key of metadata.state_dict_metadata
    # If yes, do a remapping of state_dict keys
    for key in metadata.state_dict_metadata.keys():
        if "mlp.down_projs" in key:
            # Means this is old checkpoint
            return True
    return False
