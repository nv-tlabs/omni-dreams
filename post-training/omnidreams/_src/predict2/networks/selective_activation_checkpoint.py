# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from enum import Enum

import torch

try:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts, noop_context_fn
except ImportError:
    CheckpointPolicy = None

mm_only_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten.addmm.default,
}


class CheckpointMode(str, Enum):
    """
    Enum for the different checkpoint modes.
    """

    NONE = "none"
    MM_ONLY = "mm_only"
    BLOCK_WISE = "block_wise"

    def __str__(self) -> str:
        # Optional: makes print() show just the value
        return self.value


def mm_only_policy(ctx, func, *args, **kwargs):
    """
    In newer flash-attn and TE versions, FA2 shows up in the list of ops with the name of 'flash_attn._flash_attn_forward'.
    However, FA2 is much slower (2-3x) than FA3 or cuDNN kernel. Registering cuDNN kernel would require heavy changes in TE code.
    That's why the best option is to use FA3 with small modifications to flash_attn_interface.py to register FA3 as PyTorch op.
    """
    to_save = func in mm_only_save_list or "flash_attn" in str(func)
    return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE


def mm_only_context_fn():
    return create_selective_checkpoint_contexts(mm_only_policy)


@dataclass
class SACConfig:
    mode: str = "mm_only"
    every_n_blocks: int = 1

    def get_context_fn(self):
        if self.mode == CheckpointMode.MM_ONLY:
            return mm_only_context_fn
        elif self.mode == CheckpointMode.BLOCK_WISE:
            return noop_context_fn
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
