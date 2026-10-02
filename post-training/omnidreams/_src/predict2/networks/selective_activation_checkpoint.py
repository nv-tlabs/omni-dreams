# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

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
    ATTENTION_OUTPUT = "attention_output"

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


_ATTENTION_OUTPUT_OP_MARKERS = (
    "flex_attention",
    "flash_attn",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
)
_COMPILED_ATTENTION_REGION: ContextVar[bool] = ContextVar(
    "compiled_attention_region",
    default=False,
)
_ATTENTION_OUTPUT_SAC_ACTIVE: ContextVar[bool] = ContextVar(
    "attention_output_sac_active",
    default=False,
)


@contextmanager
def compiled_attention_region() -> Iterator[None]:
    """Mark a compiled region whose output is a fused attention output."""

    token = _COMPILED_ATTENTION_REGION.set(True)
    try:
        yield
    finally:
        _COMPILED_ATTENTION_REGION.reset(token)


def is_attention_output_sac_active() -> bool:
    """Return whether execution is inside an attention-output SAC context."""

    return _ATTENTION_OUTPUT_SAC_ACTIVE.get()


@contextmanager
def _attention_output_sac_context(mode) -> Iterator[None]:
    token = _ATTENTION_OUTPUT_SAC_ACTIVE.set(True)
    try:
        with mode:
            yield
    finally:
        _ATTENTION_OUTPUT_SAC_ACTIVE.reset(token)


def is_attention_output_op(func: object) -> bool:
    """Return whether ``func`` is a fused attention op worth saving."""

    op_name = str(func).lower()
    if _COMPILED_ATTENTION_REGION.get() and "inductor_compiled_code" in op_name:
        return True
    return any(marker in op_name for marker in _ATTENTION_OUTPUT_OP_MARKERS)


def attention_output_policy(ctx, func, *args, **kwargs):
    """Save fused attention outputs and recompute every other operation."""

    del ctx, args, kwargs
    if is_attention_output_op(func):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def attention_output_context_fn():
    forward_mode, recompute_mode = create_selective_checkpoint_contexts(attention_output_policy)
    return (
        _attention_output_sac_context(forward_mode),
        _attention_output_sac_context(recompute_mode),
    )


@dataclass
class SACConfig:
    mode: str = "mm_only"
    every_n_blocks: int = 1

    def get_context_fn(self):
        if self.mode == CheckpointMode.MM_ONLY:
            return mm_only_context_fn
        elif self.mode == CheckpointMode.BLOCK_WISE:
            return noop_context_fn
        elif self.mode == CheckpointMode.ATTENTION_OUTPUT:
            return attention_output_context_fn
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
