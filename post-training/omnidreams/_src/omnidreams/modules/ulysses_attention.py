# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Autograd-safe sequence/head all-to-all transforms for Ulysses CP."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from omnidreams._src.imaginaire.utils.context_parallel import (
    broadcast,
    cat_outputs_cp_with_grad,
    split_inputs_cp,
)


class UlyssesCPManager:
    """Own replicated-input and sequence/head layouts for Ulysses CP."""

    def __init__(self, process_group: ProcessGroup | None = None) -> None:
        self.process_group = process_group

    @property
    def size(self) -> int:
        return 1 if self.process_group is None else self.process_group.size()

    @property
    def is_distributed(self) -> bool:
        return self.size > 1

    def validate_num_heads(self, num_heads: int) -> None:
        if num_heads % self.size != 0:
            raise ValueError(f"Ulysses CP requires num_heads ({num_heads}) to be divisible by CP size ({self.size})")

    def prepare_model_inputs(
        self,
        x0: torch.Tensor | None,
        condition: Any,
        epsilon: torch.Tensor | None,
        sigma: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, Any, torch.Tensor | None, torch.Tensor | None]:
        """Replicate pre-network inputs; Ulysses owns the only sequence split."""
        if not self.is_distributed:
            return x0, condition, epsilon, sigma

        if x0 is not None:
            x0 = broadcast(x0, self.process_group)
            assert isinstance(x0, torch.Tensor)
        if epsilon is not None:
            epsilon = broadcast(epsilon, self.process_group)
            assert isinstance(epsilon, torch.Tensor)
        if sigma is not None:
            sigma = broadcast(sigma, self.process_group)
            assert isinstance(sigma, torch.Tensor)
        if condition is not None:
            condition = condition.broadcast(self.process_group, split=False)
        return x0, condition, epsilon, sigma

    def split_sequence(self, x: torch.Tensor, *, dim: int) -> torch.Tensor:
        """Partition a replicated token sequence exactly once."""
        if not self.is_distributed:
            return x
        assert self.process_group is not None
        return split_inputs_cp(x, seq_dim=dim, cp_group=self.process_group)

    def gather_sequence_with_grad(self, x: torch.Tensor, *, dim: int) -> torch.Tensor:
        """Restore a replicated token sequence while preserving local gradients."""
        if not self.is_distributed:
            return x
        assert self.process_group is not None
        return cat_outputs_cp_with_grad(x, seq_dim=dim, cp_group=self.process_group)

    def sequence_to_head_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather sequence/scatter heads for packed Q/K/V."""
        if not self.is_distributed:
            return query, key, value
        assert self.process_group is not None
        return sequence_to_head_qkv(query, key, value, self.process_group)

    def head_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Restore sequence shards after head-parallel attention."""
        if not self.is_distributed:
            return x
        assert self.process_group is not None
        return head_to_sequence(x, self.process_group)


def _validate_4d_tensor(x: torch.Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"Ulysses CP expects [batch, sequence, heads, head_dim], got shape {tuple(x.shape)}")


def _sequence_to_head_impl(x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
    """Transform ``[B, S/P, H, D]`` into ``[B, S, H/P, D]``."""
    _validate_4d_tensor(x)
    world_size = dist.get_world_size(process_group)
    batch_size, local_sequence_length, num_heads, head_dim = x.shape
    if num_heads % world_size != 0:
        raise ValueError(f"Ulysses CP requires num_heads ({num_heads}) to be divisible by cp_size ({world_size})")
    local_heads = num_heads // world_size

    send = (
        x.reshape(batch_size, local_sequence_length, world_size, local_heads, head_dim)
        .permute(2, 0, 1, 3, 4)
        .contiguous()
    )
    recv = torch.empty_like(send)
    with torch.autograd.profiler.record_function("ulysses_sequence_to_head_a2a"):
        dist.all_to_all_single(recv, send, group=process_group)

    return recv.permute(1, 0, 2, 3, 4).reshape(
        batch_size,
        local_sequence_length * world_size,
        local_heads,
        head_dim,
    )


def _head_to_sequence_impl(x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
    """Transform ``[B, S, H/P, D]`` into ``[B, S/P, H, D]``."""
    _validate_4d_tensor(x)
    world_size = dist.get_world_size(process_group)
    batch_size, sequence_length, local_heads, head_dim = x.shape
    if sequence_length % world_size != 0:
        raise ValueError(
            f"Ulysses CP requires sequence length ({sequence_length}) to be divisible by cp_size ({world_size})"
        )
    local_sequence_length = sequence_length // world_size

    send = (
        x.reshape(batch_size, world_size, local_sequence_length, local_heads, head_dim)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
    recv = torch.empty_like(send)
    with torch.autograd.profiler.record_function("ulysses_head_to_sequence_a2a"):
        dist.all_to_all_single(recv, send, group=process_group)

    return recv.permute(1, 2, 0, 3, 4).reshape(
        batch_size,
        local_sequence_length,
        local_heads * world_size,
        head_dim,
    )


class _SequenceToHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
        ctx.process_group = process_group
        return _sequence_to_head_impl(x, process_group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _head_to_sequence_impl(grad_output.contiguous(), ctx.process_group), None


class _HeadToSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
        ctx.process_group = process_group
        return _head_to_sequence_impl(x, process_group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _sequence_to_head_impl(grad_output.contiguous(), ctx.process_group), None


def sequence_to_head(x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
    """Gather sequence shards while scattering attention heads."""
    if dist.get_world_size(process_group) == 1:
        return x
    return _SequenceToHead.apply(x, process_group)


def sequence_to_head_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    process_group: ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather sequence/scatter heads for Q/K/V with one packed all-to-all."""
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "Packed Ulysses Q/K/V all-to-all requires matching shapes, "
            f"got {query.shape}, {key.shape}, and {value.shape}"
        )
    packed = torch.cat((query, key, value), dim=0)
    full_packed = sequence_to_head(packed, process_group)
    return full_packed.chunk(3, dim=0)


def head_to_sequence(x: torch.Tensor, process_group: ProcessGroup) -> torch.Tensor:
    """Restore local sequence shards while gathering attention heads."""
    if dist.get_world_size(process_group) == 1:
        return x
    return _HeadToSequence.apply(x, process_group)
