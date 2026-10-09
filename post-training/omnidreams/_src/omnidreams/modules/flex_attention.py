# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from functools import lru_cache
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, Partial, Shard
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    create_block_mask,
    flex_attention,
)

from omnidreams._src.omnidreams.modules.ulysses_attention import UlyssesCPManager


@lru_cache
def create_block_mask_cached(
    score_mod: _mask_mod_signature,
    B: Optional[int],
    H: Optional[int],
    M: int,
    N: int,
    device: str = "cuda",
) -> BlockMask:
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    return q_idx >= kv_idx


def rewrite_mask_mod_for_cp(
    mask_mod: _mask_mod_signature,
    rank: int,
    shard_size: int,
) -> _mask_mod_signature:
    # since we're sharding on `seq_dim`, global q_idx is mapped to q_idx % shard_size
    # on each rank which means q_idx = q_idx_on_rank + shard_size * rank
    return lambda b, h, q_idx, kv_idx: mask_mod(b, h, q_idx + rank * shard_size, kv_idx)


def flex_attention_cp(
    query: Tensor,  # [B, H, S, D]
    key: Tensor,  # [B, H, S, D]
    value: Tensor,  # [B, H, S, D]
    process_group: Optional[ProcessGroup] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    block_mask: Optional[BlockMask] = None,
    flex_attention_fn: Callable = flex_attention,
    seq_dim: int = 2,  # sharding on this dimension
    **kwargs,
) -> Tensor:
    """Extend flex attention to support context parallel (CP)."""
    # normal flex attention, no context parallel
    if process_group is None:
        return flex_attention_fn(query, key, value, score_mod=mask_mod, block_mask=block_mask, **kwargs)
    else:
        assert query.ndim == 4 and key.ndim == 4 and value.ndim == 4, "Only support 4D input."

        world_size = torch.distributed.get_world_size(process_group)  # size of shard group
        local_rank = torch.distributed.get_rank(process_group)  # rank in shard group
        backend = torch.distributed.get_backend(process_group)
        assert backend == "nccl", "Only support NCCL backend."
        device_type = "cuda"

        device_mesh = DeviceMesh.from_group(process_group, device_type=device_type)

        if mask_mod is None:
            assert block_mask is not None, "Either mask_mod or block_mask must be provided."
            mask_mod = block_mask.mask_mod

        # manually do context parallel on attention
        # the input hook of Context Parallel
        query_is_dtensor = isinstance(query, DTensor)
        key_is_dtensor = isinstance(key, DTensor)
        value_is_dtensor = isinstance(value, DTensor)

        if not query_is_dtensor:
            q_local = query
            shard_size = query.shape[seq_dim]  # local sequence length
            seq_len = shard_size * world_size
        else:
            q_local = query.to_local()
            seq_len = query.shape[seq_dim]  # global sequence length
            shard_size = seq_len // world_size

        # kv all-gather
        # NOTE: we don't consider load-balance for now
        # NOTE: wait() is immediately called in all_gather_tensor when gather_dim != 0
        # k,v: [1, 12, 8192, 128] before gather
        if key_is_dtensor:
            k_full = key.full_tensor(grad_placements=[Partial()])
        else:
            k_local = DTensor.from_local(key, device_mesh, [Shard(seq_dim)])
            k_full = k_local.full_tensor(grad_placements=[Partial()])

        if value_is_dtensor:
            v_full = value.full_tensor(grad_placements=[Partial()])
        else:
            v_local = DTensor.from_local(value, device_mesh, [Shard(seq_dim)])
            v_full = v_local.full_tensor(grad_placements=[Partial()])

        # rewrite `block_mask`
        cp_mask_mod = rewrite_mask_mod_for_cp(mask_mod, local_rank, shard_size)
        cp_block_mask = create_block_mask_cached(cp_mask_mod, B=1, H=1, M=shard_size, N=seq_len, device=device_type)

        cp_out = flex_attention_fn(
            q_local,
            k_full,
            v_full,
            score_mod=None,
            block_mask=cp_block_mask,
            **kwargs,
        )
        assert isinstance(cp_out, torch.Tensor)

        if query_is_dtensor:
            # If the input is a DTensor, return the DTensor
            cp_out_dist = DTensor.from_local(cp_out, device_mesh, [Shard(seq_dim)])
            return cp_out_dist
        else:
            # If the input is a full tensor, return the full tensor
            return cp_out


def ulysses_flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    block_mask: BlockMask,
    cp_manager: UlyssesCPManager,
    flex_attention_fn: Callable = flex_attention,
) -> Tensor:
    """Run local FlexAttention on full-sequence Ulysses head shards.

    Inputs and output use ``[B, S / CP, H, D]`` for CP>1. Ulysses changes
    this temporarily to ``[B, S, H / CP, D]``; FlexAttention itself performs
    no distributed communication in this execution mode.
    """
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            f"Ulysses FlexAttention requires matching Q/K/V shapes, got {query.shape}, {key.shape}, and {value.shape}"
        )
    if query.ndim != 4:
        raise ValueError(f"Ulysses FlexAttention expects [B, S, H, D], got {query.shape}")

    cp_manager.validate_num_heads(query.shape[2])
    full_query, full_key, full_value = cp_manager.sequence_to_head_qkv(query, key, value)
    sequence_length = full_query.shape[1]
    padded_length = math.ceil(sequence_length / 128) * 128 - sequence_length
    if padded_length > 0:
        pad_shape = [full_query.shape[0], padded_length, full_query.shape[2], full_query.shape[3]]
        full_query = torch.cat([full_query, torch.zeros(pad_shape, device=query.device, dtype=query.dtype)], dim=1)
        full_key = torch.cat([full_key, torch.zeros(pad_shape, device=key.device, dtype=key.dtype)], dim=1)
        full_value = torch.cat([full_value, torch.zeros(pad_shape, device=value.device, dtype=value.dtype)], dim=1)

    output = flex_attention_fn(
        full_query.transpose(2, 1),
        full_key.transpose(2, 1),
        full_value.transpose(2, 1),
        block_mask=block_mask,
    ).transpose(2, 1)
    if padded_length > 0:
        output = output[:, :-padded_length]
    return cp_manager.head_to_sequence(output)
