# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FlashAttention-3/4 implementations of block-causal training attention."""

from collections.abc import Callable
from functools import cache

import torch

from omnidreams._src.imaginaire.attention.flash3 import FLASH3_SUPPORTED, flash3_attention
from omnidreams._src.omnidreams.modules.attention_backend import FLASH_ATTENTION_BACKENDS
from omnidreams._src.omnidreams.modules.ulysses_attention import UlyssesCPManager


@cache
def _load_flash4_attention() -> Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]:
    """Load FA4 lazily so Flex/FA3 do not require the CuTe DSL runtime."""
    try:
        from flash_attn.cute import flash_attn_func
    except Exception as exc:
        raise RuntimeError(
            "FlashAttention-4 training was requested, but flash-attn-4 or its CuTe DSL "
            "runtime could not be imported. Install the CUDA/PyTorch-matched FA4 environment."
        ) from exc
    return flash_attn_func


def _validate_backend_environment(query: torch.Tensor, attention_backend: str) -> None:
    if attention_backend not in FLASH_ATTENTION_BACKENDS:
        raise ValueError(
            f"Invalid FlashAttention backend {attention_backend!r}; expected one of {sorted(FLASH_ATTENTION_BACKENDS)}"
        )

    backend_label = "FlashAttention-3" if attention_backend == "flash_attn_3" else "FlashAttention-4"
    if query.device.type != "cuda":
        raise ValueError(f"{backend_label} requires CUDA tensors, got {query.device}")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"{backend_label} training requires float16 or bfloat16, got {query.dtype}")

    device_capability = torch.cuda.get_device_capability(query.device)
    if attention_backend == "flash_attn_3":
        if not FLASH3_SUPPORTED:
            raise RuntimeError(
                "FlashAttention-3 training was requested, but flash-attn-3-nv is unavailable. "
                "Install the CUDA/PyTorch-matched FA3 environment."
            )
        if device_capability != (9, 0):
            raise RuntimeError(
                "The flash-attn-3-nv backend currently requires a Hopper GPU "
                f"(compute capability 9.0), got {device_capability}"
            )
    elif device_capability[0] not in {9, 10, 11, 12}:
        raise RuntimeError(
            "The flash-attn-4 backend requires a Hopper or Blackwell GPU "
            f"(compute capability 9.x, 10.x, 11.x, or 12.x), got {device_capability}"
        )


def _run_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_backend: str,
) -> torch.Tensor:
    if attention_backend == "flash_attn_3":
        output = flash3_attention(
            query=query,
            key=key,
            value=value,
            is_causal=False,
        )
    else:
        output = _load_flash4_attention()(query, key, value, causal=False)
        if isinstance(output, tuple):
            output = output[0]

    if not isinstance(output, torch.Tensor):
        raise RuntimeError(f"{attention_backend} unexpectedly returned a non-tensor output")
    return output


def block_causal_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tokens_per_block: int,
    attention_backend: str = "flash_attn_3",
) -> torch.Tensor:
    """Apply full attention within a block and causal attention across blocks.

    The inputs use ``[batch, sequence, heads, head_dim]`` layout.  Each query
    block attends to the key/value prefix ending at that block.  Prefixes are
    tensor views, so this expresses the same mask as OmniDreams' non-interleaved
    FlexAttention path without materializing duplicated K/V tensors.

    This helper is process-group agnostic: it accepts either every attention
    head for CP=1 or a local head shard produced by Ulysses CP.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must use [batch, sequence, heads, head_dim] layout")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(f"query, key, and value shapes must match, got {query.shape}, {key.shape}, {value.shape}")
    if query.device != key.device or query.device != value.device:
        raise ValueError(f"query, key, and value devices must match, got {query.device}, {key.device}, {value.device}")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(f"query, key, and value dtypes must match, got {query.dtype}, {key.dtype}, {value.dtype}")
    if tokens_per_block <= 0:
        raise ValueError(f"tokens_per_block must be positive, got {tokens_per_block}")
    if query.shape[1] == 0:
        raise ValueError("Block-causal FlashAttention requires a non-empty sequence")
    _validate_backend_environment(query, attention_backend)

    sequence_length = query.shape[1]
    outputs = []
    for block_start in range(0, sequence_length, tokens_per_block):
        block_end = min(block_start + tokens_per_block, sequence_length)
        output = _run_flash_attention(
            query[:, block_start:block_end],
            key[:, :block_end],
            value[:, :block_end],
            attention_backend=attention_backend,
        )
        outputs.append(output)

    return torch.cat(outputs, dim=1)


def ulysses_block_causal_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tokens_per_block: int,
    cp_manager: UlyssesCPManager,
    attention_backend: str = "flash_attn_3",
) -> torch.Tensor:
    """Run exact block-causal FlashAttention through a Ulysses adapter.

    With CP>1, inputs and output use the rank-local sequence layout
    ``[B, S / CP, H, D]``. Packed Q/K/V all-to-all temporarily transforms it
    into ``[B, S, H / CP, D]`` so every rank evaluates the full causal
    sequence for an equal subset of heads. CP1 has no process group and takes
    the same API's communication-free identity path.
    """
    if not cp_manager.is_distributed:
        return block_causal_flash_attention(
            query,
            key,
            value,
            tokens_per_block=tokens_per_block,
            attention_backend=attention_backend,
        )

    full_query, full_key, full_value = cp_manager.sequence_to_head_qkv(query, key, value)
    full_output = block_causal_flash_attention(
        full_query,
        full_key,
        full_value,
        tokens_per_block=tokens_per_block,
        attention_backend=attention_backend,
    )
    return cp_manager.head_to_sequence(full_output)
