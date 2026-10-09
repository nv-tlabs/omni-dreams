# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Attention-kernel and context-parallel backend compatibility."""

FLASH_ATTENTION_BACKENDS = frozenset({"flash_attn_3", "flash_attn_4"})
TRAINING_ATTENTION_BACKENDS = frozenset({"flex", *FLASH_ATTENTION_BACKENDS})
CONTEXT_PARALLEL_BACKENDS = frozenset({"auto", "legacy", "ulysses"})

_AUTO_CONTEXT_PARALLEL_BACKEND = {
    "flex": "legacy",
    "flash_attn_3": "ulysses",
    "flash_attn_4": "ulysses",
}
_SUPPORTED_CONTEXT_PARALLEL_BACKENDS = {
    "flex": frozenset({"legacy", "ulysses"}),
    "flash_attn_3": frozenset({"ulysses"}),
    "flash_attn_4": frozenset({"ulysses"}),
}


def validate_attention_backends(
    attention_backend: str,
    context_parallel_backend: str,
) -> None:
    """Reject unknown or mathematically unsupported backend combinations."""
    if attention_backend not in TRAINING_ATTENTION_BACKENDS:
        raise ValueError(
            f"Invalid training_attention_backend={attention_backend!r}; "
            f"expected one of {sorted(TRAINING_ATTENTION_BACKENDS)}"
        )
    if context_parallel_backend not in CONTEXT_PARALLEL_BACKENDS:
        raise ValueError(
            f"Invalid context_parallel_backend={context_parallel_backend!r}; "
            f"expected one of {sorted(CONTEXT_PARALLEL_BACKENDS)}"
        )
    if context_parallel_backend == "auto":
        return
    if context_parallel_backend not in _SUPPORTED_CONTEXT_PARALLEL_BACKENDS[attention_backend]:
        supported = sorted(_SUPPORTED_CONTEXT_PARALLEL_BACKENDS[attention_backend])
        raise ValueError(
            f"Unsupported attention/CP combination: training_attention_backend={attention_backend!r}, "
            f"context_parallel_backend={context_parallel_backend!r}; supported CP backends for "
            f"{attention_backend!r}: {supported}. FlashAttention kernels require Ulysses for CP>1 "
            "because legacy sequence sharding does not communicate the missing K/V prefixes."
        )


def resolve_context_parallel_backend(
    attention_backend: str,
    context_parallel_backend: str,
    *,
    cp_size: int,
) -> str:
    """Resolve ``auto`` independently from the selected attention kernel."""
    validate_attention_backends(attention_backend, context_parallel_backend)
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if cp_size == 1:
        return "none"
    if context_parallel_backend == "auto":
        return _AUTO_CONTEXT_PARALLEL_BACKEND[attention_backend]
    return context_parallel_backend


def uses_ulysses_context_parallel(
    attention_backend: str,
    context_parallel_backend: str,
    *,
    cp_size: int,
) -> bool:
    """Return whether this execution uses Ulysses communication."""
    return (
        resolve_context_parallel_backend(
            attention_backend,
            context_parallel_backend,
            cp_size=cp_size,
        )
        == "ulysses"
    )
