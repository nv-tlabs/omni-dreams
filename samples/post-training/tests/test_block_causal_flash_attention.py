# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Correctness tests for the FlashAttention block-causal backends."""

import math

import pytest
import torch
from omnidreams._src.omnidreams.modules.block_causal_flash_attention import (
    _run_flash_attention,
    block_causal_flash_attention,
)


def _hopper_is_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _dense_block_causal_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tokens_per_block: int,
) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) / math.sqrt(
        query.shape[-1]
    )
    positions = torch.arange(query.shape[1], device=query.device)
    query_block = torch.div(positions, tokens_per_block, rounding_mode="floor")
    key_block = torch.div(positions, tokens_per_block, rounding_mode="floor")
    keep = key_block[None, :] <= query_block[:, None]
    scores = scores.masked_fill(~keep[None, None], float("-inf"))
    probabilities = scores.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, value.float())


@pytest.mark.parametrize("tokens_per_block", [0, -1])
def test_flash_attention_rejects_non_positive_block_size(tokens_per_block: int) -> None:
    inputs = [torch.empty((1, 1, 1, 64)) for _ in range(3)]

    with pytest.raises(ValueError, match="tokens_per_block must be positive"):
        block_causal_flash_attention(*inputs, tokens_per_block=tokens_per_block)


def test_flash_attention_rejects_empty_sequence() -> None:
    inputs = [torch.empty((1, 0, 1, 64)) for _ in range(3)]

    with pytest.raises(ValueError, match="requires a non-empty sequence"):
        block_causal_flash_attention(*inputs, tokens_per_block=1)


def test_flash_attention_rejects_unknown_backend() -> None:
    inputs = [torch.empty((1, 1, 1, 64)) for _ in range(3)]

    with pytest.raises(ValueError, match="Invalid FlashAttention backend"):
        block_causal_flash_attention(
            *inputs,
            tokens_per_block=1,
            attention_backend="unknown",
        )


def test_flash4_unwraps_public_api_output(monkeypatch: pytest.MonkeyPatch) -> None:
    query, key, value = [torch.randn((1, 2, 1, 8)) for _ in range(3)]
    expected = torch.randn_like(query)

    def fake_flash4(q, k, v, *, causal):
        assert q is query
        assert k is key
        assert v is value
        assert causal is False
        return expected, torch.empty(0)

    monkeypatch.setattr(
        "omnidreams._src.omnidreams.modules.block_causal_flash_attention._load_flash4_attention",
        lambda: fake_flash4,
    )

    actual = _run_flash_attention(
        query,
        key,
        value,
        attention_backend="flash_attn_4",
    )

    assert actual is expected


@pytest.mark.parametrize(
    ("shape", "tokens_per_block"),
    [
        ((1, 12, 2, 64), 4),
        # Production head_dim, B>1, and a partial final block.
        ((2, 10, 4, 128), 4),
        # A single partial block larger than the sequence is full attention.
        ((1, 5, 2, 64), 8),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["flash_attn_3", "flash_attn_4"])
@pytest.mark.skipif(
    not _hopper_is_available(),
    reason="The FlashAttention integration test requires a Hopper (SM90) GPU",
)
def test_flash_attention_matches_dense_block_causal_forward_and_gradients(
    shape: tuple[int, int, int, int],
    tokens_per_block: int,
    dtype: torch.dtype,
    backend: str,
) -> None:
    if backend == "flash_attn_3":
        pytest.importorskip("flash_attn_3_nv")
    else:
        pytest.importorskip("flash_attn.cute")

    torch.manual_seed(7)
    actual_inputs = [
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
    ]
    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in actual_inputs]
    output_gradient = torch.randn(shape, device="cuda", dtype=dtype)

    actual = block_causal_flash_attention(
        *actual_inputs,
        tokens_per_block=tokens_per_block,
        attention_backend=backend,
    )
    reference = _dense_block_causal_reference(
        *reference_inputs,
        tokens_per_block=tokens_per_block,
    )
    actual.backward(output_gradient)
    reference.backward(output_gradient.float())

    torch.testing.assert_close(actual.float(), reference, atol=3e-2, rtol=3e-2)
    for actual_input, reference_input in zip(actual_inputs, reference_inputs, strict=True):
        assert actual_input.grad is not None
        assert reference_input.grad is not None
        torch.testing.assert_close(
            actual_input.grad.float(),
            reference_input.grad.float(),
            atol=6e-2,
            rtol=6e-2,
        )
