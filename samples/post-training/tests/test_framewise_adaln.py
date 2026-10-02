# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Correctness tests for framewise AdaLN modulation."""

from copy import deepcopy

import pytest
import torch
from omnidreams._src.omnidreams.modules.framewise_adaln import (
    apply_adaln_modulation,
    make_token_frame_indices,
    prepare_adaln_block_inputs,
)


class _SelfAttentionStub(torch.nn.Module):
    def forward(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return value.tanh()


class _CrossAttentionStub(torch.nn.Module):
    def forward(self, value: torch.Tensor, context: torch.Tensor, **kwargs) -> torch.Tensor:
        del context, kwargs
        return value.sigmoid()


@pytest.mark.parametrize(
    "token_frame_indices",
    [
        torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        # A CP shard can start/end mid-frame and can use a non-contiguous layout.
        torch.tensor([0, 0, 2, 2, 2, 1, 1]),
    ],
)
def test_framewise_modulation_matches_tokenwise_forward_and_gradients(
    token_frame_indices: torch.Tensor,
) -> None:
    torch.manual_seed(31)
    frame_embedding = torch.randn(2, 3, 16, requires_grad=True)
    optimized_embedding = frame_embedding.detach().clone().requires_grad_(True)
    frame_lora = torch.randn(2, 3, 48, requires_grad=True)
    optimized_lora = frame_lora.detach().clone().requires_grad_(True)
    reference_module = torch.nn.Sequential(
        torch.nn.SiLU(),
        torch.nn.Linear(16, 8, bias=False),
        torch.nn.Linear(8, 48, bias=False),
    )
    optimized_module = deepcopy(reference_module)

    reference = reference_module(frame_embedding.index_select(1, token_frame_indices))
    reference = reference + frame_lora.index_select(1, token_frame_indices)
    optimized = apply_adaln_modulation(
        optimized_module,
        optimized_embedding,
        optimized_lora,
        token_frame_indices=token_frame_indices,
        sequence_length=token_frame_indices.numel(),
    )
    output_gradient = torch.randn_like(reference)
    reference.backward(output_gradient)
    optimized.backward(output_gradient)

    torch.testing.assert_close(optimized, reference)
    torch.testing.assert_close(optimized_embedding.grad, frame_embedding.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(optimized_lora.grad, frame_lora.grad)
    for optimized_parameter, reference_parameter in zip(
        optimized_module.parameters(),
        reference_module.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            optimized_parameter.grad,
            reference_parameter.grad,
            atol=2e-5,
            rtol=2e-5,
        )


def test_token_layout_retains_legacy_behavior() -> None:
    module = torch.nn.Linear(4, 12, bias=False)
    embedding = torch.randn(2, 5, 4)
    adaln_lora = torch.randn(2, 5, 12)

    actual = apply_adaln_modulation(
        module,
        embedding,
        adaln_lora,
        token_frame_indices=None,
        sequence_length=5,
    )

    torch.testing.assert_close(actual, module(embedding) + adaln_lora)


def test_make_token_frame_indices() -> None:
    actual = make_token_frame_indices(3, 4, device=torch.device("cpu"))

    torch.testing.assert_close(
        actual,
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
    )


@pytest.mark.parametrize("framewise", [False, True])
@pytest.mark.parametrize("with_lora", [False, True])
def test_prepare_adaln_block_inputs(framewise: bool, with_lora: bool) -> None:
    frame_embedding = torch.randn(2, 3, 4)
    frame_lora = torch.randn(2, 3, 12) if with_lora else None

    embedding, adaln_lora, token_frame_indices = prepare_adaln_block_inputs(
        frame_embedding,
        frame_lora,
        num_frames=3,
        tokens_per_frame=2,
        framewise=framewise,
    )

    if framewise:
        torch.testing.assert_close(embedding, frame_embedding)
        if frame_lora is None:
            assert adaln_lora is None
        else:
            torch.testing.assert_close(adaln_lora, frame_lora)
        torch.testing.assert_close(token_frame_indices, torch.tensor([0, 0, 1, 1, 2, 2]))
    else:
        torch.testing.assert_close(embedding, torch.repeat_interleave(frame_embedding, 2, dim=1))
        if frame_lora is None:
            assert adaln_lora is None
        else:
            torch.testing.assert_close(adaln_lora, torch.repeat_interleave(frame_lora, 2, dim=1))
        assert token_frame_indices is None


def test_causal_block_framewise_path_matches_legacy_token_path() -> None:
    from omnidreams._src.omnidreams.networks.causal_cosmos import CausalCosmosBlock

    torch.manual_seed(41)
    reference_block = CausalCosmosBlock(
        x_dim=16,
        context_dim=16,
        num_heads=2,
        use_adaln_lora=True,
        adaln_lora_dim=8,
    )
    optimized_block = deepcopy(reference_block)
    for block in (reference_block, optimized_block):
        block.self_attn = _SelfAttentionStub()
        block.cross_attn = _CrossAttentionStub()

    token_frame_indices = torch.tensor([0, 0, 2, 2, 1, 1, 1])
    reference_x = torch.randn(2, token_frame_indices.numel(), 16, requires_grad=True)
    optimized_x = reference_x.detach().clone().requires_grad_(True)
    reference_embedding = torch.randn(2, 3, 16, requires_grad=True)
    optimized_embedding = reference_embedding.detach().clone().requires_grad_(True)
    reference_lora = torch.randn(2, 3, 48, requires_grad=True)
    optimized_lora = reference_lora.detach().clone().requires_grad_(True)
    crossattn = torch.randn(2, 4, 16)

    reference = reference_block(
        reference_x,
        reference_embedding.index_select(1, token_frame_indices),
        crossattn,
        adaln_lora_B_L_3D=reference_lora.index_select(1, token_frame_indices),
    )
    optimized = optimized_block(
        optimized_x,
        optimized_embedding,
        crossattn,
        adaln_lora_B_L_3D=optimized_lora,
        adaln_token_frame_indices=token_frame_indices,
    )
    output_gradient = torch.randn_like(reference)
    reference.backward(output_gradient)
    optimized.backward(output_gradient)

    torch.testing.assert_close(optimized, reference)
    torch.testing.assert_close(optimized_x.grad, reference_x.grad)
    torch.testing.assert_close(
        optimized_embedding.grad,
        reference_embedding.grad,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(optimized_lora.grad, reference_lora.grad, atol=2e-5, rtol=2e-5)
    for optimized_parameter, reference_parameter in zip(
        optimized_block.parameters(),
        reference_block.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            optimized_parameter.grad,
            reference_parameter.grad,
            atol=2e-5,
            rtol=2e-5,
        )
