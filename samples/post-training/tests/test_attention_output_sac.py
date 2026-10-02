# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the backend-neutral attention-output SAC policy."""

import pytest
import torch
from omnidreams._src.predict2.networks.selective_activation_checkpoint import (
    CheckpointMode,
    SACConfig,
    attention_output_policy,
    compiled_attention_region,
    is_attention_output_op,
    is_attention_output_sac_active,
)
from torch.utils.checkpoint import checkpoint


class _NamedOp:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


_ATTENTION_CALLS = 0


@torch.library.custom_op("omnidreams_sac_test::flex_attention", mutates_args=())
def _attention_op(value: torch.Tensor) -> torch.Tensor:
    global _ATTENTION_CALLS
    _ATTENTION_CALLS += 1
    return value.sin()


@_attention_op.register_fake
def _attention_op_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


def _attention_op_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(inputs[0])


def _attention_op_backward(ctx, output_gradient: torch.Tensor) -> torch.Tensor:
    (value,) = ctx.saved_tensors
    return output_gradient * value.cos()


_attention_op.register_autograd(_attention_op_backward, setup_context=_attention_op_setup_context)


@pytest.mark.parametrize(
    "op_name",
    [
        "flex_attention",
        "torch.ops.higher_order.flex_attention",
        "flash_attn_3_nv::_flash_attn_forward",
        "flash_attn_4::_flash_attn_forward",
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten._scaled_dot_product_cudnn_attention.default",
    ],
)
def test_attention_output_policy_saves_fused_attention_ops(op_name: str) -> None:
    op = _NamedOp(op_name)

    assert is_attention_output_op(op)
    assert attention_output_policy(None, op) == torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE


@pytest.mark.parametrize(
    "op_name",
    [
        "aten.mm.default",
        "aten.layer_norm.default",
        "aten.silu.default",
        "aten.add.Tensor",
    ],
)
def test_attention_output_policy_recomputes_non_attention_ops(op_name: str) -> None:
    op = _NamedOp(op_name)

    assert not is_attention_output_op(op)
    assert (
        attention_output_policy(None, op)
        == torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
    )


def test_compiled_region_is_saved_only_inside_attention_scope() -> None:
    compiled_op = _NamedOp("inductor_compiled_code")

    assert not is_attention_output_op(compiled_op)
    with compiled_attention_region():
        assert is_attention_output_op(compiled_op)
        assert (
            attention_output_policy(None, compiled_op)
            == torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
        )
    assert not is_attention_output_op(compiled_op)


def test_attention_output_mode_builds_selective_checkpoint_contexts() -> None:
    config = SACConfig(mode=CheckpointMode.ATTENTION_OUTPUT)

    forward_context, recompute_context = config.get_context_fn()()

    assert forward_context is not None
    assert recompute_context is not None

    assert not is_attention_output_sac_active()
    with forward_context:
        assert is_attention_output_sac_active()
    assert not is_attention_output_sac_active()


def test_minimal_dit_config_inherits_attention_output_mode() -> None:
    from omnidreams._src.predict2.networks.minimal_v4_dit import (
        CheckpointMode as MinimalCheckpointMode,
    )
    from omnidreams._src.predict2.networks.minimal_v4_dit import SACConfig as MinimalSACConfig

    config = MinimalSACConfig(mode=MinimalCheckpointMode.ATTENTION_OUTPUT)

    forward_context, recompute_context = config.get_context_fn()()

    assert forward_context is not None
    assert recompute_context is not None


def test_attention_output_sac_does_not_replay_saved_attention_op() -> None:
    global _ATTENTION_CALLS
    _ATTENTION_CALLS = 0
    value = torch.randn(8, requires_grad=True)
    config = SACConfig(mode=CheckpointMode.ATTENTION_OUTPUT)

    output = checkpoint(
        lambda tensor: _attention_op(tensor).square(),
        value,
        use_reentrant=False,
        context_fn=config.get_context_fn(),
    )
    output.sum().backward()

    assert _ATTENTION_CALLS == 1
    assert value.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_flex_attention_output_sac_matches_reference() -> None:
    from omnidreams._src.omnidreams.modules.compiled_flex_attention_sac import (
        clear_compiled_flex_attention_sac_cache,
        compiled_flex_attention_for_output_sac,
        supports_compiled_flex_attention_output_sac,
    )
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    if not supports_compiled_flex_attention_output_sac():
        pytest.skip("requires the PyTorch 2.10 compiled FlexAttention APIs")

    torch.manual_seed(1234)
    shape = (1, 2, 256, 64)
    block_ends = torch.arange(shape[2], device="cuda") // 128 * 128 + 128

    def mask_mod(b, h, query_index, key_index):
        del b, h
        return key_index < block_ends[query_index.to(torch.long)]

    block_mask = create_block_mask(mask_mod, None, None, shape[2], shape[2], device="cuda")
    actual_inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    reference_inputs = [value.detach().clone().requires_grad_() for value in actual_inputs]
    output_gradient = torch.randn_like(actual_inputs[0])

    compiled_reference = torch.compile(flex_attention, dynamic=False)
    reference = compiled_reference(*reference_inputs, block_mask=block_mask)
    reference.backward(output_gradient)

    clear_compiled_flex_attention_sac_cache()
    config = SACConfig(mode=CheckpointMode.ATTENTION_OUTPUT)
    actual = checkpoint(
        lambda query, key, value: compiled_flex_attention_for_output_sac(
            query,
            key,
            value,
            block_mask=block_mask,
        ),
        *actual_inputs,
        use_reentrant=False,
        context_fn=config.get_context_fn(),
    )
    actual.backward(output_gradient)

    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    for actual_input, reference_input in zip(actual_inputs, reference_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, reference_input.grad, atol=2e-2, rtol=2e-2)
