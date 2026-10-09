# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU validation for the Ulysses context-parallel transforms."""

from types import SimpleNamespace

import pytest
import torch
from omnidreams._src.omnidreams.models.joint_causal_cosmos_model import CausalJointCosmosModel
from omnidreams._src.omnidreams.modules import (
    attention_backend,
    block_causal_flash_attention,
    ulysses_attention,
)
from omnidreams._src.omnidreams.networks import causal_cosmos
from omnidreams._src.omnidreams.self_forcing import dmd as self_forcing_dmd


class _FakeProcessGroup:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _FakeCondition:
    def __init__(self) -> None:
        self.split = None
        self.is_video = True

    def broadcast(self, process_group, *, split: bool):
        self.split = split
        return self


def test_packed_qkv_requires_matching_shapes() -> None:
    query = torch.empty(1, 4, 2, 8)
    key = torch.empty(1, 5, 2, 8)

    with pytest.raises(ValueError, match="requires matching shapes"):
        ulysses_attention.sequence_to_head_qkv(query, key, key, object())


def test_world_size_one_packed_qkv_is_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ulysses_attention.dist, "get_world_size", lambda group: 1)
    inputs = tuple(torch.randn(1, 4, 2, 8) for _ in range(3))

    outputs = ulysses_attention.sequence_to_head_qkv(*inputs, object())

    for actual, expected in zip(outputs, inputs, strict=True):
        torch.testing.assert_close(actual, expected)


def test_sequence_to_head_requires_head_divisibility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ulysses_attention.dist, "get_world_size", lambda group: 2)
    x = torch.empty(1, 4, 3, 8)

    with pytest.raises(ValueError, match="num_heads .* divisible by cp_size"):
        ulysses_attention._sequence_to_head_impl(x, object())


def test_ulysses_manager_replicates_pre_network_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    process_group = _FakeProcessGroup(2)
    manager = ulysses_attention.UlyssesCPManager(process_group)
    condition = _FakeCondition()
    tensors = tuple(torch.randn(1, 4) for _ in range(3))
    broadcasted = []

    def fake_broadcast(tensor, group):
        assert group is process_group
        broadcasted.append(tensor)
        return tensor

    monkeypatch.setattr(ulysses_attention, "broadcast", fake_broadcast)

    actual = manager.prepare_model_inputs(tensors[0], condition, tensors[1], tensors[2])

    assert actual[0] is tensors[0]
    assert actual[1] is condition
    assert actual[2] is tensors[1]
    assert actual[3] is tensors[2]
    assert broadcasted == list(tensors)
    assert condition.split is False


@pytest.mark.parametrize(
    ("backend", "cp_backend", "configured_split", "expected"),
    [
        ("flash_attn_3", "auto", True, False),
        ("flash_attn_4", "auto", True, False),
        ("flex", "ulysses", True, False),
        ("flex", "auto", True, True),
        ("flex", "legacy", False, False),
    ],
)
def test_cp_backend_owns_model_input_partition_policy(
    backend: str,
    cp_backend: str,
    configured_split: bool,
    expected: bool,
) -> None:
    model = SimpleNamespace(
        net=SimpleNamespace(
            training_attention_backend=backend,
            context_parallel_backend=cp_backend,
        ),
        config=SimpleNamespace(split_cp_in_model=configured_split),
    )

    assert CausalJointCosmosModel.split_cp_model_inputs.fget(model) is expected


@pytest.mark.parametrize(
    ("backend", "cp_backend"),
    [
        ("flash_attn_3", "auto"),
        ("flash_attn_4", "auto"),
        ("flex", "ulysses"),
    ],
)
def test_self_forcing_ulysses_keeps_pre_network_inputs_replicated(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    cp_backend: str,
) -> None:
    process_group = _FakeProcessGroup(2)
    condition = _FakeCondition()
    tensors = tuple(torch.randn(1, 4) for _ in range(3))
    enabled_groups = []
    net = SimpleNamespace(
        training_attention_backend=backend,
        context_parallel_backend=cp_backend,
        enable_context_parallel=enabled_groups.append,
        disable_context_parallel=lambda: None,
    )
    model = SimpleNamespace(net=net, get_context_parallel_group=lambda: process_group)

    monkeypatch.setattr(
        self_forcing_dmd,
        "broadcast_split_tensor",
        lambda *args, **kwargs: pytest.fail(
            "FlashAttention self-forcing must not pre-split model inputs"
        ),
    )
    monkeypatch.setattr(ulysses_attention, "broadcast", lambda tensor, group: tensor)

    actual = self_forcing_dmd.ImaginaireDMDBaseModel.broadcast_split_for_model_parallelsim(
        model,
        tensors[0],
        condition,
        tensors[1],
        tensors[2],
    )

    assert actual[0] is tensors[0]
    assert actual[1] is condition
    assert actual[2] is tensors[1]
    assert actual[3] is tensors[2]
    assert condition.split is False
    assert enabled_groups == [process_group]


@pytest.mark.parametrize(
    ("attention", "configured_cp", "cp_size", "expected"),
    [
        ("flex", "auto", 4, "legacy"),
        ("flex", "ulysses", 4, "ulysses"),
        ("flash_attn_3", "auto", 4, "ulysses"),
        ("flash_attn_4", "auto", 4, "ulysses"),
        ("flex", "ulysses", 1, "none"),
        ("flash_attn_3", "auto", 1, "none"),
    ],
)
def test_attention_and_cp_backends_resolve_independently(
    attention: str,
    configured_cp: str,
    cp_size: int,
    expected: str,
) -> None:
    assert (
        attention_backend.resolve_context_parallel_backend(
            attention,
            configured_cp,
            cp_size=cp_size,
        )
        == expected
    )


@pytest.mark.parametrize("backend", ["flash_attn_3", "flash_attn_4"])
def test_flash_attention_rejects_legacy_cp(backend: str) -> None:
    with pytest.raises(ValueError, match="does not communicate the missing K/V prefixes"):
        attention_backend.resolve_context_parallel_backend(
            backend,
            "legacy",
            cp_size=2,
        )


def test_cp1_ulysses_adapter_bypasses_communication(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = tuple(torch.randn(1, 8, 4, 16) for _ in range(3))
    monkeypatch.setattr(
        block_causal_flash_attention,
        "block_causal_flash_attention",
        lambda query, key, value, *, tokens_per_block, attention_backend: query,
    )

    output = block_causal_flash_attention.ulysses_block_causal_flash_attention(
        *inputs,
        tokens_per_block=8,
        cp_manager=ulysses_attention.UlyssesCPManager(),
    )

    assert output is inputs[0]


@pytest.mark.parametrize("backend", ["flash_attn_3", "flash_attn_4"])
@pytest.mark.parametrize(("cp_size", "video_t"), [(1, 1), (2, 2)])
def test_causal_attention_dispatches_to_unified_ulysses_adapter(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    cp_size: int,
    video_t: int,
) -> None:
    attention = causal_cosmos.CausalSelfAttention(
        query_dim=64,
        n_heads=4,
        head_dim=16,
        training_attention_backend=backend,
    )
    process_group = None if cp_size == 1 else _FakeProcessGroup(cp_size)
    attention.cp_group = process_group
    attention.ulysses_cp_manager = ulysses_attention.UlyssesCPManager(process_group)
    attention.q_norm = torch.nn.Identity()
    attention.k_norm = torch.nn.Identity()
    monkeypatch.setattr(attention, "_apply_rope", lambda query, key, rope: (query, key))

    called = False

    def fake_ulysses(
        query,
        key,
        value,
        *,
        tokens_per_block,
        cp_manager,
        attention_backend,
    ):
        nonlocal called
        called = True
        assert query.shape == key.shape == value.shape == (1, 8, 4, 16)
        assert tokens_per_block == 8
        assert cp_manager is attention.ulysses_cp_manager
        assert attention_backend == backend
        return query

    monkeypatch.setattr(causal_cosmos, "ulysses_block_causal_flash_attention", fake_ulysses)

    output = attention(
        torch.randn(1, 8, 64),
        rope_emb=torch.empty(0),
        video_size=causal_cosmos.VideoSize(T=video_t, H=2, W=4),
    )

    assert called
    assert output.shape == (1, 8, 64)


def test_flex_dispatches_to_ulysses_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    attention = causal_cosmos.CausalSelfAttention(
        query_dim=64,
        n_heads=4,
        head_dim=16,
        training_attention_backend="flex",
        context_parallel_backend="ulysses",
    )
    process_group = _FakeProcessGroup(2)
    attention.cp_group = process_group
    attention.ulysses_cp_manager = ulysses_attention.UlyssesCPManager(process_group)
    attention.q_norm = torch.nn.Identity()
    attention.k_norm = torch.nn.Identity()
    monkeypatch.setattr(attention, "_apply_rope", lambda query, key, rope: (query, key))

    block_mask = object()
    called = False

    def fake_ulysses_flex(
        query,
        key,
        value,
        *,
        block_mask: object,
        cp_manager,
        flex_attention_fn,
    ):
        nonlocal called
        called = True
        assert query.shape == key.shape == value.shape == (1, 8, 4, 16)
        assert block_mask is expected_block_mask
        assert cp_manager is attention.ulysses_cp_manager
        assert flex_attention_fn is causal_cosmos.flex_attention
        return query

    expected_block_mask = block_mask
    monkeypatch.setattr(causal_cosmos, "ulysses_flex_attention", fake_ulysses_flex)

    output = attention(
        torch.randn(1, 8, 64),
        rope_emb=torch.empty(0),
        video_size=causal_cosmos.VideoSize(T=2, H=2, W=4),
        block_mask=block_mask,
    )

    assert called
    assert output.shape == (1, 8, 64)
