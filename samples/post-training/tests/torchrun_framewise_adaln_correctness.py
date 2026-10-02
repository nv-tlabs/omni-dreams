# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed forward/gradient oracle for framewise AdaLN with contiguous CP.

Run from ``post-training`` in the CUDA 12.8 environment:

    uv run --extra cu128 torchrun --standalone --nproc-per-node=2 \
      ../samples/post-training/tests/torchrun_framewise_adaln_correctness.py
"""

from __future__ import annotations

import os
from copy import deepcopy

import torch
import torch.distributed as dist
from omnidreams._src.omnidreams.modules.framewise_adaln import (
    apply_adaln_modulation,
    make_token_frame_indices,
    shard_adaln_block_inputs,
)


def _local_sequence_shard(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return x.chunk(world_size, dim=1)[rank].contiguous()


def _gather_sequence(x: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=1)


def _test_framewise_adaln_oracle(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """Validate compact AdaLN when CP shards start and end inside frames."""
    batch, num_frames, embedding_dim = 2, world_size + 1, 16
    tokens_per_frame = world_size
    generator = torch.Generator(device=device).manual_seed(20260921)
    frame_embedding = torch.randn(
        batch,
        num_frames,
        embedding_dim,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    frame_lora = torch.randn(
        batch,
        num_frames,
        3 * embedding_dim,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    torch.manual_seed(20260921)
    module = torch.nn.Sequential(
        torch.nn.SiLU(),
        torch.nn.Linear(embedding_dim, embedding_dim, bias=False),
        torch.nn.Linear(embedding_dim, 3 * embedding_dim, bias=False),
    ).to(device)
    reference_module = deepcopy(module)
    reference_embedding = frame_embedding.detach().clone().requires_grad_(True)
    reference_lora = frame_lora.detach().clone().requires_grad_(True)

    global_indices = make_token_frame_indices(
        num_frames,
        tokens_per_frame,
        device=device,
    )
    local_embedding, local_lora, local_indices = shard_adaln_block_inputs(
        frame_embedding,
        frame_lora,
        global_indices,
        cp_group=dist.group.WORLD,
    )
    assert local_indices is not None
    local_output = apply_adaln_modulation(
        module,
        local_embedding,
        local_lora,
        token_frame_indices=local_indices,
        sequence_length=local_indices.numel(),
    )
    reference_output = reference_module(reference_embedding.index_select(1, global_indices))
    reference_output = reference_output + reference_lora.index_select(1, global_indices)
    torch.testing.assert_close(
        _gather_sequence(local_output.detach(), world_size),
        reference_output,
    )

    output_gradient = torch.randn(
        reference_output.shape,
        device=device,
        generator=generator,
    )
    local_output.backward(_local_sequence_shard(output_gradient, rank, world_size))
    reference_output.backward(output_gradient)

    distributed_gradients = [frame_embedding.grad, frame_lora.grad]
    distributed_gradients.extend(parameter.grad for parameter in module.parameters())
    reference_gradients = [reference_embedding.grad, reference_lora.grad]
    reference_gradients.extend(parameter.grad for parameter in reference_module.parameters())
    for actual, expected in zip(distributed_gradients, reference_gradients, strict=True):
        dist.all_reduce(actual)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    if rank == 0:
        print(
            f"PASS: framewise AdaLN contiguous CP={world_size} forward and gradients",
            flush=True,
        )


def main() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("The distributed correctness oracle requires at least two ranks")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    try:
        _test_framewise_adaln_oracle(device, dist.get_rank(), world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
