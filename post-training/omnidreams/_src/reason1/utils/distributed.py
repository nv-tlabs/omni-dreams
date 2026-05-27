# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch.distributed as dist


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get the rank (GPU device) of the worker.

    Returns:
        rank (int): The rank of the worker.
    """
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group)
    return rank


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get world size. How many GPUs are available in this job.

    Returns:
        world_size (int): The total number of GPUs available in this job.
    """
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group)
    return world_size


def broadcast(tensor, src, group=None, async_op=False):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    dist.broadcast(tensor, src=src, group=group, async_op=async_op)
