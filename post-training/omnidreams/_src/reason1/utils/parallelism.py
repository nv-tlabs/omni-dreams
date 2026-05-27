# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.distributed.tensor import Replicate, distribute_tensor

except ImportError:
    print("torch.distributed.tensor is not available. DeepSeek model will not work.")


def broadcast(tensor: torch.Tensor, cp_or_tp_mesh: DeviceMesh) -> torch.Tensor:
    tensor = tensor.to("cuda")
    if cp_or_tp_mesh.size() > 1:
        tensor = distribute_tensor(tensor, cp_or_tp_mesh, [Replicate()]).to_local()
    return tensor


def broadcast_with_shape_check(tensor: torch.Tensor, cp_or_tp_mesh: DeviceMesh) -> torch.Tensor:
    """Broadcast a tensor and check if the shape is the same across CP/TP ranks.
    If not, create a new tensor matching rank 0 and broadcast it.

    Args:
        tensor (torch.Tensor): The tensor to broadcast.
        cp_or_tp_mesh (DeviceMesh): The device mesh used to broadcast.

    Returns:
        torch.Tensor: The broadcasted tensor.
    """
    # create a tensor with the original value of the shape
    original_shape = torch.tensor(tensor.shape).cuda()

    # create a tensor that tracks the shape from rank 0.
    final_shape = torch.tensor(tensor.shape).cuda()
    final_shape = broadcast(final_shape, cp_or_tp_mesh)

    # if final shape is different from current shape, create a new tensor
    if final_shape.ne(original_shape).any():
        tensor = torch.zeros(final_shape.tolist(), dtype=tensor.dtype, device=tensor.device)

    tensor = broadcast(tensor, cp_or_tp_mesh)
    return tensor
