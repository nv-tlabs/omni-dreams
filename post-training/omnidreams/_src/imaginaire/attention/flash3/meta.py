# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Flash Attention v3 (flash3) Backend: metadata
Always safe to import (as long as torch is available.)
"""

import torch

from omnidreams._src.imaginaire.attention.utils import safe_log as log


def get_fwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """
    Returns data type choices for forward pass according to arch tag (attention.utils.get_arch_tag).

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        data_type_choices (list): a list of PyTorch data types. Empty if device is not supported.

    """

    if arch_tag != 90:
        log.debug("Flash Attention v3 (flash3) only supports compute capability 9.0 (Hopper).")
        return []

    return [torch.float16, torch.bfloat16]


def get_bwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """
    Returns data type choices for backward pass according to arch tag (attention.utils.get_arch_tag).

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        data_type_choices (list): a list of PyTorch data types. Empty if device is not supported.

    """

    if arch_tag != 90:
        log.debug("Flash Attention v3 (flash3) only supports compute capability 9.0 (Hopper).")
        return []

    return [torch.float16, torch.bfloat16]
