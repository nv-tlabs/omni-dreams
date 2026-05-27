# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Environment-related utilities.
"""

import torch

from omnidreams._src.imaginaire.utils import log


# Controls all regions guarded against torch compile
# Logs, and certain assertions cause graph breaks.
def is_torch_compiling() -> bool:
    try:
        return torch.compiler.is_compiling()
    except Exception as e:
        log.exception(f"Exception occurred checking whether in torch compiled region: {e}")
        # Assume too old to support torch compile
        return False
