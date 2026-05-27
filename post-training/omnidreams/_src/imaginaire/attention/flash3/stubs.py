# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Flash Attention v3 (flash3) Backend: intermediate API stubs
Always safe to import (as long as torch is available.)
"""

from torch import Tensor

from omnidreams._src.imaginaire.attention.masks import CausalType


def flash3_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    raise RuntimeError(
        "Tried to run Flash Attention v3, but it is not supported / available. "
        "Try running with debug logs enabled to see why."
    )
