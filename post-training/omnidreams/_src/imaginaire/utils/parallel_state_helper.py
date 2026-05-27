# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
This module contains various helper functions designed to extend the functionality of parallel states within the MCore library.

MCore is a third-party library that is infrequently updated and may introduce backward compatibility issues in our codebase, such as changes in function signatures or missing / new functions in new versions.

To mitigate these issues, this module provides stable functions that ensure the omnidreams._src.imaginaire codebase remains compatible with different versions of MCore.
"""

try:
    from megatron.core import parallel_state
except ImportError:
    print("Megatron is not installed, is_tp_cp_pp_rank0 functions will not work.")


def is_tp_cp_pp_rank0():
    return (
        parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_pipeline_model_parallel_rank() == 0
        and parallel_state.get_context_parallel_rank() == 0
    )
