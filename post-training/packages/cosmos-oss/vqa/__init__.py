# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""VQA (Video Question Answering) package for Cosmos models."""

__version__ = "1.0.0"

# Make key components available at package level
try:
    from vqa.cosmos_reason_inference import CosmosReasonModel

    __all__ = [
        "CosmosReasonModel",
    ]
except ImportError:
    # Allow package to be imported even if dependencies are not installed
    pass
