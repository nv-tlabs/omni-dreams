# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Dataset registration for cosmos datasets with support for different caption types.
"""

from typing import Dict, List, Optional, Tuple

from omnidreams._src.imaginaire import config
from omnidreams._src.imaginaire.datasets.webdataset.config.schema import DatasetInfo
from omnidreams._src.imaginaire.utils import log


DATASET_OPTIONS = {}


# embeddings are packed together. Need to clean data to reduce entropy.
_CAPTION_EMBEDDING_KEY_MAPPING_IMAGES = {
    "ai_v3p1": "ai_v3p1",
    "qwen2p5_7b_v4": "qwen2p5_7b_v4",
    "prompts": "qwen2p5_7b_v4",
}


def dataset_register(key):
    log.info(f"registering dataset {key}")

    def decorator(func):
        DATASET_OPTIONS[key] = func
        return func

    return decorator


