# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from enum import Enum


class EmbeddingConcatStrategy(str, Enum):
    FULL_CONCAT = "full_concat"  # Concatenate embeddings all layers
    MEAN_POOLING = "mean_pooling"  # Average pool embeddings all layers
    POOL_EVERY_N_LAYERS_AND_CONCAT = "pool_every_n_layers_and_concat"  # Pool every n layers and concatenatenate

    def __str__(self) -> str:
        return self.value
