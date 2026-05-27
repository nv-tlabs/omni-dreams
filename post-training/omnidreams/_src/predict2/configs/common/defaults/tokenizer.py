# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.predict2.tokenizers.cosmos import (
    Wan2pt1VAEConfig,
    Wan2pt1VAEConfig_GCP,
    Wan2pt2VAEConfig,
)



def register_tokenizer():
    cs = ConfigStore.instance()

    # Wan2pt1 and Wan2pt2 tokenizers
    cs.store(group="tokenizer", package="model.config.tokenizer", name="wan2pt1_tokenizer", node=Wan2pt1VAEConfig)
    cs.store(
        group="tokenizer", package="model.config.tokenizer", name="wan2pt1_tokenizer_gcp", node=Wan2pt1VAEConfig_GCP
    )
    cs.store(group="tokenizer", package="model.config.tokenizer", name="wan2pt2_tokenizer", node=Wan2pt2VAEConfig)

