# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.predict2.models.text2world_model import EMAConfig

PowerEMAConfig: EMAConfig = EMAConfig(
    enabled=True,
    rate=0.10,
    iteration_shift=0,
)


def register_ema():
    cs = ConfigStore.instance()
    cs.store(group="ema", package="model.config.ema", name="power", node=PowerEMAConfig)
