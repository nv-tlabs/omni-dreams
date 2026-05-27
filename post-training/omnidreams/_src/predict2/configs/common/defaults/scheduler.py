# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.configs.lr_scheduler import LambdaLinearSchedulerConfig


def register_scheduler():
    cs = ConfigStore.instance()
    cs.store(group="scheduler", package="scheduler", name="lambdalinear", node=LambdaLinearSchedulerConfig)
