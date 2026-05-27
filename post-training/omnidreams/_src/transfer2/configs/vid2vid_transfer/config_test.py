# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams._src.imaginaire.utils.config_helper import override
from omnidreams._src.transfer2.configs.vid2vid_transfer.config import make_config


def test_config():
    config = make_config()
    config = override(config)
    assert config.model is not None
    assert config.optimizer is not None
    assert config.scheduler is not None
