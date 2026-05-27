# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from omnidreams._src.imaginaire.utils.config_helper import override
from omnidreams._src.predict2.configs.text2world.config import make_config


@pytest.mark.L1
def test_make_config():
    config = make_config()
    config = override(config, ["--", "experiment=error-free_fsdp_mock-data_base-cb", "trainer.max_iter=1"])

    assert config.trainer.max_iter == 1
