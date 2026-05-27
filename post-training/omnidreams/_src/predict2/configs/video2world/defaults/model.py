# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.predict2.models.text2world_wan2pt1_model import Text2WorldModelWan2pt1Config
from omnidreams._src.predict2.models.video2world_model import Video2WorldConfig, Video2WorldModel
from omnidreams._src.predict2.models.video2world_model_rectified_flow import (
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from omnidreams._src.predict2.models.video2world_wan2pt1_model import I2VWan2pt1Model

DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(Video2WorldModel)(
        config=Video2WorldConfig(),
        _recursive_=False,
    ),
)

FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Video2WorldModel)(
        config=Video2WorldConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


FSDP_WAN2PT1_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(I2VWan2pt1Model)(
        config=Text2WorldModelWan2pt1Config(
            fsdp_shard_size=8,
            state_t=24,
        ),
        _recursive_=False,
    ),
)

FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Video2WorldModelRectifiedFlow)(
        config=Video2WorldModelRectifiedFlowConfig(
            fsdp_shard_size=8,
            state_t=24,
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp", node=DDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp", node=FSDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_wan2pt1", node=FSDP_WAN2PT1_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_rectified_flow", node=FSDP_RECTIFIED_FLOW_CONFIG)
