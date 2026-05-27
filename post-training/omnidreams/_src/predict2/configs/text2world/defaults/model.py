# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.predict2.models.text2world_model import DiffusionModel as Text2WorldModel
from omnidreams._src.predict2.models.text2world_model import Text2WorldModelConfig
from omnidreams._src.predict2.models.text2world_wan2pt1_model import Text2WorldModelWan2pt1Config
from omnidreams._src.predict2.models.text2world_wan2pt1_model import WANDiffusionModel as Text2WorldWan2pt1Model

DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(Text2WorldModel)(
        config=Text2WorldModelConfig(),
        _recursive_=False,
    ),
)

FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Text2WorldModel)(
        config=Text2WorldModelConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

FSDP_WAN2PT1_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Text2WorldWan2pt1Model)(
        config=Text2WorldModelWan2pt1Config(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp", node=DDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp", node=FSDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_wan2pt1", node=FSDP_WAN2PT1_CONFIG)
