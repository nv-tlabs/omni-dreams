# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.transfer2.models.vid2vid_model_control_vace import (
    ControlVideo2WorldConfig,
    ControlVideo2WorldModel,
)
from omnidreams._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import (
    ControlVideo2WorldModelRectifiedFlow,
    ControlVideo2WorldRectifiedFlowConfig,
)

DDP_CONFIG_CONTROL_VACE = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(ControlVideo2WorldModel)(
        config=ControlVideo2WorldConfig(),
        _recursive_=False,
    ),
)

FSDP_CONFIG_CONTROL_VACE = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ControlVideo2WorldModel)(
        config=ControlVideo2WorldConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ControlVideo2WorldModelRectifiedFlow)(
        config=ControlVideo2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp_control_vace", node=DDP_CONFIG_CONTROL_VACE)
    cs.store(group="model", package="_global_", name="fsdp_control_vace", node=FSDP_CONFIG_CONTROL_VACE)
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_control_vace_rectified_flow",
        node=FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW,
    )
