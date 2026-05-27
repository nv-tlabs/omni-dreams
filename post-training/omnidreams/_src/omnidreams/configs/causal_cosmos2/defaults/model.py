# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.omnidreams.models.joint_causal_cosmos_model import (
    CausalJointCosmosModel,
    CausalJointCosmosModelConfig,
)
from omnidreams._src.omnidreams.models.joint_causal_cosmos_model_hdmap import (
    CausalJointCosmosModelHdmap,
    CausalJointCosmosModelHdmapConfig,
)

DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(CausalJointCosmosModel)(
        config=CausalJointCosmosModelConfig(),
        _recursive_=False,
    ),
)

FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CausalJointCosmosModel)(
        config=CausalJointCosmosModelConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

FSDP_CONFIG_HDMAP = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CausalJointCosmosModelHdmap)(
        config=CausalJointCosmosModelHdmapConfig(
            fsdp_shard_size=8,
            preset_hint_keys=[
                "control_input_hdmap_bbox",
            ],
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp", node=DDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp", node=FSDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_hdmap", node=FSDP_CONFIG_HDMAP)
