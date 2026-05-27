# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.omnidreams.models.joint_causal_cosmos_mv_hdmap_model import (
    CausalJointCosmosMVModelHdmap,
    CausalJointCosmosMVModelHdmapConfig,
)
from omnidreams._src.omnidreams.models.joint_causal_cosmos_mv_model import (
    CausalJointCosmosMVModel,
    CausalJointCosmosMVModelConfig,
)

DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(CausalJointCosmosMVModel)(
        config=CausalJointCosmosMVModelConfig(),
        _recursive_=False,
    ),
)

FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CausalJointCosmosMVModel)(
        config=CausalJointCosmosMVModelConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)
DDP_CONFIG_HDMAP = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(CausalJointCosmosMVModelHdmap)(
        config=CausalJointCosmosMVModelHdmapConfig(
            preset_hint_keys=[
                "control_input_hdmap_bbox",
            ],
        ),
        _recursive_=False,
    ),
)

FSDP_CONFIG_HDMAP = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CausalJointCosmosMVModelHdmap)(
        config=CausalJointCosmosMVModelHdmapConfig(
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
    cs.store(group="model", package="_global_", name="ddp_hdmap", node=DDP_CONFIG_HDMAP)
    cs.store(group="model", package="_global_", name="fsdp_hdmap", node=FSDP_CONFIG_HDMAP)
