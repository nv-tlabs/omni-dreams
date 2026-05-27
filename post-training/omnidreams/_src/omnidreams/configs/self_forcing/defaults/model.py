# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.omnidreams.self_forcing.self_forcing_dmd import (
    DMDSelfForcingModel,
    DMDSelfForcingModelConfig,
)
from omnidreams._src.omnidreams.self_forcing.self_forcing_dmd_hdmap import (
    DMDSelfForcingModelHDMap,
    DMDSelfForcingModelHDMapConfig,
)
from omnidreams._src.omnidreams.self_forcing.self_forcing_dmd_mv_hdmap import (
    DMDSelfForcingMVModelHDMap,
    DMDSelfForcingMVModelHDMapConfig,
)

FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(DMDSelfForcingModel)(
        config=DMDSelfForcingModelConfig(
            fsdp_shard_size=32,
        ),
        _recursive_=False,
    ),
)


HDMAP_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(DMDSelfForcingModelHDMap)(
        config=DMDSelfForcingModelHDMapConfig(
            fsdp_shard_size=32,
            preset_hint_keys=[
                "control_input_hdmap_bbox",
            ],
        ),
        _recursive_=False,
    ),
)


MV_HDMAP_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(DMDSelfForcingMVModelHDMap)(
        config=DMDSelfForcingMVModelHDMapConfig(
            fsdp_shard_size=32,
            preset_hint_keys=[
                "control_input_hdmap_bbox",
            ],
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="fsdp", node=FSDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_hdmap", node=HDMAP_FSDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_mv_hdmap", node=MV_HDMAP_FSDP_CONFIG)
