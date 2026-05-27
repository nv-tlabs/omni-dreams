# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.predict2_multiview.models.multiview_vid2vid_model_rectified_flow import (
    MultiviewVid2VidModelRectifiedFlow,
    MultiviewVid2VidModelRectifiedFlowConfig,
)

FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(MultiviewVid2VidModelRectifiedFlow)(
        config=MultiviewVid2VidModelRectifiedFlowConfig(),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="fsdp_rectified_flow_multiview", node=FSDP_RECTIFIED_FLOW_CONFIG)
