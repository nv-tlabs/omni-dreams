# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.predict2.camera.models.multiview_camera_ar_video2world_model import (
    CameraConditionedARVideo2WorldModelRectifiedFlow,
    CameraConditionedARVideo2WorldRectifiedFlowConfig,
)
from omnidreams._src.predict2.camera.models.multiview_camera_frameinit_video2world_model import (
    CameraConditionedFrameinitVideo2WorldModelRectifiedFlow,
    CameraConditionedFrameinitVideo2WorldRectifiedFlowConfig,
)
from omnidreams._src.predict2.camera.models.multiview_camera_video2world_model import (
    CameraConditionedVideo2WorldModelRectifiedFlow,
    CameraConditionedVideo2WorldRectifiedFlowConfig,
)

CAMERA_CONDITIONED_FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CameraConditionedVideo2WorldModelRectifiedFlow)(
        config=CameraConditionedVideo2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

CAMERA_CONDITIONED_FRAMEINIT_FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CameraConditionedFrameinitVideo2WorldModelRectifiedFlow)(
        config=CameraConditionedFrameinitVideo2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

CAMERA_CONDITIONED_AR_FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CameraConditionedARVideo2WorldModelRectifiedFlow)(
        config=CameraConditionedARVideo2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="camera_conditioned_rectified_flow_fsdp",
        node=CAMERA_CONDITIONED_FSDP_RECTIFIED_FLOW_CONFIG,
    )
    cs.store(
        group="model",
        package="_global_",
        name="camera_conditioned_frameinit_rectified_flow_fsdp",
        node=CAMERA_CONDITIONED_FRAMEINIT_FSDP_RECTIFIED_FLOW_CONFIG,
    )
    cs.store(
        group="model",
        package="_global_",
        name="camera_conditioned_ar_rectified_flow_fsdp",
        node=CAMERA_CONDITIONED_AR_FSDP_RECTIFIED_FLOW_CONFIG,
    )
