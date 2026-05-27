# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.predict2_multiview.datasets.multiview import (
    DEFAULT_CAMERA_VIEW_MAPPING,
    DEFAULT_CAMERAS,
    DEFAULT_VIDEO_KEY_MAPPING,
)
from omnidreams._src.omnidreams.datasets.multiview import (
    AugmentationConfig,
    get_multiview_video_loader,
)

DEFAULT_CAMERA_VIEW_CONFIGS = {
    "7views": DEFAULT_CAMERAS,
    "4views": [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
    ],
    "1view": [
        "camera_front_wide_120fov",
    ],
}

INDEX_TO_CAMERA_MAPPING = {v: k for k, v in DEFAULT_CAMERA_VIEW_MAPPING.items()}
DEFAULT_CAPTION_KEY_MAPPING = dict(zip(DEFAULT_CAMERAS, [f"metas_{k}" for k in DEFAULT_CAMERAS]))
DEFAULT_CONTROL_KEY_MAPPING: dict[str, str] = {
    camera_name: f"world_scenario_{camera_name}" for i, camera_name in INDEX_TO_CAMERA_MAPPING.items()
}


