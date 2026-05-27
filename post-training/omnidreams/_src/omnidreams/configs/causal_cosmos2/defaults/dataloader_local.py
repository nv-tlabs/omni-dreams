# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hydra ConfigStore registration for the local (filesystem) multiview loader."""

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.predict2_multiview.datasets.multiview import (
    DEFAULT_VIDEO_KEY_MAPPING,
)
from omnidreams._src.omnidreams.configs.causal_cosmos2.defaults.dataloader import (
    DEFAULT_CAMERA_VIEW_CONFIGS,
    DEFAULT_CAPTION_KEY_MAPPING,
    DEFAULT_CONTROL_KEY_MAPPING,
)
from omnidreams._src.omnidreams.datasets.local_multiview import (
    get_local_multiview_video_loader,
)
from omnidreams._src.omnidreams.datasets.multiview import AugmentationConfig

# Only the front-wide camera is available locally under data/.
_LOCAL_VIEWS = {"1view": DEFAULT_CAMERA_VIEW_CONFIGS["1view"]}

_RESOLUTIONS = [
    ("480p", (480, 832)),
    ("720p", (704, 1280)),
    ("1080p", (1080, 1920)),
]
_FPS = [
    ("10fps", 3),
    ("15fps", 2),
    ("30fps", 1),
]
_NUM_FRAMES = [
    ("29frames", 29),
    ("61frames", 61),
    ("93frames", 93),
]


def register_local_multiview_dataloader(data_root: str = "data", repeat_factor: int = 1) -> None:
    """Register filesystem-backed multiview dataloader configurations.

    Names follow the pattern ``video_local_<res>_<fps>_<frames>_1view`` and
    are registered under both ``data_train`` and ``data_val``.

    ``repeat_factor`` multiplies the underlying clip list and is the
    sample-side knob for running on small datasets (see e.g. the PAI
    intersect sample, <200 clips); experiments that override the
    dataloader can also set this directly on the Hydra node.
    """
    cs = ConfigStore.instance()

    for resolution_str, resolution_hw in _RESOLUTIONS:
        for fps_str, downsample_factor in _FPS:
            for num_frames_str, num_frames in _NUM_FRAMES:
                for views_str, camera_keys in _LOCAL_VIEWS.items():
                    camera_keys_set = set(camera_keys)
                    video_key_mapping = {k: v for k, v in DEFAULT_VIDEO_KEY_MAPPING.items() if k in camera_keys_set}
                    caption_key_mapping = {k: v for k, v in DEFAULT_CAPTION_KEY_MAPPING.items() if k in camera_keys_set}
                    control_key_mapping = {k: v for k, v in DEFAULT_CONTROL_KEY_MAPPING.items() if k in camera_keys_set}
                    name = f"video_local_{resolution_str}_{fps_str}_{num_frames_str}_{views_str}"
                    augmentation_config = L(AugmentationConfig)(
                        resolution_hw=resolution_hw,
                        fps_downsample_factor=downsample_factor,
                        num_video_frames=num_frames,
                        camera_keys=camera_keys,
                        camera_video_key_mapping=video_key_mapping,
                        camera_caption_key_mapping=caption_key_mapping,
                        camera_control_key_mapping=control_key_mapping,
                    )
                    cs.store(
                        group="data_train",
                        package="dataloader_train",
                        name=name,
                        node=L(get_local_multiview_video_loader)(
                            is_train=True,
                            data_root=data_root,
                            augmentation_config=augmentation_config,
                            num_workers=4,
                            prefetch_factor=2,
                            repeat_factor=repeat_factor,
                        ),
                    )
                    cs.store(
                        group="data_val",
                        package="dataloader_val",
                        name=name,
                        node=L(get_local_multiview_video_loader)(
                            is_train=False,
                            data_root=data_root,
                            augmentation_config=augmentation_config,
                            batch_size=1,
                            num_workers=2,
                            prefetch_factor=1,
                        ),
                    )
