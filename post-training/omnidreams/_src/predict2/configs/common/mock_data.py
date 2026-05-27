# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.predict2.datasets.cached_replay_dataloader import get_cached_replay_dataloader
from omnidreams._src.predict2.datasets.data_sources.mock_data import get_image_dataset, get_video_dataset
from omnidreams._src.predict2.datasets.joint_dataloader import IterativeJointDataLoader

_IMAGE_LOADER = L(get_cached_replay_dataloader)(
    dataset=L(get_image_dataset)(
        resolution="512",
    ),
    batch_size=2,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    webdataset=False,
    cache_replay_name="image_dataloader",
)

_VIDEO_LOADER = L(
    get_cached_replay_dataloader
)(
    dataset=L(get_video_dataset)(
        resolution="512",
        num_video_frames=136,  # number of pixel frames, the number needs to agree with tokenizer encoder since tokenizer can not handle arbitrary length
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    webdataset=False,
    cache_replay_name="video_dataloader",
)

MOCK_DATA_INTERLEAVE_CONFIG = L(IterativeJointDataLoader)(
    dataloaders={
        "image_data": {
            "dataloader": _IMAGE_LOADER,
            "ratio": 1,
        },
        "video_data": {
            "dataloader": _VIDEO_LOADER,
            "ratio": 1,
        },
    }
)

MOCK_DATA_IMAGE_ONLY_CONFIG = _IMAGE_LOADER

MOCK_DATA_VIDEO_ONLY_CONFIG = _VIDEO_LOADER
