# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from omnidreams._src.imaginaire.datasets.webdataset.dataloader import DataLoader as WebDataLoader
from omnidreams._src.imaginaire.lazy_config import LazyCall as L





def register_camera_data():
    cs = ConfigStore.instance()

