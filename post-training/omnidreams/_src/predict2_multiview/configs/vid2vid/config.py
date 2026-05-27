# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams._src.imaginaire.flags import INTERNAL
from omnidreams._src.imaginaire.utils.config_helper import import_all_modules_from_package
from omnidreams._src.predict2.configs.video2world.config import make_config as vid2vid_make_config
from omnidreams._src.predict2_multiview.configs.vid2vid.defaults.callbacks import register_callbacks
from omnidreams._src.predict2_multiview.configs.vid2vid.defaults.conditioner import register_conditioner
from omnidreams._src.predict2_multiview.configs.vid2vid.defaults.dataloader import (
    register_multiview_dataloader,
)
from omnidreams._src.predict2_multiview.configs.vid2vid.defaults.dataloader_local import register_waymo_dataloader
from omnidreams._src.predict2_multiview.configs.vid2vid.defaults.model import register_model
from omnidreams._src.predict2_multiview.configs.vid2vid.defaults.net import register_net
from omnidreams._src.predict2_multiview.configs.vid2vid.defaults.optimizer import register_optimizer


def make_config():
    c = vid2vid_make_config()
    c.job.project = "cosmos_predict2_multiview"
    register_conditioner()
    register_model()
    register_net()
    register_multiview_dataloader()
    register_waymo_dataloader()
    register_callbacks()
    register_optimizer()
    import_all_modules_from_package("omnidreams._src.predict2_multiview.configs.vid2vid.experiment", reload=True)
    if not INTERNAL:
        import_all_modules_from_package("cosmos_predict2.experiments.multiview", reload=True)
    return c
