# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.predict2.callbacks.every_n_draw_sample import EveryNDrawSample
from omnidreams._src.predict2.configs.common.defaults.callbacks import (
    BASIC_CALLBACKS,
    SPEED_CALLBACKS,
    WANDB_CALLBACK,
)

_basic_callback = copy.deepcopy(BASIC_CALLBACKS)

DEBUG_CALLBACKS = dict()
LONG_RUNNING_CALLBACKS = dict()

VIZ_ONLINE_SAMPLING_CALLBACKS = dict(
    every_n_sample_reg=L(EveryNDrawSample)(
        every_n=5000,
        save_s3=True,
    ),
    every_n_sample_ema=L(EveryNDrawSample)(
        every_n=5000,
        is_ema=True,
        save_s3=True,
    ),
)


def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=_basic_callback)
    cs.store(group="callbacks", package="trainer.callbacks", name="wandb", node=WANDB_CALLBACK)
    cs.store(group="callbacks", package="trainer.callbacks", name="debug", node=DEBUG_CALLBACKS)
    cs.store(
        group="callbacks", package="trainer.callbacks", name="viz_online_sampling", node=VIZ_ONLINE_SAMPLING_CALLBACKS
    )

    cs.store(group="callbacks", package="trainer.callbacks", name="long", node=LONG_RUNNING_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="cluster_speed", node=SPEED_CALLBACKS)
