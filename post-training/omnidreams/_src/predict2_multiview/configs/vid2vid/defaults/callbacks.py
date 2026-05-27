# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.predict2_multiview.callbacks.log_weight import LogWeight
from omnidreams._src.predict2_multiview.callbacks.sigma_loss_analysis_per_frame import SigmaLossAnalysisPerFrame

LOG_SIGMA_LOSS_CALLBACKS = dict(
    sigma_loss_log=L(SigmaLossAnalysisPerFrame)(
        save_s3="${upload_reproducible_setup}",
        logging_iter_multipler=2,
        logging_viz_iter_multipler=10,
    ),
)

LOG_WEIGHT_CALLBACKS = dict(
    log_weight=L(LogWeight)(
        every_n=100,
    ),
)


def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="log_sigma_loss",
        node=LOG_SIGMA_LOSS_CALLBACKS,
    )
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="log_weight",
        node=LOG_WEIGHT_CALLBACKS,
    )
