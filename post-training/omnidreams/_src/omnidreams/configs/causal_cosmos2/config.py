# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
from typing import Any, List

import attrs

from omnidreams._src.imaginaire import config
from omnidreams._src.imaginaire.trainer import ImaginaireTrainer as Trainer
from omnidreams._src.imaginaire.utils.config_helper import import_all_modules_from_package

# modules from predict2
from omnidreams._src.predict2.configs.common.defaults.checkpoint import register_checkpoint
from omnidreams._src.predict2.configs.common.defaults.ckpt_type import register_ckpt_type
from omnidreams._src.predict2.configs.common.defaults.optimizer import register_optimizer
from omnidreams._src.predict2.configs.common.defaults.scheduler import register_scheduler

# modules from predict2
from omnidreams._src.predict2.configs.common.defaults.tokenizer import register_tokenizer
from omnidreams._src.predict2.configs.text2world.defaults.callbacks import register_callbacks
from omnidreams._src.omnidreams.configs.causal_cosmos2.defaults.conditioner import register_conditioner

from omnidreams._src.omnidreams.configs.causal_cosmos2.defaults.dataloader_local import (
    register_local_multiview_dataloader,
)
from omnidreams._src.omnidreams.configs.causal_cosmos2.defaults.model import register_model
from omnidreams._src.omnidreams.configs.causal_cosmos2.defaults.net import register_net
from omnidreams._src.omnidreams.configs.causal_cosmos2.defaults.tokenizer import (
    register_tokenizer as register_tokenizer_hf,
)



@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock"},
            {"data_val": "mock"},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"model": "fsdp"},
            {"callbacks": "basic"},
            {"net": None},
            {"conditioner": "add_fps_padding_mask"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"checkpoint": "s3"},
            {"ckpt_type": "dcp"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "cosmos_v2_causal_av"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    # Call this function to register config groups for advanced overriding. the order follows the default config groups
    register_local_multiview_dataloader()
    register_optimizer()
    register_scheduler()
    register_model()
    register_callbacks()
    register_net()
    register_conditioner()
    register_tokenizer()
    register_tokenizer_hf()
    register_checkpoint()
    register_ckpt_type()

    import_all_modules_from_package(
        "omnidreams._src.omnidreams.configs.causal_cosmos2.experiment", reload=True
    )
    try:
        if importlib.util.find_spec("omnidreams.experiments") is not None:
            import_all_modules_from_package("omnidreams.experiments", reload=True)
    except ModuleNotFoundError:
        pass
    return c
