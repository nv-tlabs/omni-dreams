# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
from typing import Any, List

import attrs

from omnidreams._src.imaginaire import config
from omnidreams._src.imaginaire.utils.config_helper import import_all_modules_from_package
from omnidreams._src.predict2.configs.common.defaults.checkpoint import register_checkpoint
from omnidreams._src.predict2.configs.common.defaults.ckpt_type import register_ckpt_type
from omnidreams._src.predict2.configs.common.defaults.optimizer import register_optimizer
from omnidreams._src.predict2.configs.common.defaults.scheduler import register_scheduler
from omnidreams._src.predict2.configs.common.defaults.tokenizer import register_tokenizer
from omnidreams._src.predict2.configs.text2world.defaults.callbacks import register_callbacks
from omnidreams._src.omnidreams.configs.causal_cosmos2.defaults.conditioner import register_conditioner

from omnidreams._src.omnidreams.configs.causal_cosmos2.defaults.dataloader_local import (
    register_local_multiview_dataloader,
)
from omnidreams._src.omnidreams.configs.causal_cosmos2.defaults.tokenizer import (
    register_tokenizer as register_tokenizer_hf,
)
from omnidreams._src.omnidreams.configs.defaults.callbacks import (
    register_callbacks as register_callbacks_causal,
)
from omnidreams._src.omnidreams.configs.defaults.ckpt_type import (
    register_ckpt_type as register_ckpt_type_distill,
)
from omnidreams._src.omnidreams.configs.self_forcing.defaults.model import register_model
from omnidreams._src.omnidreams.configs.self_forcing.defaults.net import register_net
from omnidreams._src.omnidreams.trainer.trainer_distillation import (
    ImaginaireTrainer as DistillationTrainer,
)


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "diffusion_renderer_video"},
            {"data_val": "mock"},
            {"optimizer": "adamw"},
            {"scheduler": "lambdalinear"},
            {"model": "fsdp"},
            {"callbacks": "basic"},
            {"net": "causal_wan2pt1_1pt3B"},
            {"net_real_score": "wan2pt1_14B"},
            {"net_fake_score": "wan2pt1_1pt3B"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"conditioner": "add_text_only_mask"},
            {"checkpoint": "s3"},
            # Distillation trainer hands optimizer_dict / scheduler_dict; the
            # per-key ckpt_type writes/reads each optimizer's shard separately.
            {"ckpt_type": "dcp_distill"},
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
    c.job.project = "cosmos_diffusion_v2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = DistillationTrainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 1_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    register_optimizer()
    register_scheduler()
    register_local_multiview_dataloader()
    register_conditioner()
    # register_i2v_conditioner()
    register_net()
    register_checkpoint()
    register_tokenizer()
    register_tokenizer_hf()
    register_callbacks_causal()
    register_callbacks()  # registers basic / wandb / cluster_speed / viz_online_sampling / long

    register_ckpt_type()
    register_ckpt_type_distill()
    register_model()

    import_all_modules_from_package("omnidreams._src.omnidreams.configs.self_forcing.experiment", reload=True)
    try:
        if importlib.util.find_spec("omnidreams.experiments") is not None:
            import_all_modules_from_package("omnidreams.experiments", reload=True)
    except ModuleNotFoundError:
        pass
    return c
