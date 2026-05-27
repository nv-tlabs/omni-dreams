# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import PLACEHOLDER, LazyDict
from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.predict2.utils.optim_instantiate import get_base_optimizer

AdamWConfig = L(get_base_optimizer)(
    model=PLACEHOLDER,
    lr=1e-4,
    weight_decay=0.1,
    betas=[0.9, 0.99],
    optim_type="adamw",
    eps=1e-8,
    fused=True,
)

FusedAdamWConfig: LazyDict = L(get_base_optimizer)(
    model=PLACEHOLDER,
    lr=1e-4,
    weight_decay=0.1,
    betas=[0.9, 0.99],
    optim_type="fusedadam",
    eps=1e-8,
    master_weights=True,
    capturable=True,
)


def register_optimizer():
    cs = ConfigStore.instance()
    cs.store(group="optimizer", package="optimizer", name="fusedadamw", node=FusedAdamWConfig)
    cs.store(group="optimizer", package="optimizer", name="adamw", node=AdamWConfig)
