# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import PLACEHOLDER, LazyDict
from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.predict2.utils.optim_instantiate import get_base_optimizer
from omnidreams._src.predict2_multiview.utils.optim_instantiate import get_multiple_optimizer

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

MultipleAdamWConfig = L(get_multiple_optimizer)(
    model=PLACEHOLDER,
    lr=1e-4,
    weight_decay=1e-3,
    betas=[0.9, 0.999],
    optim_type="adamw",
    eps=1e-8,
    fused=True,
    lr_overrides=[],  # New format: list of dicts with 'pattern', 'lr', and optional 'match_type'
)


MultipleFusedAdamWConfig = L(get_multiple_optimizer)(
    model=PLACEHOLDER,
    lr=1e-4,
    weight_decay=1e-3,
    betas=[0.9, 0.999],
    optim_type="fusedadam",
    eps=1e-8,
    lr_overrides=[],  # New format: list of dicts with 'pattern', 'lr', and optional 'match_type'
)


def register_optimizer():
    cs = ConfigStore.instance()
    cs.store(group="optimizer", package="optimizer", name="fusedadamw", node=FusedAdamWConfig)
    cs.store(group="optimizer", package="optimizer", name="adamw", node=AdamWConfig)
    cs.store(group="optimizer", package="optimizer", name="multipleadamw", node=MultipleAdamWConfig)
    cs.store(group="optimizer", package="optimizer", name="multiplefusedadamw", node=MultipleFusedAdamWConfig)
