# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Causal-multiview-specific checkpoint types.

Adds the ``dcp_distill`` ckpt_type used by the self-forcing distillation
trainer (see ``projects/cosmos/sil/causal_multiview/checkpointer/dcp_distill.py``).
"""

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.omnidreams.checkpointer.dcp_distill import (
    DistributedCheckpointer as DistillCheckpointer,
)

DCP_DISTILL: dict[str, str] = L(DistillCheckpointer)()


def register_ckpt_type():
    cs = ConfigStore.instance()
    cs.store(group="ckpt_type", package="checkpoint.type", name="dcp_distill", node=DCP_DISTILL)
