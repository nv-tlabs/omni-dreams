# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams._src.imaginaire.functional.lr_scheduler import LambdaLinearScheduler
from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.imaginaire.lazy_config import LazyDict

LambdaLinearSchedulerConfig: LazyDict = L(LambdaLinearScheduler)(
    warm_up_steps=[1000],
    cycle_lengths=[10000000000000],
    f_start=[1.0e-6],
    f_max=[1.0],
    f_min=[1.0],
)
