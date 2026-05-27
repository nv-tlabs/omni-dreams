# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DenoisePrediction:
    x0: Optional[torch.Tensor] = None  # clean data prediction
    F: Optional[torch.Tensor] = None  # F prediction in TrigFlow
    velocity: Optional[torch.Tensor] = None  # velocity prediction if using RF
    intermediate_features: Optional[list[torch.Tensor]] = None
