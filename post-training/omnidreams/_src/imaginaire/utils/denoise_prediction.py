# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DenoisePrediction:
    x0: torch.Tensor  # clean data prediction
    eps: Optional[torch.Tensor] = None  # noise prediction
    logvar: Optional[torch.Tensor] = None  # log variance of noise prediction, can be used a confidence / uncertainty
