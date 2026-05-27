# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import json
import os
import re
from dataclasses import dataclass
from typing import List

import omegaconf

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.imaginaire.lazy_config import LazyDict
from omnidreams._src.imaginaire.utils import log
from omnidreams._src.imaginaire.utils.config_helper import get_config_module, override
from omnidreams._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from omnidreams._src.predict2.tokenizers.wan2pt2 import Wan2pt2VAEInterface



Wan2pt1VAEConfig: LazyDict = L(Wan2pt1VAEInterface)(name="wan2pt1_tokenizer")
Wan2pt1VAEConfig_GCP: LazyDict = L(Wan2pt1VAEInterface)(
    name="wan2pt1_tokenizer_gcp",
    s3_credential_path="credentials/gcp_training.secret",
    vae_pth="s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth",
)
Wan2pt2VAEConfig: LazyDict = L(Wan2pt2VAEInterface)(name="wan2pt2_tokenizer")
