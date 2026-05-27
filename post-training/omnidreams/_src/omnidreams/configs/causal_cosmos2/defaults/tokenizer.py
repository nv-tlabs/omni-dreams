# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.imaginaire.lazy_config import LazyDict
from omnidreams._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

Wan2pt1VAEConfig_HF: LazyDict = L(Wan2pt1VAEInterface)(
    name="wan2pt1_tokenizer_hf",
    vae_pth="hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
)


def register_tokenizer():
    cs = ConfigStore.instance()

    # Wan2pt1 tokenizer from HF
    cs.store(group="tokenizer", package="model.config.tokenizer", name="wan2pt1_tokenizer_hf", node=Wan2pt1VAEConfig_HF)
