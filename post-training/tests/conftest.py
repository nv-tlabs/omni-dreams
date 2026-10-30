# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU unit tests of the post-training code.

Importing the model modules runs two CUDA checks at import time, and both fail in the main pytest process, where
.pytest.ini sets CUDA_VISIBLE_DEVICES="":
- omnidreams/_src/predict2/inference/get_umt5_emb.py: UMT5EncoderModel.__init__ has the default argument
  `device=torch.cuda.current_device()`, evaluated when the class is defined;
- omnidreams/_src/reason1/networks/qwen2_5_vl.py: `assert is_flash_attn_2_available()` at module level.
The tests in this directory use only the CPU, so without a visible GPU both checks are made to pass before any test
module is imported. No CUDA context is created.
"""

import torch
import transformers.utils

if not torch.cuda.is_available():
    torch.cuda.current_device = lambda: 0
    transformers.utils.is_flash_attn_2_available = lambda: True
