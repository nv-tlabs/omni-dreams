# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from omnidreams._src.imaginaire.utils import misc
from omnidreams._src.predict2.inference.get_umt5_emb import UMT5EncoderModel


@pytest.mark.L2
def test_encoder():
    with misc.timer("load model"):
        model = UMT5EncoderModel(
            checkpoint_path="s3://bucket/cosmos_diffusion_v2/pretrain_weights/models_t5_umt5-xxl-enc-bf16.pth"
        )
    emb = model(texts=["hello world", "hello", "world"])
    assert len(emb) == 3
    assert emb[0].shape == (512, 4096)
    assert emb[1].shape == (512, 4096)
    assert emb[2].shape == (512, 4096)
