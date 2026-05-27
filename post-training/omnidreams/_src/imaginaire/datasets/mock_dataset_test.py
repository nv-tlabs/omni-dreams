# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Usage:
    pytest -s omnidreams/_src/imaginaire/datasets/mock_dataset_test.py
"""

import pytest
import torch

from omnidreams._src.imaginaire.datasets.mock_dataset import CombinedDictDataset, LambdaDataset, RepeatDataset
from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.imaginaire.lazy_config import instantiate


@pytest.fixture
def cfg():
    return L(CombinedDictDataset)(
        key1=L(LambdaDataset)(
            length=64,
            fn=lambda: torch.randn(3, 32, 32),
        ),
        key2=L(RepeatDataset)(
            dataset=L(LambdaDataset)(
                fn=lambda: torch.randn(3, 32, 32),
            ),
        ),
    )


@pytest.mark.L0
def test_mock_dataset(cfg):
    batch_size = 4
    dataset_obj = instantiate(cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset_obj,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=1,
    )
    assert len(dataset_obj) == 64
    for ith, batch in enumerate(dataloader):
        assert batch["key1"].shape == (batch_size, 3, 32, 32)
        assert batch["key2"].shape == (batch_size, 3, 32, 32)
        if ith > 2:
            break
