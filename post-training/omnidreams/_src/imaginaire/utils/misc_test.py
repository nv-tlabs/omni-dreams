# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Usage:
    pytest -v -s omnidreams/_src/imaginaire/utils/misc_test.py
"""

import pytest
import torch

from omnidreams._src.imaginaire.utils.misc import get_data_batch_size


@pytest.mark.L0
def test_get_data_batch_size():
    """
    Test get_data_batch_size function.

    This test verifies that the function returns the correct data batch size for various inputs.
    """
    data_batch = {"images": torch.zeros(1, 3, 16, 16), "tokens": torch.zeros(1, 16)}
    assert get_data_batch_size(data_batch) == 1

    data_batch = {"images": torch.zeros(2, 3, 16, 16), "tokens": torch.zeros(2, 16)}
    assert get_data_batch_size(data_batch) == 2

    # Nested dictionary "__url__" without torch tensors - should be skipped
    data_batch = {
        "__key__": "value",
        "__url__": {"k1": "v1", "k2": "v2"},
        "images": torch.zeros(2, 3, 16, 16),
        "tokens": torch.zeros(2, 16),
    }

    assert get_data_batch_size(data_batch) == 2

    # Nested dictionary "image_dict" with torch tensors - should be counted
    data_batch = {
        "__key__": "value",
        "__url__": {"k1": "v1", "k2": "v2"},
        "image_dict": {"images": torch.zeros(2, 3, 16, 16)},
    }
    assert get_data_batch_size(data_batch) == 2

    # Invalid data_batch that should raise ValueError
    invalid_data_batch = {"__key__": "value", "__url__": {"k1": "v1", "k2": "v2"}, "tokens": [0, 1, 2]}
    with pytest.raises(ValueError):
        get_data_batch_size(invalid_data_batch)
