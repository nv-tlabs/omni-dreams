# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np


def detect_aspect_ratio(img_size):
    r"""
    Function for detecting the closest aspect ratio.
    """

    _aspect_ratios = np.array([(16 / 9), (4 / 3), 1, (3 / 4), (9 / 16)])
    _aspect_ratio_keys = ["16,9", "4,3", "1,1", "3,4", "9,16"]
    w, h = img_size
    current_ratio = w / h
    closest_aspect_ratio = np.argmin((_aspect_ratios - current_ratio) ** 2)
    return _aspect_ratio_keys[closest_aspect_ratio]
