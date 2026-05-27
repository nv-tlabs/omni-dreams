# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pickle
import re
from typing import Optional


def pkl_decoder(key: str, data: bytes) -> Optional[dict]:
    r"""
    Function to decode a pkl file.
    Args:
        key: Data key.
        data: Data dict.
    """
    extension = re.sub(r".*[.]", "", key)
    if extension == "pkl" or extension == "pickle":
        data_dict = pickle.loads(data)
        return data_dict
    else:
        return None
