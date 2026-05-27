# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import re
from typing import Optional


def json_decoder(key: str, data: bytes) -> Optional[dict]:
    r"""
    Function to decode a json file.
    Args:
        key: Data key.
        data: Data dict.
    """
    extension = re.sub(r".*[.]", "", key)
    if extension == "json":
        data_dict = json.loads(data)
        return data_dict
    else:
        return None
