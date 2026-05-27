# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
import re
from typing import Optional

from PIL import Image

Image.MAX_IMAGE_PIXELS = 933120000
_IMG_EXTENSIONS = "jpg jpeg png ppm pgm pbm pnm webp".split()


def pil_loader(key: str, data: bytes) -> Optional[Image.Image]:
    r"""
    Function to load an image.
    If the image is corrupt, it returns a black image.
    Args:
        key (str): Image key.
        data (bytes): Image data stream.
    Returns:
        PIL image
    """
    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _IMG_EXTENSIONS:
        return None

    with io.BytesIO(data) as stream:
        img = Image.open(stream)
        img.load()
        img = img.convert("RGB")

    return img
