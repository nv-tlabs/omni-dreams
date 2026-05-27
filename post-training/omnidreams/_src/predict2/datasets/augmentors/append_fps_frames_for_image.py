# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

from omnidreams._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor


class AppendFPSFramesForImage(Augmentor):
    def __init__(
        self, input_keys: Optional[list] = None, output_keys: Optional[list] = None, args: Optional[dict] = None
    ) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Remove the input keys from the data dict.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with keys removed.
        """
        data_dict["fps"] = 30.0  # set image model fps = 30, which is the most common fps we used to train video.
        data_dict["num_frames"] = 1
        return data_dict
