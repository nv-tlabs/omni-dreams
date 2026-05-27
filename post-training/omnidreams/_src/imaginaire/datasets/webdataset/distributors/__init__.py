# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams._src.imaginaire.datasets.webdataset.distributors.basic import ShardlistBasic
from omnidreams._src.imaginaire.datasets.webdataset.distributors.multi_aspect_ratio import ShardlistMultiAspectRatio
from omnidreams._src.imaginaire.datasets.webdataset.distributors.multi_aspect_ratio_v2 import ShardlistMultiAspectRatioInfinite
from omnidreams._src.imaginaire.datasets.webdataset.distributors.weighted_multi_aspect_ratio import WeightedShardlistMultiAspectRatio

distributors_list = {
    "basic": ShardlistBasic,
    "multi_aspect_ratio": ShardlistMultiAspectRatio,
    "multi_aspect_ratio_infinite": ShardlistMultiAspectRatioInfinite,
    "weighted_multi_aspect_ratio": WeightedShardlistMultiAspectRatio,
}
