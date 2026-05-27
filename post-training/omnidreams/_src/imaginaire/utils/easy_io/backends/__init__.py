# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams._src.imaginaire.flags import TRAINING
from omnidreams._src.imaginaire.utils.easy_io.backends.base_backend import BaseStorageBackend
from omnidreams._src.imaginaire.utils.easy_io.backends.http_backend import HTTPBackend
from omnidreams._src.imaginaire.utils.easy_io.backends.local_backend import LocalBackend
from omnidreams._src.imaginaire.utils.easy_io.backends.registry_utils import backends, prefix_to_backends, register_backend

__all__ = [
    "BaseStorageBackend",
    "LocalBackend",
    "HTTPBackend",
    "register_backend",
    "backends",
    "prefix_to_backends",
]

if TRAINING:
    from omnidreams._src.imaginaire.utils.easy_io.backends.boto3_backend import Boto3Backend
    from omnidreams._src.imaginaire.utils.easy_io.backends.msc_backend import MSCBackend

    __all__ += [
        "Boto3Backend",
        "MSCBackend",
    ]
