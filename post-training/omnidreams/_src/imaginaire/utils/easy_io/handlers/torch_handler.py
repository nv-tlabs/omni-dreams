# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

try:
    import torch
except ImportError:
    torch = None

from omnidreams._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler


class TorchHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return torch.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        torch.save(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError
