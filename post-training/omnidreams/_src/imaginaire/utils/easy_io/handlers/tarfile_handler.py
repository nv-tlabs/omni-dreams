# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import tarfile

from omnidreams._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler


class TarHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, mode="r|*", **kwargs):
        return tarfile.open(fileobj=file, mode=mode, **kwargs)

    def load_from_path(self, filepath, mode="r|*", **kwargs):
        return tarfile.open(filepath, mode=mode, **kwargs)

    def dump_to_fileobj(self, obj, file, mode="w", **kwargs):
        with tarfile.open(fileobj=file, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with tarfile.open(filepath, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError
