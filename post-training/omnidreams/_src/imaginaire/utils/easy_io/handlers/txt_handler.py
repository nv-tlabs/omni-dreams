# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler


class TxtHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        del kwargs
        return file.read()

    def dump_to_fileobj(self, obj, file, **kwargs):
        del kwargs
        if not isinstance(obj, str):
            obj = str(obj)
        file.write(obj)

    def dump_to_str(self, obj, **kwargs):
        del kwargs
        if not isinstance(obj, str):
            obj = str(obj)
        return obj
