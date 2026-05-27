# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pandas as pd

from omnidreams._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler  # isort:skip


class PandasHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return pd.read_csv(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        obj.to_csv(file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError("PandasHandler does not support dumping to str")
