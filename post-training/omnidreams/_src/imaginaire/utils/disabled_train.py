# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Any


def disabled_train(self: Any, mode: bool = True) -> Any:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
