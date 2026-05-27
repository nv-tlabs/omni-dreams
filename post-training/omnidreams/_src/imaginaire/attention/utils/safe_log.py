# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Safe logging utilities: logging should be disabled when in a torch.compiled
region.
"""

from omnidreams._src.imaginaire.attention.utils.environment import is_torch_compiling
from omnidreams._src.imaginaire.utils import log


def trace(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.trace(message=message, rank0_only=rank0_only)


def debug(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.debug(message=message, rank0_only=rank0_only)


def info(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.info(message=message, rank0_only=rank0_only)


def success(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.success(message=message, rank0_only=rank0_only)


def warning(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.warning(message=message, rank0_only=rank0_only)


def error(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.critical(message=message, rank0_only=rank0_only)


def critical(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.critical(message=message, rank0_only=rank0_only)


def exception(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.exception(message=message, rank0_only=rank0_only)
