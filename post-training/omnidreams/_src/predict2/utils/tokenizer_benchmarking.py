# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass


@dataclass
class BenchmarkTimes:
    """
    Class used to store times computed during tokenizer benchmarking.
    All times are in seconds.
    """

    model_invocation: float = 0.0
    # Model's invocation time + overhead
    total: float = 0.0

    @property
    def overhead(self) -> float:
        return self.total - self.model_invocation

    def __repr__(self) -> str:
        return f"BenchmarkTimes(model_invocation={self.model_invocation}, overhead={self.overhead}, total={self.total})"
