# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from omnidreams._src.imaginaire.lazy_config.registry import convert_target_to_string


def function(): ...


class Base:
    @classmethod
    def class_method(cls): ...


class Derived(Base):
    @staticmethod
    def static_method(): ...

    def instance_method(self): ...


@pytest.mark.L0
def test_convert_target_to_string():
    assert convert_target_to_string(int) == "builtins.int"
    assert convert_target_to_string(print) == "builtins.print"
    assert convert_target_to_string(Derived().instance_method) == f"{__name__}.{Derived.__qualname__}.instance_method"
    assert convert_target_to_string(Derived.static_method) == f"{__name__}.{Derived.__qualname__}.static_method"
    assert convert_target_to_string(Derived.class_method) == f"{__name__}.{Derived.__qualname__}.class_method"
    assert convert_target_to_string(function) == f"{__name__}.{function.__qualname__}"
