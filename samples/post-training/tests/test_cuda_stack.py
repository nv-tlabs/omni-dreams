# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only checks that the post-training venv uses one CUDA stack.

Run this with the selected venv's pytest, for example:

    post-training/.venv/bin/pytest samples/post-training/tests/test_cuda_stack.py -q

The tests intentionally do not invoke ``uv run`` because ``uv run`` may sync
the workspace before executing the command.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VENV = REPO_ROOT / "post-training" / ".venv"
VENV_PYTHON = VENV / "bin" / "python"
CUDA_TAG_BY_EXTRA = {
    "cu128": "+cu128",
    "cu130": "+cu130",
}
CUDA_MAJOR_BY_EXTRA = {
    "cu128": "12",
    "cu130": "13",
}


@dataclass(frozen=True)
class CudaStack:
    torch_version: str
    transformer_engine_version: str

    @property
    def extra(self) -> str:
        for extra, tag in CUDA_TAG_BY_EXTRA.items():
            if tag in self.torch_version:
                return extra
        raise AssertionError(f"torch has no expected CUDA tag: {self.torch_version}")

    @property
    def libcudart_major(self) -> str:
        return CUDA_MAJOR_BY_EXTRA[self.extra]


@dataclass(frozen=True)
class LibcudartInventory:
    majors: set[str]
    paths: list[Path]

    @property
    def formatted_paths(self) -> str:
        return "\n".join(f"  - {path}" for path in self.paths)


def _require_venv() -> None:
    if not VENV_PYTHON.exists():
        pytest.skip("post-training venv missing; run uv sync with a CUDA extra first")


def _venv_python(*statements: str) -> list[str]:
    _require_venv()
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", "\n".join(statements)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()


def _cuda_stack() -> CudaStack:
    torch_version, transformer_engine_version = _venv_python(
        "import importlib.metadata as metadata",
        "import torch",
        "print(torch.__version__)",
        "print(metadata.version('transformer-engine'))",
    )
    return CudaStack(torch_version, transformer_engine_version)


def _nvidia_lib_root() -> Path:
    purelib = _venv_python("import sysconfig", "print(sysconfig.get_path('purelib'))")[0]
    return Path(purelib) / "nvidia"


def _libcudart_inventory() -> LibcudartInventory:
    nvidia_lib_root = _nvidia_lib_root()
    if not nvidia_lib_root.is_dir():
        pytest.skip(f"venv nvidia library directory is missing: {nvidia_lib_root}")

    majors: set[str] = set()
    paths = sorted(nvidia_lib_root.rglob("libcudart.so.*"))
    for path in paths:
        match = re.search(r"libcudart\.so\.(\d+)", path.name)
        if match:
            majors.add(match.group(1))
    return LibcudartInventory(majors=majors, paths=paths)


def test_single_libcudart_major_in_venv() -> None:
    """Catch mixed cu12/cu13 venvs before Transformer Engine fails at runtime."""
    inventory = _libcudart_inventory()
    assert inventory.majors, "no libcudart.so.* files found in venv nvidia package directory"
    assert len(inventory.majors) == 1, (
        f"multiple CUDA runtime majors found in venv: {sorted(inventory.majors)}\n"
        f"libcudart paths:\n{inventory.formatted_paths}"
    )


def test_torch_and_transformer_engine_cuda_tags_align() -> None:
    stack = _cuda_stack()
    assert stack.extra in stack.transformer_engine_version, (
        f"transformer-engine {stack.transformer_engine_version!r} does not match "
        f"torch {stack.torch_version!r}"
    )


def test_libcudart_matches_torch_cuda_tag() -> None:
    """Infer the expected CUDA runtime from the installed torch build."""
    stack = _cuda_stack()
    inventory = _libcudart_inventory()
    assert inventory.majors == {stack.libcudart_major}, (
        f"torch {stack.torch_version!r} expects {stack.extra}, but libcudart majors "
        f"are {sorted(inventory.majors)}\nlibcudart paths:\n{inventory.formatted_paths}"
    )
