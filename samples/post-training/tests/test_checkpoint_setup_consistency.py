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

"""Consistency checks between setup downloads and the checkpoint registry."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_ENV = REPO_ROOT / "samples" / "post-training" / "setup_env.sh"
CHECKPOINTS = REPO_ROOT / "post-training" / "omnidreams" / "checkpoints_omnidreams.py"


def _eval_string_expr(node: ast.AST, values: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise AssertionError(f"Unknown name in string expression: {node.id}")
        return values[node.id]
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
            elif isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name):
                if part.value.id not in values:
                    raise AssertionError(f"Unknown name in f-string: {part.value.id}")
                parts.append(values[part.value.id])
            else:
                raise AssertionError(f"Unsupported f-string part: {ast.dump(part)}")
        return "".join(parts)
    raise AssertionError(f"Unsupported string expression: {ast.dump(node)}")


def _checkpoint_registry_filenames() -> set[str]:
    tree = ast.parse(CHECKPOINTS.read_text(encoding="utf-8"))
    values: dict[str, str] = {}

    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            continue

        name = statement.targets[0].id
        if name == "HF_CACHE_FILES":
            assert isinstance(statement.value, ast.Dict)
            return {_eval_string_expr(value, values) for value in statement.value.values}

        try:
            values[name] = _eval_string_expr(statement.value, values)
        except AssertionError:
            continue

    raise AssertionError("HF_CACHE_FILES was not found in checkpoints_omnidreams.py")


def _setup_checkpoint_filenames() -> set[str]:
    lines = SETUP_ENV.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "for f in \\":
            filenames: list[str] = []
            for download_line in lines[index + 1 :]:
                stripped = download_line.strip()
                if stripped == "do":
                    return set(filenames)
                filenames.append(stripped.removesuffix("\\").strip())
    raise AssertionError("Could not find setup_env.sh single-view checkpoint download loop")


def test_setup_env_downloads_match_registered_checkpoint_filenames() -> None:
    registry_filenames = _checkpoint_registry_filenames()
    setup_filenames = _setup_checkpoint_filenames()

    assert registry_filenames == setup_filenames
    assert len(registry_filenames) == 3
    assert all(filename.startswith("single_view/") for filename in registry_filenames)
    assert all(filename.endswith("_model.pt") for filename in registry_filenames)
