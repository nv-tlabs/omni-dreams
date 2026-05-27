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

"""CI smoke tests for ``samples/post-training/triton_per_rank_wrap.sh``."""

from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "samples" / "post-training" / "triton_per_rank_wrap.sh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _seed_fake_command(tmp_path: Path) -> Path:
    command = tmp_path / "record_env.sh"
    _write_executable(
        command,
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        {
          printf 'PWD=%s\\n' "$PWD"
          printf 'TRITON_CACHE_BASE=%s\\n' "${TRITON_CACHE_BASE:-}"
          printf 'TRITON_CACHE_DIR=%s\\n' "${TRITON_CACHE_DIR:-}"
          printf 'ARGS\\n'
          printf '%s\\n' "$@"
        } > "${TRITON_WRAP_TEST_LOG:?}"
        """,
    )
    return command


def _read_wrap_log(log_path: Path) -> tuple[dict[str, str], list[str]]:
    assert log_path.exists(), f"Expected Triton wrapper log was not written: {log_path}"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    args_index = lines.index("ARGS")
    metadata = dict(line.split("=", 1) for line in lines[:args_index])
    return metadata, lines[args_index + 1 :]


def _run_wrapper(
    tmp_path: Path,
    *,
    local_rank: str | None,
    extra_args: tuple[str, ...] = ("alpha", "beta"),
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    command = _seed_fake_command(tmp_path)
    log_path = tmp_path / "triton-wrap.log"
    cache_base = tmp_path / "triton-cache"

    env = os.environ.copy()
    env.update(
        {
            "TRITON_CACHE_BASE": str(cache_base),
            "TRITON_WRAP_TEST_LOG": str(log_path),
        }
    )
    if local_rank is None:
        env.pop("LOCAL_RANK", None)
    else:
        env["LOCAL_RANK"] = local_rank

    result = subprocess.run(
        ["bash", str(SCRIPT), str(command), *extra_args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return result, log_path, cache_base


def test_triton_per_rank_wrap_exports_rank_specific_cache_and_execs_command(
    tmp_path: Path,
) -> None:
    result, log_path, cache_base = _run_wrapper(tmp_path, local_rank="6")

    assert result.returncode == 0, result.stdout + result.stderr

    metadata, argv = _read_wrap_log(log_path)
    expected_cache_dir = Path(f"{cache_base}_6")
    assert metadata["PWD"] == str(tmp_path)
    assert metadata["TRITON_CACHE_BASE"] == str(cache_base)
    assert metadata["TRITON_CACHE_DIR"] == str(expected_cache_dir)
    assert expected_cache_dir.is_dir()
    assert argv == ["alpha", "beta"]


def test_triton_per_rank_wrap_requires_local_rank(tmp_path: Path) -> None:
    result, log_path, cache_base = _run_wrapper(tmp_path, local_rank=None)

    assert result.returncode == 2
    assert "LOCAL_RANK is required; run this wrapper via torchrun" in result.stderr
    assert not Path(f"{cache_base}_0").exists()
    assert not log_path.exists()
