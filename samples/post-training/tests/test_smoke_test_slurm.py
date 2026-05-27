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

"""CI smoke tests for ``samples/post-training/smoke_test.slurm``."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "samples" / "post-training" / "smoke_test.slurm"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _seed_fake_repo(tmp_path: Path) -> Path:
    fake_repo = tmp_path / "repo"
    script_dir = fake_repo / "samples" / "post-training"
    script_dir.mkdir(parents=True)
    shutil.copy2(SCRIPT, script_dir / "smoke_test.slurm")
    _write_executable(
        script_dir / "torchrun_smoke.sh",
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        {
          printf 'PWD=%s\\n' "$PWD"
          printf 'TRITON_CACHE_BASE=%s\\n' "${TRITON_CACHE_BASE:-}"
          printf 'ARGS\\n'
          printf '%s\\n' "$@"
        } > "${SLURM_TEST_LOG:?}"
        """,
    )
    return fake_repo


def _read_slurm_log(log_path: Path) -> tuple[dict[str, str], list[str]]:
    assert log_path.exists(), f"Expected fake Slurm wrapper log was not written: {log_path}"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    args_index = lines.index("ARGS")
    metadata = dict(line.split("=", 1) for line in lines[:args_index])
    return metadata, lines[args_index + 1 :]


def _run_slurm_wrapper(
    tmp_path: Path,
    *,
    experiment: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    fake_repo = _seed_fake_repo(tmp_path)
    log_path = tmp_path / "slurm.log"

    env = os.environ.copy()
    env.update(
        {
            "SLURM_SUBMIT_DIR": str(fake_repo),
            "SLURM_JOB_ID": "12345",
            "SLURM_TEST_LOG": str(log_path),
        }
    )
    if experiment is None:
        env.pop("EXPERIMENT", None)
    else:
        env["EXPERIMENT"] = experiment

    result = subprocess.run(
        ["bash", str(fake_repo / "samples" / "post-training" / "smoke_test.slurm")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return result, log_path, fake_repo


def test_smoke_test_slurm_sets_job_triton_cache_and_forwards_experiment(
    tmp_path: Path,
) -> None:
    result, log_path, fake_repo = _run_slurm_wrapper(tmp_path, experiment="3")

    assert result.returncode == 0, result.stdout + result.stderr

    metadata, argv = _read_slurm_log(log_path)
    assert metadata["PWD"] == str(tmp_path)
    assert metadata["TRITON_CACHE_BASE"] == "/tmp/triton_12345"
    assert argv == ["3"]
    assert fake_repo.is_dir()


def test_smoke_test_slurm_defaults_to_experiment_1(tmp_path: Path) -> None:
    result, log_path, _fake_repo = _run_slurm_wrapper(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr

    _metadata, argv = _read_slurm_log(log_path)
    assert argv == ["1"]


def test_smoke_test_slurm_requires_submit_dir(tmp_path: Path) -> None:
    script_path = tmp_path / "smoke_test.slurm"
    shutil.copy2(SCRIPT, script_path)

    env = os.environ.copy()
    env.pop("SLURM_SUBMIT_DIR", None)
    env["SLURM_JOB_ID"] = "12345"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "SLURM_SUBMIT_DIR: run sbatch from the repo root" in result.stderr
