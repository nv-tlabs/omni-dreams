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

"""CI smoke tests for ``samples/post-training/torchrun_smoke.sh``."""

from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "samples" / "post-training" / "torchrun_smoke.sh"
WRAP = REPO_ROOT / "samples" / "post-training" / "triton_per_rank_wrap.sh"
CONFIG_ROOT = "omnidreams/_src/omnidreams/configs"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _seed_fake_release(tmp_path: Path) -> Path:
    release_dir = tmp_path / "release"
    venv = release_dir / ".venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)

    (bin_dir / "activate").write_text(
        "\n".join(
            (
                f"VIRTUAL_ENV='{venv}'",
                "export VIRTUAL_ENV",
                'PATH="$VIRTUAL_ENV/bin:$PATH"',
                "export PATH",
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_executable(
        bin_dir / "python",
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        {
          printf 'PWD=%s\\n' "$PWD"
          printf 'PYTHONPATH=%s\\n' "${PYTHONPATH:-}"
          printf 'ARGS\\n'
          printf '%s\\n' "$@"
        } > "${TORCHRUN_TEST_LOG:?}"
        """,
    )
    return release_dir


def _run_smoke(
    tmp_path: Path,
    experiment: str,
    *extra_args: str,
    env_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    release_dir = _seed_fake_release(tmp_path)
    log_path = tmp_path / "torchrun.log"

    env = os.environ.copy()
    env.update(
        {
            "OMNI_RELEASE_DIR": str(release_dir),
            "OMNI_CACHE_DIR": str(tmp_path / "cache"),
            "OMNI_MIN_TRAIN_FREE_GB": "0",
            "TMPDIR": str(tmp_path / "tmp"),
            "TORCHRUN_TEST_LOG": str(log_path),
        }
    )
    env.update(env_overrides or {})

    result = subprocess.run(
        ["bash", str(SCRIPT), experiment, *extra_args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return result, log_path, release_dir


def _read_torchrun_log(log_path: Path) -> tuple[dict[str, str], list[str]]:
    assert log_path.exists(), f"Expected fake torchrun log was not written: {log_path}"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    args_index = lines.index("ARGS")
    metadata = dict(line.split("=", 1) for line in lines[:args_index])
    return metadata, lines[args_index + 1 :]


def _prefix() -> list[str]:
    return [
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=8",
        "--master_port=12341",
        "--no-python",
        str(WRAP),
        "python",
        "-m",
        "scripts.train",
    ]


@pytest.mark.parametrize(
    ("experiment", "headline", "expected_train_args"),
    (
        (
            "1",
            "Experiment 1: Mid-training student-init",
            [
                f"--config={CONFIG_ROOT}/causal_cosmos2/config.py",
                "--",
                "experiment=causal_cosmos2_2B_single_view_chunk2_t24_hdmap_vae",
                "job.name=causal_cosmos2_2B_single_view_chunk2_t24_hdmap_vae",
                "model.config.fsdp_shard_size=8",
                "model_parallel.context_parallel_size=8",
                "dataloader_train.repeat_factor=200",
            ],
        ),
        (
            "2",
            "Experiment 2: Mid-training teacher",
            [
                f"--config={CONFIG_ROOT}/causal_cosmos2/config.py",
                "--",
                "experiment=teacher_cosmos2_2B_single_view_t24_hdmap_vae",
                "job.name=teacher_cosmos2_2B_single_view_t24_hdmap_vae",
                "model.config.fsdp_shard_size=8",
                "model_parallel.context_parallel_size=8",
                "dataloader_train.repeat_factor=200",
            ],
        ),
        (
            "3",
            "Experiment 3: Self-forcing distillation",
            [
                f"--config={CONFIG_ROOT}/self_forcing/config.py",
                "--",
                "experiment=cosmos_v2_2b_SF_res720p_fps30_i2v_hdmap_chunk2_vae_encode_loc6_release",
                "job.wandb_mode=disabled",
                "dataloader_train.repeat_factor=200",
            ],
        ),
    ),
)
def test_torchrun_smoke_builds_expected_training_command(
    tmp_path: Path,
    experiment: str,
    headline: str,
    expected_train_args: list[str],
) -> None:
    result, log_path, release_dir = _run_smoke(tmp_path, experiment)

    assert result.returncode == 0, result.stdout + result.stderr
    assert headline in result.stdout

    metadata, argv = _read_torchrun_log(log_path)
    assert metadata["PWD"] == str(release_dir)
    assert metadata["PYTHONPATH"].startswith(
        f"{release_dir}/packages/cosmos-cuda:{release_dir}/packages/cosmos-oss:{release_dir}"
    )
    assert argv == _prefix() + expected_train_args


def test_torchrun_smoke_places_user_overrides_last(tmp_path: Path) -> None:
    overrides = (
        "model.config.fsdp_shard_size=4",
        "model_parallel.context_parallel_size=4",
        "custom.flag=true",
    )

    result, log_path, _release_dir = _run_smoke(tmp_path, "1", *overrides)

    assert result.returncode == 0, result.stdout + result.stderr
    _metadata, argv = _read_torchrun_log(log_path)
    assert "model.config.fsdp_shard_size=8" in argv
    assert "model_parallel.context_parallel_size=8" in argv
    assert argv[-len(overrides) :] == list(overrides)


def test_torchrun_smoke_rejects_unsupported_nproc(tmp_path: Path) -> None:
    result, log_path, _release_dir = _run_smoke(
        tmp_path,
        "1",
        env_overrides={"NPROC": "4"},
    )

    assert result.returncode == 2
    assert "NPROC=4 is not supported by the post-training configs" in result.stderr
    assert "supported minimum post-training launch uses NPROC=8" in result.stderr
    assert not log_path.exists()


def test_torchrun_smoke_rejects_invalid_experiment(tmp_path: Path) -> None:
    result, log_path, _release_dir = _run_smoke(tmp_path, "4")

    assert result.returncode == 2
    assert "EXPERIMENT must be 1, 2, or 3 (got: 4)" in result.stderr
    assert not log_path.exists()
