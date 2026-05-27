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

"""CI smoke tests for ``samples/post-training/setup_env.sh``.

The real setup path downloads tens of GB from Hugging Face and requires the
post-training venv, so CI runs the orchestration against a synthetic source
tree with mocked download and prepare commands. This still catches regressions
in repo-root resolution, token/cache setup, checkpoint repo selection, dataset
staging invocation, VIRTUAL_ENV patching, and TE symlink creation.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_log(path: Path) -> str:
    assert path.exists(), f"Expected setup test log was not written: {path}"
    return path.read_text(encoding="utf-8")


def _copy_setup_files(source_root: Path) -> Path:
    script_dir = source_root / "samples" / "post-training"
    script_dir.mkdir(parents=True)
    for name in ("setup_env.sh", "_env.sh", "prepare.py"):
        shutil.copy2(REPO_ROOT / "samples" / "post-training" / name, script_dir / name)
    return script_dir


def _write_executable(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _seed_fake_venv(release_dir: Path) -> Path:
    venv = release_dir / ".venv"
    bin_dir = venv / "bin"
    lib_dir = (
        venv
        / "lib"
        / "python3.10"
        / "site-packages"
        / "nvidia"
        / "fakepkg"
        / "lib"
    )
    bin_dir.mkdir(parents=True)
    lib_dir.mkdir(parents=True)

    (bin_dir / "activate").write_text(
        "VIRTUAL_ENV='/old/imaginaire4/build/path'\nexport VIRTUAL_ENV\n",
        encoding="utf-8",
    )
    _write_executable(
        bin_dir / "python",
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\n' "$*" >> "$SETUP_TEST_PREPARE_LOG"
        """,
    )
    (lib_dir / "libfake_te.so.1").write_text("", encoding="utf-8")
    return venv


def test_setup_env_runs_in_source_tree_without_git_metadata(tmp_path: Path) -> None:
    source_root = tmp_path / "omni-dreams-source"
    script_dir = _copy_setup_files(source_root)
    release_dir = source_root / "post-training"
    venv = _seed_fake_venv(release_dir)

    cache_dir = tmp_path / "cache"
    (cache_dir / "huggingface").mkdir(parents=True)
    (cache_dir / "huggingface" / "token").write_text("test-token", encoding="utf-8")

    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    uvx_log = tmp_path / "uvx.log"
    prepare_log = tmp_path / "prepare.log"
    _write_executable(
        command_dir / "uvx",
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\n' "$*" >> "$SETUP_TEST_UVX_LOG"
        """,
    )

    assert not (source_root / ".git").exists()

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{command_dir}{os.pathsep}{env['PATH']}",
            "OMNI_CACHE_DIR": str(cache_dir),
            "OMNI_DREAMS_HF_ORG": "ci-org",
            "OMNI_HF_CKPT_REVISION": "ci-revision",
            "OMNI_MIN_SETUP_FREE_GB": "0",
            "OMNI_MIN_WORKTREE_FREE_GB": "0",
            "SETUP_TEST_UVX_LOG": str(uvx_log),
            "SETUP_TEST_PREPARE_LOG": str(prepare_log),
            "TMPDIR": str(tmp_path / "tmp"),
        }
    )

    result = subprocess.run(
        ["bash", str(script_dir / "setup_env.sh")],
        cwd=source_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "fatal: not a git repository" not in output
    assert (
        "Downloading single-view checkpoints from HuggingFace "
        "(ci-org/omni-dreams-models@ci-revision)"
    ) in output
    assert "Setup complete" in output

    uvx_calls = _read_log(uvx_log)
    assert "ci-org/omni-dreams-models --repo-type model --revision ci-revision" in uvx_calls
    assert "single_view/distilled/e5cadda3-8f52-43b2-b621-aa3d4c9f0588_model.pt" in uvx_calls
    assert "nvidia/Cosmos-Predict2-2B-Video2World --repo-type model --revision main" in uvx_calls
    assert (
        "nvidia/Cosmos-Reason1-7B --repo-type model "
        "--revision 3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
    ) in uvx_calls

    prepare_call = _read_log(prepare_log)
    assert str(script_dir / "prepare.py") in prepare_call
    assert f"--stage 1 --data-dir {release_dir / 'data'}" in prepare_call

    activate = (venv / "bin" / "activate").read_text(encoding="utf-8")
    assert f"VIRTUAL_ENV='{venv}'" in activate
    te_link = (
        venv
        / "lib"
        / "python3.10"
        / "site-packages"
        / "nvidia"
        / "fakepkg"
        / "lib"
        / "libfake_te.so"
    )
    assert te_link.is_symlink()
    assert os.readlink(te_link) == "libfake_te.so.1"


def test_setup_env_reports_missing_release_tree(tmp_path: Path) -> None:
    source_root = tmp_path / "broken-source"
    script_dir = _copy_setup_files(source_root)

    result = subprocess.run(
        ["bash", str(script_dir / "setup_env.sh")],
        cwd=source_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "Could not find the post-training release tree" in result.stderr
    assert "samples/post-training/ and post-training/" in result.stderr
