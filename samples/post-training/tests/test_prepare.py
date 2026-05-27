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

"""Unit tests for the post-training dataset staging helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PREPARE_PATH = Path(__file__).resolve().parents[1] / "prepare.py"
_SPEC = importlib.util.spec_from_file_location("post_training_prepare", _PREPARE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
prepare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare)


def _write_scene(
    staging_root: Path,
    uuid: str,
    cameras: tuple[str, ...],
    *,
    prompt: bool = True,
) -> None:
    scene_dir = staging_root / uuid
    scene_dir.mkdir(parents=True)
    if prompt:
        (scene_dir / f"{uuid}.prompt.txt").write_text(f"prompt for {uuid}", encoding="utf-8")

    for camera in cameras:
        (scene_dir / f"{uuid}.{camera}_rgb.mp4").write_text(f"rgb {camera}", encoding="utf-8")
        (scene_dir / f"{uuid}.{camera}_hdmap.mp4").write_text(f"hdmap {camera}", encoding="utf-8")

    (scene_dir / f"{uuid}.unused_depth.mp4").write_text("ignored", encoding="utf-8")
    (scene_dir / "unrelated.txt").write_text("ignored", encoding="utf-8")


def _assert_link_points_to(link: Path, target: Path) -> None:
    assert link.is_symlink(), f"expected symlink at {link}"
    assert link.resolve() == target.resolve()


def test_fanout_scene_layout_creates_per_camera_symlinks(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    data_dir = tmp_path / "data"
    scene_a = "scene-0001"
    scene_b = "scene-0002"
    front = "camera_front_wide_120fov"
    tele = "camera_front_tele_30fov"

    _write_scene(staging_root, scene_a, (front, tele))
    _write_scene(staging_root, scene_b, (front,))

    num_scenes, cameras = prepare.fanout_scene_layout(staging_root, data_dir)

    assert num_scenes == 2
    assert cameras == {front, tele}

    _assert_link_points_to(
        data_dir / "video" / front / f"{scene_a}.mp4",
        staging_root / scene_a / f"{scene_a}.{front}_rgb.mp4",
    )
    _assert_link_points_to(
        data_dir / "hdmap" / front / f"{scene_a}.mp4",
        staging_root / scene_a / f"{scene_a}.{front}_hdmap.mp4",
    )
    _assert_link_points_to(
        data_dir / "caption" / front / f"{scene_a}.txt",
        staging_root / scene_a / f"{scene_a}.prompt.txt",
    )
    _assert_link_points_to(
        data_dir / "video" / tele / f"{scene_a}.mp4",
        staging_root / scene_a / f"{scene_a}.{tele}_rgb.mp4",
    )
    _assert_link_points_to(
        data_dir / "caption" / tele / f"{scene_a}.txt",
        staging_root / scene_a / f"{scene_a}.prompt.txt",
    )
    _assert_link_points_to(
        data_dir / "video" / front / f"{scene_b}.mp4",
        staging_root / scene_b / f"{scene_b}.{front}_rgb.mp4",
    )
    assert not (data_dir / "video" / "unused_depth" / f"{scene_a}.mp4").exists()


def test_fanout_scene_layout_replaces_existing_symlinks(tmp_path: Path) -> None:
    first_staging = tmp_path / "first"
    second_staging = tmp_path / "second"
    data_dir = tmp_path / "data"
    uuid = "scene-0001"
    camera = "camera_front_wide_120fov"

    _write_scene(first_staging, uuid, (camera,))
    prepare.fanout_scene_layout(first_staging, data_dir)

    video_link = data_dir / "video" / camera / f"{uuid}.mp4"
    _assert_link_points_to(video_link, first_staging / uuid / f"{uuid}.{camera}_rgb.mp4")

    _write_scene(second_staging, uuid, (camera,))
    prepare.fanout_scene_layout(second_staging, data_dir)

    _assert_link_points_to(video_link, second_staging / uuid / f"{uuid}.{camera}_rgb.mp4")


def test_fanout_scene_layout_skips_scenes_without_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    staging_root = tmp_path / "staging"
    data_dir = tmp_path / "data"
    uuid = "scene-without-prompt"
    camera = "camera_front_wide_120fov"
    _write_scene(staging_root, uuid, (camera,), prompt=False)

    num_scenes, cameras = prepare.fanout_scene_layout(staging_root, data_dir)

    assert num_scenes == 1
    assert cameras == set()
    assert not data_dir.exists()
    assert "missing prompt.txt; skipping" in capsys.readouterr().out


def test_default_repo_uses_omni_dreams_hf_org(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNI_DREAMS_HF_ORG", raising=False)
    assert prepare.default_repo() == "nvidia/omni-dreams-scenes"

    monkeypatch.setenv("OMNI_DREAMS_HF_ORG", "custom-org")
    assert prepare.default_repo() == "custom-org/omni-dreams-scenes"


def test_stage_1_dry_run_does_not_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _fail_snapshot_download(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("dry-run should not call snapshot download")

    monkeypatch.setattr(prepare, "_snapshot_download", _fail_snapshot_download)

    prepare.stage_1_hf_dataset(
        tmp_path / "data",
        repo="test-org/test-scenes",
        subpath="slice",
        force=False,
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert "would snapshot_download test-org/test-scenes/slice" in output
    assert "would fan files out into" in output
    assert not (tmp_path / "data").exists()
