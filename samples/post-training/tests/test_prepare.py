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

"""Unit tests for the post-training dataset staging helper.

The fixtures mirror the *real* ``nvidia/PhysicalAI-Autonomous-Vehicles-NuRec``
``26.01`` layout: per-scene directories whose media files are **not**
UUID-prefixed (``camera_<key>_rgb.mp4``), plus the legacy ``omni-dreams-scenes``
layout that prefixed every file with ``<uuid>.``.
"""

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
    prompt: str | bool = "per_camera",
    legacy_prefix: bool = False,
    extras: bool = True,
) -> None:
    """Write a fake scene dir.

    ``prompt`` selects the caption convention:
      - ``"per_camera"``: one ``<cam>_prompt.txt`` per camera (PAI-NuRec, default)
      - ``"scene"`` / ``True``: one scene-level ``<uuid>.prompt.txt`` (legacy)
      - ``False`` / ``"none"``: no prompt at all

    ``legacy_prefix`` reproduces the old omni-dreams-scenes naming where every
    media file was prefixed with ``<uuid>.``. ``extras`` adds the recon-only
    companions PAI-NuRec ships (``<uuid>.usdz``, ``labels.json``, the bare
    ``camera_front_wide_120fov.mp4`` preview) so tests prove they're ignored.
    """
    scene_dir = staging_root / uuid
    scene_dir.mkdir(parents=True)
    pfx = f"{uuid}." if legacy_prefix else ""

    if prompt in ("scene", True):
        (scene_dir / f"{uuid}.prompt.txt").write_text(f"prompt for {uuid}", encoding="utf-8")

    for camera in cameras:
        (scene_dir / f"{pfx}{camera}_rgb.mp4").write_text(f"rgb {camera}", encoding="utf-8")
        (scene_dir / f"{pfx}{camera}_hdmap.mp4").write_text(f"hdmap {camera}", encoding="utf-8")
        if prompt == "per_camera":
            (scene_dir / f"{pfx}{camera}_prompt.txt").write_text(
                f"prompt {camera}", encoding="utf-8"
            )

    if extras:
        (scene_dir / f"{uuid}.usdz").write_text("huge recon", encoding="utf-8")
        (scene_dir / "labels.json").write_text("{}", encoding="utf-8")
        (scene_dir / "camera_front_wide_120fov.mp4").write_text("bare preview", encoding="utf-8")


def _assert_link_points_to(link: Path, target: Path) -> None:
    assert link.is_symlink(), f"expected symlink at {link}"
    assert link.resolve() == target.resolve()


def test_fanout_scene_layout_pai_nurec_filenames(tmp_path: Path) -> None:
    """Real PAI-NuRec files carry no UUID prefix; they must still fan out."""
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
        staging_root / scene_a / f"{front}_rgb.mp4",
    )
    _assert_link_points_to(
        data_dir / "hdmap" / front / f"{scene_a}.mp4",
        staging_root / scene_a / f"{front}_hdmap.mp4",
    )
    _assert_link_points_to(
        data_dir / "caption" / front / f"{scene_a}.txt",
        staging_root / scene_a / f"{front}_prompt.txt",
    )
    _assert_link_points_to(
        data_dir / "video" / tele / f"{scene_a}.mp4",
        staging_root / scene_a / f"{tele}_rgb.mp4",
    )

    # The recon-only companions must never leak into the camera tree, and the
    # bare front-wide preview must not be mistaken for the front-wide RGB.
    assert not list((data_dir).rglob("*.usdz"))
    assert not list((data_dir).rglob("labels.json"))
    _assert_link_points_to(
        data_dir / "video" / front / f"{scene_a}.mp4",
        staging_root / scene_a / f"{front}_rgb.mp4",  # the _rgb file, not the bare .mp4
    )


def test_fanout_scene_layout_legacy_uuid_prefix(tmp_path: Path) -> None:
    """The legacy omni-dreams-scenes ``<uuid>.<cam>_rgb.mp4`` naming still works."""
    staging_root = tmp_path / "staging"
    data_dir = tmp_path / "data"
    uuid = "scene-0001"
    front = "camera_front_wide_120fov"

    _write_scene(staging_root, uuid, (front,), prompt="scene", legacy_prefix=True, extras=False)

    num_scenes, cameras = prepare.fanout_scene_layout(staging_root, data_dir)

    assert num_scenes == 1
    assert cameras == {front}
    _assert_link_points_to(
        data_dir / "video" / front / f"{uuid}.mp4",
        staging_root / uuid / f"{uuid}.{front}_rgb.mp4",
    )
    # Scene-level caption fans out to the camera.
    _assert_link_points_to(
        data_dir / "caption" / front / f"{uuid}.txt",
        staging_root / uuid / f"{uuid}.prompt.txt",
    )


def test_fanout_scene_layout_replaces_existing_symlinks(tmp_path: Path) -> None:
    first_staging = tmp_path / "first"
    second_staging = tmp_path / "second"
    data_dir = tmp_path / "data"
    uuid = "scene-0001"
    camera = "camera_front_wide_120fov"

    _write_scene(first_staging, uuid, (camera,))
    prepare.fanout_scene_layout(first_staging, data_dir)

    video_link = data_dir / "video" / camera / f"{uuid}.mp4"
    _assert_link_points_to(video_link, first_staging / uuid / f"{camera}_rgb.mp4")

    _write_scene(second_staging, uuid, (camera,))
    prepare.fanout_scene_layout(second_staging, data_dir)

    _assert_link_points_to(video_link, second_staging / uuid / f"{camera}_rgb.mp4")


def test_fanout_scene_layout_skips_scenes_without_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A recon-only scene (no rgb/hdmap/prompt) is the non-trainable case."""
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


def test_fanout_scene_layout_uses_per_camera_prompts(tmp_path: Path) -> None:
    """PAI-NuRec ships one prompt per camera; each caption links to its own."""
    staging_root = tmp_path / "staging"
    data_dir = tmp_path / "data"
    uuid = "scene-0001"
    front = "camera_front_wide_120fov"
    tele = "camera_front_tele_30fov"

    _write_scene(staging_root, uuid, (front, tele), prompt="per_camera")

    num_scenes, cameras = prepare.fanout_scene_layout(staging_root, data_dir)

    assert num_scenes == 1
    assert cameras == {front, tele}
    _assert_link_points_to(
        data_dir / "caption" / front / f"{uuid}.txt",
        staging_root / uuid / f"{front}_prompt.txt",
    )
    _assert_link_points_to(
        data_dir / "caption" / tele / f"{uuid}.txt",
        staging_root / uuid / f"{tele}_prompt.txt",
    )


def test_default_repo_is_pai_nurec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNI_HF_DATA_REPO", raising=False)
    assert prepare.default_repo() == "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"

    # Decoupled from the checkpoint org knob.
    monkeypatch.setenv("OMNI_DREAMS_HF_ORG", "some-models-org")
    assert prepare.default_repo() == "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"

    monkeypatch.setenv("OMNI_HF_DATA_REPO", "custom-org/custom-dataset")
    assert prepare.default_repo() == "custom-org/custom-dataset"


def test_default_revision_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNI_HF_DATA_REVISION", raising=False)
    assert prepare.default_revision() == "26.01"

    monkeypatch.setenv("OMNI_HF_DATA_REVISION", "main")
    assert prepare.default_revision() == "main"


def test_default_include_globs_select_training_media(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default include positively selects the per-camera training media,
    # which is what excludes the multi-TB .usdz and the bare preview mp4.
    monkeypatch.delenv("OMNI_HF_DATA_INCLUDE", raising=False)
    assert prepare._default_include_globs() == [
        "*_rgb.mp4",
        "*_hdmap.mp4",
        "*_prompt.txt",
        "*.prompt.txt",
    ]

    monkeypatch.setenv("OMNI_HF_DATA_INCLUDE", "")  # fetch the whole subpath
    assert prepare._default_include_globs() == []

    monkeypatch.setenv("OMNI_HF_DATA_INCLUDE", "**")
    assert prepare._default_include_globs() == ["**"]


def test_default_ignore_globs_empty_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNI_HF_DATA_IGNORE", raising=False)
    assert prepare._default_ignore_globs() == []

    monkeypatch.setenv("OMNI_HF_DATA_IGNORE", "*.usdz *.bin")
    assert prepare._default_ignore_globs() == ["*.usdz", "*.bin"]


def test_allow_patterns_scopes_include_to_subpath() -> None:
    # No include + no subpath -> no filter at all.
    assert prepare._allow_patterns("", []) is None
    # No include but a subpath -> whole subpath.
    assert prepare._allow_patterns("sample_set/26.01_release", []) == [
        "sample_set/26.01_release/**"
    ]
    # Include globs are scoped under the subpath. fnmatch `*` spans `/`, so a
    # single `<subpath>/*_rgb.mp4` matches at any depth (no `**/` needed).
    assert prepare._allow_patterns("sample_set/26.01_release", ["*_rgb.mp4", "*.prompt.txt"]) == [
        "sample_set/26.01_release/*_rgb.mp4",
        "sample_set/26.01_release/*.prompt.txt",
    ]
    # Include globs with no subpath match at any depth.
    assert prepare._allow_patterns("", ["*_rgb.mp4"]) == ["*_rgb.mp4"]


def test_stage_1_passes_revision_include_ignore_to_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OMNI_HF_DATA_INCLUDE", raising=False)
    monkeypatch.delenv("OMNI_HF_DATA_IGNORE", raising=False)
    monkeypatch.delenv("OMNI_HF_DATA_REVISION", raising=False)
    captured: dict[str, object] = {}

    def _fake_download(
        repo: str,
        subpath: str,
        target: Path,
        force: bool,
        include: list[str],
        ignore: list[str],
        revision: str,
    ) -> Path:
        captured["include"] = include
        captured["ignore"] = ignore
        captured["revision"] = revision
        staging = target / "scene-0001"
        staging.mkdir(parents=True)
        # A .usdz that somehow lands in staging must never be linked out.
        (staging / "scene-0001.usdz").write_text("huge", encoding="utf-8")
        (staging / "camera_x_rgb.mp4").write_text("v", encoding="utf-8")
        (staging / "camera_x_prompt.txt").write_text("p", encoding="utf-8")
        return target

    monkeypatch.setattr(prepare, "_snapshot_download", _fake_download)
    data_dir = tmp_path / "data"
    prepare.stage_1_hf_dataset(
        data_dir, repo="org/repo", subpath="", force=False, dry_run=False
    )
    assert captured["include"] == ["*_rgb.mp4", "*_hdmap.mp4", "*_prompt.txt", "*.prompt.txt"]
    assert captured["ignore"] == []
    assert captured["revision"] == "26.01"
    # the fanout links the mp4 but never the .usdz into the camera tree
    assert (data_dir / "video" / "camera_x" / "scene-0001.mp4").is_symlink()
    for sub in ("video", "hdmap", "caption"):
        assert not list((data_dir / sub).rglob("*.usdz"))


def test_stage_1_local_source_skips_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail_snapshot_download(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("local-source staging must not download")

    monkeypatch.setattr(prepare, "_snapshot_download", _fail_snapshot_download)

    staging_root = tmp_path / "s3copy"
    uuid = "scene-0001"
    front = "camera_front_wide_120fov"
    _write_scene(staging_root, uuid, (front,), prompt="per_camera")

    data_dir = tmp_path / "data"
    prepare.stage_1_hf_dataset(
        data_dir,
        repo="ignored",
        subpath="",
        force=False,
        dry_run=False,
        local_source=staging_root,
    )

    _assert_link_points_to(
        data_dir / "video" / front / f"{uuid}.mp4",
        staging_root / uuid / f"{front}_rgb.mp4",
    )


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
        revision="26.01",
    )

    output = capsys.readouterr().out
    assert "would snapshot_download test-org/test-scenes@26.01/slice" in output
    assert "would fan files out into" in output
    assert not (tmp_path / "data").exists()
