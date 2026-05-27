#!/usr/bin/env python3
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

"""Stage post-training inputs into ``samples/post-training/data/``.

Downloads the HuggingFace sample dataset and fans its per-scene file
layout out into the per-camera tree that
``LocalMultiviewVideoDataset`` expects.

Source layout on HF (one directory per scene UUID)::

    <subpath>/<uuid>/<uuid>.<camera_key>_rgb.mp4
    <subpath>/<uuid>/<uuid>.<camera_key>_hdmap.mp4
    <subpath>/<uuid>/<uuid>.prompt.txt

Target layout under ``data_root`` (symlinks; cheap + reversible)::

    data/video/<camera_key>/<uuid>.mp4
    data/hdmap/<camera_key>/<uuid>.mp4
    data/caption/<camera_key>/<uuid>.txt

Idempotent: rerunning with the same args replaces only the symlinks,
not the downloaded payload. ``setup_env.sh`` invokes this script on
every run.

Env overrides:
    OMNI_DREAMS_HF_ORG    HF org for OmniDreams repos
                          (default: nvidia)
    OMNI_HF_DATA_SUBPATH  subdir within the repo
                          (default: PAI-900_intersect_PAI-300k)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

DEFAULT_HF_ORG = "nvidia"
SCENES_REPO_NAME = "omni-dreams-scenes"
DEFAULT_SUBPATH = "PAI-900_intersect_PAI-300k"


def info(message: str) -> None:
    print(f"[prepare] {message}")


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_repo() -> str:
    return f"{os.environ.get('OMNI_DREAMS_HF_ORG', DEFAULT_HF_ORG)}/{SCENES_REPO_NAME}"


def parse_args() -> argparse.Namespace:
    repo_default = default_repo()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        choices=("all", "1"),
        default="1",
        help="Stage to run. Only stage 1 (HF sample dataset) is defined today.",
    )
    parser.add_argument(
        "--repo",
        default=repo_default,
        help=f"HF dataset repo id (default: {repo_default}; org env: OMNI_DREAMS_HF_ORG).",
    )
    parser.add_argument(
        "--subpath",
        default=os.environ.get("OMNI_HF_DATA_SUBPATH", DEFAULT_SUBPATH),
        help=f"Subdirectory within the repo (default: {DEFAULT_SUBPATH}; env: OMNI_HF_DATA_SUBPATH).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the destination dir (default: samples/post-training/data/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download payload even if files already exist on disk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be staged without making network calls.",
    )
    return parser.parse_args()


def _snapshot_download(repo: str, subpath: str, target: Path, force: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Run "
            "`uv sync --extra=cu128` from samples/post-training/ first."
        ) from exc
    target.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=[f"{subpath}/**"],
        force_download=force,
    )
    return Path(local) / subpath


# Source filename suffix → (target subdir, target extension).
# The prompt.txt has no camera tag in its name (one caption per scene); we
# fan it out to every camera dir so the per-camera caption lookup in
# LocalMultiviewVideoDataset finds it under whichever camera_keys the
# experiment selects.
_RGB_SUFFIX = "_rgb.mp4"
_HDMAP_SUFFIX = "_hdmap.mp4"
_PROMPT_SUFFIX = ".prompt.txt"


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def fanout_scene_layout(staging_root: Path, data_dir: Path) -> tuple[int, set[str]]:
    """Symlink per-scene files into the per-camera tree.

    Returns ``(num_scenes, cameras_seen)``.
    """
    scenes = sorted(p for p in staging_root.iterdir() if p.is_dir())
    cameras_seen: set[str] = set()

    for scene_dir in scenes:
        uuid = scene_dir.name
        prompt_src = scene_dir / f"{uuid}{_PROMPT_SUFFIX}"
        if not prompt_src.is_file():
            info(f"  warning: scene {uuid} missing prompt.txt; skipping")
            continue

        scene_cameras: set[str] = set()
        for src in scene_dir.iterdir():
            name = src.name
            prefix = f"{uuid}."
            if not name.startswith(prefix):
                continue
            tag = name[len(prefix):]  # e.g. "camera_front_wide_120fov_rgb.mp4"
            if tag.endswith(_RGB_SUFFIX):
                cam = tag[: -len(_RGB_SUFFIX)]
                _symlink(src.resolve(), data_dir / "video" / cam / f"{uuid}.mp4")
                scene_cameras.add(cam)
            elif tag.endswith(_HDMAP_SUFFIX):
                cam = tag[: -len(_HDMAP_SUFFIX)]
                _symlink(src.resolve(), data_dir / "hdmap" / cam / f"{uuid}.mp4")
                scene_cameras.add(cam)
            # Other tags (e.g. prompt.txt) handled below / ignored here.

        for cam in scene_cameras:
            _symlink(prompt_src.resolve(), data_dir / "caption" / cam / f"{uuid}.txt")
        cameras_seen |= scene_cameras

    return len(scenes), cameras_seen


def stage_1_hf_dataset(
    data_dir: Path, *, repo: str, subpath: str, force: bool, dry_run: bool
) -> None:
    info(f"Stage 1: HF dataset {repo}/{subpath} -> {data_dir}")
    staging_root = data_dir / f"{subpath.replace('/', '_').lower()}_staging"

    if dry_run:
        info(f"  (dry-run) would snapshot_download {repo}/{subpath} into {staging_root}")
        info(f"  (dry-run) would fan files out into {data_dir}/{{video,hdmap,caption}}/<cam>/<uuid>.<ext>")
        return

    local = _snapshot_download(repo, subpath, staging_root, force=force)
    info(f"  Downloaded to {local}")

    num_scenes, cameras = fanout_scene_layout(local, data_dir)
    info(f"  Linked {num_scenes} scene(s) across cameras: {sorted(cameras)}")


def stages_to_run(selection: str) -> Iterable[int]:
    if selection == "all":
        return (1,)
    return (int(selection),)


def main() -> int:
    args = parse_args()
    data_dir = (args.data_dir or (repo_root() / "data")).resolve()

    if args.dry_run:
        info("Dry-run mode; no network calls.")

    for stage in stages_to_run(args.stage):
        if stage == 1:
            stage_1_hf_dataset(
                data_dir,
                repo=args.repo,
                subpath=args.subpath,
                force=args.force,
                dry_run=args.dry_run,
            )

    info("Workspace inputs are ready.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
