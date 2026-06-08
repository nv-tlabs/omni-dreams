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

Downloads the HuggingFace sample dataset (or fans out an already-staged
local copy) and maps its per-scene file layout into the per-camera tree
that ``LocalMultiviewVideoDataset`` expects.

The sample dataset is the PAI-NuRec release,
``nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`` — a *separate* HF repo
from the ``omni-dreams-models`` checkpoint repo. Override it with
``OMNI_HF_DATA_REPO`` (full ``org/repo`` id), independent of the
``OMNI_DREAMS_HF_ORG`` knob that selects the checkpoint org.

The trainable scenes live on the ``26.01`` **branch** under
``sample_set/26.01_release/`` (the branch and folder names differ — both
are pinned here in ``DEFAULT_REVISION`` / ``DEFAULT_SUBPATH``, override with
``OMNI_HF_DATA_REVISION`` / ``OMNI_HF_DATA_SUBPATH``).

Source layout (one directory per scene UUID)::

    <subpath>/<uuid>/<camera_key>_rgb.mp4       # PAI-NuRec: no UUID prefix
    <subpath>/<uuid>/<camera_key>_hdmap.mp4
    <subpath>/<uuid>/<camera_key>_prompt.txt    # PAI-NuRec: per-camera caption
    <subpath>/<uuid>/<uuid>.prompt.txt          # legacy: one caption per scene
    <subpath>/<uuid>/<uuid>.usdz                # multi-TB recon; never trained on

The legacy ``omni-dreams-scenes`` layout prefixed every file with
``<uuid>.`` (e.g. ``<uuid>.<camera_key>_rgb.mp4``); that prefix is stripped
when present, so both conventions fan out correctly. Two caption flavours
are accepted: a per-camera ``_prompt.txt`` wins, falling back to a
scene-level ``.prompt.txt`` when present.

**There is no manifest marking the trainable subset** — a scene is
trainable iff its directory ships the per-camera training media. We select
it positively at download time: only ``*_rgb.mp4`` / ``*_hdmap.mp4`` /
``*_prompt.txt`` (plus the legacy scene-level ``*.prompt.txt``) are
fetched. That skips the ~740 reconstruction-only scenes, every per-scene
``.usdz`` (~1.6 TB), and the bare ``camera_front_wide_120fov.mp4`` preview
(~8.5 GiB) the dataloader never reads — ~10 GiB instead of ~1.65 TB.
Override the selector with ``OMNI_HF_DATA_INCLUDE`` (e.g. a future run that
differentiably renders the scenes wants the ``.usdz``: set
``OMNI_HF_DATA_INCLUDE='**'``).

Target layout under ``data_root`` (symlinks; cheap + reversible)::

    data/video/<camera_key>/<uuid>.mp4
    data/hdmap/<camera_key>/<uuid>.mp4
    data/caption/<camera_key>/<uuid>.txt

Idempotent: rerunning with the same args replaces only the symlinks,
not the downloaded payload. ``setup_env.sh`` invokes this script on
every run.

Env overrides:
    OMNI_HF_DATA_REPO       HF dataset repo id
                            (default: nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)
    OMNI_HF_DATA_REVISION   branch / tag / commit on that repo
    OMNI_HF_DATA_SUBPATH    subdir within the repo
                            (defaults: see DEFAULT_REVISION / DEFAULT_SUBPATH)
    OMNI_LOCAL_DATA_SOURCE  fan out from this already-downloaded per-scene
                            tree instead of hitting HuggingFace (e.g. an
                            rclone'd S3 copy). Skips snapshot_download.
    OMNI_HF_DATA_INCLUDE    space-separated globs to fetch (default: the
                            per-camera training media; '**' = everything).
    OMNI_HF_DATA_IGNORE     space-separated globs to skip on download
                            (default: none — the include selector already
                            excludes the .usdz).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

# PAI-NuRec is a standalone public dataset repo, decoupled from the
# omni-dreams-models checkpoint org (OMNI_DREAMS_HF_ORG). Override the whole
# id with OMNI_HF_DATA_REPO.
DEFAULT_DATA_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
# The trainable scenes live on this branch; the in-repo folder is
# named 26.01_release (note: branch != folder). Both are overridable.
DEFAULT_REVISION = "26.01"
DEFAULT_SUBPATH = "sample_set/26.01_release"


def info(message: str) -> None:
    print(f"[prepare] {message}")


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_repo() -> str:
    return os.environ.get("OMNI_HF_DATA_REPO", DEFAULT_DATA_REPO)


def default_revision() -> str:
    return os.environ.get("OMNI_HF_DATA_REVISION", DEFAULT_REVISION)


def parse_args() -> argparse.Namespace:
    repo_default = default_repo()
    revision_default = default_revision()
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
        help=f"HF dataset repo id (default: {repo_default}; env: OMNI_HF_DATA_REPO).",
    )
    parser.add_argument(
        "--revision",
        default=revision_default,
        help=(
            f"Branch / tag / commit on the dataset repo "
            f"(default: {revision_default}; env: OMNI_HF_DATA_REVISION)."
        ),
    )
    parser.add_argument(
        "--subpath",
        default=os.environ.get("OMNI_HF_DATA_SUBPATH", DEFAULT_SUBPATH),
        help=(
            f"Subdirectory within the repo (default: {DEFAULT_SUBPATH}; env: OMNI_HF_DATA_SUBPATH)."
        ),
    )
    parser.add_argument(
        "--local-source",
        type=Path,
        default=(
            Path(os.environ["OMNI_LOCAL_DATA_SOURCE"])
            if os.environ.get("OMNI_LOCAL_DATA_SOURCE")
            else None
        ),
        help=(
            "Fan out from this already-downloaded per-scene tree instead of "
            "downloading from HuggingFace (env: OMNI_LOCAL_DATA_SOURCE)."
        ),
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        metavar="GLOB",
        help=(
            "Globs to fetch (default: env OMNI_HF_DATA_INCLUDE, else the "
            "per-camera training media). Pass --include '**' to fetch the "
            "whole subpath, including the .usdz."
        ),
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=None,
        metavar="GLOB",
        help=(
            "Globs to skip on download (default: env OMNI_HF_DATA_IGNORE, "
            "else none — the include selector already excludes the .usdz)."
        ),
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


# Source filename suffixes. RGB / HDMap are always per-camera. Captions come
# in two flavours: PAI-NuRec ships one prompt *per camera*
# (``<cam>_prompt.txt``); the legacy omni-dreams-scenes layout shipped one
# prompt *per scene* (``<uuid>.prompt.txt``) fanned out to every camera.
# A per-camera prompt wins; the scene-level prompt is the fallback.
_RGB_SUFFIX = "_rgb.mp4"
_HDMAP_SUFFIX = "_hdmap.mp4"
_PROMPT_PER_CAMERA_SUFFIX = "_prompt.txt"
_PROMPT_SCENE_SUFFIX = ".prompt.txt"

# Positively select the per-camera training media. This *is* the trainable-
# scene filter: scenes that ship none of these (recon-only scenes) download
# nothing and fan out to nothing. Globs are matched by huggingface_hub with
# fnmatch, where `*` spans `/`, so a bare `*_rgb.mp4` matches at any depth.
# `*.prompt.txt` keeps the legacy scene-level caption; `*_prompt.txt` keeps
# the PAI-NuRec per-camera caption.
DEFAULT_INCLUDE_GLOBS = (
    f"*{_RGB_SUFFIX}",
    f"*{_HDMAP_SUFFIX}",
    f"*{_PROMPT_PER_CAMERA_SUFFIX}",
    f"*{_PROMPT_SCENE_SUFFIX}",
)


def _default_include_globs() -> list[str]:
    raw = os.environ.get("OMNI_HF_DATA_INCLUDE")
    if raw is None:
        return list(DEFAULT_INCLUDE_GLOBS)
    return raw.split()  # empty string -> [] -> whole subpath (see _allow_patterns)


def _default_ignore_globs() -> list[str]:
    raw = os.environ.get("OMNI_HF_DATA_IGNORE")
    if raw is None:
        return []
    return raw.split()


def _allow_patterns(subpath: str, include: list[str]) -> list[str] | None:
    # huggingface_hub matches allow_patterns with fnmatch, where `*` spans
    # `/`. So `<subpath>/*_rgb.mp4` matches a `_rgb.mp4` at *any* depth under
    # the subpath (a literal `**/` would instead force an intermediate dir and
    # miss files staged directly under the subpath). Scope each include glob to
    # the configured subpath (or the whole repo when empty). An empty include
    # list means "no filename filter": fall back to the whole subpath.
    if not include:
        return [f"{subpath}/**"] if subpath else None
    prefix = f"{subpath}/" if subpath else ""
    return [f"{prefix}{glob}" for glob in include]


def _snapshot_download(
    repo: str,
    subpath: str,
    target: Path,
    force: bool,
    include: list[str],
    ignore: list[str],
    revision: str,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Run "
            "`uv sync --extra=cu128` or `uv sync --extra=cu130` from "
            "post-training/ first."
        ) from exc
    target.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(target),
        allow_patterns=_allow_patterns(subpath, include),
        ignore_patterns=ignore or None,
        force_download=force,
    )
    return Path(local) / subpath if subpath else Path(local)


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
        # Legacy omni-dreams-scenes prefixed every file with "<uuid>.";
        # PAI-NuRec does not. Strip the prefix when present so both fan out.
        legacy_prefix = f"{uuid}."
        scene_prompt = scene_dir / f"{uuid}{_PROMPT_SCENE_SUFFIX}"

        rgb: dict[str, Path] = {}
        hdmap: dict[str, Path] = {}
        per_camera_prompt: dict[str, Path] = {}
        for src in scene_dir.iterdir():
            name = src.name
            tag = name[len(legacy_prefix) :] if name.startswith(legacy_prefix) else name
            if tag.endswith(_RGB_SUFFIX):
                rgb[tag[: -len(_RGB_SUFFIX)]] = src
            elif tag.endswith(_HDMAP_SUFFIX):
                hdmap[tag[: -len(_HDMAP_SUFFIX)]] = src
            elif tag.endswith(_PROMPT_PER_CAMERA_SUFFIX):
                # Excludes the scene-level "prompt.txt" (no leading "_").
                per_camera_prompt[tag[: -len(_PROMPT_PER_CAMERA_SUFFIX)]] = src

        # A scene is usable only if it carries at least one prompt (per-camera
        # or scene-level). Skip otherwise, creating nothing for it.
        if not per_camera_prompt and not scene_prompt.is_file():
            info(f"  warning: scene {uuid} missing prompt.txt; skipping")
            continue

        for cam in sorted(set(rgb) | set(hdmap)):
            if cam in rgb:
                _symlink(rgb[cam].resolve(), data_dir / "video" / cam / f"{uuid}.mp4")
            if cam in hdmap:
                _symlink(hdmap[cam].resolve(), data_dir / "hdmap" / cam / f"{uuid}.mp4")
            prompt_src = per_camera_prompt.get(cam)
            if prompt_src is None and scene_prompt.is_file():
                prompt_src = scene_prompt
            if prompt_src is not None:
                _symlink(prompt_src.resolve(), data_dir / "caption" / cam / f"{uuid}.txt")
            cameras_seen.add(cam)

    return len(scenes), cameras_seen


def stage_1_hf_dataset(
    data_dir: Path,
    *,
    repo: str,
    subpath: str,
    force: bool,
    dry_run: bool,
    revision: str | None = None,
    local_source: Path | None = None,
    include: list[str] | None = None,
    ignore: list[str] | None = None,
) -> None:
    revision = default_revision() if revision is None else revision
    include = _default_include_globs() if include is None else include
    ignore = _default_ignore_globs() if ignore is None else ignore
    # Local-source path: fan out from an already-downloaded per-scene tree
    # (e.g. an rclone'd S3 copy), no HuggingFace round-trip.
    if local_source is not None:
        info(f"Stage 1: local source {local_source} -> {data_dir}")
        if dry_run:
            info(
                f"  (dry-run) would fan {local_source} out into "
                f"{data_dir}/{{video,hdmap,caption}}/<cam>/<uuid>.<ext>"
            )
            return
        if not local_source.is_dir():
            raise RuntimeError(f"--local-source path does not exist: {local_source}")
        num_scenes, cameras = fanout_scene_layout(local_source, data_dir)
        info(f"  Linked {num_scenes} scene(s) across cameras: {sorted(cameras)}")
        return

    label = f"{repo}@{revision}/{subpath}" if subpath else f"{repo}@{revision}"
    info(f"Stage 1: HF dataset {label} -> {data_dir}")
    slug = subpath.replace("/", "_").lower() or "root"
    staging_root = data_dir / f"{slug}_staging"

    if dry_run:
        info(
            f"  (dry-run) would snapshot_download {label} into {staging_root} "
            f"(include: {include or 'everything'}; ignore: {ignore or 'nothing'})"
        )
        info(
            f"  (dry-run) would fan files out into "
            f"{data_dir}/{{video,hdmap,caption}}/<cam>/<uuid>.<ext>"
        )
        return

    local = _snapshot_download(
        repo,
        subpath,
        staging_root,
        force=force,
        include=include,
        ignore=ignore,
        revision=revision,
    )
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
                revision=args.revision,
                local_source=args.local_source,
                include=args.include,
                ignore=args.ignore,
            )

    info("Workspace inputs are ready.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
