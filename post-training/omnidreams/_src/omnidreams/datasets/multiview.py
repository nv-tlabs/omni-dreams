# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Webloaders of datasets and augmentations for visual-text multiview dataset for AV
"""

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
import io
import json
import os
import random
from typing import Any, Literal, Optional, TypeAlias

import attrs
import numpy as np
import torch
import webdataset as wds
from einops import rearrange
from torchvision.transforms import InterpolationMode, Resize

import omnidreams._src.predict2.datasets.distributor.parallel_sync_multi_aspect_ratio as parallel_sync_multi_aspect_ratio
from omnidreams._src.imaginaire.datasets.decoders.json_loader import json_decoder
from omnidreams._src.imaginaire.datasets.decoders.video_decoder import video_naive_bytes
from omnidreams._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from omnidreams._src.imaginaire.datasets.webdataset.config.schema import DatasetConfig
from omnidreams._src.imaginaire.datasets.webdataset.distributors import ShardlistBasic
from omnidreams._src.imaginaire.utils import log
from omnidreams._src.predict2.datasets.cached_replay_dataloader import get_cached_replay_dataloader
from omnidreams._src.predict2_multiview.datasets.multiview import (
    AugmentationConfig as _BaseAugmentationConfig,
)
from omnidreams._src.predict2_multiview.datasets.multiview import (
    UnpackMetas,
    collate_fn,
)
from omnidreams._src.omnidreams.datasets.sil_dataset import SILDataset
from omnidreams._src.omnidreams.datasets.wdinfo_utils import (
    DEFAULT_CATALOG,
    get_video_dataset_info,
    load_tar_filter_from_probs,
)


@attrs.define(slots=False)
class AugmentationConfig(_BaseAugmentationConfig):
    """SIL-extended augmentation config with rejection sampling fields."""

    rejection_probs_path: str | None = None
    rejection_default_prob: float = 0.0
    rejection_seed: int | None = None
    rejection_histogram: bool = False
    rejection_histogram_interval: int = 10


CameraKeyType: TypeAlias = str


class RejectionSampler:
    """Probabilistic rejection sampling for data rebalancing.

    Rejects samples based on per-clip acceptance probabilities loaded from a JSON file.
    Clips not in the file are rejected by default (``default_prob=0.0``).
    """

    def __init__(
        self,
        probs_path: str | None = None,
        default_prob: float = 1.0,
        seed: int | None = None,
    ):
        self.prob_map: dict[str, float] = {}
        self.default_prob = default_prob
        self._has_clip_probs = False
        self._has_url_patterns = False
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self._stats = {"total": 0, "accepted": 0, "rejected": 0}
        if probs_path:
            self._load_probs(probs_path)

    def _load_probs(self, path: str) -> None:
        if not os.path.exists(path):
            log.warning(f"RejectionSampler: probability file not found at {path}, accepting all samples")
            return

        with open(path) as f:
            raw = json.load(f)

        if isinstance(raw, dict) and raw.get("_mode") == "url_pattern":
            self._has_url_patterns = True
            patterns = raw.get("patterns", {})
            self.pattern_probs = sorted(patterns.items(), key=lambda x: -len(x[0]))
            log.info(f"RejectionSampler: loaded {len(self.pattern_probs)} URL patterns from {path}")
        else:
            self._has_clip_probs = True
            self.prob_map = {
                str(k): float(v) for k, v in raw.items() if not k.startswith("_") and isinstance(v, (int, float))
            }
            log.info(f"RejectionSampler: loaded {len(self.prob_map)} clip probabilities from {path}")

    def _get_prob(self, data: dict) -> float:
        prob = self.default_prob
        if self._has_clip_probs:
            clip_key = str(data.get("__key__", ""))
            prob = self.prob_map.get(clip_key, self.default_prob)
        elif self._has_url_patterns:
            url_str = str(data.get("__url__", ""))
            for pattern, p in self.pattern_probs:
                if pattern in url_str:
                    prob = p
                    break
        return prob

    def __call__(self, data: dict) -> dict | None:
        prob = self._get_prob(data)
        self._stats["total"] += 1
        if self._rng.random() < prob:
            self._stats["accepted"] += 1
            if self._stats["total"] <= 3 or self._stats["total"] % 1000 == 0:
                log.info(
                    f"RejectionSampler sample #{self._stats['accepted']}: "
                    f"__key__='{data.get('__key__', '?')[:60]}', "
                    f"prob={prob:.3f}, decision=ACCEPT, "
                    f"__url__='{str(data.get('__url__', '?'))[:80]}'"
                )
            return data
        self._stats["rejected"] += 1
        return None


class RejectionHistogramLogger:
    """Transparent dataloader wrapper that tracks per-clip sampling counts.

    Extracts ``__key__`` from each batch and logs a W&B bar-chart histogram of
    how often each tracked clip_id has been sampled.  Enable via
    ``rejection_histogram=True`` on :class:`AugmentationConfig`.

    Each rank logs independently — no distributed synchronisation required.
    Because shards are split evenly (``split_by_node=True``), each rank's
    histogram is representative; summing across ranks gives the aggregate.
    """

    def __init__(
        self,
        dataloader,
        tracked_clip_ids: set[str],
        log_interval: int = 100,
    ):
        self.dataloader = dataloader
        self.tracked_clip_ids = tracked_clip_ids
        self.log_interval = log_interval
        self.counts: dict[str, int] = {cid: 0 for cid in tracked_clip_ids}
        self._step = 0

    # Forward attributes so the training loop sees the same interface.
    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name: str):
        return getattr(self.dataloader, name)

    def __iter__(self):
        for batch in self.dataloader:
            keys = batch.get("__key__", [])
            if isinstance(keys, str):
                keys = [keys]
            for k in keys:
                if k in self.counts:
                    self.counts[k] += 1
            self._step += 1
            if self._step % self.log_interval == 0:
                self._log_wandb()
            yield batch

    def _log_wandb(self) -> None:
        _rank = 0
        _world = 1
        aggregated = {cid: cnt for cid, cnt in self.counts.items() if cnt > 0}

        try:
            import torch
            import torch.distributed as dist

            if dist.is_initialized():
                _rank = dist.get_rank()
                _world = dist.get_world_size()
                ordered_ids = sorted(self.tracked_clip_ids)
                local = torch.tensor(
                    [self.counts.get(c, 0) for c in ordered_ids],
                    dtype=torch.long,
                    device="cuda",
                )
                dist.all_reduce(local, op=dist.ReduceOp.SUM)
                aggregated = {c: int(v) for c, v in zip(ordered_ids, local.tolist()) if v > 0}
        except Exception:
            pass

        if _rank != 0:
            return

        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        if not aggregated:
            return

        sorted_ids = sorted(aggregated, key=aggregated.get, reverse=True)
        labels = [cid[:40] for cid in sorted_ids]
        values = [aggregated[cid] for cid in sorted_ids]

        table = wandb.Table(
            data=list(zip(labels, values)),
            columns=["clip_id", "sample_count"],
        )
        wandb.log(
            {
                "rejection_sampling/clip_histogram": wandb.plot.bar(
                    table,
                    "clip_id",
                    "sample_count",
                    title=f"Rejection Sampling Distribution ({_world} GPUs)",
                ),
                "rejection_sampling/total_tracked_samples": sum(values),
                "rejection_sampling/unique_clips_seen": len(aggregated),
            },
            step=self._step,
        )
        total = sum(values)
        if aggregated:
            top_count = max(values)
            top_pct = 100 * top_count / max(total, 1)
            log.info(
                f"RejectionHistogramLogger step {self._step}: "
                f"{len(aggregated)} unique clips, {total} total samples "
                f"({_world} GPUs), top clip: {top_pct:.1f}% ({top_count}/{total})"
            )
        else:
            log.info(
                f"RejectionHistogramLogger step {self._step}: 0 unique clips, {total} total samples ({_world} GPUs)"
            )


def get_multiview_dataset(
    *,
    dataset_name: str,
    is_train: bool,
    object_store: Literal["s3"] = "s3",
    dataset_keys: list[str],
    augmentations: dict[str, Augmentor],
    dataset_catalog: dict[str, dict[str, list[str]]],
    max_shards: int = 0,
    shuffle_buffer_size: int = 1,
    tar_drop_set: set[str] | None = None,
    accepted_clip_ids: set[str] | None = None,
    precomputed_offsets: dict | None = None,
    prob_map: dict[str, float] | None = None,
    default_prob: float = 1.0,
    rejection_seed: int | None = None,
    clip_to_tar: dict[str, str] | None = None,
) -> SILDataset:
    """Get video-text dataset with optional custom augmentation factory.

    Args:
        is_train: Whether this is for training
        dataset_name: Name of dataset to use for loading wdinfo files
        object_store: Object store to use ("gcs" or "s3")
        dataset_keys: List of keys to use for loading dataset
        augmentations: Augmentations map to apply to dataset
        dataset_catalog: Dataset catalog to use for loading dataset
        max_shards: If > 0, limit the number of tar shards loaded (useful for fast debug runs).
            Can also be set via the MAX_SHARDS environment variable.
    """
    # Environment variable takes precedence (bypasses Hydra config issues)
    env_max_shards = os.environ.get("MAX_SHARDS")
    if env_max_shards is not None:
        max_shards = int(env_max_shards)

    split = "train" if is_train else "val"
    log.info(f"[{split}] Loading dataset '{dataset_name}' from {object_store} (max_shards={max_shards})")

    dataset_info = get_video_dataset_info(
        dataset_name,
        object_store=object_store,
        dataset_keys=dataset_keys,
        dataset_catalog=dataset_catalog,
        max_shards=max_shards,
    )

    if (
        USE_MEGATRON
        and parallel_state.is_initialized()
        and (
            parallel_state.get_context_parallel_world_size() > 1
            or parallel_state.get_tensor_model_parallel_world_size() > 1
        )
    ):
        distributor_fn = parallel_sync_multi_aspect_ratio.ShardlistMultiAspectRatioParallelSync
    else:
        distributor_fn = ShardlistBasic

    video_data_config = DatasetConfig(
        keys=[],  # keys are defined per dataset
        buffer_size=shuffle_buffer_size,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=distributor_fn(
            shuffle=is_train,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=False,
            is_infinite_loader=is_train,
        ),
        decoders=[
            video_naive_bytes(),
            json_decoder,
        ],
        augmentation=augmentations,
        remove_extension_from_keys=True,
    )

    dataset = SILDataset(
        config=video_data_config,
        handler=wds.warn_and_continue,
        decoder_handler=wds.warn_and_continue,
        detshuffle=False,
        tar_drop_set=tar_drop_set,
        accepted_clip_ids=accepted_clip_ids,
        precomputed_offsets=precomputed_offsets,
        prob_map=prob_map,
        default_prob=default_prob,
        rejection_seed=rejection_seed,
        clip_to_tar=clip_to_tar,
    )

    if max_shards > 0:
        dataset.wdinfo.tar_files = dataset.wdinfo.tar_files[:max_shards]
        dataset.wdinfo.total_key_count = min(
            dataset.wdinfo.total_key_count,
            max_shards * dataset.wdinfo.chunk_size,
        )

    log.info(
        f"[{split}] Dataset '{dataset_name}' ready: "
        f"{len(dataset.wdinfo.tar_files)} shards, {dataset.wdinfo.total_key_count} keys"
    )

    return dataset


class ExtractFramesAndCaptions(Augmentor):
    """Extract frames from a videos."""

    def __init__(
        self,
        camera_order: list[CameraKeyType],
        num_frames: int,
        resolution_hw: tuple[int, int],
        fps_downsample_factor: int,
        caption_probability: dict[str, float],
        camera_view_mapping: dict[CameraKeyType, int],
        camera_caption_key_mapping: dict[CameraKeyType, str],
        camera_video_key_mapping: dict[CameraKeyType, str],
        camera_control_key_mapping: Optional[dict[CameraKeyType, str]] = None,
        add_view_prefix_to_caption: bool = False,
        camera_prefix_mapping: Optional[dict[CameraKeyType, str]] = None,
        single_caption_camera_name: Optional[CameraKeyType] = None,
        window_random_frame_offset_range: Optional[tuple[int, int]] = None,
    ) -> None:
        """Extracts frames and captions from video/metadata dicts.

        Args:
            camera_order: Order of cameras to extract
            num_frames: Number of frames to extract
            resolution_hw: Resolution of the extracted frames
            fps_downsample_factor: FPS downsample factor
            caption_probability: Probability of each caption type in t2w window
            camera_view_mapping: Mapping of camera keys to view indices
            camera_caption_key_mapping: Mapping of camera keys to caption keys
            camera_video_key_mapping: Mapping of camera keys to video keys
            camera_control_key_mapping: Mapping of camera keys to control keys
            add_view_prefix_to_caption: Whether to add caption prefix for all views
            camera_prefix_mapping: Mapping of camera keys to prefixes
            single_caption_camera_name: Name of the camera key to use for single caption conditioning.
                If `add_view_prefix_to_caption` is True, will still provide prefixes for other views.
            window_random_frame_offset_range: Optional range of random offset to add to the start frame of the extracted window.

        Returns:
            data: Dictionary with resized tensors of frames and captions
        """
        super().__init__([], {})
        self.camera_order = camera_order
        self.num_frames = num_frames
        self.resolution_hw = resolution_hw
        self.fps_downsample_factor = fps_downsample_factor
        self.caption_probability = caption_probability
        self.camera_view_mapping = camera_view_mapping
        self.camera_caption_key_mapping = camera_caption_key_mapping
        self.camera_video_key_mapping = camera_video_key_mapping
        self.camera_control_key_mapping = camera_control_key_mapping
        self.add_view_prefix_to_caption = add_view_prefix_to_caption
        self.camera_prefix_mapping = camera_prefix_mapping
        self.single_caption_camera_name = single_caption_camera_name
        self.window_random_frame_offset_range = window_random_frame_offset_range

        if self.add_view_prefix_to_caption and self.camera_prefix_mapping is None:
            raise ValueError("camera_prefix_mapping is required when add_view_prefix_to_caption is True")

        if set(self.camera_caption_key_mapping.keys()) != set(self.camera_video_key_mapping.keys()):
            raise ValueError(
                f"Mismatching keys {set(self.camera_caption_key_mapping.keys())} != {set(self.camera_video_key_mapping.keys())}"
            )
        if self.camera_control_key_mapping is not None:
            if set(self.camera_control_key_mapping.keys()) != set(self.camera_caption_key_mapping.keys()):
                raise ValueError(
                    f"Mismatching keys {set(self.camera_control_key_mapping.keys())} != {set(self.camera_caption_key_mapping.keys())}"
                )
        for camera_name in self.camera_caption_key_mapping.keys():
            if camera_name not in self.camera_view_mapping:
                raise ValueError(f"Camera name {camera_name} not found in camera view mapping")
        if self.single_caption_camera_name and self.single_caption_camera_name not in self.camera_order:
            raise ValueError(
                f"Single caption camera name {self.single_caption_camera_name} must appear in selected cameras"
            )

        if self.window_random_frame_offset_range is not None:
            start_range, end_range = self.window_random_frame_offset_range
            if start_range < 0 or end_range < 0:
                raise ValueError("`window_random_frame_offset_range` must be non-negative")
            if start_range > end_range:
                raise ValueError("`window_random_frame_offset_range` start must be less than end")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Extract frames from a video."""

        chunk_index, extracted_frame_ids, video_fps = None, None, None
        (
            captions,
            multiview_frames,
            multiview_control,
            view_indices,
            view_indices_selection,
            camera_keys_selection,
            original_sizes,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        # extract frames
        random_offset = 0
        if self.window_random_frame_offset_range is not None:
            random_offset = random.randint(*self.window_random_frame_offset_range)

        for camera_name in self.camera_order:
            video_key = self.camera_video_key_mapping[camera_name]
            if self.single_caption_camera_name:
                meta_key = self.camera_caption_key_mapping[self.single_caption_camera_name]
            else:
                meta_key = self.camera_caption_key_mapping[camera_name]

            if meta_key not in data:
                log.error(f"Missing meta key '{meta_key}' in data. Available keys: {list(data.keys())}")
                return None
            t2w_windows = data[meta_key]["t2w_windows"]
            if chunk_index is None:
                chunk_index = random.choice(list(range(len(t2w_windows))))
            try:
                window = t2w_windows[chunk_index]
            except IndexError:
                log.error(f"IndexError: chunk_index: {chunk_index}, len(t2w_windows): {len(t2w_windows)}")
                return None

            # extract caption
            choices = list(self.caption_probability.keys())
            weights = list(self.caption_probability.values())
            caption_style = random.choices(choices, weights=weights)[0]
            caption = ""
            if self.single_caption_camera_name:
                if camera_name == self.single_caption_camera_name:
                    caption = window[caption_style]
            else:
                caption = window[caption_style]

            assert isinstance(caption, str), f"Caption is not a string: {caption}"
            if self.add_view_prefix_to_caption:
                caption = f"{self.camera_prefix_mapping[camera_name]} {caption}"
            captions.append(caption)

            frame_start = window["start_frame"] + random_offset
            frame_end = frame_start + self.num_frames * self.fps_downsample_factor
            frame_indices = list(range(frame_start, frame_end, self.fps_downsample_factor))
            try:
                frames, original_fps, original_hw = self.extract_frames(
                    data[video_key], frame_indices, self.resolution_hw
                )
            except Exception as e:
                log.error(f"Error extracting frames for camera {camera_name}: {e}")
                return None
            assert len(frames) == self.num_frames, f"Expected {self.num_frames} frames, got {len(frames)}"
            multiview_frames.append(frames)

            # check consistency between videos
            if extracted_frame_ids is None:
                extracted_frame_ids = frame_indices
            elif frame_indices != extracted_frame_ids:
                raise ValueError("Extracted frame IDs do not match")

            if video_fps is None:
                video_fps = int(np.round(original_fps))
            elif video_fps != int(np.round(original_fps)):
                raise ValueError("Video FPS does not match")
            original_sizes.append(list(original_hw))

            # extract control frames if available
            if self.camera_control_key_mapping is not None:
                control_key = self.camera_control_key_mapping[camera_name]
                try:
                    control_frames, control_fps, _ = self.extract_frames(
                        data[control_key], frame_indices, self.resolution_hw
                    )
                except Exception as e:
                    log.error(f"Error extracting control frames for camera {camera_name}: {e}")
                    return None
                if len(control_frames) != self.num_frames:
                    raise ValueError(f"Expected {self.num_frames} frames, got {len(control_frames)}")
                if int(np.round(control_fps)) != int(np.round(original_fps)):
                    raise ValueError(f"Control FPS {control_fps} does not match video FPS {original_fps}")
                multiview_control.append(control_frames)

            view_indices.extend([self.camera_view_mapping[camera_name]] * self.num_frames)
            view_indices_selection.append(self.camera_view_mapping[camera_name])
            camera_keys_selection.append(camera_name)

        front_cam_view_idx_sample_position = (
            torch.tensor(self.camera_order.index(self.single_caption_camera_name), dtype=torch.int64)
            if self.single_caption_camera_name
            else None
        )
        if self.single_caption_camera_name and not self.add_view_prefix_to_caption:
            captions = [captions[front_cam_view_idx_sample_position]]

        if video_fps % self.fps_downsample_factor != 0:
            raise ValueError(
                f"Original FPS is not divisible by FPS downsample factor, video_fps: {video_fps}, fps_downsample_factor: {self.fps_downsample_factor}"
            )
        fps = video_fps / self.fps_downsample_factor

        sample = {
            "__key__": data["__key__"],
            "__url__": data["__url__"],
            "video": rearrange(torch.cat(multiview_frames, dim=0), "t c h w -> c t h w"),
            "ai_caption": captions,
            "view_indices": torch.tensor(view_indices, dtype=torch.int64),
            "fps": torch.tensor(fps, dtype=torch.float64),
            "chunk_index": torch.tensor(chunk_index, dtype=torch.int64),
            "frame_indices": torch.tensor(extracted_frame_ids, dtype=torch.int64),
            "num_video_frames_per_view": torch.tensor(len(extracted_frame_ids), dtype=torch.int64),
            "view_indices_selection": torch.tensor(view_indices_selection, dtype=torch.int64),
            "camera_keys_selection": camera_keys_selection,
            "sample_n_views": torch.tensor(len(camera_keys_selection), dtype=torch.int64),
            "padding_mask": torch.zeros((1, *self.resolution_hw), dtype=torch.float32),
            "ref_cam_view_idx_sample_position": torch.tensor(-1, dtype=torch.int64),
            "front_cam_view_idx_sample_position": front_cam_view_idx_sample_position,
            "original_hw": torch.tensor(original_sizes, dtype=torch.int64),
        }
        if self.camera_control_key_mapping is not None:
            sample["control_input_hdmap_bbox"] = rearrange(torch.cat(multiview_control, dim=0), "t c h w -> c t h w")
        return sample

    @staticmethod
    def extract_frames(
        video: bytes, frame_indices: list[int], resolution_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, float, tuple[int, int]]:
        """Extract frames from a video given start and end frame range."""

        from decord import VideoReader

        video_reader = VideoReader(io.BytesIO(video))
        fps = video_reader.get_avg_fps()
        frames = video_reader.get_batch(frame_indices).asnumpy()
        frames = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")
        original_h, original_w = frames.shape[-2:]
        return (
            Resize(resolution_hw, interpolation=InterpolationMode.BILINEAR, antialias=True)(frames),
            fps,
            (original_h, original_w),
        )


def make_augmentations(augmentation_config: AugmentationConfig) -> tuple[dict[str, Augmentor], list[str]]:
    """Make augmentations for multiview video dataset."""

    augmentations = dict()
    if augmentation_config.position_to_camera_mapping is not None:
        augmentations["unpack_metas"] = UnpackMetas(
            position_to_camera_mapping=augmentation_config.position_to_camera_mapping
        )

    augmentations["extract_frames_and_captions"] = ExtractFramesAndCaptions(
        camera_order=augmentation_config.camera_keys,
        num_frames=augmentation_config.num_video_frames,
        resolution_hw=augmentation_config.resolution_hw,
        fps_downsample_factor=augmentation_config.fps_downsample_factor,
        caption_probability=augmentation_config.caption_probability,
        camera_view_mapping=augmentation_config.camera_view_mapping,
        camera_caption_key_mapping=augmentation_config.camera_caption_key_mapping,
        camera_video_key_mapping=augmentation_config.camera_video_key_mapping,
        camera_control_key_mapping=augmentation_config.camera_control_key_mapping,
        add_view_prefix_to_caption=augmentation_config.add_view_prefix_to_caption,
        camera_prefix_mapping=augmentation_config.camera_prefix_mapping,
        single_caption_camera_name=augmentation_config.single_caption_camera_name,
        window_random_frame_offset_range=augmentation_config.window_random_frame_offset_range,
    )

    # define dataset keys to load
    dataset_keys = list(augmentation_config.camera_video_key_mapping.values())
    if augmentation_config.position_to_camera_mapping is not None:
        dataset_keys.append("metas")
    else:
        dataset_keys.extend(augmentation_config.camera_caption_key_mapping.values())
    if augmentation_config.camera_control_key_mapping is not None:
        dataset_keys.extend(augmentation_config.camera_control_key_mapping.values())

    return augmentations, dataset_keys


def _discover_precomputed_offsets(dataset_name: str | None) -> dict | None:
    """Auto-discover precomputed tar offsets from local cache, shared Lustre, or S3."""
    ds = dataset_name.replace("/", "_") if dataset_name else "mads-large"
    exp = os.environ.get("EXP", "")
    local_cache = os.path.join(exp, "wds_index", ds) if exp else os.path.join("wds_index", ds)

    candidates = []
    if exp:
        candidates.append(os.path.join(exp, "wds_index", ds, "tar_offsets.json"))
    candidates.append(os.path.join("wds_index", ds, "tar_offsets.json"))

    for path in candidates:
        if os.path.exists(path):
            log.info(f"Loading precomputed tar offsets: {path}")
            with open(path) as f:
                raw = json.load(f)
            offsets = {k: v for k, v in raw.items() if not k.startswith("_")}
            log.info(f"  {len(offsets)} clip offsets loaded")
            return offsets

    try:
        s3_key = f"wdinfo/{ds}/tar_offsets.json"
        cache_path = os.path.join(local_cache, "tar_offsets.json")
        log.info(f"Offsets not found locally, trying S3: {s3_key}")
        import boto3

        creds_path = "credentials/s3_data.secret"
        if os.path.exists(creds_path):
            with open(creds_path) as f:
                sc = json.load(f)
            s3c = boto3.client(
                "s3",
                endpoint_url=sc.get("endpoint_url", "https://pdx.s8k.io"),
                aws_access_key_id=sc["aws_access_key_id"],
                aws_secret_access_key=sc["aws_secret_access_key"],
            )
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            import tempfile

            tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(cache_path))
            os.close(tmp_fd)
            try:
                s3c.download_file("webdataset", s3_key, tmp_path)
                os.rename(tmp_path, cache_path)
            except Exception:
                os.unlink(tmp_path)
                raise
            log.info(f"Downloaded offsets from S3 -> {cache_path}")
            with open(cache_path) as f:
                raw = json.load(f)
            offsets = {k: v for k, v in raw.items() if not k.startswith("_")}
            log.info(f"  {len(offsets)} clip offsets loaded")
            return offsets
    except Exception as e:
        log.warning(f"Could not load offsets from S3: {e}")

    return None


def get_multiview_video_loader(
    *,
    dataset_name: str,
    is_train: bool,
    object_store: Literal["gcs", "s3", "pdx"] = "s3",
    augmentation_config: AugmentationConfig = AugmentationConfig(),
    batch_size: int = 1,
    num_workers: int = 4,
    prefetch_factor: int | None = 1,
    max_shards: int = 0,
    shuffle_buffer_size: int = 1,
    **kwargs: Any,
):
    """Get video loader for alpamayo multiview dataset.

    Args:
        dataset_name: Name of dataset to use
        is_train: Whether this is for training
        object_store: Object store to use ("gcs" or "s3")
        augmentation_config: Augmentation configuration
        batch_size: Batch size
        num_workers: Number of data loading workers
        prefetch_factor: Prefetch factor
        max_shards: If > 0, limit the number of tar shards loaded (useful for fast debug runs)
        **kwargs: Additional kwargs to tolerate `dataloaders` from inheritance
    """

    # make augmentations
    augmentations, dataset_keys = make_augmentations(augmentation_config)

    # --- Rejection sampling setup ---
    rejection_probs_path = augmentation_config.rejection_probs_path
    rejection_default_prob = augmentation_config.rejection_default_prob
    rejection_seed = augmentation_config.rejection_seed

    tar_drop_set: set[str] | None = None
    accepted_clip_ids: set[str] | None = None
    precomputed_offsets: dict | None = None
    prob_map: dict[str, float] | None = None
    tracked_clip_ids: set[str] | None = None
    _clip_to_tar: dict[str, str] | None = None

    if rejection_probs_path:
        _probs_data = None
        if os.path.exists(rejection_probs_path):
            with open(rejection_probs_path) as _f:
                _probs_data = json.load(_f)

        tar_drop_set = load_tar_filter_from_probs(rejection_probs_path, _parsed=_probs_data)

        if _probs_data is not None:
            clip_probs = {
                str(k): float(v)
                for k, v in _probs_data.items()
                if not k.startswith("_") and isinstance(v, (int, float))
            }
            tracked_clip_ids = set(clip_probs.keys())
            has_smooth = any(0 < v < 1 for v in clip_probs.values())

            if has_smooth:
                # -- Smooth mode: per-epoch pre-rolled accept set --
                # At each shard-list cycle, selective_tar_samples pre-rolls
                # which clips to accept and uses clip_to_tar for zero-waste
                # tar targeting.  No RejectionSampler augmentor needed.
                prob_map = clip_probs
                _clip_to_tar = _probs_data.get("_tar_filter", {}).get("clip_to_tar", {})
                log.info(
                    f"Smooth rejection: {len(prob_map)} clip probs, "
                    f"{len(_clip_to_tar)} clip-to-tar mappings, "
                    f"default_prob={rejection_default_prob}"
                )
            elif rejection_default_prob == 0.0:
                # -- Hard mode: deterministic accept set --
                accepted_clip_ids = {k for k, v in clip_probs.items() if v > 0}
                log.info(f"Selective tar reader: {len(accepted_clip_ids)} accepted clip IDs loaded")
            else:
                # Probs exist but all are 0 or 1 with default_prob > 0 — use
                # the RejectionSampler augmentor (original behaviour).
                from collections import OrderedDict

                augmentations = OrderedDict(
                    [
                        (
                            "rejection_sampler",
                            RejectionSampler(
                                probs_path=rejection_probs_path,
                                default_prob=rejection_default_prob,
                                seed=rejection_seed,
                            ),
                        )
                    ]
                    + list(augmentations.items())
                )

        # Auto-discover precomputed tar offsets (for selective reader paths)
        if accepted_clip_ids is not None or prob_map is not None:
            precomputed_offsets = _discover_precomputed_offsets(dataset_name)

    # get dataloader
    loader = get_cached_replay_dataloader(
        dataset=get_multiview_dataset(
            is_train=is_train,
            object_store=object_store,
            dataset_name=dataset_name,
            dataset_keys=dataset_keys,
            dataset_catalog=DEFAULT_CATALOG,
            augmentations=augmentations,
            max_shards=max_shards,
            shuffle_buffer_size=shuffle_buffer_size if is_train else 1,
            tar_drop_set=tar_drop_set,
            accepted_clip_ids=accepted_clip_ids,
            precomputed_offsets=precomputed_offsets,
            prob_map=prob_map,
            default_prob=rejection_default_prob,
            rejection_seed=rejection_seed,
            clip_to_tar=_clip_to_tar,
        ),
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=None,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,
        pin_memory=False,
        collate_fn=collate_fn,
        cache_replay_name=f"video_dataloader_{'train' if is_train else 'val'}",
    )

    if augmentation_config.rejection_histogram and tracked_clip_ids:
        loader = RejectionHistogramLogger(
            loader,
            tracked_clip_ids=tracked_clip_ids,
            log_interval=augmentation_config.rejection_histogram_interval,
        )
        log.info(
            f"RejectionHistogramLogger enabled: tracking {len(tracked_clip_ids)} clip IDs, "
            f"logging every {augmentation_config.rejection_histogram_interval} steps"
        )

    return loader
