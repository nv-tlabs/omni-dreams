# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Dict, List

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from omnidreams._src.predict2.datasets.cached_replay_dataloader import CachedReplayDataLoader, concatenate_batches


class DummyDataset(Dataset):
    """A dummy dataset that returns a dictionary with a 'videos' tensor of shape (1, 3, frames_per_batch).

    Each item is a tensor with sequential values and a batch-specific offset for uniqueness.
    """

    def __init__(self, num_batches: int, frames_per_batch: int) -> None:
        self.num_batches = num_batches
        self.frames_per_batch = frames_per_batch

    def __len__(self) -> int:
        return 999

    def __getitem__(self, index: int) -> Dict:
        # Create a tensor with shape (1, 3, frames_per_batch) with sequential values.
        video = torch.arange(self.frames_per_batch, dtype=torch.float32).unsqueeze(0).repeat(3, 1)
        video = video + index * 1000  # Offset each batch for uniqueness.
        return {"videos": video}


def temporal_slice_augmentation(batch: Dict) -> List[Dict]:
    """Augmentation function that creates multiple temporal slice variants.

    This simulates the original CachedReplayDataLoader behavior but as an external function.
    """
    videos = batch["videos"]
    total_frames = videos.shape[2]
    num_video_frames = 20  # Number of frames per slice
    replay_num = 5  # Number of slices to create

    if total_frames < num_video_frames:
        raise ValueError(f"Total frames ({total_frames}) is less than required frames ({num_video_frames}).")

    # Compute evenly spaced starting offsets along the T dimension
    if replay_num == 1:
        offsets = [0]
    else:
        max_start = total_frames - num_video_frames
        offsets = [int(round(i * max_start / (replay_num - 1))) for i in range(replay_num)]

    # Create clones with different temporal slices
    augmented_batches = []
    for offset in offsets:
        clone = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Slice the video tensor along T dimension (index 2)
        clone["videos"] = videos[:, :, offset : offset + num_video_frames, ...]
        augmented_batches.append(clone)

    return augmented_batches


def brightness_augmentation(batch: Dict) -> List[Dict]:
    """Augmentation function that creates variants with different brightness levels."""
    videos = batch["videos"]
    scales = [0.8, 1.0, 1.2]  # Brightness adjustment factors

    augmented_batches = []
    for scale in scales:
        clone = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        clone["videos"] = videos * scale
        augmented_batches.append(clone)

    return augmented_batches


@pytest.mark.L1
def test_augmentation():
    """Test that the dataloader correctly uses an external augmentation function."""
    total_frames = 100
    dataset = DummyDataset(num_batches=1, frames_per_batch=total_frames)
    data_loader = DataLoader(dataset, batch_size=1)

    cached_loader = CachedReplayDataLoader(
        data_loader,
        cache_size=10,
        cache_augmentation_fn=temporal_slice_augmentation,
    )

    # Collect augmented batches
    augmented_batches = []
    for batch in cached_loader:
        augmented_batches.append(batch)
        if len(augmented_batches) >= 5:  # We expect 5 variants from our augmentation function
            break

    # Verify each augmented batch has the correct shape
    for batch in augmented_batches:
        video = batch["videos"]
        # Expected shape after temporal slicing (1, 3, 20)
        assert video.shape[0] == 1
        assert video.shape[1] == 3
        assert video.shape[2] == 20

    # Check that the variants cover different parts of the video
    offsets = [int(video[0, 0, 0].item() % 1000) for batch in augmented_batches for video in [batch["videos"]]]
    assert len(set(offsets)) > 1  # Should have different starting offsets

    cached_loader.close()


@pytest.mark.L1
def test_batch_concatenation():
    """Test that batch concatenation works correctly."""
    total_frames = 80
    concat_size = 2

    dataset = DummyDataset(num_batches=10, frames_per_batch=total_frames)
    data_loader = DataLoader(dataset, batch_size=1)

    cached_loader = CachedReplayDataLoader(
        data_loader,
        cache_size=10,
        cache_augmentation_fn=brightness_augmentation,
        concat_size=concat_size,
    )

    # Collect a few batches and check their shapes
    for i, batch in enumerate(cached_loader):
        # Should have batch_size batches concatenated along dim 0
        assert batch["videos"].shape[0] == concat_size

        if i >= 3:
            break

    cached_loader.close()


@pytest.mark.L1
def test_external_concatenate_batches():
    """Test the concatenate_batches function separately."""
    # Create sample batches with tensors
    batch1 = {"videos": torch.ones(1, 3, 10), "labels": torch.tensor([1])}
    batch2 = {"videos": torch.zeros(1, 3, 10), "labels": torch.tensor([0])}

    # Test concatenation
    result = concatenate_batches(1, [batch1, batch2])
    assert len(result) == 2

    result = concatenate_batches(2, [batch1, batch2])
    assert len(result) == 1
    assert result[0]["videos"].shape[0] == 2
    assert result[0]["labels"].shape[0] == 2
