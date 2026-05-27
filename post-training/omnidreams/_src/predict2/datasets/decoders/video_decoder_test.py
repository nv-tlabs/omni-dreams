# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np
import pytest

from omnidreams._src.predict2.datasets.decoders.video_decoder import (
    basic_check_on_inputs,
    get_frame_indices_w_lowered_fps,
    sample_chunk_index_from_chunked_video,
)


@pytest.fixture
def set_random_seed():
    """Fixture to ensure reproducible random numbers."""
    np.random.seed(42)
    yield
    np.random.seed(None)


@pytest.mark.L0
def test_basic_functionality(set_random_seed):
    """Test basic functionality with valid inputs."""
    indices, fps = get_frame_indices_w_lowered_fps(
        n_video_frames=100, video_fps=30, min_fps_thres=4, max_fps_thres=30, n_target_frames=5
    )

    assert len(indices) == 5
    assert all(0 <= idx < 100 for idx in indices)
    assert 4 <= fps <= 30
    assert indices == sorted(indices)  # Ensure indices are monotonically increasing


@pytest.mark.L0
def test_sequence_spacing():
    """Test that frame indices are evenly spaced."""
    indices, _ = get_frame_indices_w_lowered_fps(
        n_video_frames=100, video_fps=30, min_fps_thres=4, max_fps_thres=30, n_target_frames=5
    )

    differences = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
    assert len(set(differences)) == 1  # All differences should be equal


@pytest.mark.L0
def test_fps_bounds():
    """Test that resulting FPS is within specified bounds."""
    _, fps = get_frame_indices_w_lowered_fps(
        n_video_frames=100, video_fps=30, min_fps_thres=4, max_fps_thres=30, n_target_frames=5
    )

    assert 4 <= fps <= 30


@pytest.mark.parametrize(
    "invalid_input",
    [
        {"n_video_frames": 0},
        {"video_fps": 0},
        {"min_fps_thres": 0},
        {"max_fps_thres": 3, "min_fps_thres": 4},
        {"n_target_frames": 1},
        {"n_target_frames": 101, "n_video_frames": 100},
    ],
)
@pytest.mark.L0
def test_invalid_inputs(invalid_input):
    """Test that invalid inputs raise appropriate errors."""
    default_args = {
        "n_video_frames": 100,
        "video_fps": 30,
        "min_fps_thres": 4,
        "max_fps_thres": 30,
        "n_target_frames": 5,
    }

    args = {**default_args, **invalid_input}
    message = basic_check_on_inputs(**args)
    assert message != "success"  # assert raise error message


@pytest.mark.L0
def test_stride_selection_bias(set_random_seed):
    """Test that larger strides (lower FPS) are selected more frequently."""
    results = []
    for _ in range(100):
        _, fps = get_frame_indices_w_lowered_fps(
            n_video_frames=100, video_fps=30, min_fps_thres=4, max_fps_thres=30, n_target_frames=5
        )
        results.append(fps)

    # Check that lower FPS (larger strides) are selected more often
    lower_fps_count = sum(1 for fps in results if fps < (4 + 30) / 2)
    assert lower_fps_count > len(results) * 0.6  # Should be selected roughly 75% of the time


@pytest.mark.L0
def test_extreme_case():
    """Test with minimal valid input values."""
    indices, fps = get_frame_indices_w_lowered_fps(
        n_video_frames=5, video_fps=8, min_fps_thres=4, max_fps_thres=8, n_target_frames=2
    )

    assert len(indices) == 2
    assert all(0 <= idx < 5 for idx in indices)
    assert 4 <= fps <= 8


@pytest.mark.L0
def test_no_valid_strides():
    """Test that appropriate error is raised when no valid strides exist."""
    with pytest.raises(ValueError) as exc_info:
        get_frame_indices_w_lowered_fps(
            n_video_frames=10, video_fps=30, min_fps_thres=25, max_fps_thres=29, n_target_frames=9
        )
    assert "No valid stride options available" in str(exc_info.value)


@pytest.mark.L0
def test_sample_chunk_index_from_chunked_video():
    sampled_chunk_index, n_frames_in_chunk, message = sample_chunk_index_from_chunked_video(
        n_video_frames=383,
        n_target_frames=4,
        chunk_size=256,
    )
    assert n_frames_in_chunk == 383

    n_frames_in_chunk_list = set()
    for _ in range(10):
        sampled_chunk_index, n_frames_in_chunk, message = sample_chunk_index_from_chunked_video(
            n_video_frames=641,
            n_target_frames=4,
            chunk_size=256,
        )
        n_frames_in_chunk_list.add(n_frames_in_chunk)
    assert n_frames_in_chunk_list == {256, 129}

    message = sample_chunk_index_from_chunked_video(
        n_video_frames=4,
        n_target_frames=121,
        chunk_size=256,
    )

    assert message != "success"
