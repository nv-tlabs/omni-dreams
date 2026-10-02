# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Utilities for evaluating AdaLN modulation once per latent frame."""

from collections.abc import Callable

import torch
from torch.distributed import ProcessGroup

from omnidreams._src.imaginaire.utils.context_parallel import split_inputs_cp


def make_token_frame_indices(
    num_frames: int,
    tokens_per_frame: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return an ``[L]`` map from flattened tokens to latent frames.

    ``L = num_frames * tokens_per_frame`` for the original ``[T, H, W]``
    token layout.
    """

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if tokens_per_frame <= 0:
        raise ValueError(f"tokens_per_frame must be positive, got {tokens_per_frame}")
    return torch.arange(num_frames, device=device, dtype=torch.long).repeat_interleave(tokens_per_frame)


def prepare_adaln_block_inputs(
    frame_embedding: torch.Tensor,
    frame_adaln_lora: torch.Tensor | None,
    *,
    num_frames: int,
    tokens_per_frame: int,
    framewise: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Prepare AdaLN inputs and the optional token-to-frame gather map.

    Framewise mode keeps ``[B, T, *]`` inputs compact and returns an ``[L]``
    gather map. Legacy mode materializes ``[B, L, *]`` inputs and returns no
    map.
    """

    if framewise:
        token_frame_indices = make_token_frame_indices(
            num_frames,
            tokens_per_frame,
            device=frame_embedding.device,
        )
        return frame_embedding, frame_adaln_lora, token_frame_indices

    token_embedding = torch.repeat_interleave(frame_embedding, tokens_per_frame, dim=1)
    token_adaln_lora = (
        None
        if frame_adaln_lora is None
        else torch.repeat_interleave(frame_adaln_lora, tokens_per_frame, dim=1)
    )
    return token_embedding, token_adaln_lora, None


def shard_adaln_block_inputs(
    embedding: torch.Tensor,
    adaln_lora: torch.Tensor | None,
    token_frame_indices: torch.Tensor | None,
    *,
    cp_group: ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Shard the sequence-bearing AdaLN inputs for contiguous CP.

    Legacy inputs carry a token dimension and are sharded directly. Framewise
    inputs remain replicated because only their token-to-frame gather map has a
    sequence dimension.
    """

    if token_frame_indices is not None:
        token_frame_indices = split_inputs_cp(token_frame_indices, seq_dim=0, cp_group=cp_group)
        return embedding, adaln_lora, token_frame_indices

    embedding = split_inputs_cp(embedding, seq_dim=1, cp_group=cp_group)
    if adaln_lora is not None:
        adaln_lora = split_inputs_cp(adaln_lora, seq_dim=1, cp_group=cp_group)
    return embedding, adaln_lora, None


def apply_adaln_modulation(
    module: Callable[[torch.Tensor], torch.Tensor],
    embedding: torch.Tensor,
    adaln_lora: torch.Tensor | None,
    *,
    token_frame_indices: torch.Tensor | None,
    sequence_length: int,
) -> torch.Tensor:
    """Evaluate an AdaLN MLP compactly and expand it to the local sequence.

    Shapes:
        - Legacy: ``embedding`` is ``[B, L, D]``, ``adaln_lora`` is
          ``[B, L, 3D]``, and ``token_frame_indices`` is ``None``.
        - Framewise: ``embedding`` is ``[B, T, D]``, ``adaln_lora`` is
          ``[B, T, 3D]``, and ``token_frame_indices`` is ``[L]``.
        - The returned modulation is always ``[B, L, 3D]``.

    The framewise gather supports any local CP token ordering, including
    shards that begin or end in the middle of a frame.
    """

    if embedding.ndim != 3:
        raise ValueError(f"AdaLN embedding must have shape [B, N, D], got {tuple(embedding.shape)}")
    modulation = module(embedding)
    if adaln_lora is not None:
        if adaln_lora.shape != modulation.shape:
            raise ValueError(
                "AdaLN-LoRA tensor must match modulation shape, "
                f"got {tuple(adaln_lora.shape)} and {tuple(modulation.shape)}"
            )
        modulation = modulation + adaln_lora

    if token_frame_indices is None:
        if modulation.shape[1] != sequence_length:
            raise ValueError(
                "Token-layout AdaLN modulation must match the local sequence length, "
                f"got {modulation.shape[1]} and {sequence_length}"
            )
        return modulation

    if token_frame_indices.ndim != 1:
        raise ValueError(f"token_frame_indices must be one-dimensional, got shape {tuple(token_frame_indices.shape)}")
    if token_frame_indices.dtype != torch.long:
        raise TypeError(f"token_frame_indices must use torch.long, got {token_frame_indices.dtype}")
    if token_frame_indices.device != modulation.device:
        raise ValueError(
            "token_frame_indices and modulation must be on the same device, "
            f"got {token_frame_indices.device} and {modulation.device}"
        )
    if token_frame_indices.numel() != sequence_length:
        raise ValueError(
            "token_frame_indices must match the local sequence length, "
            f"got {token_frame_indices.numel()} and {sequence_length}"
        )
    return modulation.index_select(1, token_frame_indices)
