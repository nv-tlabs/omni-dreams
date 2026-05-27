# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import torch
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup, all_gather, all_gather_into_tensor, get_process_group_ranks

from omnidreams._src.imaginaire.utils import log


def _all_gather_along_dim(x: Tensor, dim: int, group: ProcessGroup) -> Tensor:
    """
    Gather tensors from all ranks along a specific dimension.

    Args:
        x: Input tensor to gather.
        dim: Dimension along which to gather.
        group: Process group for communication.

    Returns:
        Gathered tensor with dim multiplied by world_size.
    """
    world_size = group.size()
    if world_size == 1:
        return x

    # Move the gather dimension to the front for efficient gathering
    # all_gather_into_tensor gathers along dim 0
    # Note: movedim creates a non-contiguous view, so we must call contiguous() after
    x_permuted = x.movedim(dim, 0).contiguous()
    gathered_shape = list(x_permuted.shape)
    gathered_shape[0] *= world_size
    gathered = torch.empty(gathered_shape, dtype=x.dtype, device=x.device)

    all_gather_into_tensor(gathered, x_permuted, group=group)

    # Move dimension back
    gathered = gathered.movedim(0, dim)
    return gathered


def _split_along_dim(x: Tensor, dim: int, rank: int, world_size: int) -> Tensor:
    """
    Split tensor along a dimension and return the chunk for the given rank.

    Args:
        x: Input tensor to split.
        dim: Dimension along which to split.
        rank: Current rank in the process group.
        world_size: Total number of ranks.

    Returns:
        The chunk of the tensor corresponding to the given rank.
    """
    dim_size = x.shape[dim]
    assert dim_size % world_size == 0, (
        f"Dimension {dim} with size {dim_size} must be divisible by world_size {world_size}"
    )
    chunk_size = dim_size // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size

    # Build slice tuple
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(start_idx, end_idx)
    # Note: slicing may create a non-contiguous view, make it contiguous for downstream ops
    return x[tuple(slices)].contiguous()


class _GatherVSplitL(torch.autograd.Function):
    """
    Autograd function to gather along V dimension (dim=1) and split along L dimension (dim=3).

    Forward: (B, V_local, T, L, D) -> (B, V_global, T, L_local, D)
    Backward: (B, V_global, T, L_local, D) -> (B, V_local, T, L, D)
    """

    @staticmethod
    def forward(ctx, x: Tensor, group: ProcessGroup) -> Tensor:
        ctx.group = group
        ctx.v_local = x.shape[1]

        world_size = group.size()
        rank = group.rank()

        # Gather along V (dim=1)
        gathered = _all_gather_along_dim(x, dim=1, group=group)

        # Split along L (dim=3)
        output = _split_along_dim(gathered, dim=3, rank=rank, world_size=world_size)

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        group = ctx.group
        world_size = group.size()
        rank = group.rank()

        # Backward is the inverse operation: gather along L, split along V
        # Gather along L (dim=3)
        gathered = _all_gather_along_dim(grad_output, dim=3, group=group)

        # Split along V (dim=1)
        grad_input = _split_along_dim(gathered, dim=1, rank=rank, world_size=world_size)

        return grad_input, None


class _GatherLSplitV(torch.autograd.Function):
    """
    Autograd function to gather along L dimension (dim=3) and split along V dimension (dim=1).

    Forward: (B, V_global, T, L_local, D) -> (B, V_local, T, L, D)
    Backward: (B, V_local, T, L, D) -> (B, V_global, T, L_local, D)
    """

    @staticmethod
    def forward(ctx, x: Tensor, group: ProcessGroup) -> Tensor:
        ctx.group = group
        ctx.l_local = x.shape[3]

        world_size = group.size()
        rank = group.rank()

        # Gather along L (dim=3)
        gathered = _all_gather_along_dim(x, dim=3, group=group)

        # Split along V (dim=1)
        output = _split_along_dim(gathered, dim=1, rank=rank, world_size=world_size)

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        group = ctx.group
        world_size = group.size()
        rank = group.rank()

        # Backward is the inverse operation: gather along V, split along L
        # Gather along V (dim=1)
        gathered = _all_gather_along_dim(grad_output, dim=1, group=group)

        # Split along L (dim=3)
        grad_input = _split_along_dim(gathered, dim=3, rank=rank, world_size=world_size)

        return grad_input, None


def gather_v_split_l(x: Tensor, group: ProcessGroup) -> Tensor:
    """
    Gather tensor along the V dimension (dim=1) across devices, then split along L dimension (dim=3).

    This function is used to redistribute tokens from V-parallel to L-parallel layout.
    Gradients are preserved through this operation.

    Args:
        x: Input tensor of shape (B, V_local, T, L, D) where V is split across devices.
        group: Process group for communication.

    Returns:
        Output tensor of shape (B, V_global, T, L_local, D) where L is split across devices.

    Raises:
        AssertionError: If L dimension is not divisible by the process group size.

    Example:
        With 4 GPUs and input shape (2, 3, 5, 16, 64):
        - V_local = 3, so V_global = 12
        - L = 16, so L_local = 4
        - Output shape: (2, 12, 5, 4, 64)
    """
    world_size = group.size()
    L = x.shape[3]
    assert L % world_size == 0, f"L dimension ({L}) must be divisible by process group size ({world_size})"
    return _GatherVSplitL.apply(x, group)  # type: ignore[return-value]


def gather_l_split_v(x: Tensor, group: ProcessGroup) -> Tensor:
    """
    Gather tensor along the L dimension (dim=3) across devices, then split along V dimension (dim=1).

    This function is the inverse of gather_v_split_l. It redistributes tokens from L-parallel
    to V-parallel layout. Gradients are preserved through this operation.

    Args:
        x: Input tensor of shape (B, V_global, T, L_local, D) where L is split across devices.
        group: Process group for communication.

    Returns:
        Output tensor of shape (B, V_local, T, L, D) where V is split across devices.

    Raises:
        AssertionError: If V dimension is not divisible by the process group size.

    Example:
        With 4 GPUs and input shape (2, 12, 5, 4, 64):
        - L_local = 4, so L_global = 16
        - V_global = 12, so V_local = 3
        - Output shape: (2, 3, 5, 16, 64)
    """
    world_size = group.size()
    V = x.shape[1]
    assert V % world_size == 0, f"V dimension ({V}) must be divisible by process group size ({world_size})"
    return _GatherLSplitV.apply(x, group)  # type: ignore[return-value]


def materialize_split_pattern(x_shape: tuple, dim_names: list, seq_dim: tuple, split_factors: tuple) -> tuple[str, str]:
    """
    Materialize the split pattern for the given sequence dimensions and split factors.

    Returns:
        A tuple of (pattern_from, pattern_to) where:
        - pattern_from: the pattern with merged dimensions (e.g., "d0 d1 d2 d3 d4")
        - pattern_to: the pattern with split dimensions (e.g., "d0 d1 chunk1 split1 chunk2 split2 d4")
    """
    rearrange_pattern_from = []
    rearrange_pattern_to = []

    for orig_dim_idx in range(len(x_shape)):
        if orig_dim_idx in seq_dim:
            seq_idx = seq_dim.index(orig_dim_idx)
            split_factor = split_factors[seq_idx]
            if split_factor > 1:
                # Split this dimension
                chunk_name = f"chunk{seq_idx}"
                split_name = f"split{seq_idx}"
                # In the "from" pattern, use parentheses to indicate this dimension will be split
                rearrange_pattern_from.append(f"({chunk_name} {split_name})")
                # In the "to" pattern, expand without parentheses
                rearrange_pattern_to.append(f"{chunk_name} {split_name}")
            else:
                # No split needed
                rearrange_pattern_from.append(dim_names[orig_dim_idx])
                rearrange_pattern_to.append(dim_names[orig_dim_idx])
        else:
            # Not a sequence dimension
            rearrange_pattern_from.append(dim_names[orig_dim_idx])
            rearrange_pattern_to.append(dim_names[orig_dim_idx])

    rearrange_pattern_from_str = " ".join(rearrange_pattern_from)
    rearrange_pattern_to_str = " ".join(rearrange_pattern_to)
    return rearrange_pattern_from_str, rearrange_pattern_to_str


def build_index_pattern(x_shape: tuple, seq_dim: tuple, split_factors: tuple, rank_positions: tuple) -> tuple:
    index_tuple = []
    # result_dim_idx = 0
    for orig_dim_idx in range(len(x_shape)):
        if orig_dim_idx in seq_dim:
            seq_idx = seq_dim.index(orig_dim_idx)
            split_factor = split_factors[seq_idx]
            if split_factor > 1:
                # Select the chunk for this dimension
                index_tuple.append(slice(None))  # Keep all chunks
                index_tuple.append(rank_positions[seq_idx])  # Select specific split
                # result_dim_idx += 2
            else:
                index_tuple.append(slice(None))
                # result_dim_idx += 1
        else:
            index_tuple.append(slice(None))
            # result_dim_idx += 1
    return tuple(index_tuple)


def split_inputs_cp_multidim(
    x: Tensor, seq_dim: tuple, maximum_split_factor: tuple, cp_group: ProcessGroup
) -> tuple[Tensor, tuple, tuple]:
    """
    Split input tensor along multiple dimensions for context parallelism.

    This function divides the input tensor into equal parts along the specified
    sequence dimension, based on the number of ranks in the context parallelism group.
    It then selects the part corresponding to the current rank.

    Args:
        x: Input tensor to be split.
        seq_dim: The dimensions along which to split the input (sequence dimensions).
        maximum_split_factor: The maximum split factor for each dimension. For example, if the input shape is (B, T, H, W, C) and the sequence dimensions are (1, 2, 3), and the maximum split factor is (2, 2, 2), then the split tensor will be (B, T/2, H/2, W/2, C) if cp_group is 8. If prod(maximum_split_factor) is not equal to the number of ranks in the context parallelism group, the right most dimension will be split first.
        cp_group: The process group for context parallelism.

    Returns:
        A slice of the input tensor corresponding to the current rank, and the corresponding split factors and rank positions.

    Raises:
        AssertionError: If the sequence dimension is not divisible by the number of ranks.
    """
    cp_ranks = get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)

    # Verify input dimensions match
    assert len(seq_dim) == len(maximum_split_factor), (
        f"seq_dim length {len(seq_dim)} must match maximum_split_factor length {len(maximum_split_factor)}"
    )

    # Calculate actual split factors for each dimension (split right-most dimension first)
    split_factors = [1] * len(seq_dim)
    remaining_cp_size = cp_size

    # Iterate from left to right (leftmost dimension first)
    for i in range(len(seq_dim)):
        dim_idx = seq_dim[i]
        max_factor = maximum_split_factor[i]
        dim_size = x.shape[dim_idx]

        # Determine the split factor for this dimension
        # It should be the minimum of: max_factor, remaining_cp_size, and what divides evenly
        actual_factor = min(max_factor, remaining_cp_size)

        # Find the largest factor that divides both the dimension size and is <= actual_factor
        while actual_factor > 1 and (dim_size % actual_factor != 0 or remaining_cp_size % actual_factor != 0):
            actual_factor -= 1

        split_factors[i] = actual_factor
        remaining_cp_size //= actual_factor
    log.info(f"[rank {cp_group.rank()}] split_factors: {split_factors}")
    # Verify that we've split across all ranks
    total_split = 1
    for factor in split_factors:
        total_split *= factor
    assert total_split == cp_size, (
        f"Product of split factors {split_factors} (={total_split}) must equal cp_size {cp_size}"
    )

    # Calculate the rank's position in the multi-dimensional grid
    rank = cp_group.rank()
    rank_positions = []

    # Decompose rank into positions along each dimension (right to left)
    # e.g. if rank is 7 and split_factors is (2, 2, 2), then rank_positions will be [1, 1, 1]
    remaining_rank = rank
    for i in range(len(seq_dim) - 1, -1, -1):
        rank_positions.insert(0, remaining_rank % split_factors[i])
        remaining_rank //= split_factors[i]

    # Build rearrange pattern to split dimensions
    # For each dimension, we split it into (size/split_factor, split_factor)
    # Example: (B, T, H, W, C) with seq_dim=(1,2,3) and split_factors=(2,2,2)
    # becomes (B, T/2, 2, H/2, 2, W/2, 2, C)

    # Create dimension names for rearrange
    dim_names = [f"d{i}" for i in range(len(x.shape))]
    rearrange_pattern_from, rearrange_pattern_to_str = materialize_split_pattern(
        x.shape, dim_names, seq_dim, tuple(split_factors)
    )
    log.info(
        f"[rank {cp_group.rank()}] rearrange_pattern_from: {rearrange_pattern_from}, rearrange_pattern_to_str: {rearrange_pattern_to_str}"
    )
    # Build the pattern with actual sizes
    rearrange_dict = {}
    for i, dim_idx in enumerate(seq_dim):
        if split_factors[i] > 1:
            chunk_size = x.shape[dim_idx] // split_factors[i]
            rearrange_dict[f"chunk{i}"] = chunk_size
            rearrange_dict[f"split{i}"] = split_factors[i]

    # Rearrange the tensor
    result = rearrange(x, f"{rearrange_pattern_from} -> {rearrange_pattern_to_str}", **rearrange_dict)

    # Index to select the appropriate chunk for this rank
    # Build the indexing tuple
    index_tuple = build_index_pattern(x.shape, seq_dim, tuple(split_factors), tuple(rank_positions))
    log.info(f"[rank {cp_group.rank()}] index_tuple: {index_tuple}")
    result = result[index_tuple]

    return result, tuple(split_factors), tuple(rank_positions)


def cat_outputs_cp_multidim(
    x: Tensor,
    seq_dim: tuple,
    split_factors: tuple,
    x_original_shape: tuple,
    cp_group: ProcessGroup,
    preserve_grad: bool = False,
) -> Tensor:
    """
    Concatenate outputs from different ranks in the context parallelism group along multiple dimensions.

    This function reverses the operation done by split_inputs_cp_multidim by gathering tensors from all ranks
    and reconstructing the original tensor.

    Args:
        x: Input tensor to be concatenated (split tensor from current rank).
        seq_dim: The dimensions along which the tensor was split (sequence dimensions).
        split_factors: The split factors used for each dimension.
        x_original_shape: The original shape of the input tensor.
        cp_group: The process group for context parallelism.
        preserve_grad: Whether to preserve the gradient of the input tensor.
    Returns:
        The reconstructed tensor with all splits concatenated back together.

    Raises:
        RuntimeError: If the gather operation fails.
    """
    cp_size = cp_group.size()
    x = x.contiguous()
    # Gather all tensors from all ranks
    gathered_tensors = [torch.zeros_like(x) for _ in range(cp_size)]
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to gather tensors: {e}")

    if preserve_grad:
        gathered_tensors[cp_group.rank()] = x

    # Calculate the full shape after placing all chunks in the rearranged layout
    # For each split dimension, we need to add the split factor dimension
    full_shape = list(x.shape)
    inserted_dims = 0
    for i, dim_idx in enumerate(seq_dim):
        if split_factors[i] > 1:
            # Insert the split dimension
            adjusted_idx = dim_idx + inserted_dims
            # The chunk dimension is already in x.shape
            # We need to insert the split dimension after it
            full_shape.insert(adjusted_idx + 1, split_factors[i])
            # Update full_shape[adjusted_idx] to be the full size (chunk_size * split_factor)
            # But we already have chunk_size in x.shape, so we just insert split_factor
            inserted_dims += 1

    # Create the full tensor with split dimensions
    result = torch.zeros(full_shape, dtype=x.dtype, device=x.device)

    # Place each gathered tensor at the correct position
    for rank in range(cp_size):
        # Compute rank_positions for this rank
        rank_pos = []
        remaining_rank = rank
        for i in range(len(seq_dim) - 1, -1, -1):
            rank_pos.insert(0, remaining_rank % split_factors[i])
            remaining_rank //= split_factors[i]

        # Build the index tuple for placing this tensor
        index_tuple = []
        dim_counter = 0
        for orig_dim_idx in range(len(x.shape)):
            # Check if this dimension corresponds to a split dimension
            # Need to find which seq_dim this corresponds to
            matched = False
            for seq_idx, s_dim in enumerate(seq_dim):
                if split_factors[seq_idx] > 1:
                    # This seq_dim was split, so it occupies 2 dimensions in result
                    # Check if orig_dim_idx matches (accounting for previous splits)
                    num_prev_splits = sum(1 for j in range(seq_idx) if split_factors[j] > 1)
                    adjusted_s_dim = s_dim + num_prev_splits
                    if dim_counter == adjusted_s_dim:
                        # This is a chunk dimension
                        index_tuple.append(slice(None))
                        index_tuple.append(rank_pos[seq_idx])
                        matched = True
                        dim_counter += 2
                        break

            if not matched:
                index_tuple.append(slice(None))
                dim_counter += 1
        log.info(f"gathering [rank {rank}] index_tuple: {index_tuple}")
        # Place the tensor
        result[tuple(index_tuple)] = gathered_tensors[rank]

    # Now rearrange back to merge the split dimensions
    # Build the reverse rearrange pattern
    dim_names = [f"d{i}" for i in range(len(x_original_shape))]
    rearrange_pattern_from, rearrange_pattern_to_str = materialize_split_pattern(
        x_original_shape, dim_names, seq_dim, split_factors
    )
    log.info(
        f"[rank {cp_group.rank()}] rearrange_pattern_from: {rearrange_pattern_from}, rearrange_pattern_to_str: {rearrange_pattern_to_str}"
    )
    # The reverse pattern swaps from and to
    result = rearrange(result, f"{rearrange_pattern_to_str} -> {rearrange_pattern_from}")
    log.info(f"[rank {cp_group.rank()}] result: {result.shape}")
    return result
