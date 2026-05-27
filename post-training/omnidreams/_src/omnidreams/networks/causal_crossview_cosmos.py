# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from typing import Dict, List, Optional, cast

import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformer_engine as te
from einops import rearrange
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torchvision import transforms

from omnidreams._src.imaginaire.utils import distributed, log
from omnidreams._src.imaginaire.utils.context_parallel import cat_outputs_cp, cat_outputs_cp_with_grad, split_inputs_cp
from omnidreams._src.predict2.conditioner import DataType
from omnidreams._src.predict2.networks.minimal_v4_dit import Attention
from omnidreams._src.predict2_multiview.networks.multiview_cross_dit import (
    CrossViewAttention,
    MultiCameraVideoRopePosition3DEmb,
    MultiViewSACConfig,
)
from omnidreams._src.omnidreams.networks.causal_cosmos import (
    BlockMask,
    CausalCosmosBlock,
    CosmosCausalDiT,
    VideoSize,
)
from omnidreams._src.omnidreams.utils.context_parallel import gather_l_split_v, gather_v_split_l

# Compile flex_attention for better performance
# flex_attention_compiled = torch.compile(flex_attention, dynamic=False)


def create_cross_view_cpv2_block_mask(
    cross_view_attn_map: dict[int, list[int]],
    view_indices_V_local: torch.Tensor,
    view_indices_V_global: torch.Tensor,
    V_local: int,
    V_global: int,
    L: int,
    device: str,
) -> BlockMask:
    """
    Create a BlockMask for CPV2 cross-view attention where Q is local and KV is global.

    In CPV2, we don't do gather-split-gather. Instead:
    - Query is from local views: (V_local, L) reshaped to (V_local*L,)
    - Key/Value is from all-gathered views: (V_global, L) reshaped to (V_global*L,)

    Args:
        cross_view_attn_map: Dict mapping source view ID to list of allowed neighbor view IDs.
        view_indices_V_local: Tensor of shape (V_local,) mapping local tensor position to global view ID.
        view_indices_V_global: Tensor of shape (V_global,) mapping global tensor position to global view ID.
        V_local: Number of local views on this device.
        V_global: Total number of views across all devices.
        L: Spatial sequence length per view (H * W).
        device: Device string for creating the mask.

    Returns:
        BlockMask for the local query vs global key/value.
    """
    local_seq_len = V_local * L
    global_seq_len = V_global * L

    # Pad to multiple of 128
    local_padded = math.ceil(local_seq_len / 128) * 128
    global_padded = math.ceil(global_seq_len / 128) * 128

    # Build mapping from global view ID to global tensor position
    num_total_global_views = len(cross_view_attn_map)
    global_view_to_tensor_pos = torch.full((num_total_global_views,), -1, dtype=torch.long, device=device)
    for tensor_pos in range(V_global):
        global_view_id = int(view_indices_V_global[tensor_pos].item())
        if 0 <= global_view_id < num_total_global_views:
            global_view_to_tensor_pos[global_view_id] = tensor_pos

    # Build neighbor_allowed matrix: [V_local, V_global]
    # neighbor_allowed[local_v, global_v] = True if local view can attend to global view
    neighbor_allowed = torch.zeros((V_local, V_global), dtype=torch.bool, device=device)

    for local_v in range(V_local):
        global_view_id = int(view_indices_V_local[local_v].item())
        if global_view_id in cross_view_attn_map:
            for neighbor_view_id in cross_view_attn_map[global_view_id]:
                global_tensor_pos = int(global_view_to_tensor_pos[neighbor_view_id].item())
                if global_tensor_pos >= 0:  # neighbor is present
                    neighbor_allowed[local_v, global_tensor_pos] = True

    def cpv2_mask_mod(b, h, q_idx, kv_idx):
        # Check padding validity first
        q_valid = q_idx < local_seq_len
        kv_valid = kv_idx < global_seq_len

        # Clamp indices to valid range
        q_idx_clamped = torch.clamp(q_idx, 0, local_seq_len - 1)
        kv_idx_clamped = torch.clamp(kv_idx, 0, global_seq_len - 1)

        # Map indices to view positions
        # Q is from local views: position in [0, V_local*L)
        q_view = q_idx_clamped // L

        # KV is from global views: position in [0, V_global*L)
        kv_view = kv_idx_clamped // L

        # Check if kv view is a valid neighbor for q view
        is_neighbor = neighbor_allowed[q_view, kv_view]

        return q_valid & kv_valid & is_neighbor

    block_mask = create_block_mask(
        cpv2_mask_mod,
        B=None,
        H=None,
        Q_LEN=local_padded,
        KV_LEN=global_padded,
        _compile=False,
        device=device,
    )
    return block_mask


def create_cross_view_cp_block_mask(
    cross_view_attn_map: dict[int, list[int]],
    view_indices_V: torch.Tensor,
    num_views: int,
    L_local: int,
    L_global: int,
    rank: int,
    device: str,
) -> BlockMask:
    """
    Create a BlockMask for cross-view attention with CP, using view-first rewriting.

    This handles the case where:
    - Query is local: (V, L_local) reshaped to (V*L_local,), padded to multiple of 128
    - Key/Value is global (all-gathered): (V, L_global) reshaped to (V*L_global,), padded to multiple of 128

    The view-first CP layout means:
    - Local position q_local in [0, V*L_local) maps to:
      - local_view = q_local // L_local
      - local_spatial = q_local % L_local
      - global_spatial = local_spatial + rank * L_local
      - global_q_idx = local_view * L_global + global_spatial

    Args:
        cross_view_attn_map: Dict mapping source view ID to list of allowed neighbor view IDs.
        view_indices_V: Tensor of shape (V,) mapping tensor position to global view ID.
        num_views: Number of views (V) in the tensor.
        L_local: Local spatial sequence length per view (L_global / cp_size).
        L_global: Global spatial sequence length per view (H * W).
        rank: Current rank in the CP group.
        device: Device string for creating the mask.

    Returns:
        BlockMask for the local query vs full key/value.
    """
    local_seq_len = num_views * L_local
    global_seq_len = num_views * L_global

    # Pad to multiple of 128
    local_padded = math.ceil(local_seq_len / 128) * 128
    global_padded = math.ceil(global_seq_len / 128) * 128

    # Build neighbor_allowed matrix in tensor position space:
    # neighbor_allowed[v1, v2] = True if tensor position v1 can attend to tensor position v2
    num_total_global_views = len(cross_view_attn_map)
    global_view_to_tensor_pos = torch.full((num_total_global_views,), -1, dtype=torch.long, device=device)
    for tensor_pos in range(num_views):
        global_view_id = int(view_indices_V[tensor_pos].item())
        if 0 <= global_view_id < num_total_global_views:
            global_view_to_tensor_pos[global_view_id] = tensor_pos

    neighbor_allowed = torch.zeros((num_views, num_views), dtype=torch.bool, device=device)
    for tensor_v1 in range(num_views):
        global_v1 = int(view_indices_V[tensor_v1].item())
        if global_v1 in cross_view_attn_map:
            for global_neighbor in cross_view_attn_map[global_v1]:
                tensor_v2 = int(global_view_to_tensor_pos[global_neighbor].item())
                if tensor_v2 >= 0:  # neighbor is present
                    neighbor_allowed[tensor_v1, tensor_v2] = True

    def cp_mask_mod(b, h, q_idx, kv_idx):
        # Check padding validity first
        q_valid = q_idx < local_seq_len
        kv_valid = kv_idx < global_seq_len

        # Clamp indices to valid range to avoid index errors during mask computation
        # (the validity check above ensures we return False for invalid positions)
        q_idx_clamped = torch.clamp(q_idx, 0, local_seq_len - 1)
        kv_idx_clamped = torch.clamp(kv_idx, 0, global_seq_len - 1)

        # Map local Q index to global Q index using view-first layout
        local_view = q_idx_clamped // L_local
        local_spatial = q_idx_clamped % L_local
        global_spatial = local_spatial + rank * L_local
        global_q_idx = local_view * L_global + global_spatial

        # KV index is already in global space
        global_kv_idx = kv_idx_clamped

        # Get tensor view positions for q and kv tokens
        q_tensor_view = global_q_idx // L_global
        kv_tensor_view = global_kv_idx // L_global

        # Check if kv tensor position is a valid neighbor for q tensor position
        is_neighbor = neighbor_allowed[q_tensor_view, kv_tensor_view]

        return q_valid & kv_valid & is_neighbor

    block_mask = create_block_mask(
        cp_mask_mod,
        B=None,
        H=None,
        Q_LEN=local_padded,
        KV_LEN=global_padded,
        _compile=False,
        device=device,
    )
    return block_mask


DEBUG = False


def create_cross_view_block_mask(
    device: str,
    cross_view_attn_map: dict[int, list[int]],
    view_indices_V: torch.Tensor,
    num_views: int,
    spatial_size_hw: tuple[int, int],
    cp_size: int = 1,
) -> BlockMask:
    """
    Create a BlockMask for cross-view attention using flex_attention.

    This creates a mask where each view can only attend to its neighbor views
    as specified in cross_view_attn_map, AND the neighbor view must be present
    in the current batch (determined by view_indices_V).

    The attention pattern is reformulated as self-attention on a concatenated
    sequence of all views: (BT, V*L) where L = H*W.

    Args:
        device: Device to create the mask on (as string).
        cross_view_attn_map: Dict mapping source view ID to list of allowed neighbor view IDs.
        view_indices_V: Tensor of shape (V,) mapping tensor position to global view ID.
            This determines which views are actually present in the batch.
        num_views: Total number of views (V) in the tensor.
        spatial_size_hw: Tuple of (H, W) for spatial dimensions.
        cp_size: Context parallel size for handling distributed padded sequences.

    Returns:
        BlockMask for use with flex_attention.
    """
    H, W = spatial_size_hw
    L = H * W  # spatial tokens per view
    total_length = num_views * L

    # Calculate padding - each local sequence padded to multiple of 128
    local_len = total_length // cp_size
    local_padded_len = math.ceil(local_len / 128) * 128
    total_padded_len = local_padded_len * cp_size

    # view_indices_V maps tensor position v to global view ID
    # We need to build: for tensor position v, which tensor positions contain its allowed neighbors?

    # Create position-to-view mapping for all tokens
    # token_to_tensor_view[token_idx] = tensor view position (0 to V-1)
    token_to_tensor_view = torch.zeros(total_padded_len, dtype=torch.long, device=device)
    for v in range(num_views):
        token_to_tensor_view[v * L : (v + 1) * L] = v

    # Create reverse mapping: global_view_id -> tensor position (or -1 if not present)
    num_total_global_views = len(cross_view_attn_map)
    global_view_to_tensor_pos = torch.full((num_total_global_views,), -1, dtype=torch.long, device=device)
    for tensor_pos in range(num_views):
        global_view_id = int(view_indices_V[tensor_pos].item())
        if 0 <= global_view_id < num_total_global_views:
            global_view_to_tensor_pos[global_view_id] = tensor_pos

    # Build neighbor_allowed matrix in tensor position space:
    # neighbor_allowed[v1, v2] = True if tensor position v1 can attend to tensor position v2
    # This requires:
    # 1. v2's global view ID is in cross_view_attn_map[v1's global view ID]
    # 2. v2 is actually present (which it is if it's in view_indices_V)
    neighbor_allowed = torch.zeros((num_views, num_views), dtype=torch.bool, device=device)

    for tensor_v1 in range(num_views):
        global_v1 = int(view_indices_V[tensor_v1].item())
        if global_v1 in cross_view_attn_map:
            for global_neighbor in cross_view_attn_map[global_v1]:
                # Find which tensor position has this global neighbor view
                tensor_v2 = int(global_view_to_tensor_pos[global_neighbor].item())
                if tensor_v2 >= 0:  # neighbor is present
                    neighbor_allowed[tensor_v1, tensor_v2] = True

    def attention_mask(b, h, q_idx, kv_idx):
        # Map physical (padded) indices to logical indices for CP support
        q_rank = q_idx // local_padded_len
        q_off = q_idx % local_padded_len
        q_logical = q_rank * local_len + q_off
        q_valid = q_off < local_len

        kv_rank = kv_idx // local_padded_len
        kv_off = kv_idx % local_padded_len
        kv_logical = kv_rank * local_len + kv_off
        kv_valid = kv_off < local_len

        # Get tensor view positions for q and kv tokens
        q_tensor_view = token_to_tensor_view[q_logical]
        kv_tensor_view = token_to_tensor_view[kv_logical]

        # Check if kv tensor position is a valid neighbor for q tensor position
        is_neighbor = neighbor_allowed[q_tensor_view, kv_tensor_view]

        return q_valid & kv_valid & is_neighbor

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_padded_len,
        KV_LEN=total_padded_len,
        _compile=False,
        device=device,
    )

    return block_mask


class CrossViewAttentionWithCPV2(Attention):
    """
    Efficient Cross-View Attention with Context Parallelism support (V2).

    This version implements a more efficient CP strategy:
    1. Computes Q, K, V on local views (V dimension split across devices)
    2. All-gathers K and V across devices to get full view dimension
    3. Applies attention with appropriate masking for local Q vs global K/V
    4. No gather-split-gather overhead

    Supports two backends:
    - "transformer_engine": Uses TE's DotProductAttention with padding masks.
    - "torch-flex": Uses PyTorch's flex_attention with BlockMask for efficient sparse attention.
    """

    def __init__(self, *args, cross_view_attn_map: Dict[int, List[int]], backend: str = "torch-flex", **kwargs):
        kwargs["backend"] = backend
        super().__init__(*args, **kwargs)
        del self.attn_op

        self.cross_view_attn_map = cross_view_attn_map
        self.max_neighbors = max(len(neighbors) for neighbors in cross_view_attn_map.values())
        self.backend = backend

        if backend == "transformer_engine":
            from transformer_engine.pytorch.attention import DotProductAttention

            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=self.qkv_format,
                attn_mask_type="padding",  # important
                attention_type="cross",  # important
            )
            # Cache for neighbor indices and masks
            self.neighbor_indices = None
            self.neighbor_mask = None
        elif backend == "torch-flex":
            # QK normalization for flex_attention path
            self.q_norm_flex = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm_flex = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)
            self.attn_op = None
        else:
            raise NotImplementedError(f"Backend {backend} not supported")

        self.cp_group = None

    def forward(
        self,
        x: torch.Tensor,
        view_indices_B_V: torch.Tensor,
        spatial_size: tuple[int, int],
        block_mask_cache: dict[tuple, BlockMask] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-view attention with efficient CP support.

        Args:
            x: Input tensor of shape (B, V_local, T, L, D) where:
                - B: batch size
                - V_local: number of views on this device (split across CP group)
                - T: number of time steps
                - L: spatial sequence length (H * W)
                - D: hidden dimension
            view_indices_B_V: View indices tensor of shape (B, V_local) mapping local views to global view IDs.
            spatial_size: Tuple of (H, W) representing the spatial dimensions.
            block_mask_cache: Optional shared cache for BlockMasks across blocks (torch-flex only).

        Returns:
            Output tensor of shape (B, V_local, T, L, D) with same layout as input.
        """
        assert not self.is_selfattn, "CrossViewAttentionWithCPV2 does not support self-attention"
        B, V_local, T, L, D = x.shape
        H, W = spatial_size
        assert H * W == L, f"H * W != L: {H * W} != {L}"

        # Check if CP is enabled
        cp_enabled = self.cp_group is not None and self.cp_group.size() > 1

        if not cp_enabled:
            raise ValueError("CrossViewAttentionWithCPV2 requires CP to be enabled (cp_group.size() > 1)")

        # Get CP info
        rank = torch.distributed.get_rank(self.cp_group)
        cp_size = self.cp_group.size()

        # ========== STEP 1: Move time to batch dimension ==========
        # (B, V_local, T, L, D) -> (B*T, V_local, L, D)
        x = rearrange(x, "b v t l d -> (b t) v l d")
        BT = B * T

        # ========== STEP 2: Compute Q, K, V projections on local views ==========
        # Concatenate views: (BT, V_local, L, D) -> (BT, V_local*L, D)
        x_concat = rearrange(x, "bt v l d -> bt (v l) d")
        local_seq_len = V_local * L

        query = self.q_proj(x_concat)  # [BT, V_local*L, D]
        key = self.k_proj(x_concat)
        value = self.v_proj(x_concat)

        # Reshape to [BT, seq, heads, head_dim]
        q = query.view(BT, local_seq_len, self.n_heads, self.head_dim)
        k_local = key.view(BT, local_seq_len, self.n_heads, self.head_dim)
        v_local = value.view(BT, local_seq_len, self.n_heads, self.head_dim)

        # Apply QK normalization (backend-agnostic)
        if self.backend == "torch-flex":
            q = self.q_norm_flex(q)
            k_local = self.k_norm_flex(k_local)
        else:  # transformer_engine
            q = self.q_norm(q)
            k_local = self.k_norm(k_local)
        v_normalized = self.v_norm(v_local)

        # ========== STEP 3: All-gather K and V along V dimension ==========
        # Reshape to separate V and L: [BT, V_local, L, H, D]
        k_local_vl = k_local.view(BT, V_local, L, self.n_heads, self.head_dim)
        v_local_vl = v_normalized.view(BT, V_local, L, self.n_heads, self.head_dim)

        # All-gather along V dimension
        k_gathered_list = [
            torch.zeros(BT, V_local, L, self.n_heads, self.head_dim, device=k_local.device, dtype=k_local.dtype)
            for _ in range(cp_size)
        ]
        v_gathered_list = [
            torch.zeros(BT, V_local, L, self.n_heads, self.head_dim, device=v_local.device, dtype=v_local.dtype)
            for _ in range(cp_size)
        ]

        torch.distributed.all_gather(k_gathered_list, k_local_vl, group=self.cp_group)
        torch.distributed.all_gather(v_gathered_list, v_local_vl, group=self.cp_group)

        # Gather view_indices from all ranks to get full V dimension
        view_indices_gathered = [torch.zeros_like(view_indices_B_V) for _ in range(cp_size)]
        torch.distributed.all_gather(view_indices_gathered, view_indices_B_V, group=self.cp_group)
        view_indices_B_V_global = torch.cat(view_indices_gathered, dim=1)  # (B, V_global)
        V_global = view_indices_B_V_global.shape[1]

        # Concatenate along V dimension: [BT, V_global, L, H, D]
        k_full = torch.cat(k_gathered_list, dim=1)
        v_full = torch.cat(v_gathered_list, dim=1)

        # ========== STEPS 4-7: Backend-specific attention ==========
        if self.backend == "torch-flex":
            # Reshape to [BT, V_global*L, H, D]
            global_seq_len = V_global * L
            k_full = k_full.view(BT, global_seq_len, self.n_heads, self.head_dim)
            v_full = v_full.view(BT, global_seq_len, self.n_heads, self.head_dim)

            # Pad Q, K, V for flex_attention (must be multiple of 128)
            local_padded_len = math.ceil(local_seq_len / 128) * 128
            global_padded_len = math.ceil(global_seq_len / 128) * 128
            local_pad = local_padded_len - local_seq_len
            global_pad = global_padded_len - global_seq_len

            if local_pad > 0:
                q = torch.cat(
                    [q, torch.zeros(BT, local_pad, self.n_heads, self.head_dim, device=q.device, dtype=q.dtype)], dim=1
                )
            if global_pad > 0:
                k_full = torch.cat(
                    [
                        k_full,
                        torch.zeros(
                            BT, global_pad, self.n_heads, self.head_dim, device=k_full.device, dtype=k_full.dtype
                        ),
                    ],
                    dim=1,
                )
                v_full = torch.cat(
                    [
                        v_full,
                        torch.zeros(
                            BT, global_pad, self.n_heads, self.head_dim, device=v_full.device, dtype=v_full.dtype
                        ),
                    ],
                    dim=1,
                )

            # Transpose for flex_attention: [BT, H, S, D]
            q = q.transpose(1, 2)
            k_full = k_full.transpose(1, 2)
            v_full = v_full.transpose(1, 2)

            # Create block mask
            if block_mask_cache is None:
                block_mask_cache = {}

            view_indices_V_local = view_indices_B_V[0].long()
            view_indices_V_global = view_indices_B_V_global[0].long()
            view_indices_local_tuple = tuple(view_indices_V_local.tolist())
            view_indices_global_tuple = tuple(view_indices_V_global.tolist())
            cp_mask_key = ("cpv2", V_local, V_global, H, W, rank, view_indices_local_tuple, view_indices_global_tuple)

            if cp_mask_key not in block_mask_cache:
                if DEBUG:
                    log.info(f"Creating crossview CPV2 block mask")
                cp_block_mask = create_cross_view_cpv2_block_mask(
                    cross_view_attn_map=self.cross_view_attn_map,
                    view_indices_V_local=view_indices_V_local,
                    view_indices_V_global=view_indices_V_global,
                    V_local=V_local,
                    V_global=V_global,
                    L=L,
                    device=str(x.device),
                )
                block_mask_cache[cp_mask_key] = cp_block_mask
            else:
                cp_block_mask = block_mask_cache[cp_mask_key]

            # Apply flex_attention
            out: torch.Tensor = flex_attention(q, k_full, v_full, block_mask=cp_block_mask)  # type: ignore[assignment]

            # Remove padding and transpose back
            out = out.transpose(1, 2)  # [BT, S, H, D]
            if local_pad > 0:
                out = out[:, :-local_pad]

        else:  # transformer_engine backend
            # Expand view_indices to match (B*T, V_local) and (B*T, V_global)
            view_indices_BT_V_local = view_indices_B_V.repeat_interleave(T, dim=0).long()
            view_indices_BT_V_global = view_indices_B_V_global.repeat_interleave(T, dim=0).long()

            # Create neighbor indices and mask (once per device)
            if self.neighbor_indices is None or self.neighbor_indices.device != x.device:
                num_total_views = len(self.cross_view_attn_map)
                neighbor_indices = torch.zeros((num_total_views, self.max_neighbors), dtype=torch.long, device=x.device)
                neighbor_mask = torch.zeros((num_total_views, self.max_neighbors), dtype=torch.bool, device=x.device)
                for i in range(num_total_views):
                    neighbors = self.cross_view_attn_map[i]
                    for j, neighbor_idx in enumerate(neighbors):
                        neighbor_indices[i, j] = neighbor_idx
                        neighbor_mask[i, j] = True
                self.neighbor_indices = neighbor_indices
                self.neighbor_mask = neighbor_mask

            # Build mapping from global view ID to global tensor position
            num_total_views = len(self.cross_view_attn_map)
            view_indices_to_tensor_pos_global = torch.full((BT, num_total_views), -1, dtype=torch.long, device=x.device)
            b_indices = torch.arange(BT, device=x.device).unsqueeze(1).expand(-1, V_global).long()
            view_indices_to_tensor_pos_global[b_indices, view_indices_BT_V_global] = (
                torch.arange(V_global, device=x.device).unsqueeze(0).expand(BT, -1)
            )

            # For each LOCAL view, find which GLOBAL views are its neighbors
            neighbor_view_indices = self.neighbor_indices[view_indices_BT_V_local]  # [BT, V_local, max_neighbors]
            b_indices_local = torch.arange(BT, device=x.device).unsqueeze(1).unsqueeze(2)
            gather_tensor_pos = view_indices_to_tensor_pos_global[
                b_indices_local, neighbor_view_indices
            ]  # [BT, V_local, max_neighbors]

            # Sort to move all -1 to the end
            gather_tensor_pos, sorted_indices = torch.sort(gather_tensor_pos, dim=-1, descending=True)

            # Gather neighbor K, V from global K/V
            b_indices_for_gather = torch.arange(BT, device=x.device)[:, None, None]
            neighbor_key = k_full[
                b_indices_for_gather, torch.clamp(gather_tensor_pos, min=0)
            ]  # [BT, V_local, max_neighbors, L, H, D]
            neighbor_value = v_full[
                b_indices_for_gather, torch.clamp(gather_tensor_pos, min=0)
            ]  # [BT, V_local, max_neighbors, L, H, D]

            # Reshape for TE attention: [BT*V_local, max_neighbors*L, H, D]
            k = rearrange(neighbor_key, "bt v n l h d -> (bt v) (n l) h d")
            v = rearrange(neighbor_value, "bt v n l h d -> (bt v) (n l) h d")
            q = rearrange(q, "bt (v l) h d -> (bt v) l h d", v=V_local)

            # Create attention mask for TE
            is_neighbor_present = gather_tensor_pos != -1  # [BT, V_local, max_neighbors]
            mask_for_input_views = self.neighbor_mask[view_indices_BT_V_local]  # [BT, V_local, max_neighbors]
            mask_for_input_views = torch.gather(mask_for_input_views, -1, sorted_indices)
            final_mask = is_neighbor_present & mask_for_input_views

            mask_per_view = rearrange(final_mask, "bt v n -> (bt v) n")  # [BT*V_local, max_neighbors]
            mask_kv = mask_per_view.repeat_interleave(L, dim=1)  # [BT*V_local, max_neighbors*L]

            # Reshape mask to [batch_size, 1, 1, max_seqlen_kv]
            mask = rearrange(mask_kv, "bv l_kv -> bv 1 1 l_kv")
            atten_mask_kv = ~mask  # 0 means keep, 1 means mask
            atten_mask_q = torch.zeros(
                q.shape[0], 1, 1, q.shape[1], device=atten_mask_kv.device, dtype=atten_mask_kv.dtype
            )

            # Apply TE attention
            out = self.attn_op(q, k, v, attention_mask=(atten_mask_q, atten_mask_kv))
            out = rearrange(out, "(bt v) l D -> bt (v l) D", bt=BT, v=V_local)

        # ========== STEP 8: Flatten heads, project output, and reshape ==========
        out = out.flatten(2)  # [BT, V_local*L, H*D]
        output = self.output_dropout(self.output_proj(out))

        # Reshape back to (B, V_local, T, L, D)
        output = rearrange(output, "(b t) (v l) d -> b v t l d", b=B, v=V_local, l=L).contiguous()

        return output

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        # We handle CP communication ourselves via all_gather, so we don't set CP on the attention op
        # This is different from V1 where TE's attn_op handles CP internally
        self.cp_group = process_group


class CrossViewAttentionWithCP(Attention):
    """
    Cross-View Attention with Context Parallelism support.

    When CP is enabled, this module:
    1. Gathers V dimension (views) across devices and splits L dimension (spatial tokens)
    2. Performs cross-view neighbor attention with all views but partial spatial tokens
    3. Gathers L dimension and splits V dimension back to original layout

    Supports two backends:
    - "transformer_engine": Uses TE's DotProductAttention with padding masks.
    - "torch-flex": Uses PyTorch's flex_attention with BlockMask for efficient sparse attention.
    """

    def __init__(self, *args, cross_view_attn_map: Dict[int, List[int]], **kwargs):
        super().__init__(*args, **kwargs)
        del self.attn_op
        if self.backend == "transformer_engine":
            from transformer_engine.pytorch.attention import DotProductAttention

            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=self.qkv_format,
                attn_mask_type="padding",  # important
                attention_type="cross",  # important
            )
        elif self.backend == "torch-flex":
            # For torch-flex, we use self-attention on concatenated views with sparse BlockMask
            # No separate attn_op needed - we use flex_attention directly
            self.attn_op = None
            # QK normalization for flex_attention path
            self.q_norm_flex = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm_flex = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported")
        self.cross_view_attn_map = cross_view_attn_map
        self.max_neighbors = max(len(neighbors) for neighbors in cross_view_attn_map.values())
        self.neighbor_indices = None
        self.neighbor_mask = None
        self.cp_group = None

    def forward(
        self,
        x: torch.Tensor,
        view_indices_B_V: torch.Tensor,
        spatial_size: tuple[int, int],
        block_mask_cache: dict[tuple, BlockMask] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-view attention with context parallelism support.

        Args:
            x: Input tensor of shape (B, V, T, L, D) where:
                - B: batch size
                - V: number of views (V_local if CP enabled, views are split across devices)
                - T: number of time steps
                - L: spatial sequence length (H * W)
                - D: hidden dimension
            view_indices_B_V: View indices tensor of shape (B, V) mapping local views to global view IDs.
            spatial_size: Tuple of (H, W) representing the spatial dimensions.
            block_mask_cache: Optional shared cache for BlockMasks across blocks.

        Returns:
            Output tensor of shape (B, V, T, L, D) with same layout as input.
        """
        assert not self.is_selfattn, "CrossViewAttention does not support self-attention"
        B, V_input, T, L, D = x.shape
        H, W = spatial_size
        assert H * W == L, f"H * W != L: {H * W} != {L}"

        # Context Parallel: Gather V (views) and split L (spatial tokens)
        # This allows cross-view attention with all views but only a portion of spatial tokens
        cp_enabled = self.cp_group is not None and self.cp_group.size() > 1
        if cp_enabled:
            # Input: (B, V_local, T, L, D) -> Output: (B, V_global, T, L_local, D)
            x = gather_v_split_l(x, self.cp_group)
            # Update view_indices to include all views after gathering
            # After gather, we have all V_global views, so view_indices should map to global indices
            world_size = self.cp_group.size()
            # Gather view_indices from all ranks
            view_indices_gathered = [torch.zeros_like(view_indices_B_V) for _ in range(world_size)]
            torch.distributed.all_gather(view_indices_gathered, view_indices_B_V, group=self.cp_group)
            view_indices_B_V = torch.cat(view_indices_gathered, dim=1)  # (B, V_global)

        B, V, T, L, D = x.shape  # V and L may have changed due to CP
        if DEBUG:
            log.info(f"x: {x.shape}")
            log.info(f"view_indices_B_V: {view_indices_B_V}")
            log.info(f"spatial_size: {spatial_size}")

        if self.backend == "torch-flex":
            output = self._forward_torch_flex(x, view_indices_B_V, spatial_size, V, T, L, block_mask_cache)
        else:
            output = self._forward_transformer_engine(x, view_indices_B_V, V, T, L)

        # Context Parallel: Gather L (spatial tokens) and split V (views)
        # This restores the original V-parallel layout
        if cp_enabled:
            # Input: (B, V_global, T, L_local, D) -> Output: (B, V_local, T, L, D)
            output = gather_l_split_v(output, self.cp_group)

        return output

    def _forward_transformer_engine(
        self,
        x: torch.Tensor,
        view_indices_B_V: torch.Tensor,
        V: int,
        T: int,
        L: int,
    ) -> torch.Tensor:
        """Forward pass using Transformer Engine backend with padding masks."""
        B = x.shape[0]
        D = x.shape[-1]

        # Move time dimension to batch dimension: (B, V, T, L, D) -> (B*T, V, L, D)
        x = rearrange(x, "b v t l d -> (b t) v l d")
        BT = B * T

        # Expand view_indices to match (B*T, V)
        view_indices_BT_V = view_indices_B_V.repeat_interleave(T, dim=0).long()

        # Create neighbor indices and mask on the fly, only once.
        if self.neighbor_indices is None or self.neighbor_indices.device != x.device:
            num_total_views = len(self.cross_view_attn_map)
            neighbor_indices = torch.zeros((num_total_views, self.max_neighbors), dtype=torch.long, device=x.device)
            neighbor_mask = torch.zeros((num_total_views, self.max_neighbors), dtype=torch.bool, device=x.device)
            for i in range(num_total_views):
                neighbors = self.cross_view_attn_map[i]
                for j, neighbor_idx in enumerate(neighbors):
                    neighbor_indices[i, j] = neighbor_idx
                    neighbor_mask[i, j] = True
            self.neighbor_indices = neighbor_indices
            self.neighbor_mask = neighbor_mask

        num_total_views = len(self.cross_view_attn_map)
        view_indices_to_tensor_pos = torch.full(
            (BT, num_total_views), -1, dtype=torch.long, device=x.device
        )  # include out of range view index
        b_indices = torch.arange(BT, device=x.device).unsqueeze(1).expand(-1, V).long()
        view_indices_to_tensor_pos[b_indices, view_indices_BT_V] = (
            torch.arange(V, device=x.device).unsqueeze(0).expand(BT, -1)
        )

        neighbor_view_indices = self.neighbor_indices[view_indices_BT_V]  # may include out of range view index
        gather_tensor_pos = view_indices_to_tensor_pos[
            b_indices.unsqueeze(2), neighbor_view_indices
        ]  # [BT, V, max_neighbors], out of range view index will be -1

        # Sort to move all -1 to the end, which is convenient for creating attention mask.
        gather_tensor_pos, sorted_indices = torch.sort(gather_tensor_pos, dim=-1, descending=True)

        b_indices_for_gather = torch.arange(BT, device=x.device)[:, None, None]
        # Clamp to avoid index error. Masked values will be ignored in attention.
        neighbor_features = x[
            b_indices_for_gather, torch.clamp(gather_tensor_pos, min=0)
        ]  # [BT, V, max_neighbors, L, C]

        # Prepare for attention
        query = self.q_proj(rearrange(x, "b v l c -> (b v) l c"))  # [BT*V, L, C]
        context = rearrange(neighbor_features, "b v n l c -> (b v) (n l) c")  # [BT*V, max_neighbors*L, C]
        key = self.k_proj(context)
        value = self.v_proj(context)

        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (query, key, value),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Create attention mask
        is_neighbor_present = gather_tensor_pos != -1  # [BT, V, max_neighbors]
        mask_for_input_views = self.neighbor_mask[view_indices_BT_V]  # [BT, V, n]

        # Reorder mask_for_input_views to match the sorted gather_tensor_pos
        mask_for_input_views = torch.gather(mask_for_input_views, -1, sorted_indices)
        final_mask = is_neighbor_present & mask_for_input_views

        mask_per_view = rearrange(final_mask, "b v n -> (b v) n")  # [BT*V, n]
        mask_kv = mask_per_view.repeat_interleave(L, dim=1)  # [BT*V, n*L]

        # Reshape mask to [batch_size, 1, 1, max_seqlen_kv] as per official documentation.
        mask = rearrange(mask_kv, "bv l_kv -> bv 1 1 l_kv")  # [BT*V, 1, 1, n*L]
        atten_mask_kv = ~mask  # 0 means keep, 1 means mask
        atten_mask_q = torch.zeros(query.shape[0], 1, 1, query.shape[1]).to(atten_mask_kv)

        attention_output = self.attn_op(q, k, v, attention_mask=(atten_mask_q, atten_mask_kv))
        attention_output = attention_output.flatten(2)  # [BT*V, L, H*D]
        output = self.output_dropout(self.output_proj(attention_output))
        output = rearrange(output, "(b t v) l d -> b v t l d", v=V, t=T)

        return output

    def _forward_torch_flex(
        self,
        x: torch.Tensor,
        view_indices_B_V: torch.Tensor,
        spatial_size: tuple[int, int],
        V: int,
        T: int,
        L: int,
        block_mask_cache: dict[tuple, BlockMask] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass using torch-flex backend with BlockMask.

        This reformulates cross-view attention as self-attention on concatenated views
        with a sparse BlockMask that only allows cross-view attention between neighbors.

        When CP is enabled:
        - Input has L_local = L / cp_size spatial tokens per view
        - We all-gather K/V to get full spatial dimension
        - We use a custom mask rewrite for the view-first layout

        Args:
            block_mask_cache: Optional shared cache for BlockMasks. If None, masks are not cached.
        """
        # Use local cache if no shared cache is provided
        if block_mask_cache is None:
            block_mask_cache = {}
        B = x.shape[0]
        D = x.shape[-1]
        H, W = spatial_size
        L_global = H * W  # Full spatial size per view

        # Get the view indices (use first batch element - assume same for all batches)
        # Shape: (V,) containing global view IDs for each tensor position
        view_indices_V = view_indices_B_V[0].long()

        # Check if CP is enabled
        cp_enabled = self.cp_group is not None and self.cp_group.size() > 1

        # L is the local spatial size (L_local when CP enabled, L_global otherwise)
        L_local = L

        # Cache key for view indices
        view_indices_tuple = tuple(view_indices_V.tolist())

        # Move time dimension to batch dimension: (B, V, T, L, D) -> (B*T, V, L, D)
        x = rearrange(x, "b v t l d -> (b t) v l d")
        BT = B * T

        # Concatenate views for self-attention pattern: (BT, V, L, D) -> (BT, V*L, D)
        x_concat = rearrange(x, "bt v l d -> bt (v l) d")
        local_seq_len = V * L_local
        global_seq_len = V * L_global

        # Compute Q, K, V projections
        query = self.q_proj(x_concat)  # [BT, V*L_local, D]
        key = self.k_proj(x_concat)
        value = self.v_proj(x_concat)

        # Reshape to [BT, seq, heads, head_dim]
        q = query.view(BT, local_seq_len, self.n_heads, self.head_dim)
        k = key.view(BT, local_seq_len, self.n_heads, self.head_dim)
        v = value.view(BT, local_seq_len, self.n_heads, self.head_dim)

        # Apply QK normalization
        q = self.q_norm_flex(q)
        k = self.k_norm_flex(k)

        if cp_enabled:
            # === CP-enabled path: all-gather K/V and use custom mask rewrite ===
            rank = torch.distributed.get_rank(self.cp_group)
            cp_size = self.cp_group.size()

            # All-gather K and V along the spatial dimension within each view
            # Current shape: [BT, V*L_local, H, D] - need to gather to [BT, V*L_global, H, D]
            # We need to interleave properly: gather L dimension while preserving V structure

            # Reshape to separate V and L: [BT, V, L_local, H, D]
            k_vl = k.view(BT, V, L_local, self.n_heads, self.head_dim)
            v_vl = v.view(BT, V, L_local, self.n_heads, self.head_dim)

            # All-gather along L dimension for each view
            k_gathered_list = [
                torch.zeros(BT, V, L_local, self.n_heads, self.head_dim, device=k.device, dtype=k.dtype)
                for _ in range(cp_size)
            ]
            v_gathered_list = [
                torch.zeros(BT, V, L_local, self.n_heads, self.head_dim, device=v.device, dtype=v.dtype)
                for _ in range(cp_size)
            ]

            torch.distributed.all_gather(k_gathered_list, k_vl, group=self.cp_group)
            torch.distributed.all_gather(v_gathered_list, v_vl, group=self.cp_group)

            # Concatenate along L dimension: [BT, V, L_global, H, D]
            k_full = torch.cat(k_gathered_list, dim=2)
            v_full = torch.cat(v_gathered_list, dim=2)

            # Reshape back to [BT, V*L_global, H, D]
            k_full = k_full.view(BT, global_seq_len, self.n_heads, self.head_dim)
            v_full = v_full.view(BT, global_seq_len, self.n_heads, self.head_dim)

            # Pad Q, K, V for flex_attention
            local_padded_len = math.ceil(local_seq_len / 128) * 128
            global_padded_len = math.ceil(global_seq_len / 128) * 128
            local_pad = local_padded_len - local_seq_len
            global_pad = global_padded_len - global_seq_len

            if local_pad > 0:
                q = torch.cat(
                    [q, torch.zeros(BT, local_pad, self.n_heads, self.head_dim, device=q.device, dtype=q.dtype)], dim=1
                )
            if global_pad > 0:
                k_full = torch.cat(
                    [
                        k_full,
                        torch.zeros(
                            BT, global_pad, self.n_heads, self.head_dim, device=k_full.device, dtype=k_full.dtype
                        ),
                    ],
                    dim=1,
                )
                v_full = torch.cat(
                    [
                        v_full,
                        torch.zeros(
                            BT, global_pad, self.n_heads, self.head_dim, device=v_full.device, dtype=v_full.dtype
                        ),
                    ],
                    dim=1,
                )

            # Transpose for flex_attention: [BT, H, S, D]
            q = q.transpose(1, 2)
            k_full = k_full.transpose(1, 2)
            v_full = v_full.transpose(1, 2)

            # Create CP block mask with view-first rewriting
            cp_mask_key = ("cp", V, H, W, rank, view_indices_tuple)
            if cp_mask_key not in block_mask_cache:
                if DEBUG:
                    log.info(f"creating crossview cp block mask")
                    log.info(f"self.cross_view_attn_map: {self.cross_view_attn_map}")
                    log.info(f"view_indices_V: {view_indices_V}")
                    log.info(f"V: {V}")
                    log.info(f"L_local: {L_local}")
                    log.info(f"L_global: {L_global}")
                    log.info(f"rank: {rank}")
                    log.info(f"device: {x.device}")
                cp_block_mask = create_cross_view_cp_block_mask(
                    cross_view_attn_map=self.cross_view_attn_map,
                    view_indices_V=view_indices_V,
                    num_views=V,
                    L_local=L_local,
                    L_global=L_global,
                    rank=rank,
                    device=str(x.device),
                )
                block_mask_cache[cp_mask_key] = cp_block_mask
            else:
                cp_block_mask = block_mask_cache[cp_mask_key]

            # Apply flex_attention
            out: torch.Tensor = flex_attention(q, k_full, v_full, block_mask=cp_block_mask)  # type: ignore[assignment]

            # Remove padding and transpose back
            out = out.transpose(1, 2)  # [BT, S, H, D]
            if local_pad > 0:
                out = out[:, :-local_pad]

            padded_length = local_pad  # For compatibility with output reshaping
            seq_len = local_seq_len

        else:
            # === Non-CP path: standard flex_attention ===
            seq_len = local_seq_len

            # Get or create the block mask (for non-CP, Q_LEN == KV_LEN == V*L)
            global_mask_key = ("global", V, H, W, view_indices_tuple)
            if global_mask_key not in block_mask_cache:
                log.info(f"creating crossview global block mask")
                log.info(f"self.cross_view_attn_map: {self.cross_view_attn_map}")
                log.info(f"view_indices_V: {view_indices_V}")
                log.info(f"V: {V}")
                log.info(f"spatial_size: {spatial_size}")
                log.info(f"device: {x.device}")
                global_block_mask = create_cross_view_block_mask(
                    device=str(x.device),
                    cross_view_attn_map=self.cross_view_attn_map,
                    view_indices_V=view_indices_V,
                    num_views=V,
                    spatial_size_hw=(H, W),
                    cp_size=1,
                )
                block_mask_cache[global_mask_key] = global_block_mask
            else:
                global_block_mask = block_mask_cache[global_mask_key]

            # Pad to multiple of 128 for flex_attention efficiency
            padded_length = math.ceil(seq_len / 128) * 128 - seq_len

            if padded_length > 0:
                pad_shape = [BT, padded_length, self.n_heads, self.head_dim]
                q = torch.cat([q, torch.zeros(pad_shape, device=q.device, dtype=q.dtype)], dim=1)
                k = torch.cat([k, torch.zeros(pad_shape, device=k.device, dtype=k.dtype)], dim=1)
                v = torch.cat([v, torch.zeros(pad_shape, device=v.device, dtype=v.dtype)], dim=1)

            # flex_attention expects [B, H, S, D] format
            q = q.transpose(1, 2)  # [BT, H, S, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Apply flex_attention with global block mask
            out: torch.Tensor = flex_attention(q, k, v, block_mask=global_block_mask)  # type: ignore[assignment]

            # Remove padding and transpose back
            out = out.transpose(1, 2)  # [BT, S, H, D]
            if padded_length > 0:
                out = out[:, :-padded_length]

        # Flatten heads and project output
        out = out.flatten(2)  # [BT, V*L, H*D]
        output = self.output_dropout(self.output_proj(out))

        # Reshape back to (B, V, T, L, D)
        output = rearrange(output, "(b t) (v l) d -> b v t l d", b=B, v=V, l=L_local)

        return output

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        if process_group is not None and self.attn_op is not None:
            self.attn_op.set_context_parallel_group(process_group, ranks, stream, cp_comm_type=cp_comm_type)
        self.cp_group = process_group


class CausalCrossViewCosmosBlock(CausalCosmosBlock):
    """
    CausalCosmosBlock with Cross-View Attention.

    Args:
        v_split_mode: If True, the input tensor x_B_V_L_D is assumed to have V split across
            devices for context parallelism (i.e., each device has V_local=1 view).
            In this mode, CrossViewAttentionWithCP is used which gathers V and splits L
            for cross-view attention, then gathers L and splits V back.
            If False, standard CrossViewAttention is used without CP redistribution.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "transformer_engine",
        image_context_dim: int | None = None,
        use_wan_fp32_strategy: bool = False,
        # Causal-specific parameters
        local_attn_size: int = -1,
        sink_size: int = 0,
        # Multi-view specific parameters
        cross_view_attn_map: Optional[Dict[int, List[int]]] = None,
        enable_cross_view_attn: bool = False,
        v_split_mode: bool = False,
    ):
        super().__init__(
            x_dim=x_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            backend=backend,
            image_context_dim=image_context_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        self.enable_cross_view_attn = enable_cross_view_attn
        self.cross_view_attn_map = cross_view_attn_map
        self.v_split_mode = v_split_mode

        if enable_cross_view_attn:
            assert cross_view_attn_map is not None
            if v_split_mode:
                # Use CP-enabled cross-view attention that gathers V and splits L
                self.cross_view_attn = CrossViewAttentionWithCPV2(
                    x_dim,
                    x_dim,  # context_dim, can not set to None
                    num_heads,
                    x_dim // num_heads,
                    qkv_format="bshd",
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                    cross_view_attn_map=cross_view_attn_map,
                    backend=backend,
                )
            else:
                # Use standard cross-view attention without CP redistribution
                self.cross_view_attn = CrossViewAttention(
                    x_dim,
                    x_dim,  # context_dim, can not set to None
                    num_heads,
                    x_dim // num_heads,
                    qkv_format="bshd",
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                    cross_view_attn_map=cross_view_attn_map,
                    backend=backend,
                )
            # no modulation so we set elementwise_affine=True
            self.layer_norm_cross_view_attn = nn.LayerNorm(x_dim, elementwise_affine=True, eps=1e-6)

    def init_weights(self) -> None:
        super().init_weights()
        if self.enable_cross_view_attn:
            self.layer_norm_cross_view_attn.reset_parameters()
            self.cross_view_attn.init_weights()
            # Zero-initialize the output projection
            torch.nn.init.zeros_(self.cross_view_attn.output_proj.weight)
            if self.cross_view_attn.output_proj.bias is not None:
                torch.nn.init.zeros_(self.cross_view_attn.output_proj.bias)

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        self.cp_size = None if ranks is None else len(ranks)
        if self.v_split_mode:
            # In v_split_mode, CP is handled by CrossViewAttentionWithCP which gathers V and splits L
            self.cross_view_attn.set_context_parallel_group(process_group, ranks, stream, cp_comm_type=cp_comm_type)
        else:
            # Standard CP on self-attention
            self.self_attn.set_context_parallel_group(process_group, ranks, stream, cp_comm_type=cp_comm_type)

    def forward(
        self,
        x_B_V_L_D: torch.Tensor,
        emb_B_V_L_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        view_indices_B_V: torch.Tensor,
        sv_video_size: VideoSize,
        rope_emb_V_L_1_1_D: torch.Tensor | None = None,
        adaln_lora_B_V_L_3D: torch.Tensor | None = None,
        extra_per_block_pos_emb: torch.Tensor | None = None,
        # Causal-specific parameters
        block_mask: BlockMask | None = None,
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        current_end: int = 0,
        disable_kv_cache: bool = False,
        disable_kv_cache_update: bool = False,
        video_size: VideoSize | None = None,
        view_embedding_proj_B_V_9D: Optional[torch.Tensor] = None,
        cross_view_block_mask_cache: dict[tuple, BlockMask] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the block with B_V_L_D tensor shape support.

        When v_split_mode is enabled, the input tensor is expected to have V split across
        devices (V_local=1 per device). The cross-view attention will gather all views,
        perform attention, and split back.

        Args:
            cross_view_block_mask_cache: Optional shared cache for cross-view attention BlockMasks.
        """
        B, V, L, D = x_B_V_L_D.shape

        # In v_split_mode, V should be 1 (each device has one view)
        if self.v_split_mode:
            assert V == 1, f"In v_split_mode, V dimension should be 1 (split across devices), but got V={V}"

        # Flatten V into B for single-view operations: (B*V, L, D)
        x_BV_L_D = rearrange(x_B_V_L_D, "b v l d -> (b v) l d")
        emb_BV_L_D = rearrange(emb_B_V_L_D, "b v l d -> (b v) l d")

        if adaln_lora_B_V_L_3D is not None:
            adaln_lora_BV_L_3D = rearrange(adaln_lora_B_V_L_3D, "b v l d -> (b v) l d")
        else:
            adaln_lora_BV_L_3D = None

        if extra_per_block_pos_emb is not None:
            # Handle extra pos emb if present, assume it matches x shape or broadcastable
            if extra_per_block_pos_emb.ndim == 4:  # B V L D
                x_BV_L_D = x_BV_L_D + rearrange(extra_per_block_pos_emb, "b v l d -> (b v) l d")
            else:
                x_BV_L_D = x_BV_L_D + extra_per_block_pos_emb

        # Compute AdaLN modulation
        with torch.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                shift_self, scale_self, gate_self = (
                    self.adaln_modulation_self_attn(emb_BV_L_D) + adaln_lora_BV_L_3D
                ).chunk(3, dim=-1)
                shift_cross, scale_cross, gate_cross = (
                    self.adaln_modulation_cross_attn(emb_BV_L_D) + adaln_lora_BV_L_3D
                ).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_BV_L_D) + adaln_lora_BV_L_3D).chunk(
                    3, dim=-1
                )
            else:
                shift_self, scale_self, gate_self = self.adaln_modulation_self_attn(emb_BV_L_D).chunk(3, dim=-1)
                shift_cross, scale_cross, gate_cross = self.adaln_modulation_cross_attn(emb_BV_L_D).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_BV_L_D).chunk(3, dim=-1)

        # Apply view embedding projection if provided
        if view_embedding_proj_B_V_9D is not None:
            # view_embedding_proj_B_V_9D: [B, V, 9D]
            (
                view_shift_self,
                view_scale_self,
                view_gate_self,
                view_shift_cross,
                view_scale_cross,
                view_gate_cross,
                view_shift_mlp,
                view_scale_mlp,
                view_gate_mlp,
            ) = view_embedding_proj_B_V_9D.chunk(9, dim=-1)

            # Expand to (B*V, 1, D) for broadcasting
            def expand_view_mod(v_mod):
                # v_mod: [B, V, D] -> [B*V, 1, D]
                return rearrange(v_mod, "b v d -> (b v) 1 d")

            shift_self = shift_self + expand_view_mod(view_shift_self)
            scale_self = scale_self + expand_view_mod(view_scale_self)
            gate_self = gate_self + expand_view_mod(view_gate_self)

            shift_cross = shift_cross + expand_view_mod(view_shift_cross)
            scale_cross = scale_cross + expand_view_mod(view_scale_cross)
            gate_cross = gate_cross + expand_view_mod(view_gate_cross)

            shift_mlp = shift_mlp + expand_view_mod(view_shift_mlp)
            scale_mlp = scale_mlp + expand_view_mod(view_scale_mlp)
            gate_mlp = gate_mlp + expand_view_mod(view_gate_mlp)

        shift_self = shift_self.type_as(x_BV_L_D)
        scale_self = scale_self.type_as(x_BV_L_D)
        gate_self = gate_self.type_as(x_BV_L_D)
        shift_cross = shift_cross.type_as(x_BV_L_D)
        scale_cross = scale_cross.type_as(x_BV_L_D)
        gate_cross = gate_cross.type_as(x_BV_L_D)
        shift_mlp = shift_mlp.type_as(x_BV_L_D)
        scale_mlp = scale_mlp.type_as(x_BV_L_D)
        gate_mlp = gate_mlp.type_as(x_BV_L_D)

        # 1. Self-attention (Single View)
        normed_x = self.layer_norm_self_attn(x_BV_L_D) * (1 + scale_self) + shift_self
        rope_single_view = rope_emb_V_L_1_1_D[0]  # [L, 1, 1, D]
        if DEBUG:
            log.info("before self-attention", rank0_only=True)
        attn_out = self.self_attn(
            normed_x,
            context=None,
            rope_emb=rope_single_view,
            video_size=sv_video_size,  # Use single view video size
            block_mask=block_mask,
            kv_cache=kv_cache,
            current_start=current_start,
            current_end=current_end,
            disable_kv_cache=disable_kv_cache,
            disable_kv_cache_update=disable_kv_cache_update,
        )
        x_BV_L_D = x_BV_L_D + gate_self * attn_out
        if DEBUG:
            log.info("after self-attention", rank0_only=True)
        # 2. Cross-View Attention
        if self.enable_cross_view_attn:
            T, H, W = sv_video_size
            x_B_V_L_D = rearrange(x_BV_L_D, "(b v) l d -> b v l d", b=B, v=V)

            # Apply layer norm (elementwise affine=True, no adaln modulation typically)
            normed_x_cv = self.layer_norm_cross_view_attn(x_B_V_L_D)

            if self.v_split_mode:
                # CrossViewAttentionWithCP expects (B, V, T, L, D) where L = H*W
                # and spatial_size = (H, W)
                normed_x_cv = rearrange(normed_x_cv, "b v (t h w) d -> b v t (h w) d", t=T, h=H, w=W)
                cv_out = self.cross_view_attn(
                    normed_x_cv, view_indices_B_V, (H, W), block_mask_cache=cross_view_block_mask_cache
                )
                # Reshape back to (B, V, L, D) where L = T*H*W
                cv_out = rearrange(cv_out, "b v t (h w) d -> b v (t h w) d", h=H, w=W)
            else:
                # CrossViewAttention expects (B, V, L, D) where L = T*H*W
                # and sv_video_size = VideoSize(T, H, W)
                cv_out = self.cross_view_attn(normed_x_cv, view_indices_B_V, sv_video_size)

            if DEBUG:
                log.info("after cross-view attention", rank0_only=True)
            # Residual connection
            x_B_V_L_D = x_B_V_L_D + cv_out
            x_BV_L_D = rearrange(x_B_V_L_D, "b v l d -> (b v) l d")

        # 3. Cross-attention
        normed_x = self.layer_norm_cross_attn(x_BV_L_D) * (1 + scale_cross) + shift_cross
        crossattn_emb_BV_L_D = rearrange(crossattn_emb, "b (v l) d -> (b v) l d", v=V)
        if DEBUG:
            log.info(f"normed_x.shape: {normed_x.shape}", rank0_only=True)
            log.info(f"crossattn_emb_BV_L_D.shape: {crossattn_emb_BV_L_D.shape}", rank0_only=True)
        # Note that RoPE is not applied to cross-attention. See omnidreams/_src/predict2/networks/minimal_v4_dit.py L527
        if crossattn_cache is not None:
            # Handle cache expansion if needed or assume cache handles it
            # Current CosmosCausalDiT assumes crossattn_cache is handled by caller or per-block
            # We pass it as is, assuming it matches B*V batch size if initialized correctly
            cross_out = self.cross_attn(normed_x, crossattn_emb_BV_L_D, crossattn_cache=crossattn_cache)
        else:
            cross_out = self.cross_attn(normed_x, crossattn_emb_BV_L_D)
        if DEBUG:
            log.info("after cross-attention", rank0_only=True)
        x_BV_L_D = x_BV_L_D + gate_cross * cross_out

        # 4. MLP
        normed_x = self.layer_norm_mlp(x_BV_L_D) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(normed_x)
        x_BV_L_D = x_BV_L_D + gate_mlp * mlp_out
        if DEBUG:
            log.info("after mlp", rank0_only=True)
        # Reshape back to B_V_L_D
        x_B_V_L_D = rearrange(x_BV_L_D, "(b v) l d -> b v l d", b=B, v=V)
        if DEBUG:
            distributed.barrier()
        return x_B_V_L_D


class CausalCrossViewCosmosDiT(CosmosCausalDiT):
    """
    Causal DiT model with MultiView capabilities.
    """

    def __init__(
        self,
        *args,
        state_t: int,
        n_cameras_emb: int,
        view_condition_dim: int,
        concat_view_embedding: bool,
        adaln_view_embedding: bool,
        sac_config: MultiViewSACConfig = MultiViewSACConfig(),
        enable_cross_view_attn: bool = False,
        cross_view_attn_map_str: Optional[Dict] = None,
        camera_to_view_id: Optional[Dict] = None,
        init_cross_view_attn_weight_from: Optional[str] = None,
        init_cross_view_attn_weight_credentials: Optional[str] = None,
        v_split_mode: bool = False,
        backend: str = "transformer_engine",
        **kwargs,
    ):
        self.state_t = state_t
        self.n_cameras_emb = n_cameras_emb
        self.view_condition_dim = view_condition_dim
        self.concat_view_embedding = concat_view_embedding
        self.adaln_view_embedding = adaln_view_embedding
        self.enable_cross_view_attn = enable_cross_view_attn
        self.init_cross_view_attn_weight_from = init_cross_view_attn_weight_from
        self.init_cross_view_attn_weight_credentials = init_cross_view_attn_weight_credentials
        self.v_split_mode = v_split_mode

        # Modify in_channels for view embedding
        if concat_view_embedding:
            kwargs["in_channels"] = kwargs.get("in_channels", 0) + view_condition_dim

        # Parse cross view attn map
        self.cross_view_attn_map = {}
        if cross_view_attn_map_str and camera_to_view_id:
            for source_view, target_views in cross_view_attn_map_str.items():
                self.cross_view_attn_map[int(camera_to_view_id[source_view])] = []
                for target_view in target_views:
                    self.cross_view_attn_map[int(camera_to_view_id[source_view])].append(
                        int(camera_to_view_id[target_view])
                    )

        super().__init__(*args, sac_config=sac_config, **kwargs)

        # Re-build blocks with CausalCrossViewCosmosBlock
        del self.blocks
        self.blocks = nn.ModuleList(
            [
                CausalCrossViewCosmosBlock(
                    x_dim=self.model_channels,
                    context_dim=cast(
                        int,
                        self.crossattn_emb_channels
                        if "crossattn_emb_channels" not in kwargs
                        else kwargs["crossattn_emb_channels"],
                    ),
                    num_heads=self.num_heads,
                    mlp_ratio=cast(float, self.mlp_ratio if hasattr(self, "mlp_ratio") else 4.0),
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim if hasattr(self, "adaln_lora_dim") else 256,
                    backend=backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                    local_attn_size=self.local_attn_size,
                    sink_size=self.sink_size,
                    # Multi-view args
                    cross_view_attn_map=self.cross_view_attn_map,
                    enable_cross_view_attn=self.enable_cross_view_attn,
                    v_split_mode=self.v_split_mode,
                )
                for _ in range(self.num_blocks)
            ]
        )

        if self.concat_view_embedding:
            self.view_embeddings = nn.Embedding(self.n_cameras_emb, view_condition_dim)

        if self.adaln_view_embedding:
            self.adaln_view_embedder = nn.Embedding(self.n_cameras_emb, self.model_channels)
            self.adaln_view_proj = nn.Linear(self.model_channels, self.model_channels * 9)

        # Shared cache for cross-view attention BlockMasks (used when v_split_mode is True)
        # This is initialized per-forward call and shared across all blocks
        self.cross_view_block_mask_cache: dict[tuple, BlockMask] = {}

        # Initialize new weights
        self.init_weights()

        if not self.postpone_checkpoint:
            self.enable_selective_checkpoint(sac_config, self.blocks)

    def _build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = MultiCameraVideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        len_h = self.max_img_h // self.patch_spatial
        len_w = self.max_img_w // self.patch_spatial
        len_t = self.max_frames // self.patch_temporal
        head_dim = self.model_channels // self.num_heads

        # In CosmosCausalDiT, pos_embedder is single instance.
        # In MultiViewCrossDiT, it creates a dict of embedders for different n_cameras.
        # Since CosmosCausalDiT supports dynamic resolution/frames, we might stick to a single flexible embedder
        # or dictionary if n_cameras varies.
        # Here we assume n_cameras is derived dynamically but we can use one embedder instance initialized with max n_cameras?
        # MultiCameraVideoRopePosition3DEmb takes n_cameras in init.

        self.pos_embedder_options = nn.ModuleDict()
        # We build for max expected cameras or just the one needed.
        # MultiViewCrossDiT iterates 1..n_cameras_emb.
        for n_cam in range(1, self.n_cameras_emb + 1):
            self.pos_embedder_options[f"n_cameras_{n_cam}"] = cls_type(
                head_dim=head_dim,
                len_h=len_h,
                len_w=len_w,
                len_t=len_t,
                h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
                w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
                t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
                enable_fps_modulation=self.rope_enable_fps_modulation,
                n_cameras=n_cam,
            )

        # Set self.pos_embedder to something default to satisfy base class if needed,
        # but we should override usage.
        self.pos_embedder = self.pos_embedder_options[f"n_cameras_{1}"]  # Default
        if self.extra_per_block_abs_pos_emb:
            raise NotImplementedError("extra_per_block_abs_pos_emb not fully supported in MultiView port yet")

    def init_weights(self) -> None:
        super().init_weights()
        if hasattr(self, "view_embeddings"):
            torch.nn.init.normal_(self.view_embeddings.weight, mean=0.0, std=0.02)
        if hasattr(self, "adaln_view_embedder"):
            torch.nn.init.normal_(self.adaln_view_embedder.weight, mean=0.0, std=0.05)
        if hasattr(self, "adaln_view_proj"):
            torch.nn.init.zeros_(self.adaln_view_proj.weight)
            torch.nn.init.zeros_(self.adaln_view_proj.bias)

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            # Each block uses its own v_split_mode to determine CP behavior
            block.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )

        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.enable_context_parallel(process_group=process_group)
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.enable_context_parallel(process_group=process_group)

        self._is_context_parallel_enabled = True
        self.cp_group = process_group

    def disable_context_parallel(self):
        # pos_embedder
        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.disable_context_parallel()

        # attention
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False
        self.cp_group = None

    def _prepare_multiview_input(
        self,
        x_B_C_T_H_W: torch.Tensor,
        view_indices_B_T: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        n_views: Optional[int] = None,
    ):
        """
        Prepare multiview inputs.
        Extracts view information, handles view embedding concatenation, and creates view_indices_B_V.

        Based on prepare_embedded_sequence from multiview_cross_dit.py.

        Args:
            x_B_C_T_H_W: Input tensor [B, C, T, H, W] where T = V * T_per_view
            view_indices_B_T: Optional view indices [B, V*T] mapping each frame to a view ID
            fps: Optional fps tensor (unused here but kept for interface consistency)
            n_views: Optional number of views. If not provided, will be computed from x_B_C_T_H_W.shape[2] // self.state_t

        Returns:
            x_B_C_T_H_W: Potentially modified input (with view embeddings concatenated if enabled)
            n_cameras: Number of cameras/views
            view_indices_B_V: View indices [B, V] mapping each view position to global view ID
        """
        # T_total = V * T_per_view
        if n_views is None:
            # Fall back to computing from shape only if not provided
            n_cameras = x_B_C_T_H_W.shape[2] // self.state_t
            T_per_view = self.state_t
        else:
            n_cameras = n_views
            T_per_view = x_B_C_T_H_W.shape[2] // n_views

        # Setup view indices and handle view embedding concatenation

        if view_indices_B_T is None:
            # Create default view indices [0, 1, ..., V-1]
            view_indices = torch.arange(n_cameras, device=x_B_C_T_H_W.device).clamp(max=self.n_cameras_emb - 1)
            if self.concat_view_embedding:
                view_embedding = self.view_embeddings(view_indices)  # [V, D]
                view_embedding = rearrange(view_embedding, "V D -> 1 D V 1 1 1")  # for broadcasting
                # Reshape x to split V
                x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_cameras)
                # Expand embedding
                view_embedding = view_embedding.expand(
                    x_B_C_V_T_H_W.shape[0],
                    -1,
                    -1,
                    x_B_C_V_T_H_W.shape[3],
                    x_B_C_V_T_H_W.shape[4],
                    x_B_C_V_T_H_W.shape[5],
                )

                x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)
                x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W")

            view_indices_B_V = view_indices.unsqueeze(0).expand(x_B_C_T_H_W.shape[0], -1)  # [B, V]
        else:
            # Validate view_indices_B_T shape and values
            expected_len = n_cameras * T_per_view
            if view_indices_B_T.shape[1] != expected_len:
                log.warning(
                    f"view_indices_B_T has unexpected length {view_indices_B_T.shape[1]}, "
                    f"expected {expected_len} (n_cameras={n_cameras}, T_per_view={T_per_view}, {view_indices_B_T}). "
                    f"Falling back to default view indices."
                )
                # Fall back to default indices
                view_indices = torch.arange(n_cameras, device=x_B_C_T_H_W.device).clamp(max=self.n_cameras_emb - 1)
                view_indices_B_V = view_indices.unsqueeze(0).expand(x_B_C_T_H_W.shape[0], -1)
                if self.concat_view_embedding:
                    view_embedding = self.view_embeddings(view_indices)  # [V, D]
                    view_embedding = rearrange(view_embedding, "V D -> 1 D V 1 1 1")  # for broadcasting
                    x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_cameras)
                    view_embedding = view_embedding.expand(
                        x_B_C_V_T_H_W.shape[0],
                        -1,
                        -1,
                        x_B_C_V_T_H_W.shape[3],
                        x_B_C_V_T_H_W.shape[4],
                        x_B_C_V_T_H_W.shape[5],
                    )
                    x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)
                    x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W")
            else:
                # Handle provided view indices
                view_indices_B_T = view_indices_B_T.clamp(min=0, max=self.n_cameras_emb - 1)
                view_indices_B_T = view_indices_B_T.to(x_B_C_T_H_W.device).long()
                if self.concat_view_embedding:
                    view_embedding = self.view_embeddings(view_indices_B_T)  # [B, V*T, D]
                    view_embedding = rearrange(view_embedding, "B (V T) D -> B D V T", V=n_cameras)
                    view_embedding = view_embedding.unsqueeze(-1).unsqueeze(-1)  # [B, D, V, T, 1, 1]

                    # Reshape x to split V
                    x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_cameras)
                    # Expand embedding to match spatial dimensions
                    view_embedding = view_embedding.expand(
                        x_B_C_V_T_H_W.shape[0],
                        view_embedding.shape[1],
                        view_embedding.shape[2],
                        x_B_C_V_T_H_W.shape[3],
                        x_B_C_V_T_H_W.shape[4],
                        x_B_C_V_T_H_W.shape[5],
                    )

                    x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)
                    x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W")

                # Extract view_indices_B_V from view_indices_B_T (take first timestep for each view)
                # Assumes view-first layout: view_indices_B_T has shape (B, V*T) with all frames of
                # view 0 first, then all frames of view 1, etc.
                view_indices_B_V_T = rearrange(view_indices_B_T, "B (V T) -> B V T", V=n_cameras)
                view_indices_B_V = view_indices_B_V_T[..., 0]  # [B, V]

                # Validate that view indices are consistent within each view
                # (all timesteps of a view should have the same view ID)
                if not torch.all(view_indices_B_V_T == view_indices_B_V.unsqueeze(-1)):
                    # Check if data might be in time-first layout instead
                    view_indices_T_V = rearrange(view_indices_B_T, "B (T V) -> B T V", V=n_cameras)
                    view_indices_B_V_from_t_first = view_indices_T_V[:, 0, :]  # First timestep

                    if torch.all(view_indices_T_V == view_indices_B_V_from_t_first.unsqueeze(1)):
                        log.error(
                            "view_indices_B_T appears to be in TIME-FIRST layout (T V) but code expects "
                            "VIEW-FIRST layout (V T). The data should be rearranged before passing to the model, "
                            "or the model's rearrange should use 'B (T V) -> B V T' instead of 'B (V T) -> B V T'."
                        )
                        # Use time-first layout
                        view_indices_B_V = view_indices_B_V_from_t_first
                    else:
                        log.warning(
                            "view_indices_B_T has inconsistent view IDs across timesteps and doesn't match "
                            "either view-first or time-first layout. This may cause incorrect attention."
                        )

        return x_B_C_T_H_W, n_cameras, view_indices_B_V

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int = 0,
        current_end: int = 0,
        start_frame_for_rope: int = 0,
        disable_kv_cache: bool = False,
        num_interleave: int = 0,
        img_context_emb: torch.Tensor | None = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
        n_views: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass dispatching to training or inference mode.

        Args:
            x_B_C_T_H_W: Input video tensor [B, C, T, H, W]
            condition_video_input_mask_B_C_T_H_W: Condition video input mask [B, C, T, H, W]
            timesteps_B_T: Timesteps [B, T]
            crossattn_emb: Cross-attention embeddings [B, N, D]
            fps: Frames per second (optional)
            padding_mask: Padding mask (optional)
            data_type: DataType enum
            kv_cache: KV cache for inference (None for training)
            crossattn_cache: Cross-attention cache for inference
            current_start/current_end: Token range for inference
            start_frame_for_rope: Frame offset for RoPE
            disable_kv_cache: Skip KV cache usage
            num_interleave: Interleave factor for multi-step generation
            img_context_emb: Image context for I2V
            view_indices_B_T: View indices [B, T]
            n_views: Number of views. If not provided, will be computed from x_B_C_T_H_W.shape[2] // self.state_t
        """
        del kwargs
        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        timesteps_B_T = timesteps_B_T * self.timestep_scale
        if kv_cache is not None:
            return self._forward_inference(
                x_B_C_T_H_W=x_B_C_T_H_W,
                timesteps_B_T=timesteps_B_T,
                crossattn_emb=crossattn_emb,
                fps=fps,
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                start_frame_for_rope=start_frame_for_rope,
                disable_kv_cache=disable_kv_cache,
                img_context_emb=img_context_emb,
                view_indices_B_T=view_indices_B_T,
                n_views=n_views,
            )
        else:
            return self._forward_train(
                x_B_C_T_H_W=x_B_C_T_H_W,
                timesteps_B_T=timesteps_B_T,
                crossattn_emb=crossattn_emb,
                fps=fps,
                padding_mask=padding_mask,
                data_type=data_type,
                num_interleave=num_interleave,
                img_context_emb=img_context_emb,
                view_indices_B_T=view_indices_B_T,
            )

    def _forward_train(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        num_interleave: int = 0,
        img_context_emb: torch.Tensor | None = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # 1. Prepare Multiview Inputs (View Embedding Concat)

        if self.concat_padding_mask and padding_mask is not None:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_C_T_H_W, n_cameras, view_indices_B_V = self._prepare_multiview_input(x_B_C_T_H_W, view_indices_B_T, fps)
        device = self.x_embedder.proj[1].weight.device

        # 2. Reshape to B V T H W for internal processing
        # Note: x_embedder expects B C T H W.
        # We can pass flattened T (V*T) to embedder, then reshape.

        # Patch embedding
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        # Reshape to B V T H W D
        x_B_V_T_H_W_D = rearrange(x_B_T_H_W_D, "b (v t) h w d -> b v t h w d", v=n_cameras)

        video_size = VideoSize(T=x_B_V_T_H_W_D.shape[2], H=x_B_V_T_H_W_D.shape[3], W=x_B_V_T_H_W_D.shape[4])

        # 3. Construct block-causal mask
        # We use single view T for masking if we treat each view as independent causal sequence in self-attn
        # In self-attn, we flatten to (B*V, L, D).
        # So we need a mask for length L = T*H*W.
        frame_seqlen = video_size.H * video_size.W
        num_frames = video_size.T
        cp_size = 1
        if self._is_context_parallel_enabled and self.cp_group is not None:
            cp_size = self.cp_group.size()
        # Effective cp_size for self-attention mask: 1 if v_split_mode (self-attn has no CP), else actual cp_size
        effective_cp_size = 1 if self.v_split_mode else cp_size
        mask_key = f"mask_f{num_frames}_seqlen{frame_seqlen}_block{self.num_frame_per_block}_cp{effective_cp_size}"
        if mask_key not in self.block_mask_dict:
            block_mask = self._prepare_blockwise_causal_attn_mask(
                device=device,
                num_frames=num_frames // (num_interleave + 1),
                frame_seqlen=frame_seqlen,
                num_frame_per_block=self.num_frame_per_block,
                num_interleave=num_interleave,
                cp_size=effective_cp_size,
            )
            self.block_mask_dict[mask_key] = block_mask
        else:
            block_mask = self.block_mask_dict[mask_key]

        # 4. RoPE embeddings
        # Use MultiCamera pos embedder
        pos_embedder = self.pos_embedder_options[f"n_cameras_{n_cameras}"]
        # rope_freq_L_1_1_D = pos_embedder(x_B_T_H_W_D, fps=fps)  # [(LV), 1, 1, D]
        rope_freq_VT_H_W_D = pos_embedder.generate_embeddings(x_B_T_H_W_D.shape, fps=fps)

        rope_freq_V_L_1_1_D = rearrange(rope_freq_VT_H_W_D, "(v t) h w d -> v (t h w) 1 1 d", v=n_cameras)

        # 5. Time embedding
        with torch.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_emb_B_T_D = self.t_embedding_norm(t_emb_B_T_D)

        # 6. View AdaLN embedding
        if self.adaln_view_embedding:
            with torch.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
                # view_indices_B_V
                view_embedding_B_V = self.adaln_view_embedder(view_indices_B_V)
                view_embedding_proj_B_V_9D = self.adaln_view_proj(view_embedding_B_V)
        else:
            view_embedding_proj_B_V_9D = None

        # 7. Context embeddings
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None and self.extra_image_context_dim is not None:
            img_context_emb = self.img_context_proj(img_context_emb)

        # 8. Prepare Block Inputs
        # Flatten x to B V L D
        x_B_V_L_D = rearrange(x_B_V_T_H_W_D, "b v t h w d -> b v (t h w) d")

        # Expand Time Embeddings
        # Ensure t_emb covers all views and frames: target shape is [B, V*T, D]
        # t_emb_B_T_D may have shape [B, 1], [B, T], or [B, V*T]
        total_VT = n_cameras * video_size.T
        if t_emb_B_T_D.shape[1] == 1:
            t_emb_B_T_D = t_emb_B_T_D.repeat(1, total_VT, 1)
        elif t_emb_B_T_D.shape[1] == video_size.T and n_cameras > 1:
            # Per-view timesteps only; replicate across all views
            t_emb_B_T_D = t_emb_B_T_D.repeat(1, n_cameras, 1)

        frame_seqlen_spatial = video_size.H * video_size.W
        t_emb_B_L_D = torch.repeat_interleave(t_emb_B_T_D, frame_seqlen_spatial, dim=1)  # [B, V*T*H*W, D]
        t_emb_B_V_L_D = rearrange(t_emb_B_L_D, "b (v l) d -> b v l d", v=n_cameras)

        if adaln_lora_B_T_3D is not None:
            if adaln_lora_B_T_3D.shape[1] == 1:
                adaln_lora_B_T_3D = adaln_lora_B_T_3D.repeat(1, total_VT, 1)
            elif adaln_lora_B_T_3D.shape[1] == video_size.T and n_cameras > 1:
                adaln_lora_B_T_3D = adaln_lora_B_T_3D.repeat(1, n_cameras, 1)
            adaln_lora_B_L_3D = torch.repeat_interleave(adaln_lora_B_T_3D, frame_seqlen_spatial, dim=1)
            adaln_lora_B_V_L_3D = rearrange(adaln_lora_B_L_3D, "b (v l) d -> b v l d", v=n_cameras)
        else:
            adaln_lora_B_V_L_3D = None

        # Context parallel: split inputs
        x_original_shape = x_B_V_L_D.shape
        cp_enabled = self._is_context_parallel_enabled and self.cp_group is not None
        T_before_cp = video_size.T
        n_cameras_before_cp = n_cameras
        if cp_enabled and self.cp_group.size() > 1:
            if self.v_split_mode:
                # v_split_mode: Split along V dimension (views distributed across devices)
                assert n_cameras % self.cp_group.size() == 0, (
                    f"n_cameras {n_cameras} is not divisible by cp_group.size() {self.cp_group.size()}"
                )
                x_B_V_L_D = split_inputs_cp(x_B_V_L_D, seq_dim=1, cp_group=self.cp_group)
                t_emb_B_V_L_D = split_inputs_cp(t_emb_B_V_L_D, seq_dim=1, cp_group=self.cp_group)
                rope_freq_V_L_1_1_D = split_inputs_cp(rope_freq_V_L_1_1_D, seq_dim=0, cp_group=self.cp_group)
                n_cameras = n_cameras // self.cp_group.size()
                if adaln_lora_B_V_L_3D is not None:
                    adaln_lora_B_V_L_3D = split_inputs_cp(adaln_lora_B_V_L_3D, seq_dim=1, cp_group=self.cp_group)
                if view_embedding_proj_B_V_9D is not None:
                    # In v_split_mode, split view_embedding_proj along V dimension
                    view_embedding_proj_B_V_9D = split_inputs_cp(
                        view_embedding_proj_B_V_9D, seq_dim=1, cp_group=self.cp_group
                    )
                # Update view_indices to match local views
                view_indices_B_V = split_inputs_cp(view_indices_B_V, seq_dim=1, cp_group=self.cp_group)
                # Split crossattn_emb along V dimension: (B, V*L_txt, D) -> (B, V_local*L_txt, D)
                # First reshape to (B, V, L_txt, D), split, then reshape back
                crossattn_emb = rearrange(crossattn_emb, "b (v l) d -> b v l d", v=n_cameras_before_cp)
                crossattn_emb = split_inputs_cp(crossattn_emb, seq_dim=1, cp_group=self.cp_group)
                crossattn_emb = rearrange(crossattn_emb, "b v l d -> b (v l) d")
            else:
                # Standard mode: Split along L dimension (sequence tokens distributed across devices)
                assert video_size.T % self.cp_group.size() == 0, (
                    f"video_size.T {video_size.T} is not divisible by cp_group.size() {self.cp_group.size()}"
                )
                x_B_V_L_D = split_inputs_cp(x_B_V_L_D, seq_dim=2, cp_group=self.cp_group)
                t_emb_B_V_L_D = split_inputs_cp(t_emb_B_V_L_D, seq_dim=2, cp_group=self.cp_group)
                rope_freq_V_L_1_1_D = split_inputs_cp(rope_freq_V_L_1_1_D, seq_dim=1, cp_group=self.cp_group)
                video_size = VideoSize(T=int(video_size.T // self.cp_group.size()), H=video_size.H, W=video_size.W)
                if adaln_lora_B_V_L_3D is not None:
                    adaln_lora_B_V_L_3D = split_inputs_cp(adaln_lora_B_V_L_3D, seq_dim=2, cp_group=self.cp_group)
                # In standard mode, view_embedding_proj is NOT split

            if distributed.get_rank() == 0 and DEBUG:
                print(
                    f"CP split shapes (train): x={x_B_V_L_D.shape}, t_emb={t_emb_B_V_L_D.shape}, rope={rope_freq_V_L_1_1_D.shape}"
                )
                if adaln_lora_B_V_L_3D is not None:
                    print(f"adaln_lora={adaln_lora_B_V_L_3D.shape}")

        # 9. Process Blocks
        # Initialize/clear cross-view block mask cache for this forward pass (v_split_mode only)
        cross_view_block_mask_cache = self.cross_view_block_mask_cache if self.v_split_mode else None
        if cross_view_block_mask_cache is not None:
            cross_view_block_mask_cache.clear()

        for block in self.blocks:
            if torch.is_grad_enabled() and self.on_the_fly_checkpoint:
                # Implement checkpointing logic adapted for B_V_L_D
                x_B_V_L_D = torch.utils.checkpoint.checkpoint(
                    block,
                    x_B_V_L_D,
                    t_emb_B_V_L_D,
                    crossattn_emb,
                    view_indices_B_V,
                    video_size,  # sv_video_size
                    rope_freq_V_L_1_1_D,
                    adaln_lora_B_V_L_3D,
                    None,  # extra_pos
                    block_mask,
                    None,
                    None,
                    0,
                    0,
                    False,
                    False,  # cache args
                    None,  # full video size (optional)
                    view_embedding_proj_B_V_9D,
                    cross_view_block_mask_cache,
                    use_reentrant=False,
                )
            else:
                x_B_V_L_D = block(
                    x_B_V_L_D=x_B_V_L_D,
                    emb_B_V_L_D=t_emb_B_V_L_D,
                    crossattn_emb=crossattn_emb,
                    view_indices_B_V=view_indices_B_V,
                    sv_video_size=video_size,
                    rope_emb_V_L_1_1_D=rope_freq_V_L_1_1_D,
                    adaln_lora_B_V_L_3D=adaln_lora_B_V_L_3D,
                    block_mask=block_mask,
                    view_embedding_proj_B_V_9D=view_embedding_proj_B_V_9D,
                    cross_view_block_mask_cache=cross_view_block_mask_cache,
                )
            if DEBUG:
                log.info("passed block", rank0_only=True)

        # 10. Final Layer
        # x_B_V_L_D -> B V T H W D
        if cp_enabled and self.cp_group is not None:
            if DEBUG:
                log.info("before cat_outputs_cp_with_grad", rank0_only=True)
            if self.v_split_mode:
                # v_split_mode: Gather along V dimension
                x_B_V_L_D = cat_outputs_cp_with_grad(x_B_V_L_D, seq_dim=1, cp_group=self.cp_group)
                n_cameras = n_cameras_before_cp  # Restore full n_cameras
            else:
                # Standard mode: Gather along L dimension
                x_B_V_L_D = cat_outputs_cp_with_grad(x_B_V_L_D, seq_dim=2, cp_group=self.cp_group)
            if DEBUG:
                log.info("after cat_outputs_cp_with_grad", rank0_only=True)
        x_BV_T_H_W_D = rearrange(x_B_V_L_D, "b v (t h w) d -> (b v) t h w d", h=video_size.H, w=video_size.W)

        # Prepare t_emb for final layer
        t_emb_BV_T_D = rearrange(t_emb_B_T_D, "b (v l) d -> (b v) l d", v=n_cameras_before_cp)
        if adaln_lora_B_T_3D is not None:
            adaln_lora_BV_T_3D = rearrange(adaln_lora_B_T_3D, "b (v l) d -> (b v) l d", v=n_cameras_before_cp)
        else:
            adaln_lora_BV_T_3D = None
        # x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_BV_T_H_W_O = self.final_layer(x_BV_T_H_W_D, t_emb_BV_T_D, adaln_lora_B_T_3D=adaln_lora_BV_T_3D)

        # Unpatchify
        # Output should be B C (V T) H W
        t = T_before_cp
        h = video_size.H
        w = video_size.W
        x_BV_C_T_H_W = rearrange(
            x_BV_T_H_W_O,
            "bv t h w (nt nh nw d) -> bv d (t nt) (h nh) (w nw)",
            nt=self.patch_temporal,
            nh=self.patch_spatial,
            nw=self.patch_spatial,
            t=t,
            h=h,
            w=w,
            d=self.out_channels,
        )
        x_B_C_VT_H_W = rearrange(x_BV_C_T_H_W, "(b v) c t h w -> b c (v t) h w", v=n_cameras_before_cp)

        return x_B_C_VT_H_W

    def _forward_inference(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int = 0,
        current_end: int = 0,
        start_frame_for_rope: int = 0,
        disable_kv_cache: bool = False,
        img_context_emb: torch.Tensor | None = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
        n_views: Optional[int] = None,
    ) -> torch.Tensor:
        """Inference forward pass with KV-caching."""
        # 1. Prepare Multiview Inputs (View Embedding Concat)
        if self.concat_padding_mask and padding_mask is not None:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_C_T_H_W, n_cameras, view_indices_B_V = self._prepare_multiview_input(
            x_B_C_T_H_W, view_indices_B_T, fps, n_views=n_views
        )

        device = cast(torch.device, self.x_embedder.proj[1].weight.device)

        # 2. Reshape to B V T H W for internal processing
        # Note: x_embedder expects B C T H W.
        # We can pass flattened T (V*T) to embedder, then reshape.

        # Patch embedding
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        # Reshape to B V T H W D
        x_B_V_T_H_W_D = rearrange(x_B_T_H_W_D, "b (v t) h w d -> b v t h w d", v=n_cameras)
        video_size = VideoSize(T=x_B_V_T_H_W_D.shape[2], H=x_B_V_T_H_W_D.shape[3], W=x_B_V_T_H_W_D.shape[4])

        # 3. RoPE embeddings with offset
        pos_embedder = self.pos_embedder_options[f"n_cameras_{n_cameras}"]

        if start_frame_for_rope > 0:
            T_chunk = video_size.T
            full_T_per_view = start_frame_for_rope + T_chunk

            # Construct dummy shape for generation
            full_shape = list(x_B_T_H_W_D.shape)
            full_shape[1] = n_cameras * full_T_per_view

            full_rope_VT_H_W_D = pos_embedder.generate_embeddings(torch.Size(full_shape), fps=fps)
            full_rope_freq_V_T_H_W_D = rearrange(full_rope_VT_H_W_D, "(v t) h w d -> v t h w d", v=n_cameras)
            rope_freq_V_T_H_W_D = full_rope_freq_V_T_H_W_D[:, start_frame_for_rope : start_frame_for_rope + T_chunk]
            rope_freq_V_L_1_1_D = rearrange(rope_freq_V_T_H_W_D, "v t h w d -> v (t h w) 1 1 d")
        else:
            rope_freq_VT_H_W_D = pos_embedder.generate_embeddings(x_B_T_H_W_D.shape, fps=fps)
            rope_freq_V_L_1_1_D = rearrange(rope_freq_VT_H_W_D, "(v t) h w d -> v (t h w) 1 1 d", v=n_cameras)

        # 5. Time Embedding
        with torch.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_emb_B_T_D = self.t_embedding_norm(t_emb_B_T_D)

        # 6. View Embedding
        if self.adaln_view_embedding:
            with torch.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
                view_embedding_B_V = self.adaln_view_embedder(view_indices_B_V)
                view_embedding_proj_B_V_9D = self.adaln_view_proj(view_embedding_B_V)
        else:
            view_embedding_proj_B_V_9D = None

        # 7. Context
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None and self.extra_image_context_dim is not None:
            img_context_emb = self.img_context_proj(img_context_emb)

        # 8. Prepare Block Inputs
        x_B_V_L_D = rearrange(x_B_V_T_H_W_D, "b v t h w d -> b v (t h w) d")

        # Expand Time Embeddings
        # Ensure t_emb covers all views and frames: target shape is [B, V*T, D]
        total_VT = n_cameras * video_size.T
        if t_emb_B_T_D.shape[1] == 1:
            t_emb_B_T_D = t_emb_B_T_D.repeat(1, total_VT, 1)
        elif t_emb_B_T_D.shape[1] == video_size.T and n_cameras > 1:
            t_emb_B_T_D = t_emb_B_T_D.repeat(1, n_cameras, 1)

        frame_seqlen_spatial = video_size.H * video_size.W
        t_emb_B_L_D = torch.repeat_interleave(t_emb_B_T_D, frame_seqlen_spatial, dim=1)
        t_emb_B_V_L_D = rearrange(t_emb_B_L_D, "b (v l) d -> b v l d", v=n_cameras)

        if adaln_lora_B_T_3D is not None:
            if adaln_lora_B_T_3D.shape[1] == 1:
                adaln_lora_B_T_3D = adaln_lora_B_T_3D.repeat(1, total_VT, 1)
            elif adaln_lora_B_T_3D.shape[1] == video_size.T and n_cameras > 1:
                adaln_lora_B_T_3D = adaln_lora_B_T_3D.repeat(1, n_cameras, 1)
            adaln_lora_B_L_3D = torch.repeat_interleave(adaln_lora_B_T_3D, frame_seqlen_spatial, dim=1)
            adaln_lora_B_V_L_3D = rearrange(adaln_lora_B_L_3D, "b (v l) d -> b v l d", v=n_cameras)
        else:
            adaln_lora_B_V_L_3D = None

        # Context parallel: split inputs
        x_original_shape = x_B_V_L_D.shape
        cp_enabled = self._is_context_parallel_enabled and self.cp_group is not None
        T_before_cp = video_size.T
        n_cameras_before_cp = n_cameras
        if cp_enabled and self.cp_group.size() > 1:
            if self.v_split_mode:
                # v_split_mode: Split along V dimension (views distributed across devices)
                assert n_cameras % self.cp_group.size() == 0, (
                    f"n_cameras {n_cameras} is not divisible by cp_group.size() {self.cp_group.size()}"
                )
                x_B_V_L_D = split_inputs_cp(x_B_V_L_D, seq_dim=1, cp_group=self.cp_group)
                t_emb_B_V_L_D = split_inputs_cp(t_emb_B_V_L_D, seq_dim=1, cp_group=self.cp_group)
                rope_freq_V_L_1_1_D = split_inputs_cp(rope_freq_V_L_1_1_D, seq_dim=0, cp_group=self.cp_group)
                n_cameras = n_cameras // self.cp_group.size()
                if adaln_lora_B_V_L_3D is not None:
                    adaln_lora_B_V_L_3D = split_inputs_cp(adaln_lora_B_V_L_3D, seq_dim=1, cp_group=self.cp_group)
                if view_embedding_proj_B_V_9D is not None:
                    # In v_split_mode, split view_embedding_proj along V dimension
                    view_embedding_proj_B_V_9D = split_inputs_cp(
                        view_embedding_proj_B_V_9D, seq_dim=1, cp_group=self.cp_group
                    )
                # Update view_indices to match local views
                view_indices_B_V = split_inputs_cp(view_indices_B_V, seq_dim=1, cp_group=self.cp_group)
                # Split crossattn_emb along V dimension: (B, V*L_txt, D) -> (B, V_local*L_txt, D)
                # First reshape to (B, V, L_txt, D), split, then reshape back
                crossattn_emb = rearrange(crossattn_emb, "b (v l) d -> b v l d", v=n_cameras_before_cp)
                crossattn_emb = split_inputs_cp(crossattn_emb, seq_dim=1, cp_group=self.cp_group)
                crossattn_emb = rearrange(crossattn_emb, "b v l d -> b (v l) d")
            else:
                # Standard mode: Split along L dimension (sequence tokens distributed across devices)
                assert video_size.T % self.cp_group.size() == 0, (
                    f"video_size.T {video_size.T} is not divisible by cp_group.size() {self.cp_group.size()}"
                )
                x_B_V_L_D = split_inputs_cp(x_B_V_L_D, seq_dim=2, cp_group=self.cp_group)
                t_emb_B_V_L_D = split_inputs_cp(t_emb_B_V_L_D, seq_dim=2, cp_group=self.cp_group)
                rope_freq_V_L_1_1_D = split_inputs_cp(rope_freq_V_L_1_1_D, seq_dim=1, cp_group=self.cp_group)
                video_size = VideoSize(T=int(video_size.T // self.cp_group.size()), H=video_size.H, W=video_size.W)
                if adaln_lora_B_V_L_3D is not None:
                    adaln_lora_B_V_L_3D = split_inputs_cp(adaln_lora_B_V_L_3D, seq_dim=2, cp_group=self.cp_group)
                # In standard mode, view_embedding_proj is NOT split

            if distributed.get_rank() == 0 and DEBUG:
                print(
                    f"CP split shapes (inference): x={x_B_V_L_D.shape}, t_emb={t_emb_B_V_L_D.shape}, rope={rope_freq_V_L_1_1_D.shape}"
                )
                if adaln_lora_B_V_L_3D is not None:
                    print(f"adaln_lora={adaln_lora_B_V_L_3D.shape}")

        # 9. Process Blocks
        # Initialize/clear cross-view block mask cache for this forward pass (v_split_mode only)
        cross_view_block_mask_cache = self.cross_view_block_mask_cache if self.v_split_mode else None
        if cross_view_block_mask_cache is not None:
            cross_view_block_mask_cache.clear()

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block_idx, block in enumerate(self.blocks):
            block_kv_cache = kv_cache[block_idx] if kv_cache is not None and not disable_kv_cache else None
            block_crossattn_cache = crossattn_cache[block_idx] if crossattn_cache is not None else None

            if torch.is_grad_enabled() and self.on_the_fly_checkpoint:
                x_B_V_L_D = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x_B_V_L_D,
                    t_emb_B_V_L_D,
                    crossattn_emb,
                    view_indices_B_V,
                    video_size,
                    rope_freq_V_L_1_1_D,
                    adaln_lora_B_V_L_3D,
                    None,
                    None,  # block_mask
                    block_kv_cache,
                    block_crossattn_cache,
                    current_start,
                    current_end,
                    disable_kv_cache,
                    False,  # disable_kv_cache_update
                    None,  # full video size
                    view_embedding_proj_B_V_9D,
                    cross_view_block_mask_cache,
                    use_reentrant=False,
                )
            else:
                x_B_V_L_D = block(
                    x_B_V_L_D=x_B_V_L_D,
                    emb_B_V_L_D=t_emb_B_V_L_D,
                    crossattn_emb=crossattn_emb,
                    view_indices_B_V=view_indices_B_V,
                    sv_video_size=video_size,
                    rope_emb_V_L_1_1_D=rope_freq_V_L_1_1_D,
                    adaln_lora_B_V_L_3D=adaln_lora_B_V_L_3D,
                    view_embedding_proj_B_V_9D=view_embedding_proj_B_V_9D,
                    block_mask=None,
                    kv_cache=block_kv_cache,
                    crossattn_cache=block_crossattn_cache,
                    current_start=current_start,
                    current_end=current_end,
                    disable_kv_cache=disable_kv_cache,
                    cross_view_block_mask_cache=cross_view_block_mask_cache,
                )

        # 10. Final Layer
        if cp_enabled and self.cp_group is not None:
            if torch.is_grad_enabled():
                if DEBUG:
                    log.info("before cat_outputs_cp_with_grad", rank0_only=True)
                if self.v_split_mode:
                    # v_split_mode: Gather along V dimension
                    x_B_V_L_D = cat_outputs_cp_with_grad(x_B_V_L_D, seq_dim=1, cp_group=self.cp_group)
                    n_cameras = n_cameras_before_cp  # Restore full n_cameras
                else:
                    # Standard mode: Gather along L dimension
                    x_B_V_L_D = cat_outputs_cp_with_grad(x_B_V_L_D, seq_dim=2, cp_group=self.cp_group)
                if DEBUG:
                    log.info("after cat_outputs_cp_with_grad", rank0_only=True)
            else:
                if DEBUG:
                    log.info("before cat_outputs_cp", rank0_only=True)
                if self.v_split_mode:
                    # v_split_mode: Gather along V dimension
                    x_B_V_L_D = cat_outputs_cp(x_B_V_L_D, seq_dim=1, cp_group=self.cp_group)
                    n_cameras = n_cameras_before_cp  # Restore full n_cameras
                else:
                    # Standard mode: Gather along L dimension
                    x_B_V_L_D = cat_outputs_cp(x_B_V_L_D, seq_dim=2, cp_group=self.cp_group)
                if DEBUG:
                    log.info("after cat_outputs_cp", rank0_only=True)
        x_BV_T_H_W_D = rearrange(x_B_V_L_D, "b v (t h w) d -> (b v) t h w d", h=video_size.H, w=video_size.W)

        t_emb_BV_T_D = rearrange(t_emb_B_T_D, "b (v l) d -> (b v) l d", v=n_cameras_before_cp)
        if adaln_lora_B_T_3D is not None:
            adaln_lora_BV_T_3D = rearrange(adaln_lora_B_T_3D, "b (v l) d -> (b v) l d", v=n_cameras_before_cp)
        else:
            adaln_lora_BV_T_3D = None

        x_BV_T_H_W_O = self.final_layer(x_BV_T_H_W_D, t_emb_BV_T_D, adaln_lora_B_T_3D=adaln_lora_BV_T_3D)

        # Unpatchify
        t = T_before_cp
        h = video_size.H
        w = video_size.W
        x_BV_C_T_H_W = rearrange(
            x_BV_T_H_W_O,
            "bv t h w (nt nh nw d) -> bv d (t nt) (h nh) (w nw)",
            nt=self.patch_temporal,
            nh=self.patch_spatial,
            nw=self.patch_spatial,
            t=t,
            h=h,
            w=w,
            d=self.out_channels,
        )
        x_B_C_VT_H_W = rearrange(x_BV_C_T_H_W, "(b v) c t h w -> b c (v t) h w", v=n_cameras_before_cp)

        return x_B_C_VT_H_W
