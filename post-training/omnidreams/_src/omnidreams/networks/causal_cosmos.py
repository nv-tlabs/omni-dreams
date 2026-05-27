# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
CosmosCausalDiT: A causal DiT model combining Cosmos architecture with
block-causal attention masking and KV-caching for autoregressive generation.
"""

import math
from collections import namedtuple

import torch
import torch.amp as amp
import torch.nn as nn
import transformer_engine as te
from einops import rearrange
from torch.distributed import ProcessGroup, get_process_group_ranks, get_rank, is_initialized
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torchvision import transforms

from omnidreams._src.imaginaire.utils import distributed

try:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
except ImportError:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb

from transformer_engine.pytorch.attention import DotProductAttention

from omnidreams._src.imaginaire.utils import log
from omnidreams._src.imaginaire.utils.context_parallel import cat_outputs_cp, cat_outputs_cp_with_grad
from omnidreams._src.predict2.conditioner import DataType
from omnidreams._src.predict2.networks.minimal_v4_dit import (
    Attention,
    CheckpointMode,
    FinalLayer,
    GPT2FeedForward,
    I2VCrossAttention,
    LearnablePosEmbAxis,
    PatchEmbed,
    SACConfig,
    TimestepEmbedding,
    Timesteps,
    VideoRopePosition3DEmb,
)
from omnidreams._src.predict2.networks.model_weights_stats import WeightTrainingStat
from omnidreams._src.omnidreams.modules.flex_attention import flex_attention_cp

# Compile flex_attention for better performance
flex_attention = torch.compile(flex_attention, dynamic=False)

VideoSize = namedtuple("VideoSize", ["T", "H", "W"])
DEBUG = False


class CausalSelfAttention(nn.Module):
    """
    Self-attention module with block-causal mask support and KV-caching.

    Training mode: Uses flex_attention with BlockMask for efficient causal attention.
    Inference mode: Uses KV-caching with optional local attention window and sink tokens.

    API aligned with Attention class from minimal_v4_dit.py.

    Args:
        query_dim: The dimensionality of the query vectors
        context_dim: The dimensionality of the context vectors (None for self-attention)
        n_heads: Number of attention heads
        head_dim: The dimension of each attention head
        dropout: Dropout probability applied to the output
        qkv_format: Format specification for QKV tensors
        backend: Backend to use for the attention operation
        use_wan_fp32_strategy: Whether to use WAN's FP32 strategy for RoPE
        local_attn_size: Window size for local attention (-1 for global attention)
        sink_size: Number of sink frames to keep at the start of KV cache
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
        backend: str = "transformer_engine",
        use_wan_fp32_strategy: bool = False,
        local_attn_size: int = -1,
        sink_size: int = 0,
    ):
        super().__init__()
        log.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{n_heads} heads with a dimension of {head_dim}."
        )
        self.is_selfattn = context_dim is None  # self attention

        assert backend in ["transformer_engine", "torch", "minimal_a2a"], f"Invalid backend: {backend}"
        self.backend = backend

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        # QKV projections (matching Attention class)
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        # QK normalization (matching Attention class)
        self.q_norm = te.pytorch.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = te.pytorch.RMSNorm(head_dim, eps=1e-6)
        self.v_norm = nn.Identity()

        # Attention operator for inference (no mask)
        self.attn_op = DotProductAttention(
            n_heads,
            head_dim,
            num_gqa_groups=n_heads,
            attention_dropout=0,
            qkv_format=qkv_format,
            attn_mask_type="no_mask",
        )

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim

        self.cp_group: ProcessGroup | None = None

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rope_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K."""
        original_dtype = q.dtype
        if self.use_wan_fp32_strategy:
            q = q.to(torch.float32)
            k = k.to(torch.float32)

        q = apply_rotary_pos_emb(q, rope_emb, tensor_format="bshd", fused=True)
        k = apply_rotary_pos_emb(k, rope_emb, tensor_format="bshd", fused=True)

        if self.use_wan_fp32_strategy:
            q = q.to(original_dtype)
            k = k.to(original_dtype)

        return q, k

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        video_size: VideoSize | None = None,
        # Causal-specific parameters
        block_mask: BlockMask | None = None,
        kv_cache: dict | None = None,
        current_start: int = 0,
        current_end: int = 0,
        disable_kv_cache: bool = False,
        disable_kv_cache_update: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with block-causal attention or KV-caching.

        API aligned with Attention.forward(x, context, rope_emb, video_size).

        Args:
            x: The query tensor of shape [B, L, D]
            context: The key tensor (unused for self-attention, kept for API compatibility)
            rope_emb: RoPE embedding tensor [L, 1, 1, head_dim]
            video_size: VideoSize namedtuple with T, H, W
            block_mask: BlockMask for training (None for inference)
            kv_cache: KV cache dict for inference
            current_start: Start token index for current chunk
            current_end: End token index for current chunk
            disable_kv_cache: Skip KV cache usage
            disable_kv_cache_update: Skip updating KV cache
        """
        del context  # Unused for self-attention

        b, s, _ = x.shape
        n, d = self.n_heads, self.head_dim

        # Compute Q, K, V
        q = self.q_norm(self.q_proj(x).view(b, s, n, d))
        k = self.k_norm(self.k_proj(x).view(b, s, n, d))
        v = self.v_proj(x).view(b, s, n, d)

        if kv_cache is None:
            # Training mode: use flex_attention with block mask
            roped_q, roped_k = self._apply_rope(q, k, rope_emb)
            roped_q = roped_q.type_as(v)
            roped_k = roped_k.type_as(v)

            # Pad to multiple of 128 for flex_attention
            padded_length = math.ceil(s / 128) * 128 - s

            if padded_length > 0:
                pad_shape = [b, padded_length, n, d]
                roped_q = torch.cat([roped_q, torch.zeros(pad_shape, device=q.device, dtype=v.dtype)], dim=1)
                roped_k = torch.cat([roped_k, torch.zeros(pad_shape, device=k.device, dtype=v.dtype)], dim=1)
                v_padded = torch.cat([v, torch.zeros(pad_shape, device=v.device, dtype=v.dtype)], dim=1)
            else:
                v_padded = v

            # Apply flex_attention with context parallel support
            # flex_attention expects [B, H, S, D] format
            out = flex_attention_cp(
                query=roped_q.transpose(2, 1),
                key=roped_k.transpose(2, 1),
                value=v_padded.transpose(2, 1),
                block_mask=block_mask,
                process_group=self.cp_group,
                flex_attention_fn=flex_attention,
            )

            # Remove padding and transpose back
            if padded_length > 0:
                out = out[:, :, :-padded_length].transpose(2, 1)
            else:
                out = out.transpose(2, 1)

        elif disable_kv_cache:
            # Inference without KV cache
            roped_q, roped_k = self._apply_rope(q, k, rope_emb)
            out = self.attn_op(roped_q.type_as(v), roped_k.type_as(v), v, video_size)

        else:
            # Inference mode with KV caching
            roped_q, roped_k = self._apply_rope(q, k, rope_emb)
            roped_q = roped_q.type_as(v)
            roped_k = roped_k.type_as(v)

            frame_seqlen = video_size.H * video_size.W
            if self.cp_group is not None:
                assert frame_seqlen % self.cp_group.size() == 0
                frame_seqlen = frame_seqlen // self.cp_group.size()

            assert current_end == current_start + roped_q.shape[1]

            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_q.shape[1]

            # Handle local attention window with rolling cache
            if (
                self.local_attn_size != -1
                and (current_end > kv_cache["global_end_index"].item())
                and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)
            ):
                # Roll the cache to make room for new tokens
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()

                local_end_index = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                    - num_evicted_tokens
                )
                local_start_index = local_end_index - num_new_tokens

                if not disable_kv_cache_update:
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_k
                    kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Standard cache update
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                if torch.is_grad_enabled():
                    # Gradient-enabled update (for self-forcing training)
                    kv_cache["k"] = torch.cat(
                        [kv_cache["k"][:, :local_start_index], roped_k, kv_cache["k"][:, local_end_index:]],
                        dim=1,
                    )
                    kv_cache["v"] = torch.cat(
                        [kv_cache["v"][:, :local_start_index], v, kv_cache["v"][:, local_end_index:]],
                        dim=1,
                    )
                else:
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_k
                    kv_cache["v"][:, local_start_index:local_end_index] = v

            # Retrieve cached K, V for attention
            if disable_kv_cache_update:
                cached_k = torch.cat([kv_cache["k"][:, :local_start_index], roped_k], dim=1)
                cached_v = torch.cat([kv_cache["v"][:, :local_start_index], v], dim=1)
            else:
                if self.local_attn_size != -1:
                    window_tokens = (self.local_attn_size - self.sink_size) * frame_seqlen
                    cached_k = torch.cat(
                        [
                            kv_cache["k"][:, :sink_tokens],
                            kv_cache["k"][:, max(sink_tokens, local_end_index - window_tokens) : local_end_index],
                        ],
                        dim=1,
                    )
                    cached_v = torch.cat(
                        [
                            kv_cache["v"][:, :sink_tokens],
                            kv_cache["v"][:, max(sink_tokens, local_end_index - window_tokens) : local_end_index],
                        ],
                        dim=1,
                    )
                else:
                    cached_k = kv_cache["k"][:, :local_end_index]
                    cached_v = kv_cache["v"][:, :local_end_index]

            out = self.attn_op(roped_q, cached_k, cached_v, video_size)

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # Output projection (matching Attention class pattern)
        out = out.flatten(2)
        return self.output_dropout(self.output_proj(out))

    def set_context_parallel_group(self, process_group: ProcessGroup | None, ranks, stream, cp_comm_type: str = "p2p"):
        self.attn_op.set_context_parallel_group(process_group, ranks, stream, cp_comm_type=cp_comm_type)
        self.cp_group = process_group


class CausalCosmosBlock(nn.Module):
    """
    Transformer block with causal self-attention, cross-attention, and MLP.

    Uses AdaLN modulation for timestep conditioning following the Cosmos architecture.
    API aligned with Block class from minimal_v4_dit.py.
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
    ):
        super().__init__()
        self.x_dim = x_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        # Self-attention with causal masking
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = CausalSelfAttention(
            query_dim=x_dim,
            context_dim=None,
            n_heads=num_heads,
            head_dim=x_dim // num_heads,
            qkv_format="bshd",
            backend=backend,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )

        # Cross-attention (using standard Attention from minimal_v4_dit)
        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        if image_context_dim is None:
            self.cross_attn = Attention(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                qkv_format="bshd",
                backend=backend,
            )
        else:
            self.cross_attn = I2VCrossAttention(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                img_latent_dim=image_context_dim,
                qkv_format="bshd",
                backend=backend,
            )

        # MLP
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        # AdaLN modulation
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.cp_size: int | None = None

    def init_weights(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        self.cp_size = None if ranks is None else len(ranks)
        self.self_attn.set_context_parallel_group(process_group, ranks, stream, cp_comm_type=cp_comm_type)

    def forward(
        self,
        x_B_L_D: torch.Tensor,
        emb_B_L_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_B_L_D: torch.Tensor | None = None,
        adaln_lora_B_L_3D: torch.Tensor | None = None,
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
    ) -> torch.Tensor:
        """
        Forward pass through the block.

        API aligned with Block.forward() from minimal_v4_dit.py.

        Args:
            x_B_L_D: Input tensor [B, L, D]
            emb_B_L_D: Time embedding [B, L, D]
            crossattn_emb: Cross-attention context
            rope_emb_B_L_D: RoPE embeddings [L, 1, 1, D] or [B, L, 1, 1, D]
            adaln_lora_B_L_3D: AdaLN LoRA embeddings [B, L, 3D]
            extra_per_block_pos_emb: Extra positional embeddings [B, L, D]
            block_mask: Block-causal mask for training (causal-specific)
            kv_cache: KV cache for inference (causal-specific)
            crossattn_cache: Cross-attention cache (causal-specific)
            current_start/current_end: Token range for inference (causal-specific)
            disable_kv_cache: Skip KV cache (causal-specific)
            disable_kv_cache_update: Skip cache updates (causal-specific)
            video_size: VideoSize tuple (T, H, W)
        """
        if extra_per_block_pos_emb is not None:
            x_B_L_D = x_B_L_D + extra_per_block_pos_emb

        # Compute AdaLN modulation
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                shift_self, scale_self, gate_self = (
                    self.adaln_modulation_self_attn(emb_B_L_D) + adaln_lora_B_L_3D
                ).chunk(3, dim=-1)
                shift_cross, scale_cross, gate_cross = (
                    self.adaln_modulation_cross_attn(emb_B_L_D) + adaln_lora_B_L_3D
                ).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_B_L_D) + adaln_lora_B_L_3D).chunk(
                    3, dim=-1
                )
            else:
                shift_self, scale_self, gate_self = self.adaln_modulation_self_attn(emb_B_L_D).chunk(3, dim=-1)
                shift_cross, scale_cross, gate_cross = self.adaln_modulation_cross_attn(emb_B_L_D).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_L_D).chunk(3, dim=-1)

        # No reshape needed as inputs are already B L D and can broadcast to B L D

        shift_self = shift_self.type_as(x_B_L_D)
        scale_self = scale_self.type_as(x_B_L_D)
        gate_self = gate_self.type_as(x_B_L_D)

        shift_cross = shift_cross.type_as(x_B_L_D)
        scale_cross = scale_cross.type_as(x_B_L_D)
        gate_cross = gate_cross.type_as(x_B_L_D)

        shift_mlp = shift_mlp.type_as(x_B_L_D)
        scale_mlp = scale_mlp.type_as(x_B_L_D)
        gate_mlp = gate_mlp.type_as(x_B_L_D)

        # Self-attention (API aligned with Attention.forward)
        normed_x = self.layer_norm_self_attn(x_B_L_D) * (1 + scale_self) + shift_self

        attn_out = self.self_attn(
            normed_x,
            context=None,  # self-attention
            rope_emb=rope_emb_B_L_D,
            video_size=video_size,
            block_mask=block_mask,
            kv_cache=kv_cache,
            current_start=current_start,
            current_end=current_end,
            disable_kv_cache=disable_kv_cache,
            disable_kv_cache_update=disable_kv_cache_update,
        )
        x_B_L_D = x_B_L_D + gate_self * attn_out

        # Cross-attention
        # Note that RoPE is not applied to cross-attention. See omnidreams/_src/predict2/networks/minimal_v4_dit.py L527
        normed_x = self.layer_norm_cross_attn(x_B_L_D) * (1 + scale_cross) + shift_cross

        if crossattn_cache is not None:
            cross_out = self.cross_attn(normed_x, crossattn_emb, crossattn_cache=crossattn_cache)
        else:
            cross_out = self.cross_attn(normed_x, crossattn_emb)

        x_B_L_D = x_B_L_D + gate_cross * cross_out

        # MLP
        normed_x = self.layer_norm_mlp(x_B_L_D) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(normed_x)
        x_B_L_D = x_B_L_D + gate_mlp * mlp_out

        return x_B_L_D


class CosmosCausalDiT(WeightTrainingStat):
    """
    Causal DiT model for video generation with block-causal attention and KV-caching.

    Combines the Cosmos DiT architecture with causal attention masking for
    autoregressive video generation.
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        extra_image_context_dim: int | None = None,
        pos_emb_cls: str = "rope3d",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        timestep_scale: float = 1.0,
        # Causal-specific parameters
        local_attn_size: int = -1,
        sink_size: int = 0,
        sac_config: SACConfig = SACConfig(),
        postpone_checkpoint: bool = False,
        on_the_fly_checkpoint: bool = False,
        use_wan_fp32_strategy: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.timestep_scale = timestep_scale
        # add 1 for the condition mask
        self.in_channels = in_channels + 1
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_layers = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.on_the_fly_checkpoint = on_the_fly_checkpoint

        # Positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.extra_image_context_dim = extra_image_context_dim

        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = crossattn_proj_in_channels

        if on_the_fly_checkpoint:
            self.postpone_checkpoint = True
        else:
            self.postpone_checkpoint = postpone_checkpoint

        # Build embeddings
        self._build_patch_embed()
        self._build_pos_embed()

        # Time embeddings
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        self.t_embedding_norm = te.pytorch.RMSNorm(model_channels, eps=1e-6)

        # Transformer blocks (API aligned with Block from minimal_v4_dit)
        self.blocks = nn.ModuleList(
            [
                CausalCosmosBlock(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend="transformer_engine",
                    image_context_dim=None if extra_image_context_dim is None else model_channels,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        # Optional projections
        if extra_image_context_dim is not None:
            self.img_context_proj = nn.Sequential(
                nn.Linear(extra_image_context_dim, model_channels, bias=True),
                nn.GELU(),
            )

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        # Initialize weights
        self.init_weights()

        # Selective activation checkpointing
        self.sac_config = sac_config
        if not postpone_checkpoint:
            self.enable_selective_checkpoint(sac_config, self.blocks)

        # Block mask cache
        self.block_mask_dict: dict[str, BlockMask] = {}
        self.num_frame_per_block = 1

        # Context parallel
        self.cp_group: ProcessGroup | None = None
        self._is_context_parallel_enabled = False

    def _build_patch_embed(self) -> None:
        in_ch = self.in_channels + 1 if self.concat_padding_mask else self.in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_ch,
            out_channels=self.model_channels,
        )

    def _build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        len_h = self.max_img_h // self.patch_spatial
        len_w = self.max_img_w // self.patch_spatial
        len_t = self.max_frames // self.patch_temporal
        head_dim = self.model_channels // self.num_heads

        self.pos_embedder = cls_type(
            head_dim=head_dim,
            len_h=len_h,
            len_w=len_w,
            len_t=len_t,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )

        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = LearnablePosEmbAxis(
                interpolation=self.pos_emb_interpolation,
                model_channels=self.model_channels,
                len_h=len_h,
                len_w=len_w,
                len_t=len_t,
            )

    def init_weights(self) -> None:
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()

        self.t_embedder[1].init_weights()
        self.t_embedding_norm.reset_parameters()

        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()

        if self.extra_image_context_dim is not None:
            self.img_context_proj[0].reset_parameters()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        num_interleave: int = 0,
        cp_size: int = 1,
    ) -> BlockMask:
        """
        Prepare block-wise causal attention mask for flex_attention.

        The token sequence is divided into blocks of num_frame_per_block frames.
        Tokens can attend to all tokens in previous blocks and within their own block
        up to and including their position (causal within block).
        """
        log.info(
            f"Constructing block mask: num_frames={num_frames}, frame_seqlen={frame_seqlen}, "
            f"num_frame_per_block={num_frame_per_block}, cp_size={cp_size}"
        )

        total_length = num_frames * frame_seqlen * (1 + num_interleave)

        # Calculate padding considering CP distribution
        # Each rank pads its local sequence to multiple of 128
        local_len = total_length // cp_size
        local_padded_len = math.ceil(local_len / 128) * 128
        total_padded_len = local_padded_len * cp_size
        padded_length = total_padded_len - total_length

        log.info(f"Block maskpadded_length={padded_length}")

        if num_interleave == 0:
            # Standard block-wise causal mask
            # Note: ends array is constructed in logical space (size total_length)
            # We map physical indices to logical indices in the mask function
            ends = torch.zeros(total_padded_len, device=device, dtype=torch.long)
            frame_indices = torch.arange(
                start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device
            )
            for idx in frame_indices:
                ends[idx : idx + frame_seqlen * num_frame_per_block] = idx + frame_seqlen * num_frame_per_block

            def attention_mask(b, h, q_idx, kv_idx):
                # Map physical (padded) indices to logical indices
                q_rank = q_idx // local_padded_len
                q_off = q_idx % local_padded_len
                q_logical = q_rank * local_len + q_off
                q_valid = q_off < local_len

                kv_rank = kv_idx // local_padded_len
                kv_off = kv_idx % local_padded_len
                kv_logical = kv_rank * local_len + kv_off
                kv_valid = kv_off < local_len
                # Causal logic in logical space
                causal_check = (
                    q_valid & kv_valid & ((kv_logical < ends[q_logical.to(torch.long)]) | (q_logical == kv_logical))
                )

                return causal_check

        else:
            # Interleaved pattern for multi-step generation
            num_interleaved_frames_per_time = 1 + num_interleave
            # Construct properties in logical space
            frame_types = torch.zeros(total_padded_len, device=device, dtype=torch.long)
            frame_indices = torch.zeros(total_padded_len, device=device, dtype=torch.long)
            total_indices_per_time = frame_seqlen * num_frame_per_block * num_interleaved_frames_per_time

            position = 0
            time_frame_idx = 0
            while position < total_length:
                for _ in range(num_frame_per_block):
                    for interleave_idx in range(num_interleaved_frames_per_time):
                        end_pos = min(position + frame_seqlen, total_length)
                        frame_types[position:end_pos] = interleave_idx
                        frame_indices[position:end_pos] = time_frame_idx
                        position = end_pos
                        if position >= total_length:
                            break
                time_frame_idx += 1

            attend_frame_type = num_interleaved_frames_per_time - 1

            def attention_mask(b, h, q_idx, kv_idx):
                # Map physical (padded) indices to logical indices
                q_rank = q_idx // local_padded_len
                q_off = q_idx % local_padded_len
                q_logical = q_rank * local_len + q_off
                q_valid = q_off < local_len

                kv_rank = kv_idx // local_padded_len
                kv_off = kv_idx % local_padded_len
                kv_logical = kv_rank * local_len + kv_off
                kv_valid = kv_off < local_len
                # Causal logic in logical space
                q_frame_type = frame_types[q_logical]
                q_frame_idx = frame_indices[q_logical]
                kv_frame_type = frame_types[kv_logical]
                kv_frame_idx = frame_indices[kv_logical]

                same_block = ((q_logical // total_indices_per_time) == (kv_logical // total_indices_per_time)) & (
                    kv_frame_type == q_frame_type
                )
                cross_block_attend = (kv_frame_type == attend_frame_type) & (kv_frame_idx < q_frame_idx)

                return q_valid & kv_valid & (same_block | cross_block_attend)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_padded_len,
            KV_LEN=total_padded_len,
            _compile=True,
            device=device,
        )

        if not is_initialized() or get_rank() == 0:
            log.info(f"Cached block-wise causal mask with block size {num_frame_per_block} frames")
            log.debug(f"Block mask: {block_mask}")

        return block_mask

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
    ) -> torch.Tensor:
        """Training forward pass with block-causal attention."""
        device = self.x_embedder.proj[1].weight.device

        # Construct block-causal mask
        frame_seqlen = x_B_C_T_H_W.shape[-2] * x_B_C_T_H_W.shape[-1] // (self.patch_spatial * self.patch_spatial)
        num_frames = x_B_C_T_H_W.shape[2]
        # Determine CP size
        cp_size = 1
        if self._is_context_parallel_enabled and self.cp_group is not None:
            cp_size = self.cp_group.size()

        mask_key = f"mask_f{num_frames}_seqlen{frame_seqlen}_block{self.num_frame_per_block}_cp{cp_size}"

        if mask_key not in self.block_mask_dict:
            block_mask = self._prepare_blockwise_causal_attn_mask(
                device=device,
                num_frames=num_frames // (num_interleave + 1),
                frame_seqlen=frame_seqlen,
                num_frame_per_block=self.num_frame_per_block,
                num_interleave=num_interleave,
                cp_size=cp_size,
            )
            self.block_mask_dict[mask_key] = block_mask
        else:
            block_mask = self.block_mask_dict[mask_key]

        # Prepare inputs
        if self.concat_padding_mask and padding_mask is not None:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )

        # Patch embedding
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        video_size = VideoSize(T=x_B_T_H_W_D.shape[1], H=x_B_T_H_W_D.shape[2], W=x_B_T_H_W_D.shape[3])

        # RoPE embeddings
        if num_interleave > 0:
            rope_shape = list(x_B_T_H_W_D.shape)
            rope_shape[1] = rope_shape[1] // (num_interleave + 1)
            rope_freq = self.pos_embedder.generate_embeddings(torch.Size(rope_shape), fps=fps)
            rope_freq = rearrange(rope_freq, "(t h w) 1 1 d -> t h w d", h=rope_shape[2], w=rope_shape[3])
            rope_freq = rope_freq.repeat(num_interleave + 1, 1, 1, 1)
            rope_freq = rearrange(rope_freq, "(n t) h w d -> (t n h w) 1 1 d", n=num_interleave + 1)
        else:
            rope_freq = self.pos_embedder.generate_embeddings(x_B_T_H_W_D.shape, fps=fps)

        # Extra positional embeddings
        extra_pos_emb = None
        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)

        # Time embedding
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_emb_B_T_D = self.t_embedding_norm(t_emb_B_T_D)

        # Context embeddings
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None and self.extra_image_context_dim is not None:
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        # Flatten inputs for blocks
        x_B_L_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")

        frame_seqlen = video_size.H * video_size.W
        t_emb_B_L_D = torch.repeat_interleave(t_emb_B_T_D, frame_seqlen, dim=1)

        if adaln_lora_B_T_3D is not None:
            adaln_lora_B_L_3D = torch.repeat_interleave(adaln_lora_B_T_3D, frame_seqlen, dim=1)
        else:
            adaln_lora_B_L_3D = None

        if extra_pos_emb is not None:
            extra_pos_emb = rearrange(extra_pos_emb, "b t h w d -> b (t h w) d")

        # Context parallel: split inputs
        cp_enabled = self._is_context_parallel_enabled and self.cp_group is not None
        if cp_enabled and self.cp_group.size() > 1:
            from omnidreams._src.imaginaire.utils.context_parallel import split_inputs_cp

            x_B_L_D = split_inputs_cp(x_B_L_D, seq_dim=1, cp_group=self.cp_group)
            t_emb_B_L_D = split_inputs_cp(t_emb_B_L_D, seq_dim=1, cp_group=self.cp_group)
            rope_freq = split_inputs_cp(rope_freq, seq_dim=0, cp_group=self.cp_group)

            if adaln_lora_B_L_3D is not None:
                adaln_lora_B_L_3D = split_inputs_cp(adaln_lora_B_L_3D, seq_dim=1, cp_group=self.cp_group)

            if extra_pos_emb is not None:
                extra_pos_emb = split_inputs_cp(extra_pos_emb, seq_dim=1, cp_group=self.cp_group)

            if distributed.get_rank() == 0 and DEBUG:
                print(f"CP split shapes (train): x={x_B_L_D.shape}, t_emb={t_emb_B_L_D.shape}, rope={rope_freq.shape}")
                if adaln_lora_B_L_3D is not None:
                    print(f"adaln_lora={adaln_lora_B_L_3D.shape}")
                if extra_pos_emb is not None:
                    print(f"extra_pos_emb={extra_pos_emb.shape}")

        # Process blocks
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.on_the_fly_checkpoint:
                x_B_L_D = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x_B_L_D,
                    t_emb_B_L_D,
                    context_input,
                    rope_freq,
                    adaln_lora_B_L_3D,
                    extra_pos_emb,
                    block_mask,
                    None,  # kv_cache
                    None,  # crossattn_cache
                    0,  # current_start
                    0,  # current_end
                    False,  # disable_kv_cache
                    False,  # disable_kv_cache_update
                    video_size,
                    use_reentrant=False,
                )
            else:
                x_B_L_D = block(
                    x_B_L_D,
                    t_emb_B_L_D,
                    context_input,
                    rope_emb_B_L_D=rope_freq,
                    adaln_lora_B_L_3D=adaln_lora_B_L_3D,
                    extra_per_block_pos_emb=extra_pos_emb,
                    block_mask=block_mask,
                    video_size=video_size,
                )

        # Context parallel: gather outputs
        if cp_enabled and self.cp_group is not None:
            # Gather before FinalLayer
            x_B_L_D = cat_outputs_cp_with_grad(x_B_L_D, seq_dim=1, cp_group=self.cp_group)

        # Unflatten for FinalLayer
        x_B_T_H_W_D = rearrange(x_B_L_D, "b (t h w) d -> b t h w d", t=video_size.T, h=video_size.H, w=video_size.W)

        # Final layer
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_emb_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)

        # Unpatchify
        t, h, w = video_size
        x_B_C_T_H_W = rearrange(
            x_B_T_H_W_O,
            "b t h w (nt nh nw d) -> b d (t nt) (h nh) (w nw)",
            nt=self.patch_temporal,
            nh=self.patch_spatial,
            nw=self.patch_spatial,
            t=t,
            h=h,
            w=w,
            d=self.out_channels,
        )

        return x_B_C_T_H_W

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
    ) -> torch.Tensor:
        """Inference forward pass with KV-caching."""
        # Prepare inputs
        if self.concat_padding_mask and padding_mask is not None:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )

        # Patch embedding
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        video_size = VideoSize(T=x_B_T_H_W_D.shape[1], H=x_B_T_H_W_D.shape[2], W=x_B_T_H_W_D.shape[3])

        # RoPE with frame offset for autoregressive generation
        rope_freq = self.pos_embedder.generate_embeddings(
            x_B_T_H_W_D.shape,
            fps=fps,
        )
        # Adjust for start frame offset
        if start_frame_for_rope > 0:
            full_shape = list(x_B_T_H_W_D.shape)
            full_shape[1] = start_frame_for_rope + full_shape[1]
            full_rope = self.pos_embedder.generate_embeddings(torch.Size(full_shape), fps=fps)
            start_idx = start_frame_for_rope * video_size.H * video_size.W
            end_idx = start_idx + video_size.T * video_size.H * video_size.W
            rope_freq = full_rope[start_idx:end_idx]

        # Extra positional embeddings
        extra_pos_emb = None
        if self.extra_per_block_abs_pos_emb:
            if start_frame_for_rope > 0:
                full_shape = list(x_B_T_H_W_D.shape)
                full_shape[1] = start_frame_for_rope + full_shape[1]
                full_extra_pos_emb = self.extra_pos_embedder.generate_embeddings(torch.Size(full_shape), fps=fps)
                extra_pos_emb = full_extra_pos_emb[:, start_frame_for_rope:, :, :, :]
            else:
                extra_pos_emb = self.extra_pos_embedder.generate_embeddings(x_B_T_H_W_D.shape, fps=fps)

            extra_pos_emb = rearrange(extra_pos_emb, "b t h w d -> b (t h w) d")

        # Time embedding
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_emb_B_T_D = self.t_embedding_norm(t_emb_B_T_D)

        # Context embeddings
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None and self.extra_image_context_dim is not None:
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        # Flatten inputs for blocks
        x_B_L_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")

        frame_seqlen = video_size.H * video_size.W
        t_emb_B_L_D = torch.repeat_interleave(t_emb_B_T_D, frame_seqlen, dim=1)

        if adaln_lora_B_T_3D is not None:
            adaln_lora_B_L_3D = torch.repeat_interleave(adaln_lora_B_T_3D, frame_seqlen, dim=1)
        else:
            adaln_lora_B_L_3D = None

        # Context parallel: split inputs
        cp_enabled = self._is_context_parallel_enabled and self.cp_group is not None
        if cp_enabled and self.cp_group.size() > 1:
            from omnidreams._src.imaginaire.utils.context_parallel import split_inputs_cp

            x_B_L_D = split_inputs_cp(x_B_L_D, seq_dim=1, cp_group=self.cp_group)
            t_emb_B_L_D = split_inputs_cp(t_emb_B_L_D, seq_dim=1, cp_group=self.cp_group)
            rope_freq = split_inputs_cp(rope_freq, seq_dim=0, cp_group=self.cp_group)

            if adaln_lora_B_L_3D is not None:
                adaln_lora_B_L_3D = split_inputs_cp(adaln_lora_B_L_3D, seq_dim=1, cp_group=self.cp_group)

            if extra_pos_emb is not None:
                extra_pos_emb = split_inputs_cp(extra_pos_emb, seq_dim=1, cp_group=self.cp_group)

            if distributed.get_rank() == 0 and DEBUG:
                print(
                    f"CP split shapes (inference): x={x_B_L_D.shape}, t_emb={t_emb_B_L_D.shape}, rope={rope_freq.shape}"
                )
                if adaln_lora_B_L_3D is not None:
                    print(f"adaln_lora={adaln_lora_B_L_3D.shape}")
                if extra_pos_emb is not None:
                    print(f"extra_pos_emb={extra_pos_emb.shape}")

        # Process blocks with KV caching
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block_idx, block in enumerate(self.blocks):
            block_kv_cache = kv_cache[block_idx] if kv_cache is not None and not disable_kv_cache else None
            block_crossattn_cache = crossattn_cache[block_idx] if crossattn_cache is not None else None

            if torch.is_grad_enabled() and self.on_the_fly_checkpoint:
                x_B_L_D = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x_B_L_D,
                    t_emb_B_L_D,
                    context_input,
                    rope_freq,
                    adaln_lora_B_L_3D,
                    extra_pos_emb,  # extra_per_block_pos_emb
                    None,  # block_mask
                    block_kv_cache,
                    block_crossattn_cache,
                    current_start,
                    current_end,
                    disable_kv_cache,
                    False,  # disable_kv_cache_update
                    video_size,
                    use_reentrant=False,
                )
            else:
                x_B_L_D = block(
                    x_B_L_D,
                    t_emb_B_L_D,
                    context_input,
                    rope_emb_B_L_D=rope_freq,
                    adaln_lora_B_L_3D=adaln_lora_B_L_3D,
                    block_mask=None,
                    kv_cache=block_kv_cache,
                    crossattn_cache=block_crossattn_cache,
                    current_start=current_start,
                    current_end=current_end,
                    disable_kv_cache=disable_kv_cache,
                    video_size=video_size,
                )

        # Context parallel: gather outputs
        if cp_enabled and self.cp_group is not None:
            # Gather before FinalLayer
            if torch.is_grad_enabled():
                x_B_L_D = cat_outputs_cp_with_grad(x_B_L_D, seq_dim=1, cp_group=self.cp_group)
            else:
                x_B_L_D = cat_outputs_cp(x_B_L_D, seq_dim=1, cp_group=self.cp_group)

        # Unflatten for FinalLayer
        x_B_T_H_W_D = rearrange(x_B_L_D, "b (t h w) d -> b t h w d", t=video_size.T, h=video_size.H, w=video_size.W)

        # Final layer
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_emb_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)

        # Unpatchify
        t, h, w = video_size
        x_B_C_T_H_W = rearrange(
            x_B_T_H_W_O,
            "b t h w (nt nh nw d) -> b d (t nt) (h nh) (w nw)",
            nt=self.patch_temporal,
            nh=self.patch_spatial,
            nw=self.patch_spatial,
            t=t,
            h=h,
            w=w,
            d=self.out_channels,
        )

        return x_B_C_T_H_W

    def init_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[dict]:
        """
        Initialize KV caches for all blocks.

        Args:
            batch_size: Batch size
            max_seq_len: Maximum sequence length (total tokens across all frames)
            device: Device to create cache on
            dtype: Data type for cache tensors

        Returns:
            List of KV cache dicts, one per block
        """
        kv_caches = []
        for _ in range(self.num_blocks):
            cache = {
                "k": torch.zeros(
                    batch_size,
                    max_seq_len,
                    self.num_heads,
                    self.model_channels // self.num_heads,
                    device=device,
                    dtype=dtype,
                ),
                "v": torch.zeros(
                    batch_size,
                    max_seq_len,
                    self.num_heads,
                    self.model_channels // self.num_heads,
                    device=device,
                    dtype=dtype,
                ),
                "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
                "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
            }
            kv_caches.append(cache)
        return kv_caches

    def reset_kv_cache(self, kv_caches: list[dict]) -> None:
        """Reset all KV caches to initial state."""
        for cache in kv_caches:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()

    def enable_selective_checkpoint(self, sac_config: SACConfig, blocks: nn.ModuleList) -> None:
        if sac_config.mode == CheckpointMode.NONE:
            return

        log.info(
            f"Enable selective checkpoint with {sac_config.mode}, "
            f"for every {sac_config.every_n_blocks} blocks. Total blocks: {len(blocks)}"
        )
        _context_fn = sac_config.get_context_fn()

        for block_id, block in blocks.named_children():
            if int(block_id) % sac_config.every_n_blocks == 0:
                log.info(f"Enable selective checkpoint for block {block_id}")
                block = ptd_checkpoint_wrapper(
                    block,
                    # checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    context_fn=_context_fn,
                    preserve_rng_state=False,
                )
                blocks.register_module(block_id, block)

        self.register_module(
            "final_layer",
            ptd_checkpoint_wrapper(
                self.final_layer,
                # checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                context_fn=_context_fn,
                preserve_rng_state=False,
            ),
        )

    def fully_shard(self, mesh, **fsdp_kwargs) -> None:
        """Apply FSDP sharding to model components."""
        for block in self.blocks:
            fully_shard(block, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.x_embedder, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)

    def enable_context_parallel(self, process_group: ProcessGroup | None = None) -> None:
        """Enable context parallelism for all attention layers."""
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )
        # ! xuanchir: disable pos_embedder and extra_pos_embedder; since for casual model, we assume we do split in the network
        # self.pos_embedder.enable_context_parallel(process_group)
        # if self.extra_per_block_abs_pos_emb:
        #     self.extra_pos_embedder.enable_context_parallel(process_group)
        self._is_context_parallel_enabled = True
        self.cp_group = process_group

    def disable_context_parallel(self) -> None:
        """Disable context parallelism."""
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )
        self.pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.disable_context_parallel()
        self._is_context_parallel_enabled = False
        self.cp_group = None

    @property
    def is_context_parallel_enabled(self) -> bool:
        return self._is_context_parallel_enabled
