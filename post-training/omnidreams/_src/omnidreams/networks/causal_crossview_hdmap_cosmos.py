# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bidirectional CrossView Cosmos DiT with B-V-L serialization and CP support."""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed._composable.fsdp import fully_shard
from torchvision import transforms

from omnidreams._src.imaginaire.utils import distributed, log
from omnidreams._src.imaginaire.utils.context_parallel import cat_outputs_cp, cat_outputs_cp_with_grad, split_inputs_cp
from omnidreams._src.predict2.conditioner import DataType
from omnidreams._src.predict2.networks.minimal_v4_dit import PatchEmbed
from omnidreams._src.predict2_multiview.networks.multiview_cross_dit import (
    VideoSize,
)
from omnidreams._src.omnidreams.networks.causal_crossview_cosmos import (
    CausalCrossViewCosmosDiT,
)

DEBUG = False


class CausalCrossViewCosmosDiTHDMapConcat(CausalCrossViewCosmosDiT):
    def __init__(
        self,
        *args,
        additional_concat_ch: int = 16,
        additional_init_method: str = "random_init",
        use_control_gate: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.additional_concat_ch = additional_concat_ch
        self.additional_init_method = additional_init_method
        self.use_control_gate = use_control_gate
        if self.additional_concat_ch is not None and self.additional_concat_ch != 0:
            self.additional_patch_embedding = PatchEmbed(
                spatial_patch_size=self.patch_spatial,
                temporal_patch_size=self.patch_temporal,
                in_channels=additional_concat_ch,
                out_channels=self.model_channels,
            )
            if use_control_gate:
                self.control_gate = nn.Parameter(torch.zeros(1), requires_grad=True)
            assert additional_init_method in ["random_init"], (
                f"additional_init_method must be 'random_init', got {additional_init_method}"
            )

    def init_weights(self) -> None:
        super().init_weights()
        if hasattr(self, "additional_concat_ch") and self.additional_concat_ch != 0:
            self.additional_patch_embedding.init_weights()
            if self.use_control_gate:
                torch.nn.init.zeros_(self.control_gate)

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
        control_input_hdmap_bbox: Optional[torch.Tensor] = None,
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
            control_input_hdmap_bbox: Control input HD map bbox [B, T, 4]
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
                control_input_hdmap_bbox=control_input_hdmap_bbox,
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
                control_input_hdmap_bbox=control_input_hdmap_bbox,
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
        control_input_hdmap_bbox: Optional[torch.Tensor] = None,
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
        if self.additional_concat_ch != 0:
            assert control_input_hdmap_bbox is not None, (
                "control_input_hdmap_bbox must be provided if additional_concat_ch is not 0"
            )
            addition_x_B_C_T_H_W = self.additional_patch_embedding(control_input_hdmap_bbox)
            if self.use_control_gate:
                x_B_T_H_W_D = x_B_T_H_W_D + self.control_gate * addition_x_B_C_T_H_W
            else:
                x_B_T_H_W_D = x_B_T_H_W_D + addition_x_B_C_T_H_W

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
        control_input_hdmap_bbox: Optional[torch.Tensor] = None,
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

        # 2. Reshape to B V T H W for internal processing
        # Note: x_embedder expects B C T H W.
        # We can pass flattened T (V*T) to embedder, then reshape.

        # Patch embedding
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        if self.additional_concat_ch != 0:
            assert control_input_hdmap_bbox is not None, (
                "control_input_hdmap_bbox must be provided if additional_concat_ch is not 0"
            )
            addition_x_B_C_T_H_W = self.additional_patch_embedding(control_input_hdmap_bbox)
            if self.use_control_gate:
                x_B_T_H_W_D = x_B_T_H_W_D + self.control_gate * addition_x_B_C_T_H_W
            else:
                x_B_T_H_W_D = x_B_T_H_W_D + addition_x_B_C_T_H_W

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
        if t_emb_B_T_D.shape[1] == 1 and video_size.T > 1:
            t_emb_B_T_D = t_emb_B_T_D.repeat(1, video_size.T, 1)

        frame_seqlen_spatial = video_size.H * video_size.W
        t_emb_B_L_D = torch.repeat_interleave(t_emb_B_T_D, frame_seqlen_spatial, dim=1)
        t_emb_B_V_L_D = rearrange(t_emb_B_L_D, "b (v l) d -> b v l d", v=n_cameras)

        if adaln_lora_B_T_3D is not None:
            if adaln_lora_B_T_3D.shape[1] == 1 and video_size.T > 1:
                adaln_lora_B_T_3D = adaln_lora_B_T_3D.repeat(1, video_size.T, 1)
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

    def fully_shard(self, mesh, **fsdp_kwargs):
        super().fully_shard(mesh, **fsdp_kwargs)
        if hasattr(self, "additional_concat_ch") and self.additional_concat_ch != 0:
            fully_shard(self.additional_patch_embedding, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
