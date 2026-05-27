# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
CosmosCausalHdmapDiT: A causal DiT model with hdmap/bbox control input support.
Extends CosmosCausalDiT with additional patch embedding for hdmap inputs.
"""

import torch
import torch.amp as amp
from einops import rearrange
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torchvision import transforms

from omnidreams._src.imaginaire.utils import distributed
from omnidreams._src.imaginaire.utils.context_parallel import cat_outputs_cp, cat_outputs_cp_with_grad
from omnidreams._src.predict2.conditioner import DataType
from omnidreams._src.predict2.networks.minimal_v4_dit import PatchEmbed
from omnidreams._src.omnidreams.networks.causal_cosmos import (
    DEBUG,
    CosmosCausalDiT,
    VideoSize,
)


class CosmosCausalHdmapDiT(CosmosCausalDiT):
    """
    CosmosCausalDiT with additional hdmap/bbox control input support.

    Adds an additional patch embedding for hdmap/bbox inputs that gets added
    to the main video embedding.
    """

    def __init__(
        self,
        *args,
        additional_concat_ch: int = 16,
        additional_init_method: str = "random_init",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.additional_concat_ch = additional_concat_ch
        self.additional_init_method = additional_init_method

        if self.additional_concat_ch != 0:
            self.additional_patch_embedding = PatchEmbed(
                spatial_patch_size=self.patch_spatial,
                temporal_patch_size=self.patch_temporal,
                in_channels=additional_concat_ch,
                out_channels=self.model_channels,
            )
            assert additional_init_method in ("random_init", "zero_init"), (
                "only support random_init or zero_init for additional_concat_ch"
            )
            if additional_init_method == "random_init":
                self.additional_patch_embedding.init_weights()
            else:  # zero_init
                self._zero_init_additional_patch_embedding()

    def init_weights(self) -> None:
        super().init_weights()
        if hasattr(self, "additional_concat_ch") and self.additional_concat_ch != 0:
            if self.additional_init_method == "random_init":
                self.additional_patch_embedding.init_weights()
            else:  # zero_init
                self._zero_init_additional_patch_embedding()

    def _zero_init_additional_patch_embedding(self) -> None:
        """Zero initialize the additional patch embedding weights."""
        for module in self.additional_patch_embedding.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
        control_input_hdmap_bbox: torch.Tensor | None = None,
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
            control_input_hdmap_bbox: Hdmap/bbox control input [B, C, T, H, W]
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
        control_input_hdmap_bbox: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training forward pass with block-causal attention and hdmap support."""
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

        # Add hdmap embedding
        if self.additional_concat_ch != 0:
            assert control_input_hdmap_bbox is not None, (
                "control_input_hdmap_bbox must be provided if additional_concat_ch is not 0"
            )
            addition_x_B_T_H_W_D = self.additional_patch_embedding(control_input_hdmap_bbox)
            x_B_T_H_W_D = x_B_T_H_W_D + addition_x_B_T_H_W_D

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
        control_input_hdmap_bbox: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference forward pass with KV-caching and hdmap support."""
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

        # Add hdmap embedding
        if self.additional_concat_ch != 0:
            assert control_input_hdmap_bbox is not None, (
                "control_input_hdmap_bbox must be provided if additional_concat_ch is not 0"
            )
            addition_x_B_T_H_W_D = self.additional_patch_embedding(control_input_hdmap_bbox)
            x_B_T_H_W_D = x_B_T_H_W_D + addition_x_B_T_H_W_D

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

    def fully_shard(self, mesh, **fsdp_kwargs) -> None:
        """Apply FSDP sharding to model components."""
        super().fully_shard(mesh, **fsdp_kwargs)

        # Shard hdmap patch embedding
        if hasattr(self, "additional_concat_ch") and self.additional_concat_ch != 0:
            fully_shard(self.additional_patch_embedding, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
