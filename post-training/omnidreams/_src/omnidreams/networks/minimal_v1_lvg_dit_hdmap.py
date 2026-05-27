# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List, Optional, Tuple

import torch
import torch.amp as amp
from einops import rearrange
from torch import nn
from torch.distributed._composable.fsdp import fully_shard

from omnidreams._src.imaginaire.utils import log
from omnidreams._src.predict2.conditioner import DataType
from omnidreams._src.predict2.networks.minimal_v4_dit import MiniTrainDIT, PatchEmbed


class MinimalV1LVGDiTHdmapConcat(MiniTrainDIT):
    def __init__(
        self,
        *args,
        timestep_scale: float = 1.0,
        additional_concat_ch: int = None,
        additional_init_method: str = "random_init",
        **kwargs,
    ):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        self.timestep_scale = timestep_scale

        # remove num_layers from kwargs
        if "num_layers" in kwargs:
            del kwargs["num_layers"]

        super().__init__(*args, **kwargs)

        self.additional_concat_ch = additional_concat_ch
        self.additional_init_method = additional_init_method
        if self.additional_concat_ch is not None and self.additional_concat_ch != 0:
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

    def init_weights(self):
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
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        control_input_hdmap_bbox: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )

        ##
        timesteps_B_T = timesteps_B_T * self.timestep_scale
        ##

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if self.additional_concat_ch != 0:
            assert control_input_hdmap_bbox is not None, (
                "control_input_hdmap_bbox must be provided if additional_concat_ch is not 0"
            )
            addition_x_B_C_T_H_W = self.additional_patch_embedding(control_input_hdmap_bbox)
            x_B_T_H_W_D = x_B_T_H_W_D + addition_x_B_C_T_H_W  # let's not touch the data point here

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set if img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        B, T, H, W, D = x_B_T_H_W_D.shape
        # x_B_THW_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)

        # x_B_T_H_W_D = rearrange(x_B_THW_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        # O = out_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if intermediate_feature_ids:
            if len(intermediate_features_outputs) != len(intermediate_feature_ids):
                log.warning(
                    f"Collected {len(intermediate_features_outputs)} intermediate features, "
                    f"but expected {len(intermediate_feature_ids)}. "
                    f"Requested IDs: {intermediate_feature_ids}"
                )
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        return x_B_C_Tt_Hp_Wp

    def fully_shard(self, mesh, **fsdp_kwargs) -> None:
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)


        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        if self.extra_per_block_abs_pos_emb:
            fully_shard(self.extra_pos_embedder, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)
        if self.extra_image_context_dim is not None:
            fully_shard(self.img_context_proj, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)

        if hasattr(self, "additional_concat_ch") and self.additional_concat_ch != 0:
            fully_shard(self.additional_patch_embedding, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
