# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List, Optional, Tuple

import torch
from torch.distributed._composable.fsdp import fully_shard

from omnidreams._src.predict2.conditioner import DataType
from omnidreams._src.predict2.networks.minimal_v4_dit import MiniTrainDIT


class MinimalV1LVGDiT(MiniTrainDIT):
    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        self.timestep_scale = timestep_scale

        # remove num_layers from kwargs
        if "num_layers" in kwargs:
            del kwargs["num_layers"]

        super().__init__(*args, **kwargs)

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
        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T * self.timestep_scale,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            intermediate_feature_ids=intermediate_feature_ids,
            img_context_emb=img_context_emb,
        )

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
