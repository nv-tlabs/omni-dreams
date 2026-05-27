# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bidirectional CrossView Cosmos DiT with B-V-L serialization and CP support."""

import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed._composable.fsdp import fully_shard
from torchvision import transforms

from omnidreams._src.imaginaire.utils import log
from omnidreams._src.imaginaire.utils.context_parallel import cat_outputs_cp_with_grad, split_inputs_cp
from omnidreams._src.predict2.conditioner import DataType
from omnidreams._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from omnidreams._src.predict2.networks.minimal_v4_dit import Attention, Block
from omnidreams._src.predict2_multiview.networks.multiview_cross_dit import (
    CrossViewAttention,
    MultiCameraSinCosPosEmbAxis,
    MultiCameraVideoRopePosition3DEmb,
    MultiViewSACConfig,
    VideoSize,
)
from omnidreams._src.omnidreams.networks.causal_crossview_cosmos import (
    CrossViewAttentionWithCPV2,
)


class BidirectionalCrossViewCosmosBlock(Block):
    """Block with bidirectional self-attention and optional cross-view attention."""

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
        enable_cross_view_attn: bool = False,
        cross_view_attn_map: dict[int, list[int]] | None = None,
        v_split_mode: bool = False,
    ) -> None:
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
        )
        self.enable_cross_view_attn: bool = enable_cross_view_attn
        self.cross_view_attn_map: dict[int, list[int]] | None = cross_view_attn_map
        self.v_split_mode: bool = v_split_mode
        self.cp_size: int | None = None
        self.cross_view_attn: Attention | None = None
        self.layer_norm_cross_view_attn: nn.LayerNorm | None = None

        if enable_cross_view_attn:
            assert cross_view_attn_map is not None
            if v_split_mode:
                cross_view_attn_cls = CrossViewAttentionWithCPV2
            else:
                cross_view_attn_cls = CrossViewAttention

            self.cross_view_attn = cross_view_attn_cls(
                x_dim,
                x_dim,
                num_heads,
                x_dim // num_heads,
                qkv_format="bshd",
                use_wan_fp32_strategy=use_wan_fp32_strategy,
                cross_view_attn_map=cross_view_attn_map,
                backend=backend,
            )

            self.layer_norm_cross_view_attn = nn.LayerNorm(x_dim, elementwise_affine=True, eps=1e-6)

    def init_weights(self) -> None:
        super().init_weights()
        if self.enable_cross_view_attn:
            self.layer_norm_cross_view_attn.reset_parameters()
            self.cross_view_attn.init_weights()
            torch.nn.init.zeros_(self.cross_view_attn.output_proj.weight)
            if self.cross_view_attn.output_proj.bias is not None:
                torch.nn.init.zeros_(self.cross_view_attn.output_proj.bias)

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p") -> None:
        self.cp_size = None if ranks is None else len(ranks)
        if self.enable_cross_view_attn and self.v_split_mode:
            self.cross_view_attn.set_context_parallel_group(process_group, ranks, stream, cp_comm_type=cp_comm_type)
        # v_split_mode = True do not requires context parallel!
        if self.v_split_mode is False:
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
        view_embedding_proj_B_V_9D: torch.Tensor | None = None,
        cross_view_block_mask_cache: dict[tuple, object] | None = None,
    ) -> torch.Tensor:
        B, V, L, D = x_B_V_L_D.shape

        if self.v_split_mode:
            assert V == 1, f"In v_split_mode, V dimension should be 1, but got V={V}"

        x_BV_L_D = rearrange(x_B_V_L_D, "b v l d -> (b v) l d")
        emb_BV_L_D = rearrange(emb_B_V_L_D, "b v l d -> (b v) l d")

        if adaln_lora_B_V_L_3D is not None:
            adaln_lora_BV_L_3D = rearrange(adaln_lora_B_V_L_3D, "b v l d -> (b v) l d")
        else:
            adaln_lora_BV_L_3D = None

        if extra_per_block_pos_emb is not None:
            if extra_per_block_pos_emb.ndim == 4:
                x_BV_L_D = x_BV_L_D + rearrange(extra_per_block_pos_emb, "b v l d -> (b v) l d")
            else:
                x_BV_L_D = x_BV_L_D + extra_per_block_pos_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
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

        if view_embedding_proj_B_V_9D is not None:
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

            def expand_view_mod(v_mod: torch.Tensor) -> torch.Tensor:
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

        normed_x = self.layer_norm_self_attn(x_BV_L_D) * (1 + scale_self) + shift_self
        rope_single_view = rope_emb_V_L_1_1_D[0] if rope_emb_V_L_1_1_D is not None else None
        attn_out = self.self_attn(
            normed_x,
            context=None,
            rope_emb=rope_single_view,
            video_size=sv_video_size,
        )
        x_BV_L_D = x_BV_L_D + gate_self * attn_out

        if self.enable_cross_view_attn:
            T, H, W = sv_video_size
            x_B_V_L_D = rearrange(x_BV_L_D, "(b v) l d -> b v l d", b=B, v=V)
            normed_x_cv = self.layer_norm_cross_view_attn(x_B_V_L_D)

            if self.v_split_mode:
                normed_x_cv = rearrange(normed_x_cv, "b v (t h w) d -> b v t (h w) d", h=H, w=W)
                cv_out = self.cross_view_attn(
                    normed_x_cv, view_indices_B_V, (H, W), block_mask_cache=cross_view_block_mask_cache
                )
                cv_out = rearrange(cv_out, "b v t (h w) d -> b v (t h w) d", h=H, w=W)
            else:
                cv_out = self.cross_view_attn(normed_x_cv, view_indices_B_V, sv_video_size)

            x_B_V_L_D = x_B_V_L_D + cv_out
            x_BV_L_D = rearrange(x_B_V_L_D, "b v l d -> (b v) l d")

        normed_x = self.layer_norm_cross_attn(x_BV_L_D) * (1 + scale_cross) + shift_cross
        crossattn_emb_BV_L_D = rearrange(crossattn_emb, "b (v l) d -> (b v) l d", v=V)
        cross_out = self.cross_attn(normed_x, crossattn_emb_BV_L_D)
        x_BV_L_D = x_BV_L_D + gate_cross * cross_out

        normed_x = self.layer_norm_mlp(x_BV_L_D) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(normed_x)
        x_BV_L_D = x_BV_L_D + gate_mlp * mlp_out

        return rearrange(x_BV_L_D, "(b v) l d -> b v l d", b=B, v=V)


class BidirectionalCrossViewCosmosDiT(MinimalV1LVGDiT):
    """Bidirectional multi-view DiT with CrossView attention and CP splitting."""

    def __init__(
        self,
        *args,
        timestep_scale: float = 1.0,
        crossattn_emb_channels: int = 1024,
        mlp_ratio: float = 4.0,
        num_layers: int,
        state_t: int,
        n_cameras_emb: int,
        view_condition_dim: int,
        concat_view_embedding: bool,
        adaln_view_embedding: bool,
        layer_mask: list[bool] | None = None,
        sac_config: MultiViewSACConfig = MultiViewSACConfig(),
        enable_cross_view_attn: bool = False,
        cross_view_attn_map_str: dict | None = None,
        camera_to_view_id: dict | None = None,
        v_split_mode: bool = False,
        backend: str = "transformer_engine",
        **kwargs,
    ) -> None:
        self.crossattn_emb_channels: int = crossattn_emb_channels
        self.mlp_ratio: float = mlp_ratio
        self.state_t: int = state_t
        self.n_cameras_emb: int = n_cameras_emb
        self.view_condition_dim: int = view_condition_dim
        self.concat_view_embedding: bool = concat_view_embedding
        self.adaln_view_embedding: bool = adaln_view_embedding
        self.enable_cross_view_attn: bool = enable_cross_view_attn
        self.v_split_mode: bool = v_split_mode
        if "atten_backend" not in kwargs:
            kwargs["atten_backend"] = backend
        self.backend: str = kwargs["atten_backend"]

        assert not (self.adaln_view_embedding and self.concat_view_embedding), (
            "adaln_view_embedding and concat_view_embedding cannot be True at the same time"
        )
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += self.view_condition_dim if self.concat_view_embedding else 0
        assert layer_mask is None, "layer_mask is not supported for BidirectionalCrossViewCosmosDiT"
        if "n_cameras" in kwargs:
            del kwargs["n_cameras"]

        super().__init__(
            *args,
            mlp_ratio=mlp_ratio,
            timestep_scale=timestep_scale,
            crossattn_emb_channels=crossattn_emb_channels,
            sac_config=sac_config,
            **kwargs,
        )

        self.cross_view_attn_map: dict[int, list[int]] = {}
        if cross_view_attn_map_str and camera_to_view_id:
            for source_view, target_views in cross_view_attn_map_str.items():
                self.cross_view_attn_map[int(camera_to_view_id[source_view])] = []
                for target_view in target_views:
                    self.cross_view_attn_map[int(camera_to_view_id[source_view])].append(
                        int(camera_to_view_id[target_view])
                    )

        del self.blocks
        self.blocks = nn.ModuleList(
            [
                BidirectionalCrossViewCosmosBlock(
                    x_dim=self.model_channels,
                    context_dim=self.crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
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

        self.cross_view_block_mask_cache: dict[tuple, object] = {}
        self.cp_group: ProcessGroup | None = None

        self.init_weights()
        self.enable_selective_checkpoint(sac_config, self.blocks)

    def fully_shard(self, mesh, **fsdp_kwargs):
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)

        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                fully_shard(extra_pos_embedder, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)
        if self.extra_image_context_dim is not None:
            fully_shard(self.img_context_proj, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)

        if hasattr(self, "view_embeddings"):
            fully_shard(self.view_embeddings, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)
        if hasattr(self, "adaln_view_embedder"):
            fully_shard(self.adaln_view_embedder, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)
        if hasattr(self, "adaln_view_proj"):
            fully_shard(self.adaln_view_proj, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)

    def build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = MultiCameraVideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        self.pos_embedder_options: nn.ModuleDict = nn.ModuleDict()
        self.extra_pos_embedders_options: nn.ModuleDict = nn.ModuleDict()
        for n_cameras in range(1, self.n_cameras_emb + 1):
            kwargs = dict(
                model_channels=self.model_channels,
                len_h=self.max_img_h // self.patch_spatial,
                len_w=self.max_img_w // self.patch_spatial,
                len_t=self.max_frames // self.patch_temporal,
                max_fps=self.max_fps,
                min_fps=self.min_fps,
                is_learnable=self.pos_emb_learnable,
                interpolation=self.pos_emb_interpolation,
                head_dim=self.model_channels // self.num_heads,
                h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
                w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
                t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
                enable_fps_modulation=self.rope_enable_fps_modulation,
                n_cameras=n_cameras,
            )
            self.pos_embedder_options[f"n_cameras_{n_cameras}"] = cls_type(**kwargs)

            if self.extra_per_block_abs_pos_emb:
                extra_kwargs = dict(
                    interpolation=self.pos_emb_interpolation,
                    model_channels=self.model_channels,
                    len_h=self.max_img_h // self.patch_spatial,
                    len_w=self.max_img_w // self.patch_spatial,
                    len_t=self.max_frames // self.patch_temporal,
                    h_extrapolation_ratio=self.extra_h_extrapolation_ratio,
                    w_extrapolation_ratio=self.extra_w_extrapolation_ratio,
                    t_extrapolation_ratio=self.extra_t_extrapolation_ratio,
                    n_cameras=n_cameras,
                )
                self.extra_pos_embedders_options[f"n_cameras_{n_cameras}"] = MultiCameraSinCosPosEmbAxis(**extra_kwargs)

        self.pos_embedder: nn.Module = self.pos_embedder_options["n_cameras_1"]

    def init_weights(self):
        self.x_embedder.init_weights()
        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.init_weights()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

        if self.extra_image_context_dim is not None:
            self.img_context_proj[0].reset_parameters()

        if hasattr(self, "view_embeddings"):
            torch.nn.init.normal_(self.view_embeddings.weight, mean=0.0, std=0.02)

        if hasattr(self, "adaln_view_embedder"):
            torch.nn.init.normal_(self.adaln_view_embedder.weight, mean=0.0, std=0.05)

        if hasattr(self, "adaln_view_proj"):
            torch.nn.init.zeros_(self.adaln_view_proj.weight)
            torch.nn.init.zeros_(self.adaln_view_proj.bias)

    def enable_context_parallel(self, process_group: ProcessGroup | None = None) -> None:
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
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

    def disable_context_parallel(self) -> None:
        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.disable_context_parallel()

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
        view_indices_B_T: torch.Tensor | None = None,
        fps: torch.Tensor | None = None,
        n_views: int | None = None,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        del fps
        if n_views is None:
            n_cameras = x_B_C_T_H_W.shape[2] // self.state_t
            t_per_view = self.state_t
        else:
            n_cameras = n_views
            t_per_view = x_B_C_T_H_W.shape[2] // n_views

        if view_indices_B_T is None:
            view_indices = torch.arange(n_cameras, device=x_B_C_T_H_W.device).clamp(max=self.n_cameras_emb - 1)
            if self.concat_view_embedding:
                view_embedding = self.view_embeddings(view_indices)
                view_embedding = rearrange(view_embedding, "v d -> 1 d v 1 1 1")
                x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "b c (v t) h w -> b c v t h w", v=n_cameras)
                view_embedding = view_embedding.expand(
                    x_B_C_V_T_H_W.shape[0],
                    -1,
                    -1,
                    x_B_C_V_T_H_W.shape[3],
                    x_B_C_V_T_H_W.shape[4],
                    x_B_C_V_T_H_W.shape[5],
                )
                x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)
                x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, "b c v t h w -> b c (v t) h w")
            view_indices_B_V = view_indices.unsqueeze(0).expand(x_B_C_T_H_W.shape[0], -1)
        else:
            expected_len = n_cameras * t_per_view
            if view_indices_B_T.shape[1] != expected_len:
                log.warning(
                    "view_indices_B_T has unexpected length "
                    f"{view_indices_B_T.shape[1]}, expected {expected_len}. Falling back to default view indices."
                )
                view_indices = torch.arange(n_cameras, device=x_B_C_T_H_W.device).clamp(max=self.n_cameras_emb - 1)
                view_indices_B_V = view_indices.unsqueeze(0).expand(x_B_C_T_H_W.shape[0], -1)
                if self.concat_view_embedding:
                    view_embedding = self.view_embeddings(view_indices)
                    view_embedding = rearrange(view_embedding, "v d -> 1 d v 1 1 1")
                    x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "b c (v t) h w -> b c v t h w", v=n_cameras)
                    view_embedding = view_embedding.expand(
                        x_B_C_V_T_H_W.shape[0],
                        -1,
                        -1,
                        x_B_C_V_T_H_W.shape[3],
                        x_B_C_V_T_H_W.shape[4],
                        x_B_C_V_T_H_W.shape[5],
                    )
                    x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)
                    x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, "b c v t h w -> b c (v t) h w")
            else:
                view_indices_B_T = (
                    view_indices_B_T.clamp(min=0, max=self.n_cameras_emb - 1).to(x_B_C_T_H_W.device).long()
                )
                if self.concat_view_embedding:
                    view_embedding = self.view_embeddings(view_indices_B_T)
                    view_embedding = rearrange(view_embedding, "b (v t) d -> b d v t", v=n_cameras)
                    view_embedding = view_embedding.unsqueeze(-1).unsqueeze(-1)
                    x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "b c (v t) h w -> b c v t h w", v=n_cameras)
                    view_embedding = view_embedding.expand(
                        x_B_C_V_T_H_W.shape[0],
                        view_embedding.shape[1],
                        view_embedding.shape[2],
                        x_B_C_V_T_H_W.shape[3],
                        x_B_C_V_T_H_W.shape[4],
                        x_B_C_V_T_H_W.shape[5],
                    )
                    x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)
                    x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, "b c v t h w -> b c (v t) h w")

                view_indices_B_V_T = rearrange(view_indices_B_T, "b (v t) -> b v t", v=n_cameras)
                view_indices_B_V = view_indices_B_V_T[..., 0]

                if not torch.all(view_indices_B_V_T == view_indices_B_V.unsqueeze(-1)):
                    view_indices_T_V = rearrange(view_indices_B_T, "b (t v) -> b t v", v=n_cameras)
                    view_indices_B_V_from_t_first = view_indices_T_V[:, 0, :]
                    if torch.all(view_indices_T_V == view_indices_B_V_from_t_first.unsqueeze(1)):
                        log.error("view_indices_B_T appears to be TIME-FIRST (T V) but code expects VIEW-FIRST (V T).")
                        view_indices_B_V = view_indices_B_V_from_t_first
                    else:
                        log.warning(
                            "view_indices_B_T has inconsistent view IDs across timesteps; check input ordering."
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
        view_indices_B_T: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [
                    x_B_C_T_H_W,
                    torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device),
                ],
                dim=1,
            )

        timesteps_B_T = timesteps_B_T * self.timestep_scale

        if self.concat_padding_mask and padding_mask is not None:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )

        x_B_C_T_H_W, n_cameras, view_indices_B_V = self._prepare_multiview_input(x_B_C_T_H_W, view_indices_B_T, fps)
        assert crossattn_emb.ndim == 3, f"crossattn_emb must be 3D, got {crossattn_emb.ndim}D"
        assert crossattn_emb.shape[0] == x_B_C_T_H_W.shape[0], (
            f"crossattn_emb batch size must match x batch size: {crossattn_emb.shape[0]} != {x_B_C_T_H_W.shape[0]}"
        )
        assert crossattn_emb.shape[1] % n_cameras == 0, (
            f"crossattn_emb length {crossattn_emb.shape[1]} is not divisible by n_cameras {n_cameras}"
        )

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        x_B_V_T_H_W_D = rearrange(x_B_T_H_W_D, "b (v t) h w d -> b v t h w d", v=n_cameras)
        video_size = VideoSize(T=x_B_V_T_H_W_D.shape[2], H=x_B_V_T_H_W_D.shape[3], W=x_B_V_T_H_W_D.shape[4])

        pos_embedder = self.pos_embedder_options[f"n_cameras_{n_cameras}"]
        rope_freq_VT_H_W_D = pos_embedder.generate_embeddings(x_B_T_H_W_D.shape, fps=fps)
        rope_freq_V_L_1_1_D = rearrange(rope_freq_VT_H_W_D, "(v t) h w d -> v (t h w) 1 1 d", v=n_cameras)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_embedder = self.extra_pos_embedders_options[f"n_cameras_{n_cameras}"]
            extra_pos_emb_B_T_H_W_D = extra_pos_embedder.generate_embeddings(x_B_T_H_W_D.shape, fps=fps)
            extra_pos_emb_B_V_L_D = rearrange(extra_pos_emb_B_T_H_W_D, "b (v t) h w d -> b v (t h w) d", v=n_cameras)
        else:
            extra_pos_emb_B_V_L_D = None

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_emb_B_T_D = self.t_embedding_norm(t_emb_B_T_D)

        if self.adaln_view_embedding:
            with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
                view_embedding_B_V = self.adaln_view_embedder(view_indices_B_V)
                view_embedding_proj_B_V_9D = self.adaln_view_proj(view_embedding_B_V)
        else:
            view_embedding_proj_B_V_9D = None

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        x_B_V_L_D = rearrange(x_B_V_T_H_W_D, "b v t h w d -> b v (t h w) d")

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

        cp_enabled = self._is_context_parallel_enabled and self.cp_group is not None
        T_before_cp = video_size.T
        n_cameras_before_cp = n_cameras

        if cp_enabled and self.cp_group.size() > 1:
            if self.v_split_mode:
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
                    view_embedding_proj_B_V_9D = split_inputs_cp(
                        view_embedding_proj_B_V_9D, seq_dim=1, cp_group=self.cp_group
                    )
                if extra_pos_emb_B_V_L_D is not None:
                    extra_pos_emb_B_V_L_D = split_inputs_cp(extra_pos_emb_B_V_L_D, seq_dim=1, cp_group=self.cp_group)

                view_indices_B_V = split_inputs_cp(view_indices_B_V, seq_dim=1, cp_group=self.cp_group)
                crossattn_emb = rearrange(crossattn_emb, "b (v l) d -> b v l d", v=n_cameras_before_cp)
                crossattn_emb = split_inputs_cp(crossattn_emb, seq_dim=1, cp_group=self.cp_group)
                crossattn_emb = rearrange(crossattn_emb, "b v l d -> b (v l) d")
            else:
                assert video_size.T % self.cp_group.size() == 0, (
                    f"video_size.T {video_size.T} is not divisible by cp_group.size() {self.cp_group.size()}"
                )
                x_B_V_L_D = split_inputs_cp(x_B_V_L_D, seq_dim=2, cp_group=self.cp_group)
                t_emb_B_V_L_D = split_inputs_cp(t_emb_B_V_L_D, seq_dim=2, cp_group=self.cp_group)
                rope_freq_V_L_1_1_D = split_inputs_cp(rope_freq_V_L_1_1_D, seq_dim=1, cp_group=self.cp_group)
                video_size = VideoSize(
                    T=video_size.T // self.cp_group.size(), H=video_size.H, W=video_size.W
                )  # new T is not used.
                if adaln_lora_B_V_L_3D is not None:
                    adaln_lora_B_V_L_3D = split_inputs_cp(adaln_lora_B_V_L_3D, seq_dim=2, cp_group=self.cp_group)
                if extra_pos_emb_B_V_L_D is not None:
                    extra_pos_emb_B_V_L_D = split_inputs_cp(extra_pos_emb_B_V_L_D, seq_dim=2, cp_group=self.cp_group)

        cross_view_block_mask_cache = self.cross_view_block_mask_cache if self.v_split_mode else None
        if cross_view_block_mask_cache is not None:
            cross_view_block_mask_cache.clear()

        for block in self.blocks:
            x_B_V_L_D = block(
                x_B_V_L_D=x_B_V_L_D,
                emb_B_V_L_D=t_emb_B_V_L_D,
                crossattn_emb=crossattn_emb,
                view_indices_B_V=view_indices_B_V,
                sv_video_size=video_size,
                rope_emb_V_L_1_1_D=rope_freq_V_L_1_1_D,
                adaln_lora_B_V_L_3D=adaln_lora_B_V_L_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_V_L_D,
                view_embedding_proj_B_V_9D=view_embedding_proj_B_V_9D,
                cross_view_block_mask_cache=cross_view_block_mask_cache,
            )

        if cp_enabled and self.cp_group is not None:
            if self.v_split_mode:
                x_B_V_L_D = cat_outputs_cp_with_grad(x_B_V_L_D, seq_dim=1, cp_group=self.cp_group)
                n_cameras = n_cameras_before_cp
            else:
                x_B_V_L_D = cat_outputs_cp_with_grad(x_B_V_L_D, seq_dim=2, cp_group=self.cp_group)

        x_BV_T_H_W_D = rearrange(x_B_V_L_D, "b v (t h w) d -> (b v) t h w d", h=video_size.H, w=video_size.W)
        t_emb_BV_T_D = rearrange(t_emb_B_T_D, "b (v l) d -> (b v) l d", v=n_cameras_before_cp)
        if adaln_lora_B_T_3D is not None:
            adaln_lora_BV_T_3D = rearrange(adaln_lora_B_T_3D, "b (v l) d -> (b v) l d", v=n_cameras_before_cp)
        else:
            adaln_lora_BV_T_3D = None

        x_BV_T_H_W_O = self.final_layer(x_BV_T_H_W_D, t_emb_BV_T_D, adaln_lora_B_T_3D=adaln_lora_BV_T_3D)
        x_BV_C_T_H_W = self.unpatchify(x_BV_T_H_W_O)
        x_B_C_VT_H_W = rearrange(x_BV_C_T_H_W, "(b v) c t h w -> b c (v t) h w", v=n_cameras_before_cp)
        return x_B_C_VT_H_W
