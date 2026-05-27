# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.imaginaire.lazy_config import LazyDict
from omnidreams._src.predict2.networks.minimal_v4_dit import SACConfig
from omnidreams._src.omnidreams.networks.bidirectional_crossview_cosmos import BidirectionalCrossViewCosmosDiT
from omnidreams._src.omnidreams.networks.causal_crossview_cosmos import CausalCrossViewCosmosDiT
from omnidreams._src.omnidreams.networks.causal_crossview_hdmap_cosmos import (
    CausalCrossViewCosmosDiTHDMapConcat,
)

COSMOS_V2_2B_NET_MININET: LazyDict = L(CausalCrossViewCosmosDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_layers=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    # atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    # fp32_timestep_modulation=True,
    # fp32_rope=True,
    sac_config=SACConfig(mode="block_wise"),
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    crossattn_emb_channels=1024,
    use_wan_fp32_strategy=True,
    n_cameras_emb=7,
    view_condition_dim=6,
    concat_view_embedding=False,
    adaln_view_embedding=True,
    enable_cross_view_attn=True,
    layer_mask=None,
    postpone_checkpoint=False,
)

COSMOS_V2_2B_NET_MININET_HDMAP: LazyDict = L(CausalCrossViewCosmosDiTHDMapConcat)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_layers=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    # atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    # fp32_timestep_modulation=True,
    # fp32_rope=True,
    sac_config=SACConfig(mode="block_wise"),
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    crossattn_emb_channels=1024,
    use_wan_fp32_strategy=True,
    n_cameras_emb=7,
    view_condition_dim=6,
    concat_view_embedding=False,
    adaln_view_embedding=True,
    enable_cross_view_attn=True,
    layer_mask=None,
    additional_concat_ch=16,
    additional_init_method="random_init",
    use_control_gate=False,
    postpone_checkpoint=False,
)

BIDIRECTIONAL_COSMOS_V2_2B_NET_MININET: LazyDict = L(BidirectionalCrossViewCosmosDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_layers=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    # atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    # fp32_timestep_modulation=True,
    # fp32_rope=True,
    timestep_scale=0.001,
    sac_config=SACConfig(mode="block_wise"),
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    crossattn_emb_channels=1024,
    use_wan_fp32_strategy=True,
    n_cameras_emb=7,
    view_condition_dim=6,
    concat_view_embedding=False,
    adaln_view_embedding=True,
    enable_cross_view_attn=True,
    layer_mask=None,
    v_split_mode=True,
)


def register_net():
    cs = ConfigStore.instance()
    cs.store(
        group="net", package="model.config.net", name="cosmos_v2_2b_causal_multiview", node=COSMOS_V2_2B_NET_MININET
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="bidirectional_crossview_cosmos",
        node=BIDIRECTIONAL_COSMOS_V2_2B_NET_MININET,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="causal_crossview_cosmos_hdmap",
        node=COSMOS_V2_2B_NET_MININET_HDMAP,
    )
