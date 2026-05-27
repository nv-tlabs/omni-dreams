# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.imaginaire.lazy_config import LazyDict
from omnidreams._src.predict2.networks.minimal_v4_dit import SACConfig
from omnidreams._src.omnidreams.networks.causal_cosmos import CosmosCausalDiT
from omnidreams._src.omnidreams.networks.causal_cosmos_hdmap import CosmosCausalHdmapDiT

# teacher net
# from omnidreams._src.predict2.configs.video2world.defaults.net import COSMOS_V1_2B_NET_MININET
from omnidreams._src.omnidreams.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from omnidreams._src.omnidreams.networks.minimal_v1_lvg_dit_hdmap import MinimalV1LVGDiTHdmapConcat

COSMOS_V2_2B_NET_MININET: LazyDict = L(CosmosCausalDiT)(
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
)

COSMOS_V2_14B_NET_MININET = copy.deepcopy(COSMOS_V2_2B_NET_MININET)
COSMOS_V2_14B_NET_MININET.model_channels = 5120
COSMOS_V2_14B_NET_MININET.num_blocks = 36
COSMOS_V2_14B_NET_MININET.num_heads = 40

COSMOS_V2_2B_NET_MININET_HDMAP: LazyDict = L(CosmosCausalHdmapDiT)(
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
)

## teacher part
COSMOS_V1_2B_NET_MININET: LazyDict = L(MinimalV1LVGDiT)(
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
    # sac_config=SACConfig(),
    sac_config=SACConfig(mode="block_wise"),
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    crossattn_emb_channels=1024,
    use_wan_fp32_strategy=True,
)

COSMOS_V1_2B_NET_MININET_HDMAP: LazyDict = L(MinimalV1LVGDiTHdmapConcat)(
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
    # sac_config=SACConfig(),
    sac_config=SACConfig(mode="block_wise"),
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    crossattn_emb_channels=1024,
    use_wan_fp32_strategy=True,
)


def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="cosmos_v2_2b_causal", node=COSMOS_V2_2B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v2_14b_causal", node=COSMOS_V2_14B_NET_MININET)
    cs.store(
        group="net", package="model.config.net", name="cosmos_v2_2b_causal_hdmap", node=COSMOS_V2_2B_NET_MININET_HDMAP
    )
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B", node=COSMOS_V1_2B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B_hdmap", node=COSMOS_V1_2B_NET_MININET_HDMAP)
