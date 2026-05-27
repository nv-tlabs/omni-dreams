# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.imaginaire.lazy_config import LazyDict
from omnidreams._src.predict2.networks.minimal_v4_dit import MiniTrainDIT, SACConfig
from omnidreams._src.predict2.networks.wan2pt1 import WanModel

COSMOS_V1_7B_NET_MININET: LazyDict = L(MiniTrainDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=4096,
    num_blocks=28,
    num_heads=32,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=True,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=2.0,
    sac_config=SACConfig(),
)
COSMOS_V1_2B_NET_MININET = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
COSMOS_V1_2B_NET_MININET.model_channels = 2048
COSMOS_V1_2B_NET_MININET.num_heads = 16
COSMOS_V1_2B_NET_MININET.num_blocks = 28
COSMOS_V1_2B_NET_MININET.extra_per_block_abs_pos_emb = False
COSMOS_V1_2B_NET_MININET.rope_t_extrapolation_ratio = 1.0

COSMOS_V1_14B_NET_MININET = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
COSMOS_V1_14B_NET_MININET.model_channels = 5120
COSMOS_V1_14B_NET_MININET.num_heads = 40
COSMOS_V1_14B_NET_MININET.num_blocks = 36
COSMOS_V1_14B_NET_MININET.extra_per_block_abs_pos_emb = False
COSMOS_V1_14B_NET_MININET.rope_t_extrapolation_ratio = 1.0

COSMOS_V2_0P6B_NET = copy.deepcopy(COSMOS_V1_2B_NET_MININET)
COSMOS_V2_0P6B_NET.model_channels = 20 * 64
COSMOS_V2_0P6B_NET.num_heads = 20
COSMOS_V2_0P6B_NET.num_blocks = 20

COSMOS_V2_1B_NET = copy.deepcopy(COSMOS_V1_2B_NET_MININET)
COSMOS_V2_1B_NET.model_channels = 24 * 64
COSMOS_V2_1B_NET.num_heads = 24
COSMOS_V2_1B_NET.num_blocks = 24

mini_net = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
mini_net.model_channels = 1024
mini_net.num_heads = 8
mini_net.num_blocks = 2
mini_net.rope_t_extrapolation_ratio = 1.0

COSMOS_V2_2B_NET: LazyDict = L(MiniTrainDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    extra_per_block_abs_pos_emb=True,
)


WAN2PT1_1PT3B: LazyDict = L(WanModel)(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=16,
    model_type="t2v",
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(mode="block_wise"),
)


def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="mini_net", node=mini_net)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_7B", node=COSMOS_V1_7B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B", node=COSMOS_V1_2B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_14B", node=COSMOS_V1_14B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v2_2B", node=COSMOS_V2_2B_NET)
    cs.store(group="net", package="model.config.net", name="cosmos_v2_0p6B", node=COSMOS_V2_0P6B_NET)
    cs.store(group="net", package="model.config.net", name="cosmos_v2_1B", node=COSMOS_V2_1B_NET)
    cs.store(group="net", package="model.config.net", name="wan2pt1_1pt3B", node=WAN2PT1_1PT3B)
