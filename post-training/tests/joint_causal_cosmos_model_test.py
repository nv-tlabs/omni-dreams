# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU test: CausalJointCosmosModel sets num_frame_per_block on the net that builds the attention mask."""

from types import SimpleNamespace

import pytest
import torch

from omnidreams._src.omnidreams.models.joint_causal_cosmos_model import (
    CausalJointCosmosModel,
    CausalJointCosmosModelConfig,
)
from omnidreams._src.predict2.models.text2world_model_rectified_flow import Text2WorldModelRectifiedFlow


class _TinyCausalNet(torch.nn.Module):
    """Like CosmosCausalDiT: the block size defaults to 1 and the block-causal mask is built from it."""

    def __init__(self):
        super().__init__()
        self.num_frame_per_block = 1
        self.q_proj = torch.nn.Linear(4, 4, bias=False)


def _base_init(self, config):
    """Stands in for Text2WorldModelRectifiedFlow.__init__ (tokenizer, conditioner, text encoder, net on CUDA): only
    what CausalJointCosmosModel.__init__ reads, with the net wrapped by the model's own add_lora under use_lora."""
    torch.nn.Module.__init__(self)
    self.config = config
    self.tensor_kwargs_fp32 = {"device": "cpu", "dtype": torch.float32}
    net = _TinyCausalNet()
    if config.use_lora:
        net = self.add_lora(net, lora_rank=2, lora_alpha=2, lora_target_modules="q_proj")
    self.net = net


@pytest.mark.level(0)
@pytest.mark.parametrize("use_lora", [False, True], ids=["full", "use_lora"])
def test_num_frame_per_block_reaches_the_net(monkeypatch, use_lora):
    monkeypatch.setattr(Text2WorldModelRectifiedFlow, "__init__", _base_init)
    config = CausalJointCosmosModelConfig(
        net=SimpleNamespace(num_blocks=1, num_heads=1), use_lora=use_lora, num_frame_per_block=2
    )

    model = CausalJointCosmosModel(config)

    net = model.net.get_base_model() if use_lora else model.net
    assert net is not model.net if use_lora else net is model.net
    assert net.num_frame_per_block == 2
    assert model.net.num_frame_per_block == 2
