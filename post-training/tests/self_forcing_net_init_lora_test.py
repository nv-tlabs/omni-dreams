# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests: the self-forcing net-init loaders refuse LoRA adapters that the net has no parameters for, instead of
loading only the frozen base weights."""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer

from omnidreams._src.omnidreams.self_forcing.utils import (
    load_consolidated_pt_to_net,
    load_internal_dcp_checkpoint_to_net,
)

# torch.distributed.checkpoint warns when it runs without a process group and about its own TypedStorage use.
pytestmark = [
    pytest.mark.filterwarnings("ignore:torch.distributed is disabled, unavailable or uninitialized:UserWarning"),
    pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning"),
]


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.layer1 = torch.nn.Linear(8, 16)

    def forward(self, x):
        return self.layer1(self.q_proj(x))


def _trained_lora_net() -> torch.nn.Module:
    torch.manual_seed(0)
    net = inject_adapter_in_model(LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "layer1"]), _Net())
    with torch.no_grad():
        for name, param in net.named_parameters():
            if "lora_B" in name:
                param.normal_()
    return net


def _lora_checkpoint_state_dict() -> dict[str, torch.Tensor]:
    """A `use_lora` training checkpoint: base weights under `<m>.weight` next to the adapter weights."""
    net = _trained_lora_net()
    return {f"net.{k.replace('base_layer.', '')}": v.detach().clone() for k, v in net.state_dict().items()}


@pytest.mark.level(0)
def test_pt_loader_refuses_unmerged_lora_adapters(tmp_path):
    path = tmp_path / "student.pt"
    torch.save(_lora_checkpoint_state_dict(), path)

    with pytest.raises(ValueError, match="LoRA"):
        load_consolidated_pt_to_net(_Net(), str(path), message="load net from .pt")


@pytest.mark.level(0)
def test_dcp_loader_refuses_unmerged_lora_adapters(tmp_path):
    dcp.save(_lora_checkpoint_state_dict(), checkpoint_id=str(tmp_path / "model"))

    with pytest.raises(ValueError, match="LoRA"):
        load_internal_dcp_checkpoint_to_net(_Net(), str(tmp_path / "model"))


@pytest.mark.level(0)
def test_pt_loader_loads_merged_lora_checkpoint(tmp_path):
    lora_net = _trained_lora_net()
    x = torch.randn(3, 8)
    expected = lora_net(x)
    with torch.no_grad():
        for module in lora_net.modules():
            if isinstance(module, LoraLayer):
                module.merge()
    merged = {f"net.{k.replace('base_layer.', '')}": v for k, v in lora_net.state_dict().items() if "lora_" not in k}
    path = tmp_path / "student.pt"
    torch.save(merged, path)
    net = _Net()

    load_consolidated_pt_to_net(net, str(path), message="load net from .pt")

    torch.testing.assert_close(net(x), expected)
