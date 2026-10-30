# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of scripts/convert_dcp_to_pt.py on small LoRA checkpoints in both key layouts the trainers save."""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from peft import LoraConfig, inject_adapter_in_model

from omnidreams._src.omnidreams.self_forcing.utils import load_consolidated_pt_to_net
from scripts.convert_dcp_to_pt import main, merge_lora_weights

RANK, ALPHA = 4, 8

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
        self.norm = torch.nn.LayerNorm(16)

    def forward(self, x):
        return self.norm(self.layer1(self.q_proj(x)))


def _trained_lora_net() -> torch.nn.Module:
    """A net with LoRA adapters as after training: B is nonzero."""
    torch.manual_seed(0)
    config = LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=["q_proj", "layer1"])
    net = inject_adapter_in_model(config, _Net())
    with torch.no_grad():
        for name, param in net.named_parameters():
            if "lora_B" in name:
                param.normal_()
    return net


def _checkpoint_state_dict(net: torch.nn.Module, layout: str) -> dict[str, torch.Tensor]:
    """The model state dict as saved: the predict2 checkpointer of a `use_lora` model maps `base_layer.` away
    ("base"); adapters injected in place keep it ("base_layer")."""
    state_dict = {f"net.{k}": v.detach().clone() for k, v in net.state_dict().items()}
    if layout == "base":
        state_dict = {k.replace("base_layer.", ""): v for k, v in state_dict.items()}
    return state_dict


def _plain_net_from(state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    net = _Net()
    net.load_state_dict({k.removeprefix("net."): v for k, v in state_dict.items()}, strict=True)
    return net


@pytest.mark.level(0)
@pytest.mark.parametrize("layout", ["base", "base_layer"])
def test_merged_net_computes_the_adapted_net(layout):
    lora_net = _trained_lora_net()

    merged, num_merged = merge_lora_weights(_checkpoint_state_dict(lora_net, layout), lora_alpha=ALPHA)

    assert num_merged == 2
    assert not [k for k in merged if ".lora_" in k or ".base_layer." in k]
    x = torch.randn(3, 8)
    torch.testing.assert_close(_plain_net_from(merged)(x), lora_net(x))


@pytest.mark.level(0)
def test_merge_keeps_the_base_weight_dtype():
    state_dict = {k: v.to(torch.bfloat16) for k, v in _checkpoint_state_dict(_trained_lora_net(), "base").items()}

    merged, _ = merge_lora_weights(state_dict, lora_alpha=ALPHA)

    assert {v.dtype for v in merged.values()} == {torch.bfloat16}


@pytest.mark.level(0)
@pytest.mark.parametrize("layout", ["base", "base_layer"])
def test_convert_dcp_with_lora_adapters(tmp_path, layout):
    lora_net = _trained_lora_net()
    dcp.save(_checkpoint_state_dict(lora_net, layout), checkpoint_id=str(tmp_path / "model"))
    out = tmp_path / "model.pt"

    main(["--dcp", str(tmp_path / "model"), "--out", str(out), "--lora-alpha", str(ALPHA)])

    x = torch.randn(3, 8)
    torch.testing.assert_close(_plain_net_from(torch.load(out))(x), lora_net(x))


@pytest.mark.level(0)
@pytest.mark.parametrize("layout", ["base", "base_layer"])
def test_converted_lora_checkpoint_loads_into_a_self_forcing_net(tmp_path, layout):
    """The converted file, given as a self-forcing `net_ckpt`, gives a net without adapters the adapted net."""
    lora_net = _trained_lora_net()
    dcp.save(_checkpoint_state_dict(lora_net, layout), checkpoint_id=str(tmp_path / "model"))
    out = tmp_path / "student.pt"
    main(["--dcp", str(tmp_path / "model"), "--out", str(out), "--lora-alpha", str(ALPHA)])
    net = _Net()

    load_consolidated_pt_to_net(net, str(out), message="load net from .pt")

    x = torch.randn(3, 8)
    torch.testing.assert_close(net(x), lora_net(x))


@pytest.mark.level(0)
def test_convert_refuses_lora_adapters_without_lora_alpha(tmp_path):
    dcp.save(_checkpoint_state_dict(_trained_lora_net(), "base"), checkpoint_id=str(tmp_path / "model"))
    out = tmp_path / "model.pt"

    with pytest.raises(SystemExit):
        main(["--dcp", str(tmp_path / "model"), "--out", str(out)])

    assert not out.exists()


@pytest.mark.level(0)
def test_convert_dcp_without_adapters(tmp_path):
    torch.manual_seed(0)
    state_dict = {f"net.{k}": v for k, v in _Net().state_dict().items()}
    dcp.save(state_dict, checkpoint_id=str(tmp_path / "model"))
    out = tmp_path / "model.pt"

    main(["--dcp", str(tmp_path / "model"), "--out", str(out)])

    converted = torch.load(out)
    assert converted.keys() == state_dict.keys()
    assert all(torch.equal(converted[k], v) for k, v in state_dict.items())
