# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests: LoRA adapters on the self-forcing (DMD) generator (`use_lora`)."""

import math
import socket

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from peft import LoraConfig, inject_adapter_in_model
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.omnidreams.self_forcing.dmd import ImaginaireDMDBaseModel, ImaginaireDMDBaseModelConfig
from omnidreams._src.omnidreams.self_forcing.utils import (
    build_net,
    load_consolidated_pt_to_net,
    load_internal_dcp_checkpoint_to_net,
)
from omnidreams._src.predict2.checkpointer.dcp import DefaultLoadPlanner, ModelWrapper
from omnidreams._src.predict2.models.text2world_model import EMAConfig

LORA = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "output_proj"])


class _Block(torch.nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.output_proj = torch.nn.Linear(dim, dim, bias=False)
        self.norm = torch.nn.LayerNorm(dim)

    def init_weights(self):
        # Like the DiT blocks: the Linear weights are initialized through the module's `.weight`.
        for linear in (self.q_proj, self.output_proj):
            torch.nn.init.trunc_normal_(linear.weight, std=0.02)
        self.norm.reset_parameters()

    def forward(self, x):
        return self.norm(x + self.output_proj(self.q_proj(x)))


class _Net(torch.nn.Module):
    """Stands in for a DiT: `init_weights` and `fully_shard` as build_net uses them."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(2)])

    def init_weights(self):
        for block in self.blocks:
            block.init_weights()

    def fully_shard(self, mesh, **kwargs):
        for block in self.blocks:
            fully_shard(block, mesh=mesh, **kwargs)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@pytest.fixture(autouse=True)
def materialize_on_cpu(monkeypatch):
    """build_net materializes the net on "cuda"; materialize it on the CPU instead."""
    to_empty = torch.nn.Module.to_empty
    monkeypatch.setattr(
        torch.nn.Module, "to_empty", lambda self, *, device, recurse=True: to_empty(self, device="cpu", recurse=recurse)
    )


@pytest.fixture
def hsdp_mesh():
    """A (replicate, shard) = (1, 1) CPU mesh, as hsdp_device_mesh builds it, in a one-process gloo group."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1, 1), mesh_dim_names=("replicate", "shard"))
    finally:
        dist.destroy_process_group()


def _save_student_init(path, drop: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    state_dict = {f"net.{k}": v for k, v in _Net().state_dict().items() if f"net.{k}" not in drop}
    torch.save(state_dict, path)
    return state_dict


def _assert_lora_init(net: torch.nn.Module):
    params = dict(net.named_parameters())
    trainable = [name for name, param in params.items() if param.requires_grad]
    assert len(trainable) == 8 and all("lora_" in name for name in trainable)
    for name, param in params.items():
        local = param.full_tensor() if isinstance(param, DTensor) else param
        if "lora_A" in name:
            assert 0 < local.abs().max() <= 1 / math.sqrt(local.shape[1])
        if "lora_B" in name:
            assert torch.count_nonzero(local) == 0


def _assert_base_loaded(net: torch.nn.Module, checkpoint: dict[str, torch.Tensor]):
    state_dict = get_model_state_dict(net)
    base_keys = [k for k in state_dict if "lora_" not in k]
    assert len(base_keys) == len(checkpoint)
    for key in base_keys:
        value = state_dict[key].full_tensor() if isinstance(state_dict[key], DTensor) else state_dict[key]
        assert torch.equal(value, checkpoint["net." + key.replace(".base_layer.", ".")]), key


@pytest.mark.level(0)
def test_build_net_injects_trainable_lora_adapters():
    net = build_net(L(_Net)(), lora_config=LORA)

    assert "blocks.0.q_proj.base_layer.weight" in dict(net.named_parameters())
    _assert_lora_init(net)
    net(torch.randn(3, 8)).square().mean().backward()
    for name, param in net.named_parameters():
        assert (param.grad is not None) == ("lora_" in name), name


@pytest.mark.level(0)
def test_pt_loader_maps_base_weights_onto_adapted_layers(tmp_path):
    checkpoint = _save_student_init(tmp_path / "student.pt")
    net = inject_adapter_in_model(LORA, _Net())

    load_consolidated_pt_to_net(net, str(tmp_path / "student.pt"), message="load net from .pt")

    _assert_base_loaded(net, checkpoint)


@pytest.mark.level(0)
def test_pt_loader_refuses_checkpoint_without_an_adapted_weight(tmp_path):
    _save_student_init(tmp_path / "student.pt", drop=("net.blocks.1.output_proj.weight",))
    net = inject_adapter_in_model(LORA, _Net())

    with pytest.raises(ValueError, match="LoRA-adapted"):
        load_consolidated_pt_to_net(net, str(tmp_path / "student.pt"), message="load net from .pt")


def _lora_dcp_state_dict(layout: str) -> dict[str, torch.Tensor]:
    """A LoRA generator's model state dict as saved: `<m>.weight` next to the adapters ("base", as the predict2
    checkpointer saves `use_lora` runs) or `<m>.base_layer.weight` ("base_layer", as a generator with in-place adapters
    is saved)."""
    torch.manual_seed(2)
    trained = inject_adapter_in_model(LORA, _Net())
    state_dict = {f"net.{k}": v.detach().clone() for k, v in trained.state_dict().items()}
    if layout == "base":
        state_dict = {k.replace("base_layer.", ""): v for k, v in state_dict.items()}
    return state_dict


@pytest.mark.level(0)
@pytest.mark.parametrize("layout", ["base", "base_layer"])
@pytest.mark.filterwarnings("ignore:torch.distributed is disabled, unavailable or uninitialized:UserWarning")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")
def test_dcp_loader_loads_a_lora_checkpoint_into_the_adapted_generator(tmp_path, layout):
    checkpoint = _lora_dcp_state_dict(layout)
    dcp.save(checkpoint, checkpoint_id=str(tmp_path / "model"))
    net = inject_adapter_in_model(LORA, _Net())

    load_internal_dcp_checkpoint_to_net(net, str(tmp_path / "model"))

    state_dict = net.state_dict()
    assert len(state_dict) == len(checkpoint)
    for key, value in state_dict.items():
        checkpoint_key = "net." + (key if layout == "base_layer" else key.replace(".base_layer.", "."))
        assert torch.equal(value, checkpoint[checkpoint_key]), key


@pytest.mark.level(0)
@pytest.mark.filterwarnings("ignore:torch.distributed is disabled, unavailable or uninitialized:UserWarning")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")
def test_dcp_loader_refuses_checkpoint_without_an_adapted_weight(tmp_path):
    checkpoint = _lora_dcp_state_dict("base")
    del checkpoint["net.blocks.1.output_proj.weight"]
    dcp.save(checkpoint, checkpoint_id=str(tmp_path / "model"))

    with pytest.raises(ValueError, match="LoRA-adapted"):
        load_internal_dcp_checkpoint_to_net(inject_adapter_in_model(LORA, _Net()), str(tmp_path / "model"))


@pytest.mark.level(0)
@pytest.mark.filterwarnings("ignore:DTensor random operators may not have complete support on cpu device mesh")
def test_fsdp_generator_with_lora_starts_as_its_student_init(tmp_path, hsdp_mesh):
    checkpoint = _save_student_init(tmp_path / "student.pt")
    net = build_net(L(_Net)(), hsdp_mesh, lora_config=LORA)
    assert all(isinstance(param, DTensor) for param in net.parameters())
    _assert_lora_init(net)

    load_consolidated_pt_to_net(net, str(tmp_path / "student.pt"), message="load net from .pt")

    _assert_base_loaded(net, checkpoint)
    student_init = _Net()
    student_init.load_state_dict({k.removeprefix("net."): v for k, v in checkpoint.items()})
    x = torch.randn(3, 8)
    torch.testing.assert_close(net(x), student_init(x))


def _dmd_model(tmp_path, **config_kwargs) -> ImaginaireDMDBaseModel:
    """A DMD model with only what set_up_model uses, and three small nets loaded from one student-init file."""
    path = str(tmp_path / "student.pt")
    model = ImaginaireDMDBaseModel.__new__(ImaginaireDMDBaseModel)
    torch.nn.Module.__init__(model)
    model.config = ImaginaireDMDBaseModelConfig(
        net=L(_Net)(),
        net_ckpt=path,
        net_real_score=L(_Net)(),
        net_real_score_ckpt=path,
        net_fake_score=L(_Net)(),
        net_fake_score_ckpt=path,
        conditioner=L(torch.nn.Identity)(),
        ema=EMAConfig(enabled=False),
        **config_kwargs,
    )
    model.fsdp_device_mesh = None
    model.mixed_precision_policy_root_module = None
    model.mixed_precision_policy_internal_layers = None
    return model


@pytest.mark.level(0)
def test_dmd_use_lora_adapts_the_generator_only(tmp_path):
    checkpoint = _save_student_init(tmp_path / "student.pt")
    model = _dmd_model(tmp_path, use_lora=True, lora_rank=4, lora_alpha=8, lora_target_modules="q_proj,output_proj")

    model.set_up_model()

    _assert_lora_init(model.net)
    _assert_base_loaded(model.net, checkpoint)
    for net, trainable in ((model.net_real_score, False), (model.net_fake_score, True)):
        assert not [name for name, _ in net.named_parameters() if "lora_" in name]
        assert all(param.requires_grad == trainable for param in net.parameters())


@pytest.mark.level(0)
@pytest.mark.filterwarnings("ignore:torch.distributed is disabled, unavailable or uninitialized:UserWarning")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")
def test_dmd_use_lora_checkpoint_saves_base_keys_and_resumes(tmp_path):
    """The self-forcing checkpointer saves and resumes the model through the predict2 ModelWrapper, which maps
    `base_layer.` away for `use_lora` models (as in the other stages) and back on load."""
    _save_student_init(tmp_path / "student.pt")
    lora_kwargs = dict(use_lora=True, lora_rank=4, lora_alpha=8, lora_target_modules="q_proj,output_proj")
    trained = _dmd_model(tmp_path, **lora_kwargs)
    trained.set_up_model()
    with torch.no_grad():
        for name, param in trained.net.named_parameters():
            if "lora_" in name:
                param.normal_()

    saved = ModelWrapper(trained).state_dict()
    assert not [k for k in saved if ".base_layer." in k]
    assert {"net.blocks.0.q_proj.weight", "net.blocks.0.q_proj.lora_B.default.weight"} <= saved.keys()
    dcp.save(saved, checkpoint_id=str(tmp_path / "model"))

    resumed = _dmd_model(tmp_path, **lora_kwargs)
    resumed.set_up_model()
    wrapper = ModelWrapper(resumed)
    state_dict = wrapper.state_dict()
    dcp.load(state_dict, checkpoint_id=str(tmp_path / "model"), planner=DefaultLoadPlanner(allow_partial_load=True))
    wrapper.load_state_dict(state_dict)

    expected = trained.net.state_dict()
    for key, value in resumed.net.state_dict().items():
        assert torch.equal(value, expected[key]), key


@pytest.mark.level(0)
def test_dmd_use_lora_refuses_ema(tmp_path):
    _save_student_init(tmp_path / "student.pt")
    model = _dmd_model(tmp_path, use_lora=True)
    model.config.ema = EMAConfig(enabled=True)

    with pytest.raises(AssertionError, match="ema"):
        model.set_up_model()
