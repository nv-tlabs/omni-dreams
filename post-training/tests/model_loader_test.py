# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests: a consolidated base checkpoint loaded into a LoRA (`use_lora=True`) model.

The release experiments set `checkpoint.load_path` to a registered URI that ends in `/model` and resolves to a local
consolidated `.pt` file, so the URI itself does not end in `.pt`.
"""

import pytest
import torch

from omnidreams._src.imaginaire.utils import checkpoint_db, log
from omnidreams._src.omnidreams.utils import model_loader as omnidreams_model_loader
from omnidreams._src.predict2.models.text2world_model import EMAConfig
from omnidreams._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)
from omnidreams._src.predict2.utils import model_loader as predict2_model_loader

REGISTERED_URI = "s3://bucket/project/job/checkpoints/iter_000000100/model"


class _Mlp(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layer1 = torch.nn.Linear(dim, 2 * dim, bias=False)
        self.layer2 = torch.nn.Linear(2 * dim, dim, bias=False)


class _Block(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)
        self.output_proj = torch.nn.Linear(dim, dim, bias=False)
        self.mlp = _Mlp(dim)
        self.norm = torch.nn.LayerNorm(dim)


class _TinyNet(torch.nn.Module):
    """Module names match the default `lora_target_modules`."""

    def __init__(self, dim: int = 8, num_blocks: int = 2):
        super().__init__()
        self.x_embedder = torch.nn.Linear(dim, dim)
        self.blocks = torch.nn.ModuleList([_Block(dim) for _ in range(num_blocks)])


def _lora_model() -> Text2WorldModelRectifiedFlow:
    """A model with only what loading uses: its config and a net wrapped by the model's own `add_lora`."""
    model = Text2WorldModelRectifiedFlow.__new__(Text2WorldModelRectifiedFlow)
    torch.nn.Module.__init__(model)
    model.config = Text2WorldModelRectifiedFlowConfig(
        use_lora=True, lora_rank=4, lora_alpha=4, ema=EMAConfig(enabled=False)
    )
    torch.manual_seed(0)
    model.net = model.add_lora(_TinyNet(), lora_rank=4, lora_alpha=4)
    return model


def _save_base_checkpoint(path, drop: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    state_dict = {f"net.{k}": v for k, v in _TinyNet().state_dict().items() if f"net.{k}" not in drop}
    torch.save(state_dict, path)
    return state_dict


@pytest.fixture
def registered_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "student_model.pt"
    monkeypatch.setattr(checkpoint_db, "get_checkpoint_path", lambda uri: str(path) if uri == REGISTERED_URI else uri)
    return path


@pytest.mark.level(0)
@pytest.mark.parametrize("loader", [predict2_model_loader, omnidreams_model_loader], ids=["predict2", "omnidreams"])
def test_lora_model_loads_base_weights_from_registered_uri(loader, registered_checkpoint, monkeypatch):
    base = _save_base_checkpoint(registered_checkpoint)
    model = _lora_model()
    warnings = []
    monkeypatch.setattr(log, "warning", lambda msg, *args, **kwargs: warnings.append(str(msg)))

    loader.load_model_state_dict_from_checkpoint(model, config=None, s3_checkpoint_dir=REGISTERED_URI)

    inner = model.net.get_base_model().state_dict()
    base_keys = [k for k in inner if "lora_" not in k]
    assert len(base_keys) == len(base)
    for key in base_keys:
        checkpoint_key = "net." + key.replace(".base_layer.", ".")
        assert torch.equal(inner[key], base[checkpoint_key]), f"{key} was not loaded from {checkpoint_key}"
    lora_b = [v for k, v in inner.items() if "lora_B" in k]
    assert lora_b and all(torch.count_nonzero(v) == 0 for v in lora_b)
    # The adapters, which a base checkpoint does not have, are not reported as missing.
    assert warnings == []


@pytest.mark.level(0)
def test_non_strict_load_state_dict_reports_missing_keys():
    model = _lora_model()
    state_dict = model.state_dict()
    dropped = "net.base_model.model.blocks.1.mlp.layer2.base_layer.weight"
    del state_dict[dropped]

    load_info = model.load_state_dict(state_dict, strict=False)

    assert load_info is not None
    assert load_info.missing_keys == [dropped.removeprefix("net.")]
    assert load_info.unexpected_keys == []


@pytest.mark.level(0)
def test_loader_reports_missing_keys_instead_of_success(registered_checkpoint, monkeypatch):
    dropped = "net.blocks.1.mlp.layer2.weight"
    _save_base_checkpoint(registered_checkpoint, drop=(dropped,))
    model = _lora_model()
    messages = []
    for level in ("info", "warning", "critical"):
        monkeypatch.setattr(log, level, lambda msg, *args, _level=level, **kwargs: messages.append((_level, str(msg))))

    predict2_model_loader.load_model_state_dict_from_checkpoint(model, config=None, s3_checkpoint_dir=REGISTERED_URI)

    assert not any("successfully" in msg for _, msg in messages)
    warnings = [msg for level, msg in messages if level == "warning" and "1 missing keys" in msg]
    assert len(warnings) == 1 and "blocks.1.mlp.layer2.base_layer.weight" in warnings[0]
    assert not any("lora_" in msg for level, msg in messages if level == "warning")
