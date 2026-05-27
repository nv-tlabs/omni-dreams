# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import collections
import copy
from typing import Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.tensor import DTensor, distribute_tensor

from omnidreams._src.imaginaire.lazy_config import LazyDict
from omnidreams._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from omnidreams._src.imaginaire.utils import log, misc
from omnidreams._src.predict2.checkpointer.dcp import ModelWrapper
from omnidreams._src.predict2.utils.dtensor_helper import broadcast_dtensor_model_states
from omnidreams._src.omnidreams.checkpointer.dcp import get_storage_reader
from omnidreams._src.omnidreams.utils.misc import sync_timer


def build_net(
    net_config: LazyDict,
    device_mesh=None,
    mixed_precision_policy_root_module: Optional[torch.distributed.fsdp.MixedPrecisionPolicy] = None,
    mixed_precision_policy_internal_layers: Optional[torch.distributed.fsdp.MixedPrecisionPolicy] = None,
):
    # NOTE: (ruiyuang) always use meta device, no need to use cpu
    if mixed_precision_policy_root_module is not None:
        root_fsdp_kwargs = {"mp_policy": mixed_precision_policy_root_module}
    else:
        root_fsdp_kwargs = {}

    if mixed_precision_policy_internal_layers is not None:
        internal_fsdp_kwargs = {"mp_policy": mixed_precision_policy_internal_layers}
    else:
        internal_fsdp_kwargs = {}

    init_device = "meta"
    with misc.timer("Creating PyTorch model"):
        with sync_timer("net instantiate"):
            with torch.device(init_device):
                net: torch.nn.Module = lazy_instantiate(net_config)

        if device_mesh:
            net.fully_shard(mesh=device_mesh, **internal_fsdp_kwargs)
            net = fully_shard(net, mesh=device_mesh, reshard_after_forward=True, **root_fsdp_kwargs)

        with misc.timer("meta to cuda and broadcast model states"):
            net.to_empty(device="cuda")
            # IMPORTANT: (qsh) model init should not depends on current tensor shape, or it can handle Dtensor shape.
            net.init_weights()

        if device_mesh:

            broadcast_dtensor_model_states(net, device_mesh)
            for name, param in net.named_parameters():
                assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
    return net


def load_self_forcing_public_ckpt_to_net(
    net,
    checkpoint_path: str,
    message: str = "",
    model_key: str | None = None,
):
    # Read public checkpoint
    assert checkpoint_path.endswith(".pt"), f"checkpoint_path should end with .pt, got {checkpoint_path}"

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if model_key is not None:
        state_dict = state_dict[model_key]
    elif "generator" in state_dict:
        state_dict = state_dict["generator"]
    elif "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    # Match the state dict of the public checkpoint to the internal model
    _state_dict = get_model_state_dict(net)
    for key, v in state_dict.items():
        tgt_key = key.replace("model.", "") if key.startswith("model.") else key
        tgt = _state_dict[tgt_key]

        # If target param/buffer is a DTensor and checkpoint value is not, distribute it
        if isinstance(tgt, DTensor) and not isinstance(v, DTensor):
            # Match device, dtype, and placements from the target DTensor
            v = v.to(tgt.device, dtype=tgt.dtype, copy=False)
            # Special handling for patch embedding module as the internal model uses Linear while
            # the public checkpoint uses Conv2d
            if tgt_key == "patch_embedding.weight":
                v = v.reshape_as(tgt)
            v = distribute_tensor(v, tgt.device_mesh, tgt.placements)
            _state_dict[tgt_key] = v
        # If target is a plain Tensor but checkpoint is a DTensor, bring it local
        elif not isinstance(tgt, DTensor) and isinstance(v, DTensor):
            v = v.to_local().to(tgt.device, dtype=tgt.dtype, copy=False)
            # Special handling for patch embedding module as the internal model uses Linear while
            # the public checkpoint uses Conv2d
            if tgt_key == "patch_embedding.weight":
                v = v.reshape_as(tgt)
            _state_dict[tgt_key] = v
        else:
            if tgt_key == "patch_embedding.weight":
                v = v.reshape_as(tgt)
            _state_dict[tgt_key] = v

    # Load and log message
    log.critical(
        f"{message}: " + str(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=True))),
        rank0_only=False,
    )




def load_consolidated_pt_to_net(
    net,
    checkpoint_path: str,
    message: str = "",
    net_prefix: str = "net.",
):
    """Load a consolidated `.pt` checkpoint into a bare `nn.Module` (e.g. `self.net`,
    `self.net_real_score`, `self.net_fake_score`).

    Supports the output format of `predict2/scripts/convert_distcp_to_pt.py`:
    - `model.pt`: full DCP-converted state with mixed prefixes (`net.*`, `net_ema.*`, etc.).
      Only `net_prefix`-prefixed keys are loaded; everything else is dropped.
    - `model_ema_fp32.pt` / `model_ema_bf16.pt`: EMA-only state where keys have been
      renamed `net_ema.* -> net.*` by the conversion script (line 116).

    Also tolerates an outer `{"model": <state_dict>}` or `{"state_dict": <state_dict>}`
    wrapper for compatibility with hand-rolled `.pt` files.

    For public self-forcing checkpoints (with `generator` / `generator_ema` /
    `model.<layer>` keys) keep using `load_self_forcing_public_ckpt_to_net`.
    """
    assert checkpoint_path.endswith(".pt"), f"checkpoint_path should end with .pt, got {checkpoint_path}"

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Unwrap common containers
    if isinstance(state_dict, dict):
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

    _state_dict = get_model_state_dict(net)

    loaded_count = 0
    for key, v in state_dict.items():
        if not key.startswith(net_prefix):
            # `net_ema.*`, optimizer state, scheduler state, etc.
            continue
        tgt_key = key[len(net_prefix) :]
        if tgt_key not in _state_dict:
            continue
        tgt = _state_dict[tgt_key]

        # DTensor / tensor mismatch handling (mirrors load_self_forcing_public_ckpt_to_net)
        if isinstance(tgt, DTensor) and not isinstance(v, DTensor):
            v = v.to(tgt.device, dtype=tgt.dtype, copy=False)
            v = distribute_tensor(v, tgt.device_mesh, tgt.placements)
        elif not isinstance(tgt, DTensor) and isinstance(v, DTensor):
            v = v.to_local().to(tgt.device, dtype=tgt.dtype, copy=False)
        else:
            v = v.to(tgt.device, dtype=tgt.dtype, copy=False)
        _state_dict[tgt_key] = v
        loaded_count += 1

    if loaded_count == 0:
        raise ValueError(
            f"No keys with prefix {net_prefix!r} found in {checkpoint_path}. "
            f"Top-level keys (sample): {list(state_dict.keys())[:5]}"
        )

    log.critical(
        f"{message} (loaded {loaded_count}/{len(_state_dict)} from {checkpoint_path}): "
        + str(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=True))),
        rank0_only=False,
    )


def load_internal_dcp_checkpoint_to_net(
    net,
    checkpoint_path: str,
    net_prefix: str = "net.",
    patch_embedding_reshape: bool = True,
    allow_partial_load: bool = True,
    credential_path="credentials/s3_inference.secret",
):
    assert checkpoint_path.endswith("/model"), f"checkpoint_path should end with /model, got {checkpoint_path}"

    storage_reader = get_storage_reader(checkpoint_path, credential_path)
    _state_dict = get_model_state_dict(net)
    _new_state_dict = collections.OrderedDict()
    # To manually check if the checkpoint is loaded correctly
    _key_to_check = None
    _value_to_check = None
    loaded_st_to_st_mapping = {}
    for k in _state_dict.keys():
        # try to remember a key and check afterwards
        if _value_to_check is None and "weight" in k and "lora" not in k:
            _key_to_check = k
            _value_to_check = torch.clone(_state_dict[k])

        # key renaming process to match the standard checkpoint
        if "_extra_state" in k:
            continue
        _name_to_load = f"{net_prefix}{k}"
        _new_state_dict[_name_to_load] = _state_dict[k]
        loaded_st_to_st_mapping[_name_to_load] = k

    dcp.load(
        _new_state_dict,
        storage_reader=storage_reader,
        planner=DefaultLoadPlanner(allow_partial_load=allow_partial_load),
    )
    for k in _new_state_dict.keys():
        _state_dict[loaded_st_to_st_mapping[k]] = _new_state_dict[k]
    log.info(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=True)))

    # double check if the checkpoint is loaded correctly
    if _key_to_check is not None:
        _current_value = get_model_state_dict(net)[_key_to_check]
        if (_current_value == _value_to_check).all():
            log.warning(
                f"The value of {_key_to_check} remain unchanged, please double check!"
                f"before: {_value_to_check}, after: {_current_value}"
            )
    del _state_dict, _new_state_dict, _current_value, _key_to_check, _value_to_check
