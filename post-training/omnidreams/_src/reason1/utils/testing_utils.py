# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import torch
import torch.distributed.checkpoint as dcp

from omnidreams._src.imaginaire.lazy_config import instantiate
from omnidreams._src.imaginaire.utils import distributed, log
from omnidreams._src.reason1.parallelisms.parallel_dims import ParallelDims
from omnidreams._src.reason1.parallelisms.parallelize_qwen import parallelize_qwen
from omnidreams._src.reason1.parallelisms.torchtitan_utilts import device_module, device_type


def setup_training_model(config, seed=0, checkpoint_load_path=None):
    torch.manual_seed(seed)
    world_size = distributed.get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    with torch.device("meta"):
        model = instantiate(config.model)

    world_mesh = None
    if world_size > 1:
        log.info(f"Initializing distributed process group with world size {world_size}")
        parallel_dims = ParallelDims(
            dp_shard=model.config.training.data_parallel_shard_degree,
            dp_replicate=model.config.training.data_parallel_replicate_degree,
            cp=model.config.training.context_parallel_degree,
            tp=model.config.training.tensor_parallel_degree,
            pp=model.config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not model.config.training.disable_loss_parallel,
        )
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = torch.device(f"{device_type}:{local_rank}")
        device_module.set_device(device)
        world_mesh = parallel_dims.build_mesh(device_type=device_type)
        log.info(world_mesh)
        parallelize_qwen(model, world_mesh, parallel_dims, model.config)
    else:
        device = None
    model.to_empty(device=device_type)

    # * unit test require calling `init_weights`
    # PYTHONPATH=. torchrun --nproc_per_node=2 -m pytest -rs --L1 projects/cosmos/reasoning/v1/models/vlm_simple_test.py::test_maybe_freeze
    # * unit test fail if calling `init_weights`
    # PYTHONPATH=. torchrun --nproc_per_node=2 -m pytest -rs --L1 projects/cosmos/reasoning/v1/scripts/training_tp_test.py::test_training_loss_and_gradient_consistency
    # PYTHONPATH=. torchrun --nproc_per_node=4 -m pytest -rs --L1 projects/cosmos/reasoning/v1/parallelisms/dcp_checkpointer_test.py::test_checkpoint_tp_load
    # What are the weight value is not calling init weights?
    # model.init_weights()

    if not model.config.use_rope_from_torchtitan:
        model.model.rope.init_weights()
    if checkpoint_load_path:
        # Load checkpoint
        state_dict = model.state_dict()
        log.info(f"Loading chkpt at: {checkpoint_load_path}")
        dcp.load(state_dict, checkpoint_id=checkpoint_load_path)
    model.train()
    return model, device
