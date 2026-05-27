# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Usage:
PYTHONPATH=. torchrun --nproc_per_node=2 -m pytest -rs projects/cosmos/reasoning/v1/parallelisms/dcp_checkpointer_test.py
or
PYTHONPATH=. torchrun --nproc_per_node=4 projects/cosmos/reasoning/v1/parallelisms/dcp_checkpointer_test.py
"""

import importlib
import os

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor import DTensor

from omnidreams._src.imaginaire.utils import distributed, log, misc
from omnidreams._src.imaginaire.utils.config_helper import get_config_module, override
from omnidreams._src.imaginaire.utils.helper_test import RunIf
from omnidreams._src.reason1.parallelisms.dcp_checkpointer import ModelWrapper
from omnidreams._src.reason1.parallelisms.torchtitan_utilts import get_train_context
from omnidreams._src.reason1.utils.testing_utils import setup_training_model


def build_model_and_forward(tp, load_model_path=None, save_model_path=None):
    distributed.init()

    # current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_file = "omnidreams/_src/reason1/configs/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(
        config,
        [
            "--",
            "experiment=sft_exp000_000_qwen7b_joint",
            "trainer.max_iter=5",
            "model.model_config.n_layers=2",
            f"model.model_config.training.tensor_parallel_degree={tp}",
        ],
    )
    model, device = setup_training_model(config=config, seed=42)

    if save_model_path:
        os.makedirs(save_model_path, exist_ok=True)
        storage_writer = dcp.filesystem.FileSystemWriter(save_model_path, thread_count=1)
        state_dict = ModelWrapper(model).state_dict()
        dcp.save(state_dict, storage_writer=storage_writer)
        print(f"DCP model saved to {save_model_path}")
    if load_model_path:
        print(f"Loading DCP model from {load_model_path}")
        state_dict = ModelWrapper(model).state_dict()
        dcp.load(state_dict, checkpoint_id=load_model_path)
    # Forward and get output and gradient
    vocab_size = 131072
    batch_size = 1
    seq_length = 32
    seed = 42
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), generator=generator, device=device)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_length), generator=generator, device=device)

    train_context = get_train_context(
        not model.config.training.disable_loss_parallel,
        model.config.experimental.enable_compiled_autograd,
    )

    with train_context():
        output = model(input_ids, data_batch={})
        loss = torch.nn.functional.cross_entropy(output.view(-1, vocab_size), target_ids.view(-1))
        loss.backward()
    print(f"device: {device}, loss: {loss}, input_ids: {input_ids}")
    loss_tp = loss.detach()
    gradient_tp = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    if isinstance(loss_tp, DTensor):
        loss_tp = loss_tp.full_tensor()
    gradient_tp = [p.full_tensor() if isinstance(p, DTensor) else p for p in gradient_tp]
    # if torch.distributed.is_initialized():
    #     torch.distributed.destroy_process_group()
    return loss_tp, gradient_tp


@pytest.mark.L1
@RunIf(
    min_gpus=8,
    requires_file=[
        "credentials/s3_training.secret",
    ],
)
def test_checkpoint_tp_load():
    """Checkpoint a model in tp2 and directly load it to tp1 or tp4.
    The loss and gradient should be the same.
    """

    # Fixed random seed
    misc.set_random_seed(seed=0)

    save_model_path = "/tmp/dcp_checkpointer_test"
    os.makedirs(save_model_path, exist_ok=True)
    loss_tp2, gradient_tp2 = build_model_and_forward(tp=2)  # , load_model_path=None, save_model_path=save_model_path)
    loss_tp1, gradient_tp1 = build_model_and_forward(tp=1)  # , load_model_path=save_model_path, save_model_path=None)
    loss_tp4, gradient_tp4 = build_model_and_forward(tp=4)  # , load_model_path=save_model_path, save_model_path=None)
    log.info(f"Loss TP2: {loss_tp2}, Loss TP1: {loss_tp1}, Loss TP4: {loss_tp4}")
    assert torch.isclose(loss_tp2, loss_tp1, atol=1e-5, rtol=1e-5), "Loss values differ between TP=2 and TP=1"
    assert torch.isclose(loss_tp2, loss_tp4, atol=1e-5, rtol=1e-5), "Loss values differ between TP=2 and TP=4"

    for g1, g2 in zip(gradient_tp2, gradient_tp1):
        # log.info(f"Gradient TP2: {g1.shape}, Gradient TP1: {g2.shape}, diff: {torch.norm(g1 - g2)}")
        # log.info(f"g1: {g1}, g2: {g2}")
        assert torch.allclose(g1, g2, atol=1e-5, rtol=1e-5), "Gradients differ between TP=2 and TP=1"
    for g1, g2 in zip(gradient_tp2, gradient_tp4):
        # log.info(f"Gradient TP2: {g1.shape}, Gradient TP4: {g2.shape}, diff: {torch.norm(g1 - g2)}")
        assert torch.allclose(g1, g2, atol=1e-5, rtol=1e-5), "Gradients differ between TP=2 and TP=4"
    print("Test passed: Loss and gradient consistency between TP=2 and TP=1")
    print("Test passed: Loss and gradient consistency between TP=2 and TP=4")


if __name__ == "__main__":
    test_checkpoint_tp_load()
