# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DCP checkpointer for causal-multiview distillation trainers.

The distillation trainer (``trainer_distillation.ImaginaireTrainer``) carries
multiple optimizers and schedulers — one per sub-module exposed by
``model.model_dict()`` (e.g. ``"net"`` and ``"fake_score"``) — and hands them
to the checkpointer as dicts rather than single objects. The base
``DistributedCheckpointer`` from ``predict2`` assumes single instances and
crashes when given a dict. This subclass mirrors the per-key save/load
convention already in use by ``predict2_distill``: each entry is written under
its own DCP sub-directory (``optim_<k>/``, ``scheduler_<k>/``).
"""

import gc
import os
import time
from typing import Any

import torch
import torch.distributed.checkpoint as dcp

from omnidreams._src.imaginaire.model import ImaginaireModel
from omnidreams._src.imaginaire.utils import distributed, log, misc
from omnidreams._src.predict2.checkpointer.dcp import (
    AsyncMode,
    DefaultLoadPlanner,
    DefaultSavePlanner,
    ModelWrapper,
    OptimizerWrapper,
)
from omnidreams._src.predict2.checkpointer.dcp import (
    DistributedCheckpointer as _DistributedCheckpointer,
)


def _reclaim_cuda_cache_before_save(iteration: int) -> None:
    """Free the allocator's idle reserved cache before a checkpoint save.

    On the E3 self-forcing recipe (8xH100 80 GB, three resident networks) the
    caching allocator holds ~20 GB reserved beyond live tensors at save time
    (peak reserved ~75 GB vs ~54 GB allocated), while NVML-free is <1 GB. The
    save is sharded (``get_{model,optimizer}_state_dict`` with no
    ``full_state_dict`` + ``dedup_save_to_lowest_rank``), so it does not gather
    full tensors, but its transients still need NVML-free headroom the driver
    can hand out: NCCL's per-collective workspace for the save plan/metadata
    exchange and the FileSystem writer's host/device staging copies. Both
    allocate *outside* the caching allocator, so they OOM with <1 GB free
    despite the idle reserved cache. Returning that cache to the driver here
    (more effective under ``expandable_segments``, which can release whole
    segments) restores ~20 GB of headroom; mirrors the ``empty_cache()`` in
    ``load()``. If this proves insufficient, the next lever is
    ``StateDictOptions(cpu_offload=True)`` on the model/optimizer state dicts.

    Logs the reclaimed headroom once per save (rank0).
    """
    if not torch.cuda.is_available():
        gc.collect()
        return
    gib = 1024**3
    free0, _ = torch.cuda.mem_get_info()
    resv0 = torch.cuda.memory_reserved()
    alloc0 = torch.cuda.memory_allocated()
    gc.collect()
    torch.cuda.empty_cache()
    free1, _ = torch.cuda.mem_get_info()
    resv1 = torch.cuda.memory_reserved()
    log.info(
        f"[ckpt-save iter {iteration}] reclaim before/after empty_cache: "
        f"reserved {resv0 / gib:.1f}->{resv1 / gib:.1f} GiB | "
        f"allocated {alloc0 / gib:.1f} GiB | "
        f"nvml_free {free0 / gib:.1f}->{free1 / gib:.1f} GiB "
        f"(+{(free1 - free0) / gib:.1f} GiB headroom for the save)",
        rank0_only=True,
    )


class DistributedCheckpointer(_DistributedCheckpointer):
    """DCP checkpointer that handles ``optimizer_dict`` / ``scheduler_dict``."""

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer_dict: dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        model_dict = model.model_dict()
        resume_keys, checkpoint_path = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0

        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            for key in resume_keys:
                load_planner = DefaultLoadPlanner(allow_partial_load=True)
                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                log.critical(f"Start loading checkpoint from {checkpoint_path}")
                torch.distributed.barrier()
                log.critical(f"starting {cur_key_ckpt_full_path}", rank0_only=False)
                if key == "model":
                    storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                    log.info("- Loading the model...")
                    _model_wrapper = ModelWrapper(model)
                    _state_dict = _model_wrapper.state_dict()
                    dcp.load(_state_dict, storage_reader=storage_reader, planner=load_planner)
                    _model_wrapper.load_state_dict(_state_dict)
                elif key == "optim":
                    if optimizer_dict is None:
                        raise ValueError("resume_keys includes 'optim' but optimizer_dict is None")
                    for k, v in optimizer_dict.items():
                        storage_reader = self.get_storage_reader(f"{cur_key_ckpt_full_path}_{k}")
                        log.info(f"- Loading the optimizer for {k}...")
                        _optim_wrapper = OptimizerWrapper(model_dict[k], v)
                        _state_dict = _optim_wrapper.state_dict()
                        dcp.load(_state_dict, storage_reader=storage_reader, planner=load_planner)
                        _optim_wrapper.load_state_dict(_state_dict)
                elif key == "scheduler":
                    if scheduler_dict is None:
                        raise ValueError("resume_keys includes 'scheduler' but scheduler_dict is None")
                    for k, v in scheduler_dict.items():
                        storage_reader = self.get_storage_reader(f"{cur_key_ckpt_full_path}_{k}")
                        log.info(f"- Loading the scheduler for {k}...")
                        _state_dict = v.state_dict()
                        dcp.load(_state_dict, storage_reader=storage_reader, planner=load_planner)
                        v.load_state_dict(_state_dict)
                elif key == "trainer":
                    if grad_scaler is None:
                        raise ValueError("resume_keys includes 'trainer' but grad_scaler is None")
                    storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                    log.info("- Loading the trainer...")
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    dcp.load(_state_dict, storage_reader=storage_reader, planner=load_planner)
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")
            if self.callbacks is not None:
                self.callbacks.on_load_checkpoint(model, state_dict=_state_dict)
            log.critical(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info(
                "Training from scratch (no DCP checkpoint to resume). "
                "If you loaded a consolidated .pt via load_path, the model was already loaded earlier; "
                "only optimizer/scheduler/iteration start from zero."
            )
        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration

    def save_state_dict_worker(self, to_save_dict: dict[str, tuple[Any, str]], checkpoint_file: str) -> None:
        for k, (v, full_checkpoint_path) in to_save_dict.items():
            if k in ("optim", "scheduler"):
                for key_net, state_dict in v.items():
                    storage_writer = self.get_storage_writer(f"{full_checkpoint_path}_{key_net}")
                    dcp.save(
                        state_dict,
                        storage_writer=storage_writer,
                        planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
                    )
            else:
                storage_writer = self.get_storage_writer(full_checkpoint_path)
                dcp.save(
                    v,
                    storage_writer=storage_writer,
                    planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
                )

        if distributed.is_rank0():
            self._write_latest_checkpoint_file(checkpoint_file)
        log.critical(f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}", rank0_only=True)

    def save(
        self,
        model: ImaginaireModel,
        optimizer_dict: dict[str, torch.optim.Optimizer],
        scheduler_dict: dict[str, torch.optim.lr_scheduler.LRScheduler],
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.get_previous_checkpoint_results(wait_for=0)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_start(model, iteration)

        # Reclaim the training step's idle reserved cache so the sharded DCP
        # save's out-of-allocator transients have NVML-free headroom; see the
        # helper's docstring for the full rationale.
        _reclaim_cuda_cache_before_save(iteration)

        model_dict = model.model_dict()
        checkpoint_file = f"iter_{iteration:09}"
        to_save_dict: dict[str, Any] = {
            "model": ModelWrapper(model).state_dict(),
            "optim": {k: OptimizerWrapper(model_dict[k], v).state_dict() for k, v in optimizer_dict.items()},
            "scheduler": {k: v.state_dict() for k, v in scheduler_dict.items()},
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
            },
        }
        for k in to_save_dict.keys():
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{k}")
            to_save_dict[k] = (to_save_dict[k], output_dirname)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint(model, state_dict=to_save_dict)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            start_time = time.monotonic()
            try:
                self.save_state_dict_worker(to_save_dict, checkpoint_file)
            finally:
                if self.callbacks is not None:
                    self.callbacks.on_save_checkpoint_success(
                        iteration=iteration, elapsed_time=time.monotonic() - start_time
                    )

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)
