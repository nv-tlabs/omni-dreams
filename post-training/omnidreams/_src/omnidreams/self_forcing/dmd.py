# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import collections
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import attrs
import torch
import torch.nn as nn
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.modules.module import _IncompatibleKeys

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.imaginaire.lazy_config import LazyDict
from omnidreams._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from omnidreams._src.imaginaire.model import ImaginaireModel
from omnidreams._src.imaginaire.utils import log, misc
from omnidreams._src.imaginaire.utils.checkpoint_db import download_checkpoint
from omnidreams._src.imaginaire.utils.checkpointer import non_strict_load_model
from omnidreams._src.imaginaire.utils.context_parallel import broadcast, broadcast_split_tensor
from omnidreams._src.imaginaire.utils.count_params import count_params
from omnidreams._src.imaginaire.utils.ema import FastEmaModelUpdater
from omnidreams._src.imaginaire.utils.fsdp_helper import hsdp_device_mesh
from omnidreams._src.imaginaire.utils.optim_instantiate import get_base_scheduler
from omnidreams._src.predict2.models.text2world_model import EMAConfig
from omnidreams._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig

from omnidreams._src.omnidreams.self_forcing.utils import (
    build_net,
    load_consolidated_pt_to_net,
    load_internal_dcp_checkpoint_to_net,
)


def _load_net_init_ckpt(net, ckpt_path: str, *, role: str, credential_path: str) -> None:
    """Dispatch a net-init checkpoint to the right loader based on its (resolved)
    extension.

    The incoming ``ckpt_path`` may be a registered URI (``s3://bucket/...``,
    UUID, or ``hf://...``); we resolve through the registry first so:
      - non-INTERNAL: registered URIs become local HF cache paths
      - INTERNAL:    URIs pass through; the DCP loader handles s3 directly
      - plain local paths pass through unchanged

    Mirrors the resolution that ``causal_multiview/utils/model_loader.py`` does
    for the standard (non-DMD) model code path
    (``load_model_state_dict_from_checkpoint`` calls ``get_checkpoint_path``,
    an alias of ``download_checkpoint``).

    After resolution:
      - ``.pt`` ending → consolidated state dict (e.g. output of
        ``convert_distcp_to_pt.py``) → ``load_consolidated_pt_to_net``.
        For the legacy public self-forcing ``.pt`` format (top-level
        ``generator`` / ``generator_ema`` keys) call
        ``load_self_forcing_public_ckpt_to_net`` directly instead.
      - otherwise → DCP directory (must end with ``/model``) →
        ``load_internal_dcp_checkpoint_to_net``.
    """
    ckpt_path = download_checkpoint(ckpt_path)
    if ckpt_path.endswith(".pt"):
        load_consolidated_pt_to_net(net, ckpt_path, message=f"load {role} from .pt")
    else:
        load_internal_dcp_checkpoint_to_net(net, ckpt_path, credential_path=credential_path)


from omnidreams._src.predict2.configs.common.defaults.optimizer import AdamWConfig
from omnidreams._src.predict2.utils.dtensor_helper import DTensorFastEmaModelUpdater
from omnidreams._src.omnidreams.utils.torch_future import clip_grad_norm_

IS_PREPROCESSED_KEY = "is_preprocessed"
_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


@attrs.define(slots=False)
class ImaginaireDMDBaseModelConfig:
    s3_credential_path: str = "credentials/s3_inference.secret"

    net: LazyDict | None = None
    net_ckpt: str = ""

    net_real_score: LazyDict | None = None
    net_real_score_ckpt: str = ""

    net_fake_score: LazyDict | None = None
    net_fake_score_ckpt: str = ""
    optimizer_fake_score_config: LazyDict = AdamWConfig

    ema: EMAConfig = EMAConfig()
    ema_weight: float = 0.99
    ema_start_step: int = 200

    fsdp_shard_size: int = 1
    precision: str = "bfloat16"
    use_torch_compile: bool = False
    input_data_key: str = "video"  # key to fetch input data from data_batch
    input_image_key: str = "images"  # key to fetch input image from data_batch

    dfake_gen_update_ratio: int = 5
    dfake_warm_up_steps: int = -1  # -1 means no warm up

    # NOTE (ruilongl):
    # 1. keep original net dtype (fp32) and FSDP gradiant reduction with fp32 is essential to the performance.
    # 2. original self-forcing public code uses FSDP1 to wrap the model, which is equivalent
    #    to `cast_forward_inputs=True` on the root module, but cast_forward_inputs=False on the internal layers.
    keep_original_net_dtype: bool = True
    mixed_precision_policy_internal_layers: LazyDict = L(torch.distributed.fsdp.MixedPrecisionPolicy)(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    mixed_precision_policy_root_module: LazyDict = L(torch.distributed.fsdp.MixedPrecisionPolicy)(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )

    tokenizer: LazyDict | None = None
    conditioner: LazyDict | None = None

    text_encoder_class: str = "reason1p1_7B"
    text_encoder_config: TextEncoderConfig = TextEncoderConfig(
        embedding_concat_strategy="full_concat",
        compute_online=True,
        ckpt_path="s3://bucket/cosmos_reasoning1/pretrained/Qwen_tokenizer/Qwen/Qwen2.5-VL-7B-Instruct",
    )
    input_caption_key: str = "ai_caption"  # Key used to fetch input captions
    split_cp_in_model: bool = False


class ImaginaireDMDBaseModel(ImaginaireModel):
    def __init__(self, config: ImaginaireDMDBaseModelConfig):
        super().__init__()
        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.device = "cuda"
        self.dtype = self.precision
        self.tensor_kwargs = {"device": self.device, "dtype": self.dtype}

        log.warning(f"DiffusionModel: precision {self.precision}")

        # 1. set data keys and data information
        self.setup_data_key()

        with misc.timer("DiffusionModel: set_up_tokenizer"):
            self.tokenizer = lazy_instantiate(config.tokenizer)

        # 5. create fsdp mesh if needed
        if config.fsdp_shard_size > 1:
            self.fsdp_device_mesh = hsdp_device_mesh(
                sharding_group_size=config.fsdp_shard_size,
            )
            self.mixed_precision_policy_internal_layers = lazy_instantiate(
                config.mixed_precision_policy_internal_layers
            )
            self.mixed_precision_policy_root_module = lazy_instantiate(config.mixed_precision_policy_root_module)
        else:
            self.fsdp_device_mesh = None
            self.mixed_precision_policy_internal_layers = None
            self.mixed_precision_policy_root_module = None

        self.set_up_model()

        self.text_encoder = None
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            self.text_encoder = TextEncoder(self.config.text_encoder_config)
        self.input_caption_key = self.config.input_caption_key

        # 7. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent)

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self):
        config = self.config
        with misc.timer("Creating PyTorch model and ema if enabled"):
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, (
                "conditioner should not have learnable parameters"
            )

            self.net = build_net(
                config.net,
                self.fsdp_device_mesh,
                self.mixed_precision_policy_root_module,
                self.mixed_precision_policy_internal_layers,
            )
            if config.net_ckpt != "":
                _load_net_init_ckpt(
                    self.net,
                    config.net_ckpt,
                    role="net",
                    credential_path=self.config.s3_credential_path,
                )

            if config.net_real_score is not None:
                self.net_real_score = build_net(
                    config.net_real_score,
                    self.fsdp_device_mesh,
                    self.mixed_precision_policy_root_module,
                    self.mixed_precision_policy_internal_layers,
                )
                _load_net_init_ckpt(
                    self.net_real_score,
                    config.net_real_score_ckpt,
                    role="net_real_score",
                    credential_path=self.config.s3_credential_path,
                )
                self.net_real_score.requires_grad_(False)
            else:
                self.net_real_score = None

            if config.net_fake_score is not None:
                self.net_fake_score = build_net(
                    config.net_fake_score,
                    self.fsdp_device_mesh,
                    self.mixed_precision_policy_root_module,
                    self.mixed_precision_policy_internal_layers,
                )
                _load_net_init_ckpt(
                    self.net_fake_score,
                    config.net_fake_score_ckpt,
                    role="net_fake_score",
                    credential_path=self.config.s3_credential_path,
                )
            else:
                self.net_fake_score = None

            self._param_count = count_params(self.net, verbose=False)

            if config.ema.enabled:
                self.net_ema = build_net(
                    config.net,
                    self.fsdp_device_mesh,
                    self.mixed_precision_policy_root_module,
                    self.mixed_precision_policy_internal_layers,
                )
                self.net_ema.requires_grad_(False)

                if self.fsdp_device_mesh:
                    self.net_ema_worker = DTensorFastEmaModelUpdater()
                else:
                    self.net_ema_worker = FastEmaModelUpdater()

                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)
        torch.cuda.empty_cache()

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        """We hanlde two types of data_batch. One comes from a joint_dataloader where "dataset_name" can be used to differenciate image_batch and video_batch.
        Another comes from a dataloader which we by default assumes as video_data for video model training.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert is_image != is_video, (
            "Only one of the input_image_key or input_data_key should be present in the data_batch."
        )
        return is_image

    def init_optimizer_scheduler(self, optimizer_config: LazyDict, scheduler_config: LazyDict) -> None:
        """Creates the optimizer and scheduler for the model.

        Args:
            config_model (ModelConfig): The config object for the model.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """
        optimizer = lazy_instantiate(optimizer_config, model=self.net)
        self.optimizer_dict = {"net": optimizer}

        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        self.scheduler_dict = {"net": scheduler}

        if self.net_fake_score is not None:
            optimizer_fake_score = lazy_instantiate(self.config.optimizer_fake_score_config, model=self.net_fake_score)
            scheduler_fake_score = get_base_scheduler(optimizer_fake_score, self, scheduler_config)
            self.optimizer_dict["fake_score"] = optimizer_fake_score
            self.scheduler_dict["fake_score"] = scheduler_fake_score

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)

        kwargs = {"device": self.device} if self.config.keep_original_net_dtype else self.tensor_kwargs
        self.net = self.net.to(memory_format=memory_format, **kwargs)
        if self.net_real_score is not None:
            self.net_real_score = self.net_real_score.to(memory_format=memory_format, **kwargs)
        if self.net_fake_score is not None:
            self.net_fake_score = self.net_fake_score.to(memory_format=memory_format, **kwargs)

        if hasattr(self.config, "use_torch_compile") and self.config.use_torch_compile:  # compatible with old config
            if torch.__version__ < "2.3":
                log.warning(
                    "torch.compile in Pytorch version older than 2.3 doesn't work well with activation checkpointing.\n"
                    "It's very likely there will be no significant speedup from torch.compile.\n"
                    "Please use at least 24.04 Pytorch container, or imaginaire4:v7 container."
                )
            # Increasing cache size. It's required because of the model size and dynamic input shapes resulting in
            # multiple different triton kernels. For 28 TransformerBlocks, the cache limit of 256 should be enough for
            # up to 9 different input shapes, as 28*9 < 256. If you have more Blocks or input shapes, and you observe
            # graph breaks at each Block (detectable with torch._dynamo.explain) or warnings about
            # exceeding cache limit, you may want to increase this size.
            # Starting with 24.05 Pytorch container, the default value is 256 anyway.
            # You can read more about it in the comments in Pytorch source code under path torch/_dynamo/cache_size.py.
            torch._dynamo.config.accumulated_cache_size_limit = 256
            # dynamic=False means that a separate kernel is created for each shape. It incurs higher compilation costs
            # at initial iterations, but can result in more specialized and efficient kernels.
            # dynamic=True currently throws errors in pytorch 2.3.
            self.net = torch.compile(self.net, dynamic=False, disable=not self.config.use_torch_compile)
            if self.net_real_score is not None:
                self.net_real_score = torch.compile(
                    self.net_real_score, dynamic=False, disable=not self.config.use_torch_compile
                )
            if self.net_fake_score is not None:
                self.net_fake_score = torch.compile(
                    self.net_fake_score, dynamic=False, disable=not self.config.use_torch_compile
                )

    def is_student_phase(self, iteration: int):
        return (
            self.config.dfake_warm_up_steps == -1 or iteration > self.config.dfake_warm_up_steps
        ) and iteration % self.config.dfake_gen_update_ratio == 0

    @staticmethod
    def get_context_parallel_group():
        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """
        update the net_ema
        """
        if not self.is_student_phase(iteration):
            return


        if self.config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(iteration)
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights
        """
        if iteration < self.config.ema_start_step:
            return 0.0
        return self.config.ema_weight

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """
        Get the optimizers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if self.is_student_phase(iteration):
            return [self.optimizer_dict["net"]]
        else:
            return [self.optimizer_dict["fake_score"]]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """
        Get the lr schedulers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if self.is_student_phase(iteration):
            return [self.scheduler_dict["net"]]
        else:
            return [self.scheduler_dict["fake_score"]]

    def optimizers_schedulers_step(self, grad_scaler: torch.cuda.amp.GradScaler, iteration: int) -> None:
        """
        Step the optimizer and scheduler step based on the iteration,
        and gradient scaler is also updated
        """
        for optimizer in self.get_optimizers(iteration):
            optimizer.step()

        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    def optimizers_zero_grad(self, iteration: int) -> None:
        """
        Zero the gradients of the optimizers based on the iteration
        """
        for optimizer in self.get_optimizers(iteration):
            optimizer.zero_grad()

    def model_dict(self) -> Dict[str, nn.Module]:
        model_dict = {"net": self.net}
        if self.net_fake_score:
            model_dict["fake_score"] = self.net_fake_score
        return model_dict

    def state_dict(self) -> Dict[str, Any]:
        net_state_dict = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)
        if self.net_fake_score:
            fake_score_state_dict = self.net_fake_score.state_dict(prefix="net_fake_score.")
            net_state_dict.update(fake_score_state_dict)
        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        _fake_score_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v
            elif k.startswith("net_fake_score."):
                _fake_score_state_dict[k.replace("net_fake_score.", "")] = v
            else:
                raise ValueError(f"Invalid key: {k}")

        if strict:
            reg_results: _IncompatibleKeys = self.net.load_state_dict(_reg_state_dict, strict=strict, assign=assign)
            missing_keys = reg_results.missing_keys
            unexpected_keys = reg_results.unexpected_keys
            if self.config.ema.enabled:
                ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(
                    _ema_state_dict, strict=strict, assign=assign
                )
                missing_keys += ema_results.missing_keys
                unexpected_keys += ema_results.unexpected_keys
            if self.net_fake_score:
                fake_score_results: _IncompatibleKeys = self.net_fake_score.load_state_dict(
                    _fake_score_state_dict, strict=strict, assign=assign
                )
                missing_keys += fake_score_results.missing_keys
                unexpected_keys += fake_score_results.unexpected_keys
            return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=unexpected_keys)

        else:
            log.critical("load model in non-strict mode")
            log.critical(str(non_strict_load_model(self.net, _reg_state_dict)), rank0_only=False)
            if self.config.ema.enabled:
                log.critical("load ema model in non-strict mode")
                log.critical(str(non_strict_load_model(self.net_ema, _ema_state_dict)), rank0_only=False)
            if self.net_fake_score:
                log.critical("load fake score model in non-strict mode")
                log.critical(str(non_strict_load_model(self.net_fake_score, _fake_score_state_dict)), rank0_only=False)

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ):
        if self.net_fake_score:
            for param in self.net_fake_score.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)
            clip_grad_norm_(
                self.net_fake_score.parameters(),
                max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            )
        return clip_grad_norm_(
            self.net.parameters(),
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        ).to(next(self.net.parameters()).device)  # cast potentially tensor(0) to same device as parameters

    def inplace_compute_text_embeddings_online(
        self,
        data_batch: dict[str, torch.Tensor],
        use_negative_prompt: bool = True,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
    ) -> None:
        """Compute text embeddings online using the model's text encoder.

        This method computes text embeddings using cosmos_reason instead of umt5.

        Args:
            data_batch: Dictionary containing input data with captions.
            use_negative_prompt: Whether to compute negative prompt embeddings.
            negative_prompt: The negative prompt text for classifier-free guidance.
        """
        if (
            self.config.text_encoder_config is not None
            and self.config.text_encoder_config.compute_online
            and self.text_encoder is not None
        ):
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

            # Compute negative prompt embeddings for classifier-free guidance
            if use_negative_prompt:
                batch_size = text_embeddings.shape[0]
                neg_data_batch = {self.input_caption_key: [negative_prompt] * batch_size, "images": None}
                neg_text_embeddings = self.text_encoder.compute_text_embeddings_online(
                    neg_data_batch, self.input_caption_key
                )
                data_batch["neg_t5_text_embeddings"] = neg_text_embeddings

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the diffusion model.

        This method is responsible for executing one iteration of the model's training. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        raise NotImplementedError("DMDModel: training_step is not implemented")

    def denoise(
        self,
        scheduler,
        net_choice: Literal["generator", "real_score", "fake_score"],
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        start_frame_for_rope: Optional[int] = None,
        block_mask: Optional[BlockMask] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if net_choice == "generator":
            model = self.net
            uniform_timestep = False
        elif net_choice == "real_score":
            model = self.net_real_score
            uniform_timestep = True
        elif net_choice == "fake_score":
            model = self.net_fake_score
            uniform_timestep = True
        else:
            raise ValueError(f"Invalid net choice: {net_choice}")

        if uniform_timestep:
            # [B, F] -> [B, 1]
            # NOTE(ruilong): the public code uses [:, 0] but internal model needs [:, :1]
            input_timestep = timestep[:, :1]
        else:
            input_timestep = timestep

        xt_B_C_T_H_W = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        timesteps_B_T = input_timestep
        ############### copied from joint_causal_cosmos_model.py ##############
        condition_video_mask = None

        if True:  # conditional_dict['is_video']
            condition_state_in_B_C_T_H_W = conditional_dict["gt_frames"].type_as(xt_B_C_T_H_W)
            assert conditional_dict["use_video_condition"], "use_video_condition should be True for distillation"
            if not conditional_dict["use_video_condition"]:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = (
                conditional_dict["condition_video_input_mask_B_C_T_H_W"].repeat(1, C, 1, 1, 1).type_as(xt_B_C_T_H_W)
            )

            # Make the first few frames of x_t be the ground truth frames
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )

                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )

                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = (
                    timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T
                )  # add dimension for batch

        # X0 prediction
        if kv_cache is not None:
            flow_pred = model(
                xt_B_C_T_H_W.to(**self.tensor_kwargs),
                timesteps_B_T.to(**self.tensor_kwargs),
                kv_cache=kv_cache,
                crossattn_cache=None,
                current_start=current_start,
                current_end=current_end,
                start_frame_for_rope=start_frame_for_rope,
                block_mask=block_mask,
                **conditional_dict,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = model(
                xt_B_C_T_H_W.to(**self.tensor_kwargs),
                timesteps_B_T.to(**self.tensor_kwargs),
                block_mask=block_mask,
                **conditional_dict,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            scheduler=scheduler,
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])


        if self.config.denoise_replace_gt_frames:
            gt_frames_x0 = conditional_dict["gt_frames"].type_as(pred_x0)
            pred_x0 = (
                gt_frames_x0 * condition_video_mask + pred_x0.permute(0, 2, 1, 3, 4) * (1 - condition_video_mask)
            ).permute(0, 2, 1, 3, 4)

        return flow_pred, pred_x0

    def _convert_flow_pred_to_x0(
        self, scheduler, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def _convert_x0_to_flow_pred(
        self, scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]):
        raise NotImplementedError("DMDModel: get_data_and_condition is not implemented")

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            _flag = data_batch.get(IS_PREPROCESSED_KEY, False)
            if isinstance(_flag, torch.Tensor):
                try:
                    _flag = bool(_flag.bool().all().item())
                except Exception:
                    _flag = False
            else:
                _flag = bool(_flag)

            if _flag:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all((data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)), (
                    f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
                )
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _augment_image_dim_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            _flag = data_batch.get(IS_PREPROCESSED_KEY, False)
            if isinstance(_flag, torch.Tensor):
                try:
                    _flag = bool(_flag.bool().all().item())
                except Exception:
                    _flag = False
            else:
                _flag = bool(_flag)

            if _flag:
                assert data_batch[input_key].shape[2] == 1, (
                    f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                )
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    def broadcast_split_for_model_parallelsim(self, x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T, split=False):
        """
        Broadcast and split the input data and condition for model parallelism.
        Currently, we only support context parallelism.
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            x0_B_C_T_H_W = broadcast_split_tensor(x0_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            epsilon_B_C_T_H_W = broadcast_split_tensor(epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] == 1:  # single sigma is shared across all frames
                    sigma_B_T = broadcast(sigma_B_T, cp_group)
                else:  # different sigma for each frame
                    sigma_B_T = broadcast_split_tensor(sigma_B_T, seq_dim=1, process_group=cp_group)
            if condition is not None:
                condition = condition.broadcast(cp_group, split=split)
            self.net.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()

        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T


