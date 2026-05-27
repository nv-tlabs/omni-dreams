# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from typing import Dict, List, Literal, Optional, Tuple

import attrs
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state
from torch.nn.attention.flex_attention import BlockMask

from omnidreams._src.imaginaire.utils import misc
from omnidreams._src.imaginaire.utils.context_parallel import (
    broadcast,
    broadcast_split_tensor,
    cat_outputs_cp_with_grad,
)
from omnidreams._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from omnidreams._src.predict2.conditioner import DataType
from omnidreams._src.omnidreams.self_forcing.dmd import ImaginaireDMDBaseModel, ImaginaireDMDBaseModelConfig
from omnidreams._src.omnidreams.third_party.self_forcing.loss import FlowPredLoss
from omnidreams._src.omnidreams.third_party.self_forcing.pipeline import SelfForcingTrainingPipeline
from omnidreams._src.omnidreams.third_party.self_forcing.scheduler import FlowMatchScheduler


def print_rank0(*msgs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*msgs)
    else:
        print(*msgs)


@attrs.define(slots=False)
class BaseModelConfig(ImaginaireDMDBaseModelConfig):
    denoising_step_list: List[int] = [1000, 750, 500, 250]
    warp_denoising_step: bool = True
    num_train_timestep: int = 1000


class BaseModel(ImaginaireDMDBaseModel):
    config: BaseModelConfig

    def __init__(self, config: BaseModelConfig):
        super().__init__(config)
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(config.num_train_timestep, training=True)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

        self.denoising_step_list = torch.LongTensor(config.denoising_step_list)

        if self.config.warp_denoising_step:
            # NOTE (ruilongl): This will slightly change the denoising step list,
            # from long type to float type.
            timesteps = torch.cat(
                (
                    self.scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            self.denoising_step_list = timesteps[self.config.num_train_timestep - self.denoising_step_list]

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long,
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long,
            )
            # make the noise level the same within every block
            timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


@attrs.define(slots=False)
class SelfForcingModelConfig(BaseModelConfig):
    num_training_frames: int = 21  # Not enabled at the moment, num_training_frames is set from data_batch
    num_gradient_enabled_frames: int = 21  # Not enabled at the moment
    num_frame_per_block: int = 3

    same_step_across_blocks: bool = True
    last_step_only: bool = False

    model_type: str = "t2v"
    i2v_zero_latent_condition: bool = False

    # KV cache noise level
    context_noise: float = 0.0
    add_context_noise_in_training: bool = False
    # Varying steps across generation
    independent_denoising_step_list: bool = False
    shrink_list: list[int] = [3]

    # cosmos I2V configs
    min_num_conditional_frames: int = 0  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 1  # Maximum number of latent conditional frames, set to 0 for t2v
    conditional_frame_timestep: float = (
        -1.0
    )  # Noise level used for conditional frames; default is -1 which will not take effective
    conditioning_strategy: str = "frame_replace"  # What strategy to use for conditioning
    denoise_replace_gt_frames: bool = False  # Whether to denoise the ground truth frames
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames


class SelfForcingModel(BaseModel):
    config: SelfForcingModelConfig

    def __init__(self, config: SelfForcingModelConfig):
        super().__init__(config)
        self.denoising_loss_func = FlowPredLoss()
        if hasattr(self.net, "num_frame_per_block"):
            self.net.num_frame_per_block = config.num_frame_per_block
        if hasattr(self.net_real_score, "num_frame_per_block"):
            self.net_real_score.num_frame_per_block = config.net_real_score.state_t
        if hasattr(self.net_fake_score, "num_frame_per_block"):
            self.net_fake_score.num_frame_per_block = config.net_fake_score.state_t

        if self.config.add_context_noise_in_training:
            self.denoising_step_list = torch.cat(
                [self.denoising_step_list, torch.tensor([self.config.context_noise]).to(self.denoising_step_list)]
            )

        if self.config.independent_denoising_step_list:
            self.denoising_step_list = [self.denoising_step_list] * (
                self.config.num_training_frames // config.num_frame_per_block
            )

            for shrink_list in config.shrink_list:
                for i in range(shrink_list, len(self.denoising_step_list)):
                    if self.denoising_step_list[i].shape[0] > 2:
                        self.denoising_step_list[i] = torch.cat(
                            [self.denoising_step_list[i][:-2], self.denoising_step_list[i][-1:]]
                        )
                    else:
                        self.denoising_step_list[i] = self.denoising_step_list[i][:-1]
            print_rank0("self.denoising_step_list: ", self.denoising_step_list)

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int], Optional[int]]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [self.config.num_gradient_enabled_frames, self.config.num_training_frames],
        # but still being a multiple of self.config.num_frame_per_block.
        # Note: for now the sampling is disabled.
        min_num_frames = noise_shape[1]
        max_num_frames = noise_shape[1]

        assert max_num_frames % self.config.num_frame_per_block == 0
        assert min_num_frames % self.config.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.config.num_frame_per_block
        min_num_blocks = min_num_frames // self.config.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        if dist.is_initialized():
            dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.config.num_frame_per_block

        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self.generate_samples_from_batch(
            is_training=True,
            noise_B_T_C_H_W=torch.randn(
                noise_shape, device=self.tensor_kwargs["device"], dtype=self.tensor_kwargs["dtype"]
            ),
            image_or_video_shape=noise_shape,
            conditional_dict=conditional_dict,
        )

        # if pred_image_or_video.shape[1] > self.config.num_gradient_enabled_frames:
        #     # Slice last 21 frames
        #     raise NotImplementedError("Not implemented")
        # else:
        pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            gradient_mask[:, : self.config.num_frame_per_block] = False
        else:
            gradient_mask = None

        # print_rank0("pred_image_or_video: ", pred_image_or_video.shape, pred_image_or_video.sum())
        # print_rank0("gradient_mask", gradient_mask)
        # print_rank0("denoised_timestep_from: ", denoised_timestep_from)
        # print_rank0("denoised_timestep_to: ", denoised_timestep_to)

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return (
            pred_image_or_video_last_21,
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
        )

    def _consistency_backward_simulation(self, noise: torch.Tensor, **conditional_dict: dict) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(noise=noise, **conditional_dict)

    def generator(self, *args, **kwargs):
        return self.denoise(net_choice="generator", scheduler=self.scheduler, *args, **kwargs)

    def real_score(self, *args, **kwargs):
        return self.denoise(net_choice="real_score", scheduler=self.scheduler, *args, **kwargs)

    def fake_score(self, *args, **kwargs):
        return self.denoise(net_choice="fake_score", scheduler=self.scheduler, *args, **kwargs)


@attrs.define(slots=False)
class DMDSelfForcingModelConfig(SelfForcingModelConfig):
    real_guidance_scale: float = 3.0
    fake_guidance_scale: float = 0.0
    timestep_shift: float = 5.0
    ts_schedule: bool = False
    ts_schedule_max: bool = False
    min_score_timestep: int = 0
    image_or_video_shape: list[int] = [16, 21, 60, 104]  # [C, T, H, W] used when the dataset is text only
    state_ch: int = 16
    state_t: int = 24

    # Default noise_scheme set to diffusion_forcing when model is pre-trained with Diffusion Forcing.
    # I2V model under teacher_forcing has variants during inference.
    noise_scheme: str = "diffusion_forcing"

    # Using VidProm dataset to override the text embeddings [for debugging]
    use_vidprom_dataset: bool = False


NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class DMDSelfForcingModel(SelfForcingModel):
    config: DMDSelfForcingModelConfig

    def __init__(self, config: DMDSelfForcingModelConfig):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(config)

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: Optional[SelfForcingTrainingPipeline] = None

        # Step 2: Initialize all dmd hyperparameters
        self.min_step = int(0.02 * self.config.num_train_timestep)
        self.max_step = int(0.98 * self.config.num_train_timestep)

        self.frame_seq_length = None  # dynamic updated in self.generate_samples_from_batch

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

        n_views = noisy_image_or_video.shape[1] // self.config.state_t

        if uniform_timestep:
            if n_views == 1:
                # [B, F] -> [B, 1]
                # NOTE(ruilong): the public code uses [:, 0] but internal model needs [:, :1]
                input_timestep = timestep[:, :1]
            else:
                # NOTE(wjay): multiview needs [B, V * T]
                input_timestep = timestep
        else:
            input_timestep = timestep

        xt_B_C_T_H_W = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        timesteps_B_T = input_timestep
        ############### copied from joint_causal_cosmos_model.py ##############
        condition_video_mask = None

        if True:  # conditional_dict['is_video']
            condition_state_in_B_C_T_H_W = conditional_dict["gt_frames"].type_as(xt_B_C_T_H_W)
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

        # Enable CP for the chosen model
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1 and hasattr(model, "enable_context_parallel"):
            model.enable_context_parallel(cp_group)
        if cp_size == 1 and hasattr(model, "disable_context_parallel"):
            model.disable_context_parallel()

        if net_choice in ["real_score", "fake_score"]:
            # NOTE(wjay):
            # For teacher models handling a single view (n_views == 1), context parallelism (CP) is applied at the model level.
            # For multi-view teacher models (n_views > 1), CP is managed internally within the network itself.
            if cp_size > 1 and n_views == 1:
                input_xt_B_C_T_H_W = broadcast_split_tensor(xt_B_C_T_H_W, seq_dim=2, process_group=cp_group)
                if timesteps_B_T.shape[1] > 1:
                    input_timesteps_B_T = broadcast_split_tensor(timesteps_B_T, seq_dim=1, process_group=cp_group)
                else:
                    input_timesteps_B_T = timesteps_B_T

                # Broadcast/split conditions based on whether they have temporal dimension
                # Pull these from conditional_dict (all may be None)
                gt_frames = conditional_dict.get("gt_frames")
                cond_mask = conditional_dict.get("condition_video_input_mask_B_C_T_H_W")
                view_indices = conditional_dict.get("view_indices_B_T")
                control_input_hdmap_bbox = conditional_dict.get("control_input_hdmap_bbox")
                state_t = self.config.state_t

                # Start with broadcasting non-special keys
                input_conditional_dict = {}
                for k, v in conditional_dict.items():
                    if k in {"gt_frames", "condition_video_input_mask_B_C_T_H_W", "view_indices_B_T"}:
                        continue
                    if v is None:
                        input_conditional_dict[k] = None
                    elif not isinstance(v, torch.Tensor):
                        input_conditional_dict[k] = v
                    else:
                        input_conditional_dict[k] = broadcast(v, cp_group)

                # Now handle the special three keys
                if gt_frames is not None and cond_mask is not None and view_indices is not None:
                    _, _, T, _, _ = gt_frames.shape
                    assert T % state_t == 0, f"T must be a multiple of state_t. Got T={T} and state_t={state_t}."
                    if T > 1 and cp_group.size() > 1:
                        n_views = T // state_t
                        gt_frames = rearrange(gt_frames, "B C (V T) H W -> B C V T H W", V=n_views)
                        cond_mask = rearrange(cond_mask, "B C (V T) H W -> B C V T H W", V=n_views)
                        view_indices = rearrange(view_indices, "B (V T) -> B V T", V=n_views)

                        gt_frames = broadcast_split_tensor(gt_frames, seq_dim=3, process_group=cp_group)
                        cond_mask = broadcast_split_tensor(cond_mask, seq_dim=3, process_group=cp_group)
                        view_indices = broadcast_split_tensor(view_indices, seq_dim=2, process_group=cp_group)

                        gt_frames = rearrange(gt_frames, "B C V T H W -> B C (V T) H W", V=n_views)
                        cond_mask = rearrange(cond_mask, "B C V T H W -> B C (V T) H W", V=n_views)
                        view_indices = rearrange(view_indices, "B V T -> B (V T)", V=n_views)
                        if control_input_hdmap_bbox is not None:
                            control_input_hdmap_bbox_B_C_V_T_H_W = rearrange(
                                control_input_hdmap_bbox, "B C (V T) H W -> B C V T H W", V=n_views
                            )
                            control_input_hdmap_bbox_B_C_V_T_H_W = broadcast_split_tensor(
                                control_input_hdmap_bbox_B_C_V_T_H_W, seq_dim=3, process_group=cp_group
                            )
                            control_input_hdmap_bbox = rearrange(
                                control_input_hdmap_bbox_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_views
                            )
                    else:
                        gt_frames = broadcast(gt_frames, cp_group)
                        cond_mask = broadcast(cond_mask, cp_group)
                        view_indices = broadcast(view_indices, cp_group)
                        if control_input_hdmap_bbox is not None:
                            control_input_hdmap_bbox = broadcast(control_input_hdmap_bbox, cp_group)

                input_conditional_dict["gt_frames"] = gt_frames
                input_conditional_dict["condition_video_input_mask_B_C_T_H_W"] = cond_mask
                input_conditional_dict["view_indices_B_T"] = view_indices
                input_conditional_dict["control_input_hdmap_bbox"] = control_input_hdmap_bbox
            else:
                input_xt_B_C_T_H_W, input_conditional_dict, input_timesteps_B_T = (
                    xt_B_C_T_H_W,
                    conditional_dict,
                    timesteps_B_T,
                )

            flow_pred = model(
                input_xt_B_C_T_H_W.to(**self.tensor_kwargs),
                input_timesteps_B_T.to(**self.tensor_kwargs),
                block_mask=block_mask,
                **input_conditional_dict,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)

            # Gather outputs from all CP ranks
            if cp_size > 1 and n_views == 1:
                flow_pred = cat_outputs_cp_with_grad(flow_pred.contiguous(), seq_dim=1, cp_group=cp_group)

        else:
            assert net_choice == "generator"
            assert kv_cache is not None
            flow_pred = model(
                xt_B_C_T_H_W.to(**self.tensor_kwargs),
                timesteps_B_T.to(**self.tensor_kwargs),
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                start_frame_for_rope=start_frame_for_rope,
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

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        n_views = noisy_image_or_video.shape[1] // self.config.state_t

        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            n_views=n_views,
        )

        if self.config.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                n_views=n_views,
            )
            pred_fake_image = (
                pred_fake_image_cond + (pred_fake_image_cond - pred_fake_image_uncond) * self.config.fake_guidance_scale
            )
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            n_views=n_views,
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            n_views=n_views,
        )

        pred_real_image = (
            pred_real_image_cond + (pred_real_image_cond - pred_real_image_uncond) * self.config.real_guidance_scale
        )

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = pred_fake_image - pred_real_image


        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = estimated_clean_image_or_video - pred_real_image
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            # "timestep": timestep.detach(),
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: Optional[int] = None,
        denoised_timestep_to: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = (
                denoised_timestep_to
                if self.config.ts_schedule and denoised_timestep_to is not None
                else self.config.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.config.ts_schedule_max and denoised_timestep_from is not None
                else self.config.num_train_timestep
            )
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.config.num_frame_per_block,
                uniform_timestep=True,
            )
            if self.config.timestep_shift > 1:
                # Note: potential change to `timestep = self.scheduler.timesteps[timestep]`
                timestep = (
                    self.config.timestep_shift
                    * (timestep / self.config.num_train_timestep)
                    / (1 + (self.config.timestep_shift - 1) * (timestep / self.config.num_train_timestep))
                    * self.config.num_train_timestep
                )
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = (
                self.scheduler.add_noise(
                    image_or_video.flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep.flatten(0, 1),
                )
                .detach()
                .unflatten(0, (batch_size, num_frame))
            )

            # Broadcast when CP > 1
            cp_group = self.get_context_parallel_group()
            cp_size = 1 if cp_group is None else cp_group.size()
            if cp_size > 1:
                noisy_latent = broadcast(noisy_latent.contiguous(), cp_group)
                timestep = broadcast(timestep, cp_group)

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(),
                (original_latent.double() - grad.double()).detach(),
                reduction="mean",
            )
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        conditional_dict_score_models: dict = None,
        unconditional_dict_score_models: dict = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict
            if conditional_dict_score_models is None
            else conditional_dict_score_models,
            unconditional_dict=unconditional_dict
            if unconditional_dict_score_models is None
            else unconditional_dict_score_models,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        conditional_dict_score_models: dict = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        n_views = image_or_video_shape[1] // self.config.state_t

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
            )

        # Step 2: Compute the fake prediction
        assert (not self.config.ts_schedule) and (not self.config.ts_schedule_max), "TS schedule is not supported now!"
        min_timestep = (
            denoised_timestep_to
            if self.config.ts_schedule and denoised_timestep_to is not None
            else self.config.min_score_timestep
        )
        max_timestep = (
            denoised_timestep_from
            if self.config.ts_schedule_max and denoised_timestep_from is not None
            else self.config.num_train_timestep
        )
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.config.num_frame_per_block,
            uniform_timestep=True,
        )

        if self.config.timestep_shift > 1:
            critic_timestep = (
                self.config.timestep_shift
                * (critic_timestep / self.config.num_train_timestep)
                / (1 + (self.config.timestep_shift - 1) * (critic_timestep / self.config.num_train_timestep))
                * self.config.num_train_timestep
            )

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, image_or_video_shape[:2])

        # Broadcast when CP > 1
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            critic_noise = broadcast(critic_noise, cp_group)
            noisy_generated_image = broadcast(noisy_generated_image, cp_group)
            critic_timestep = broadcast(critic_timestep, cp_group)

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict
            if conditional_dict_score_models is None
            else conditional_dict_score_models,
            timestep=critic_timestep,
            n_views=n_views,
        )

        # Step 3: Compute the denoising loss for the fake critic
        flow_pred = self._convert_x0_to_flow_pred(
            scheduler=self.scheduler,
            x0_pred=pred_fake_image.flatten(0, 1),
            xt=noisy_generated_image.flatten(0, 1),
            timestep=critic_timestep.flatten(0, 1),
        )
        pred_fake_noise = None

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=None,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
        )

        # Step 5: Debugging Log
        # critic_log_dict = {"critic_timestep": critic_timestep.detach()}
        critic_log_dict = {}

        return denoising_loss, critic_log_dict

    def get_data_batch_with_latent_view_indices(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "latent_view_indices_B_T" in data_batch:
            return data_batch
        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
        view_indices_B_V_T = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=n_views)
        latent_view_indices_B_V_T = view_indices_B_V_T[:, :, 0 : self.config.state_t]
        latent_view_indices_B_T = rearrange(latent_view_indices_B_V_T, "B V T -> B (V T)")
        data_batch["latent_view_indices_B_T"] = latent_view_indices_B_T  # [B, V * T]
        return data_batch

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor], with_uncondition: bool = True):
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Condition
        self.inplace_compute_text_embeddings_online(
            data_batch,
            use_negative_prompt=with_uncondition,
        )

        data_batch_original = copy.deepcopy(data_batch)
        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        latent_state = self.encode(raw_state).contiguous().float()

        # Condition
        condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        condition_original, uncondition_original = self.conditioner.get_condition_with_negative_prompt(
            data_batch_original
        )
        condition_original = condition_original.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition_original = uncondition_original.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        state_t = int(
            (data_batch["num_video_frames_per_view"].cpu().item() - 1) // self.tokenizer.temporal_compression_factor + 1
        )
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = None

        condition = condition.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,  # overrides random_min_num_conditional_frames_per_view and random_max_num_conditional_frames_per_view
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition_original = condition_original.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,  # overrides random_min_num_conditional_frames_per_view and random_max_num_conditional_frames_per_view
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        uncondition = uncondition.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,  # overrides random_min_num_conditional_frames_per_view and random_max_num_conditional_frames_per_view
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        uncondition_original = uncondition_original.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,  # overrides random_min_num_conditional_frames_per_view and random_max_num_conditional_frames_per_view
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        if with_uncondition:
            return raw_state, latent_state, (condition, uncondition, condition_original, uncondition_original)
        else:
            return raw_state, latent_state, (condition, condition_original)

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        train_generator = self.is_student_phase(iteration)

        self.eval()  # prevent any randomness (e.g. dropout)
        if iteration % 20 == 0:
            torch.cuda.empty_cache()

        data_batch = self.get_data_batch_with_latent_view_indices(data_batch)

        _, x0_B_C_T_H_W, (condition, uncondition, condition_original, uncondition_original) = (
            self.get_data_and_condition(data_batch, with_uncondition=True)
        )

        # broadcast when CP size > 1, do not handle splitting
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            x0_B_C_T_H_W = broadcast(x0_B_C_T_H_W, cp_group)
            condition = condition.broadcast(cp_group, split=False)
            uncondition = uncondition.broadcast(cp_group, split=False)
            condition_original = condition_original.broadcast(cp_group, split=False)
            uncondition_original = uncondition_original.broadcast(cp_group, split=False)

        if train_generator:
            # Step 3: Store gradients for the generator (if training the generator)
            generator_loss, generator_log_dict = self.generator_loss(
                image_or_video_shape=list(x0_B_C_T_H_W.permute(0, 2, 1, 3, 4).shape),
                conditional_dict=condition.to_dict(),
                unconditional_dict=uncondition.to_dict(),
                conditional_dict_score_models=condition_original.to_dict(),
                unconditional_dict_score_models=uncondition_original.to_dict(),
            )
            generator_log_dict.update({"generator_loss": generator_loss.detach()})
            return generator_log_dict, generator_loss
        else:
            # Step 4: Store gradients for the critic (if training the critic)
            critic_loss, critic_log_dict = self.critic_loss(
                image_or_video_shape=list(x0_B_C_T_H_W.permute(0, 2, 1, 3, 4).shape),
                conditional_dict=condition.to_dict(),
                conditional_dict_score_models=condition_original.to_dict(),
            )
            critic_log_dict.update({"critic_loss": critic_loss.detach()})
            return critic_log_dict, critic_loss

    def get_x0_fn_from_batch(
        self,
        data_batch: dict[str, torch.Tensor] | None = None,
        guidance: float = 1.0,
        is_negative_prompt: bool = False,
        conditional_dict: dict = None,
    ):
        assert data_batch is not None or conditional_dict is not None, "data_batch or conditional_dict must be provided"

        if data_batch is not None:
            data_batch = self.get_data_batch_with_latent_view_indices(data_batch)
            if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
                num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
            else:
                num_conditional_frames = None

        if conditional_dict is None:
            _, latent_state, _ = self.get_data_and_condition(
                data_batch, with_uncondition=False
            )  # we need always process the data batch first.
            is_image_batch = self.is_image_batch(data_batch)

            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)

            # Enable CP and broadcast conditions (no temporal split; net handles CP internally)
            condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
            _, condition, _, _ = self.broadcast_split_for_model_parallelsim(None, condition, None, None)

            state_t = int(
                (data_batch["num_video_frames_per_view"].cpu().item() - 1) // self.tokenizer.temporal_compression_factor
                + 1
            )

            condition = condition.set_video_condition(
                state_t=state_t,
                gt_frames=latent_state.to(**self.tensor_kwargs),
                condition_locations=["first_random_n"],
                random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
                random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
                num_conditional_frames_per_view=num_conditional_frames,  # overrides random_min_num_conditional_frames_per_view and random_max_num_conditional_frames_per_view
                view_condition_dropout_max=0,
                conditional_frames_probs=self.config.conditional_frames_probs,
            )

            conditional_dict = condition.to_dict()


        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def x0_fn(
            noise_x: torch.Tensor,
            timestep: torch.Tensor,
            kv_cache: Optional[List[dict]] = None,
            i2v_force_add_into_cache: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            assert self.config.model_type == "i2v"

            noise_x = noise_x.permute(0, 2, 1, 3, 4)
            new_condition_dict = copy.deepcopy(conditional_dict)

            if (
                new_condition_dict["gt_frames"] is not None
                and new_condition_dict["gt_frames"].shape[2] != noise_x.shape[2]
            ):
                assert kwargs.get("start_frame_for_rope", None) is not None, "start_frame_for_rope is not provided"
                start_frame = kwargs.get("start_frame_for_rope")
                end_frame = start_frame + noise_x.shape[2]

                # Slice condition tensors along T dimension
                new_condition_dict["gt_frames"] = new_condition_dict["gt_frames"][:, :, start_frame:end_frame, :, :]
                if new_condition_dict["condition_video_input_mask_B_C_T_H_W"] is not None:
                    new_condition_dict["condition_video_input_mask_B_C_T_H_W"] = new_condition_dict[
                        "condition_video_input_mask_B_C_T_H_W"
                    ][:, :, start_frame:end_frame, :, :]
                if new_condition_dict["view_indices_B_T"] is not None:
                    new_condition_dict["view_indices_B_T"] = new_condition_dict["view_indices_B_T"][
                        :, start_frame:end_frame
                    ]

            _, denoised_pred = self.generator(
                noisy_image_or_video=noise_x.permute(0, 2, 1, 3, 4),
                conditional_dict=new_condition_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                **kwargs,
            )
            return denoised_pred

        return x0_fn

    def generate_samples_from_batch(
        self,
        data_batch: dict[str, torch.Tensor] | None = None,
        guidance: float = 1.0,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        start_latents: Optional[torch.Tensor] = None,
        verbose: bool = False,
        conditional_dict: dict = None,
        image_or_video_shape: Tuple | None = None,
        noise_B_T_C_H_W: Optional[torch.Tensor] = None,
        is_training: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        # Note (zianw): Reference: projects/cosmos/sil/causal_wan/causal_training/models/t2v_model_causal.py
        # The tensor shape of this function uses: B_T_C_H_W, different from the B_C_T_H_W used in reference.
        # This function is not only for inference but also for SF generation during training

        if data_batch is not None:
            self._normalize_video_databatch_inplace(data_batch)
            self._augment_image_dim_inplace(data_batch)
            is_image_batch = self.is_image_batch(data_batch)
            input_key = self.input_image_key if is_image_batch else self.input_data_key

            if n_sample is None:
                n_sample = data_batch[input_key].shape[0]
            if state_shape is None:
                _T, _H, _W = data_batch[input_key].shape[-3:]
                state_shape = [
                    self.tokenizer.get_latent_num_frames(_T),
                    self.config.state_ch,
                    _H // self.tokenizer.spatial_compression_factor,
                    _W // self.tokenizer.spatial_compression_factor,
                ]
            else:
                state_shape = (state_shape[1], state_shape[0], *state_shape[2:])

        assert state_shape is not None or image_or_video_shape is not None, (
            "data_batch or image_or_video_shape must be provided"
        )

        if noise_B_T_C_H_W is None:
            noise_B_T_C_H_W = misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape) if image_or_video_shape is None else image_or_video_shape,
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            misc.set_random_seed(seed=seed, by_rank=False)

        self.frame_seq_length = int(noise_B_T_C_H_W.shape[-1] * noise_B_T_C_H_W.shape[-2] / 4)

        # CP broadcast
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            self.net.enable_context_parallel(cp_group)
            # broadcast (no split; net handles CP split internally)
            noise_B_T_C_H_W = broadcast(noise_B_T_C_H_W.contiguous(), cp_group)
            if start_latents is not None:
                start_latents = broadcast(start_latents.contiguous(), cp_group)
        else:
            # Some network variants may not expose the property until enabled; use getattr
            assert not getattr(self.net, "is_context_parallel_enabled", False), (
                "context parallel should be disabled if parallel_state is not initialized"
            )

        if cp_group is not None and not is_tp_cp_pp_rank0():
            verbose = False

        flow_pred_fn = self.get_x0_fn_from_batch(
            data_batch=data_batch,
            guidance=guidance,
            is_negative_prompt=is_negative_prompt,
            conditional_dict=conditional_dict,
        )

        def x0_fn(
            noisy_image_or_video: torch.Tensor,
            timestep: torch.Tensor,
            kv_cache: Optional[List[dict]] = None,
            crossattn_cache: Optional[List[dict]] = None,
            current_start: Optional[int] = None,
            current_end: Optional[int] = None,
            start_frame_for_rope: Optional[int] = None,
        ):
            return flow_pred_fn(
                noise_x=noisy_image_or_video,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                start_frame_for_rope=start_frame_for_rope,
            )

        batch_size, num_frames, num_channels, height, width = noise_B_T_C_H_W.shape

        num_input_frames = 0
        num_output_frames = num_frames + num_input_frames

        output_B_T_C_H_W = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise_B_T_C_H_W.device,
            dtype=noise_B_T_C_H_W.dtype,
        )

        assert num_frames % self.config.num_frame_per_block == 0
        num_blocks = num_frames // self.config.num_frame_per_block

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=self.tensor_kwargs["dtype"],
            device=self.tensor_kwargs["device"],
            num_training_frames=num_output_frames,
            is_training=is_training,
        )
        # if not torch.is_grad_enabled():
        #     # Avoid cross attn cache during training. Do cross attn cache during inference.
        #     self._initialize_crossattn_cache(
        #         batch_size=batch_size, dtype=self.tensor_kwargs["dtype"], device=self.tensor_kwargs["device"]
        #     )
        # else:
        self.crossattn_cache = None

        # Step 2: Cache context feature (start_latents is not handled yet)
        current_start_frame = 0

        # Step 3: Temporal denoising loop
        all_num_frames = [self.config.num_frame_per_block] * num_blocks

        exit_flags = []
        if self.config.independent_denoising_step_list:
            for block_index in range(len(all_num_frames)):
                if is_training:
                    exit_flag = self.generate_and_sync_list(
                        1, len(self.denoising_step_list[block_index]), device=noise_B_T_C_H_W.device
                    )
                else:
                    exit_flag = [len(self.denoising_step_list[block_index]) - 1]
                exit_flags.append(exit_flag[0])
        else:
            num_denoising_steps = len(self.denoising_step_list)
            if is_training:
                exit_flags = self.generate_and_sync_list(
                    len(all_num_frames), num_denoising_steps, device=noise_B_T_C_H_W.device
                )
            else:
                exit_flags = [num_denoising_steps - 1] * len(all_num_frames)

        # Note: current implementation assuems all frames have gradients
        # start_gradient_frame_index = num_output_frames - self.config.num_gradient_enabled_frames
        start_gradient_frame_index = 0

        for block_index, current_num_frames in enumerate(all_num_frames):
            current_end_frame = current_start_frame + current_num_frames
            noisy_input = noise_B_T_C_H_W[
                :,
                current_start_frame - num_input_frames : current_end_frame - num_input_frames,
            ]

            # Step 3.1: Spatial denoising loop
            denoising_step_list = (
                self.denoising_step_list[block_index]
                if self.config.independent_denoising_step_list
                else self.denoising_step_list
            )
            for index, current_timestep in enumerate(denoising_step_list):
                if self.config.same_step_across_blocks:
                    exit_flag = index == exit_flags[0]
                else:
                    exit_flag = (
                        index == exit_flags[block_index]
                    )  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames],
                        device=noise_B_T_C_H_W.device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )

                if not exit_flag:
                    with torch.no_grad():
                        denoised_pred = x0_fn(
                            noisy_image_or_video=noisy_input,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length // cp_size,
                            current_end=current_end_frame * self.frame_seq_length // cp_size,
                            start_frame_for_rope=current_start_frame,
                        )
                        next_timestep = denoising_step_list[index + 1]
                        current_noise = torch.randn_like(denoised_pred.flatten(0, 1))
                        if cp_size > 1:
                            current_noise = broadcast(current_noise.contiguous(), cp_group)
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            current_noise,
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise_B_T_C_H_W.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])

                else:
                    # for getting real output
                    if current_start_frame < start_gradient_frame_index or not is_training:
                        with torch.no_grad():
                            denoised_pred = x0_fn(
                                noisy_image_or_video=noisy_input,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length // cp_size,
                                current_end=current_end_frame * self.frame_seq_length // cp_size,
                                start_frame_for_rope=current_start_frame,
                            )
                    else:
                        denoised_pred = x0_fn(
                            noisy_image_or_video=noisy_input,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length // cp_size,
                            current_end=current_end_frame * self.frame_seq_length // cp_size,
                            start_frame_for_rope=current_start_frame,
                        )
                    break

            # Step 3.2: record the model's output
            output_B_T_C_H_W[:, current_start_frame:current_end_frame] = denoised_pred

            # Step 3.3: rerun with timestep at self.config.context_noise (e.g. 0 or 128) to update the cache
            context_timestep = torch.ones_like(timestep) * self.config.context_noise
            if self.config.context_noise > 0:
                # add context noise
                current_noise = torch.randn_like(denoised_pred.flatten(0, 1))
                if cp_size > 1:
                    current_noise = broadcast(current_noise.contiguous(), cp_group)
                denoised_pred = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    current_noise,
                    context_timestep
                    * torch.ones(
                        [batch_size * current_num_frames],
                        device=noise_B_T_C_H_W.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                x0_fn(
                    noisy_image_or_video=denoised_pred,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length // cp_size,
                    current_end=current_end_frame * self.frame_seq_length // cp_size,
                    start_frame_for_rope=current_start_frame,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame = current_end_frame

        # Step 3.5: Return the denoised timestep
        # Deleted some code here, now the denoised_timestep_from and denoised_timestep_to are all None!!!
        denoised_timestep_from, denoised_timestep_to = None, None

        if is_training:
            return output_B_T_C_H_W, denoised_timestep_from, denoised_timestep_to
        else:
            return output_B_T_C_H_W.permute(0, 2, 1, 3, 4)

    def _initialize_kv_cache(self, batch_size, dtype, device, num_training_frames=None, is_training=False):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        if num_training_frames is None:
            num_training_frames = self.config.num_training_frames

        local_attn_size = getattr(self.net, "local_attn_size", -1)
        if local_attn_size == -1 or is_training:
            # global attention
            kv_cache_size = self.frame_seq_length * num_training_frames
        else:
            if local_attn_size > num_training_frames:
                raise ValueError(
                    f"local_attn_size {local_attn_size} is larger than num_training_frames {num_training_frames}, "
                    f"which is not supported"
                )
            kv_cache_size = self.frame_seq_length * local_attn_size

        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            assert kv_cache_size % cp_size == 0, "kv_cache_size must be divisible by cp_size"
            kv_cache_size = kv_cache_size // cp_size

        kv_cache1 = []
        for _ in range(self.net.num_layers):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, int(kv_cache_size), self.net.num_heads, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, int(kv_cache_size), self.net.num_heads, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.net.num_layers):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, self.net.num_heads, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, self.net.num_heads, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device) -> List[int]:
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(low=0, high=num_denoising_steps, size=(num_blocks,), device=device)
            if self.config.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()
