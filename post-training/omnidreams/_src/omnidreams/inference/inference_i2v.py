# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Inference script for causal multiview I2V model using CausalJointCosmosModel.

This script is based on omnidreams/_src/predict2_multiview/scripts/inference.py
with adaptations for KV cache management from inference_i2v_av_wan.py.

To run inference on the training data (as visualization/debugging), use:
```bash
EXP=your_experiment_name
ckpt_path=path/to/checkpoint/
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m omnidreams._src.omnidreams.inference.inference_i2v \
    --experiment ${EXP} \
    --ckpt_path ${ckpt_path} \
    --context_parallel_size 8 \
    --input_is_train_data \
    --max_samples 1 \
    --num_conditional_frames 0 \
    --guidance 3 \
    --save_root results/causal_multiview/
```
"""


import argparse
import collections
import collections.abc
import os
from typing import Any

import torch
from einops import rearrange
from megatron.core import parallel_state

from omnidreams._src.imaginaire.lazy_config import instantiate
from omnidreams._src.imaginaire.utils import distributed, log, misc
from omnidreams._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from omnidreams._src.imaginaire.visualize.video import save_img_or_video
from omnidreams._src.omnidreams.utils.misc import sync_timer
from omnidreams._src.omnidreams.utils.model_loader import load_model_from_checkpoint

IS_PREPROCESSED_KEY = "is_preprocessed"
NUM_CONDITIONAL_FRAMES_KEY = "num_conditional_frames"

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def to_with_skip_tensor(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    memory_format: torch.memory_format = torch.preserve_format,
    key: str | None = None,
) -> Any:
    """Recursively cast data into the specified device, dtype, and/or memory_format.

    The input data can be a tensor, a list of tensors, a dict of tensors.
    See the documentation for torch.Tensor.to() for details.

    Args:
        data (Any): Input data.
        device (str | torch.device): GPU device (default: None).
        dtype (torch.dtype): data type (default: None).
        memory_format (torch.memory_format): memory organization format (default: torch.preserve_format).
        key (str | None): Key name for skip tensor logic.

    Returns:
        data (Any): Data cast to the specified device, dtype, and/or memory_format.
    """
    skip_tensor_name = [
        "camera",
        "depth",
        "intrinsics",
        "buffer_depths",
        "buffer_w2cs",
        "target_w2cs",
        "buffer_intrinsics",
        "target_intrinsics",
        "buffer_points",
        "buffer_masks",
        "num_video_frames_per_view",
    ]
    assert device is not None or dtype is not None or memory_format is not None, (
        "at least one of device, dtype, memory_format should be specified"
    )
    if isinstance(data, torch.Tensor):
        if (
            memory_format == torch.channels_last
            and data.dim() != 4
            or memory_format == torch.channels_last_3d
            and data.dim() != 5
        ):
            memory_format = torch.preserve_format  # do not change the memory format
        is_cpu = (isinstance(device, str) and device == "cpu") or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        if key is not None and key in skip_tensor_name:
            data = data.to(
                device=device,
                dtype=torch.float32,
                memory_format=memory_format,
                non_blocking=(not is_cpu),
            )
        else:
            data = data.to(
                device=device,
                dtype=dtype,
                memory_format=memory_format,
                non_blocking=(not is_cpu),
            )
        return data
    elif isinstance(data, collections.abc.Mapping):
        converted = {
            key: to_with_skip_tensor(data[key], device=device, dtype=dtype, memory_format=memory_format, key=key)
            for key in data
        }
        return type(data)(converted)  # type: ignore[call-arg]
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        converted_list = [
            to_with_skip_tensor(elem, device=device, dtype=dtype, memory_format=memory_format, key=key) for elem in data
        ]
        return type(data)(converted_list)  # type: ignore[call-arg]
    else:
        return data


def to_model_input(data_batch: dict, model: torch.nn.Module) -> dict:
    """
    Convert data batch to model input format, avoiding converting uint8 "video" to float.

    Args:
        data_batch: Dictionary containing input data.
        model: The model to get tensor kwargs from.

    Returns:
        Data batch with tensors moved to proper device and dtype.
    """
    for k, v in data_batch.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)  # type: ignore[arg-type]
        data_batch[k] = _v
    return data_batch


def save_output(to_show: list[torch.Tensor], vid_save_path: str, fps: int = 16) -> None:
    """Save output video for visualization.

    Args:
        to_show: List of tensors to visualize, each of shape [b, c, t, h, w].
        vid_save_path: Path to save the video (without extension).
        fps: Frames per second for the output video.
    """
    legancy_to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]

    video_array = (rearrange(legancy_to_show, "n b c t h w -> t (n h) (b w) c") * 255).to(torch.uint8).cpu().numpy()
    log.info(
        f"video_array.shape: {video_array.shape} value: {video_array.max()}, {video_array.min()}, save to {vid_save_path}"
    )
    save_img_or_video(
        rearrange(legancy_to_show, "n b c t h w -> c t (n h) (b w)"),
        vid_save_path.split(".mp4")[0],  # remove .mp4
        fps=fps,
    )
    log.info(f"save video to {vid_save_path}", rank0_only=True)


class I2VInference:
    """
    Handles the I2V inference process for CausalJointCosmosModel, including model loading,
    data preparation, and video generation. Supports context parallelism.
    """

    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        config_file: str = "omnidreams/_src/omnidreams/configs/causal_cosmos2/config.py",
        context_parallel_size: int = 1,
        guidance: float = 5.0,
        shift: float = 5.0,
        num_sampling_steps: int = 35,
        seed: int = 1,
    ):
        """
        Initializes the I2VInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path. Sets up distributed processing if needed.

        Args:
            experiment_name: Name of the experiment configuration.
            ckpt_path: Path to the model checkpoint (local or S3).
            config_file: Path to the configuration file.
            context_parallel_size: Number of GPUs for context parallelism.
            guidance: Classifier-free guidance scale.
            shift: Shift parameter for the diffusion process.
            num_sampling_steps: Number of sampling steps.
            seed: Random seed for reproducibility.
        """
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.config_file = config_file
        self.context_parallel_size = context_parallel_size
        self.guidance = guidance
        self.shift = shift
        self.num_sampling_steps = num_sampling_steps
        self.process_group = None

        if "RANK" in os.environ:
            self._init_distributed()

        misc.set_random_seed(seed=seed, by_rank=True)

        # Load the model and config
        self.model, self.config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=self.config_file,
            load_ema_to_reg=False,
            instantiate_ema=False,
            cache_text_encoder=True,
            local_cache_dir=os.path.expanduser(os.getenv("IMAGINAIRE_CACHE_DIR", "~/.cache/imaginaire")),
        )

        # Enable context parallel on the model if using context parallelism
        self.rank0 = True
        if self.context_parallel_size > 1:
            self.model.net.enable_context_parallel(self.process_group)
            self.rank0 = distributed.get_rank() == 0

        self.model.eval()
        self.model = self.model.to(dtype=torch.bfloat16)

        # reset self.model.net.pos_embedder to ensure values are in fp32
        if hasattr(self.model, "net") and hasattr(self.model.net, "pos_embedder"):
            log.info("Resetting pos_embedder parameters to restore float32 precision after bf16 cast")
            self.model.net.pos_embedder.reset_parameters()
        else:
            log.warning("self.model.net.pos_embedder not available, skipping reset_parameters()")


        self.model.config.split_cp_in_model = False
        self.batch_size = 1
        self.generate_cnt = 0
        torch.cuda.empty_cache()

    def _init_distributed(self) -> None:
        """Initialize distributed processing for context parallelism."""
        # Initialize distributed environment
        distributed.init()

        # Initialize model parallel states
        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
        )

        # Get the process group for context parallel
        self.process_group = parallel_state.get_context_parallel_group()

        log.info(f"Initialized context parallel with size {self.context_parallel_size}")
        log.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def clear_cache(self) -> None:
        """Clear KV caches for the model."""
        self.model.kv_cache1 = None
        self.model.kv_cache2 = None

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
            self.model.config.text_encoder_config is not None
            and self.model.config.text_encoder_config.compute_online
            and self.model.text_encoder is not None
        ):
            text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                data_batch, self.model.input_caption_key
            )
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

            # Compute negative prompt embeddings for classifier-free guidance
            if use_negative_prompt:
                batch_size = text_embeddings.shape[0]
                neg_data_batch = {self.model.input_caption_key: [negative_prompt] * batch_size, "images": None}
                neg_text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                    neg_data_batch, self.model.input_caption_key
                )
                data_batch["neg_t5_text_embeddings"] = neg_text_embeddings


    def generate_from_batch(
        self,
        data_batch: dict,
        guidance: float | None = None,
        seed: int = 1,
        num_steps: int | None = None,
        shift: float | None = None,
        use_negative_prompt: bool = True,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        save_output_for_viz: bool = False,
        output_path: str | None = None,
        output_name: str | None = None,
    ) -> torch.Tensor:
        """Generate video tensor from batch.

        Args:
            data_batch: Dictionary containing input data.
            guidance: Classifier-free guidance scale. Uses instance default if None.
            seed: Random seed for generation.
            num_steps: Number of sampling steps. Uses instance default if None.
            shift: Shift parameter. Uses instance default if None.
            use_negative_prompt: Whether to use negative prompt.
            negative_prompt: Custom negative prompt for classifier-free guidance.
            save_output_for_viz: Whether to save output for visualization.
            output_path: Path to save visualization output.
            output_name: Name of the output video.
        Returns:
            Tensor with values in the range [-1, 1], shape (B, C, T, H, W).
        """
        guidance = guidance if guidance is not None else self.guidance
        num_steps = num_steps if num_steps is not None else self.num_sampling_steps
        shift = shift if shift is not None else self.shift

        # Preprocess video data
        if "video" in data_batch:
            data_batch["video"] = data_batch["video"].float()
            # Normalize if not already preprocessed
            if not data_batch.get(IS_PREPROCESSED_KEY, False):
                data_batch["video"] = data_batch["video"] / 127.5 - 1.0
            data_batch["video"] = torch.clamp(data_batch["video"], -1, 1)

        # Preprocess hdmap condition if present
        if "control_input_hdmap_bbox" in data_batch:
            data_batch["control_input_hdmap_bbox"] = data_batch["control_input_hdmap_bbox"].float()
            # Normalize if not already preprocessed
            if not data_batch.get(IS_PREPROCESSED_KEY, False):
                data_batch["control_input_hdmap_bbox"] = data_batch["control_input_hdmap_bbox"] / 127.5 - 1.0
            data_batch["control_input_hdmap_bbox"] = torch.clamp(data_batch["control_input_hdmap_bbox"], -1, 1)

        data_batch[IS_PREPROCESSED_KEY] = True
        data_batch = to_with_skip_tensor(data_batch, **self.model.tensor_kwargs)

        # Compute text embeddings online using cosmos_reason
        self.inplace_compute_text_embeddings_online(
            data_batch,
            use_negative_prompt=use_negative_prompt,
            negative_prompt=negative_prompt,
        )

        # Store hdmap for visualization before model processing
        control_input_hdmap_bbox_viz = data_batch.get("control_input_hdmap_bbox")
        data_batch = self.model.get_data_batch_with_latent_view_indices(data_batch)
        raw_data, x0, condition = self.model.get_data_and_condition(data_batch)

        with torch.no_grad():
            log.info("Start inference", rank0_only=True)
            with sync_timer("generate_samples_from_batch"):
                sample = self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=guidance,
                    shift=shift,
                    state_shape=x0.shape[1:],
                    n_sample=x0.shape[0],
                    seed=seed,
                    num_steps=num_steps,
                    is_negative_prompt=use_negative_prompt,
                    verbose=True,
                )
            with sync_timer("decode"):
                video = self.model.decode(sample)
            log.info("End inference", rank0_only=True)

        if save_output_for_viz and output_path is not None:
            os.makedirs(output_path, exist_ok=True)

            if output_name is not None:
                base_fp_wo_ext = os.path.join(output_path, output_name + "_with_hdmap.mp4")
            else:
                base_fp_wo_ext = os.path.join(output_path, f"_Sample_Iter{self.generate_cnt:03d}.mp4")
            self.generate_cnt += 1
            # to_show = [video.float().cpu(), raw_data.float().cpu()]   # uncomment to visualize first RGB frame
            to_show = [
                video.float().cpu(),
            ]
            # Include hdmap in visualization if present
            if control_input_hdmap_bbox_viz is not None:
                to_show.insert(0, control_input_hdmap_bbox_viz.float().cpu())
            if self.context_parallel_size > 1:
                if is_tp_cp_pp_rank0():
                    save_output(to_show, base_fp_wo_ext)
            else:
                save_output(to_show, base_fp_wo_ext)

        return video

    def cleanup(self) -> None:
        """Clean up distributed resources."""
        if "RANK" in os.environ:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the I2V inference script."""
    parser = argparse.ArgumentParser(description="Causal I2V inference script")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specified in the config",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="omnidreams/_src/omnidreams/configs/causal_cosmos2/config.py",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )
    # Generation parameters
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance value")
    parser.add_argument("--shift", type=float, default=5.0, help="Shift parameter for diffusion")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output video")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of sampling steps")
    parser.add_argument("--num_conditional_frames", type=int, default=1, help="Number of conditional frames")
    parser.add_argument(
        "--use_negative_prompt",
        action="store_true",
        default=True,
        help="Use negative prompt for classifier-free guidance (default: True)",
    )
    parser.add_argument(
        "--no_negative_prompt",
        action="store_false",
        dest="use_negative_prompt",
        help="Disable negative prompt for classifier-free guidance",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Custom negative prompt for classifier-free guidance. If not specified, uses default negative prompt.",
    )
    # Input/output
    parser.add_argument(
        "--input_is_train_data",
        action="store_true",
        help="Inference on the training data, the input_root will be ignored if this is set",
    )
    parser.add_argument("--input_root", type=str, default="assets/i2v", help="Input root")
    parser.add_argument("--save_root", type=str, default="results/causal_i2v", help="Save root")
    parser.add_argument("--max_samples", type=int, default=20, help="Maximum number of samples to generate")
    parser.add_argument(
        "--save_output_for_viz",
        action="store_true",
        help="Save output videos with ground truth for visualization",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["NVTE_FUSED_ATTN"] = "0"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_grad_enabled(False)

    args = parse_arguments()

    # Initialize the inference handler with context parallel support
    i2v_cli = I2VInference(
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        context_parallel_size=args.context_parallel_size,
        guidance=args.guidance,
        shift=args.shift,
        num_sampling_steps=args.num_steps,
        seed=args.seed,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    os.makedirs(args.save_root, exist_ok=True)

    if args.input_is_train_data:
        # fix data loader order for inference
        dataloader = instantiate(i2v_cli.config.dataloader_train, is_train=False, num_workers=0)
        for i, batch in enumerate(dataloader):  # type: ignore[arg-type]
            if i >= args.max_samples:
                break

            # Clear KV cache before each sample
            i2v_cli.clear_cache()

            # Set number of conditional frames
            batch[NUM_CONDITIONAL_FRAMES_KEY] = args.num_conditional_frames

            video = i2v_cli.generate_from_batch(
                batch,
                guidance=args.guidance,
                seed=args.seed,
                num_steps=args.num_steps,
                shift=args.shift,
                use_negative_prompt=args.use_negative_prompt,
                negative_prompt=args.negative_prompt,
                save_output_for_viz=args.save_output_for_viz,
                output_path=args.save_root,
            )

            if rank0:
                # Normalize to [0, 1] for saving
                video_normalized = ((video + 1.0) / 2.0).clamp(0, 1)
                save_name = f"infer_from_train_{i}"
                save_img_or_video(video_normalized[0], f"{args.save_root}/{save_name}", fps=args.fps)
                log.info(f"Saved sample {i} to {args.save_root}/{save_name}")
    else:
        raise NotImplementedError(
            "Custom input inference not implemented yet. Use --input_is_train_data "
        )

    # Cleanup
    i2v_cli.cleanup()
