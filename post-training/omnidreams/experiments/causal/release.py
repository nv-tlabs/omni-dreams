# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams.checkpoints_omnidreams import (
    S3_URI_L1B_TEACHER,
    S3_URI_L2A_STUDENT_INIT,
    register_checkpoints,
)

# Chains predict2's registry first, then registers the omnidreams SV
# ancestry checkpoints (L0 distilled, L1b teacher, L2a student-init).
# `cosmos-oss/scripts/train.py` calls `download_checkpoint(load_path)` on these
# s3 URIs and resolves them via `CheckpointConfig.hf` to the local HF cache path
# (pre-staged offline by `ckpt_ancestry/link_hf_cache.sh`). Once the weights are
# actually published, only `_HF_REVISION` in `checkpoints_omnidreams.py`
# needs to change — these configs stay as-is.
register_checkpoints()


from omnidreams._src.imaginaire.lazy_config import LazyDict

"""
Mid-training experiments mirroring the SV ancestry's L1b (189-frame teacher) and L2a
(16N causal student-init / chunk2) — both resume from local consolidated `.pt`
checkpoints rather than DCP dirs in object store.

Run via:
    torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
        --config=omnidreams/_src/omnidreams/configs/causal_cosmos2/config.py \
        -- experiment="causal_cosmos2_2B_single_view_chunk2_t24_hdmap_vae"
"""


# L2a parallel: 16N causal student-init, chunk2, hdmap+VAE encoding, single front-wide camera.
COSMOS2_2B_DF_HDMAP_VAE_CHUNK2: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "video_local_720p_30fps_93frames_1view"},
            {"override /data_val": "video_local_720p_30fps_93frames_1view"},
            {"override /model": "fsdp_hdmap"},
            {"override /net": "cosmos_v2_2b_causal_hdmap"},
            {"override /conditioner": "video_prediction_multiview_causal_conditioner_per_view_dropout_hdmap"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {
                "override /callbacks": [
                    "basic",
                    "wandb",
                    "cluster_speed",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer_hf"},
            "_self_",
        ],
        job=dict(
            group="omnidreams",
            name="causal_cosmos2_2B_single_view_chunk2_t24_hdmap_vae",
            wandb_mode="disabled",
        ),
        optimizer=dict(
            lr=3e-5,
            weight_decay=1e-3,
        ),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[100],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                # chunk2 student: causal, 2 latent frames per block.
                num_frame_per_block=2,
                max_latent_frames_per_gpu=24,
                state_t=24,
                fsdp_shard_size=8,
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="uniform",
                split_cp_in_model=False,
                # Hdmap conditioning via VAE-encoded bbox channels (16-ch concat).
                hdmap_process_method="vae_encoding",
                hdmap_selection_mode="all",
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                denoise_replace_gt_frames=True,
                ema=dict(enabled=False),
                net=dict(
                    additional_concat_ch=16,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    timestep_scale=0.001,
                ),
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.2,
                        use_empty_string=False,
                    ),
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy="full_concat",
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/pretrained/Qwen_tokenizer/Qwen/Qwen2.5-VL-7B-Instruct",
                    s3_credential_path="credentials/s3_checkpoint.secret",
                ),
            )
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        checkpoint=dict(
            save_iter=10,
            save_to_object_store=dict(enabled=False),
            # Local .pt — bypass object-store load.
            load_from_object_store=dict(enabled=False),
            load_path=S3_URI_L2A_STUDENT_INIT,
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=150_000,
            logging_iter=10,
            callbacks=dict(
                grad_clip=dict(clip_norm=0.1),
                iter_speed=dict(hit_thres=100),
            ),
        ),
        dataloader_train=dict(
            augmentation_config=dict(
                resolution_hw=(704, 1280),
                num_video_frames=93,
                camera_keys=[
                    "camera_front_wide_120fov",
                ],
            ),
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


# L1b parallel: 32N bidirectional teacher, chunk-24 (full block), hdmap+VAE encoding,
# 189-frame target window. Local data tops out at 93 frames so the dataloader uses
# that; the model still operates over `state_t=24` latent frames per gpu.
TEACHER_COSMOS2_2B_HDMAP_VAE: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "video_local_720p_30fps_93frames_1view"},
            {"override /data_val": "video_local_720p_30fps_93frames_1view"},
            {"override /model": "fsdp_hdmap"},
            {"override /net": "cosmos_v1_2B_hdmap"},
            {"override /conditioner": "video_prediction_multiview_causal_conditioner_per_view_dropout_hdmap"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {
                "override /callbacks": [
                    "basic",
                    "wandb",
                    "cluster_speed",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer_hf"},
            "_self_",
        ],
        job=dict(
            group="omnidreams",
            name="teacher_cosmos2_2B_single_view_t24_hdmap_vae",
            wandb_mode="disabled",
        ),
        optimizer=dict(
            lr=3e-5,
            weight_decay=1e-3,
        ),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[100],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                # Bidirectional teacher: full-block (state_t == num_frame_per_block).
                num_frame_per_block=24,
                max_latent_frames_per_gpu=24,
                state_t=24,
                fsdp_shard_size=8,
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="uniform",
                split_cp_in_model=True,
                hdmap_process_method="vae_encoding",
                hdmap_selection_mode="all",
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                denoise_replace_gt_frames=True,
                ema=dict(enabled=False),
                net=dict(
                    additional_concat_ch=16,
                    additional_init_method="random_init",
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    timestep_scale=0.001,
                ),
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.2,
                        use_empty_string=False,
                    ),
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy="full_concat",
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/pretrained/Qwen_tokenizer/Qwen/Qwen2.5-VL-7B-Instruct",
                    s3_credential_path="credentials/s3_checkpoint.secret",
                ),
            )
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        checkpoint=dict(
            save_iter=1000,
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
            load_path=S3_URI_L1B_TEACHER,
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=150_000,
            logging_iter=10,
            callbacks=dict(
                grad_clip=dict(clip_norm=0.1),
                iter_speed=dict(hit_thres=100),
            ),
            run_validation=True,
            run_validation_on_start=True,
            validation_iter=100,
            max_val_iter=50,
        ),
        dataloader_train=dict(
            augmentation_config=dict(
                resolution_hw=(704, 1280),
                num_video_frames=93,
                camera_keys=[
                    "camera_front_wide_120fov",
                ],
            ),
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


cs = ConfigStore.instance()
for _item in [
    COSMOS2_2B_DF_HDMAP_VAE_CHUNK2,
    TEACHER_COSMOS2_2B_HDMAP_VAE,
]:
    cs.store(group="experiment", package="_global_", name=_item["job"]["name"], node=_item)
