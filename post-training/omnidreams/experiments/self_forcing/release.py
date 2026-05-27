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
# `cosmos-oss/scripts/train.py` calls `download_checkpoint(load_path)` on the
# `s3://bucket/...` URIs below and resolves them via `CheckpointConfig.hf` to
# the local HF cache path (pre-staged by `ckpt_ancestry/link_hf_cache.sh`).
# Once weights are actually published, only `_HF_REVISION` in
# `checkpoints_omnidreams.py` needs updating — this file stays as-is.
register_checkpoints()


from omnidreams._src.imaginaire.lazy_config import LazyDict

"""
Self-forcing distillation parallel of the SV ancestry's L0 final checkpoint.
Mirrors source recipe `cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk2_vae_encode_loc6_gcp`
(omnidreams/_src/omnidreams/configs/self_forcing/experiment/exp2_i2v_hdmap.py:727)
but uses the registered local consolidated `.pt` checkpoints:
- net_ckpt            = L2a (student-init; same as the source recipe)
- net_real_score_ckpt = L1b (189-frame teacher; closest registered ancestor of L2b)
- net_fake_score_ckpt = L1b

Run via:
    torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
        --config=omnidreams/_src/omnidreams/configs/self_forcing/config.py \
        -- experiment="cosmos_v2_2b_SF_res720p_fps30_i2v_hdmap_chunk2_vae_encode_loc6_release"
"""


COSMOS2_2B_SF_RES720P_FPS30_I2V_HDMAP_CHUNK2_VAE_ENCODE_LOC6: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "video_local_720p_30fps_93frames_1view"},
            {"override /data_val": "video_local_720p_30fps_93frames_1view"},
            {"override /model": "fsdp_hdmap"},
            {"override /net": "cosmos_v2_2b_causal_hdmap"},
            {"override /net_real_score": "cosmos_v1_2B_hdmap"},
            {"override /net_fake_score": "cosmos_v1_2B_hdmap"},
            {"override /conditioner": "video_prediction_multiview_causal_conditioner_per_view_dropout_hdmap"},
            {"override /optimizer": "adamw"},
            {
                "override /callbacks": [
                    "basic",
                    "wandb_dmd",
                    "cluster_speed",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer_hf"},
            # NOTE: do NOT override /ckpt_type here. The upstream self_forcing
            # config defaults to "dcp_distill", which routes through the
            # per-key DistributedCheckpointer that handles the trainer's
            # optimizer_dict / scheduler_dict. Forcing "dcp" routes through
            # the bare predict2 OptimizerWrapper and crashes at first save
            # (it iterates dict keys as strings).
            "_self_",
        ],
        job=dict(
            group="omnidreams",
            name="cosmos_v2_2b_SF_res720p_fps30_i2v_hdmap_chunk2_vae_encode_loc6_release",
            # wandb_dmd callback calls wandb.login() unconditionally; disable
            # to keep the release runnable on clusters without wandb auth.
            wandb_mode="disabled",
        ),
        model=dict(
            config=dict(
                # chunk2 self-forcing distillation, local attention window of 6.
                num_frame_per_block=2,
                ema=dict(enabled=False),
                model_type="i2v",
                i2v_zero_latent_condition=True,
                use_vidprom_dataset=False,
                fsdp_shard_size=8,
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                denoise_replace_gt_frames=True,
                context_noise=128,
                # Hdmap conditioning via VAE-encoded bbox channels (16-ch concat).
                hdmap_process_method="vae_encoding",
                hdmap_selection_mode="all",
                # 720p RoPE matched to the chunk2 student / teacher.
                net=dict(
                    additional_concat_ch=16,
                    local_attn_size=6,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    timestep_scale=0.001,
                ),
                net_real_score=dict(
                    additional_concat_ch=16,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    timestep_scale=0.001,
                ),
                net_fake_score=dict(
                    additional_concat_ch=16,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    timestep_scale=0.001,
                ),
                # Distillation parents — resolved through the checkpoint registry.
                # `download_checkpoint(s3_uri)` (called from cosmos-oss/scripts/train.py)
                # returns the local HF cache path, which `dmd.py:_load_net_init_ckpt`
                # dispatches as a consolidated `.pt` via `load_consolidated_pt_to_net`.
                net_ckpt=S3_URI_L2A_STUDENT_INIT,
                net_real_score_ckpt=S3_URI_L1B_TEACHER,
                net_fake_score_ckpt=S3_URI_L1B_TEACHER,
                # Teacher/fake_score optimizer (aligns with public self-forcing code).
                optimizer_fake_score_config=dict(
                    lr=4e-7,
                    weight_decay=1e-2,
                    betas=(0.0, 0.999),
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.0,
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
        # Public-self-forcing-aligned hyperparams.
        optimizer=dict(
            lr=2e-6,
            weight_decay=1e-2,
            betas=(0.0, 0.999),
        ),
        # Constant learning rate.
        scheduler=dict(
            f_max=[1.0],
            f_min=[1.0],
            f_start=[1.0],
            warm_up_steps=[0],
            cycle_lengths=[10_000],
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            save_iter=50,
            save_to_object_store=dict(enabled=False),
            # net_ckpt / net_real_score_ckpt / net_fake_score_ckpt are loaded
            # inside `ImaginaireDMDBaseModel.set_up_model` via the registry; the
            # top-level `load_path` is unused here (no DCP resume).
            load_from_object_store=dict(enabled=False),
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=10_000,
            logging_iter=10,
            callbacks=dict(
                grad_clip=dict(clip_norm=10.0),
                iter_speed=dict(hit_thres=50),
                compile_tokenizer=dict(enabled=False),
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


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=COSMOS2_2B_SF_RES720P_FPS30_I2V_HDMAP_CHUNK2_VAE_ENCODE_LOC6["job"]["name"],
    node=COSMOS2_2B_SF_RES720P_FPS30_I2V_HDMAP_CHUNK2_VAE_ENCODE_LOC6,
)
