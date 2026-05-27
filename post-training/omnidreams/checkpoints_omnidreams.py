# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint registry for Cosmos-CausalMultiview-2B.

The registry is the single source of truth for the (uuid, s3-uri, hf-uri)
mapping. Configs reference checkpoints by these module-level constants — the
loader (`omnidreams._src.imaginaire.utils.checkpoint_db.download_checkpoint`,
called from `cosmos-oss/scripts/train.py`) translates the s3 URI into the
local HF cache path automatically. When the weights are actually published,
only `_HF_REVISION` and the filenames need to change here; experiment configs
stay untouched.

Until upload, the local consolidated `.pt` files (see
`ckpt_ancestry/ANCESTRY.md`) are pre-staged into `$HF_HOME` so that
`hf download` resolves to them offline. See `ckpt_ancestry/link_hf_cache.sh`.
"""

import functools
import os

from omnidreams._src.imaginaire.utils.checkpoint_db import (
    CheckpointConfig,
    CheckpointDirS3,
    CheckpointFileHf,
    register_checkpoint,
)

_HF_ORG = os.environ.get("OMNI_DREAMS_HF_ORG", "nvidia")
_HF_REPOSITORY = f"{_HF_ORG}/omni-dreams-models"

# `main` until we pin a release SHA. Experiment configs do not reference this.
_HF_REVISION = "main"


# UUIDs (stable; safe to reference from configs).
UUID_L0_DISTILLED = "e5cadda3-8f52-43b2-b621-aa3d4c9f0588"
UUID_L1B_TEACHER = "3b4c21d0-7b77-4694-9d9d-6ac9b6dbba51"
UUID_L2A_STUDENT_INIT = "a12bf26e-8855-4ff0-a651-e7c2a0cb0697"


# Registered s3 URIs (sanitized form: literal "bucket"). These are what
# `download_checkpoint(uri)` looks up in the registry; the actual local path is
# resolved transparently via `CheckpointConfig.download()`.
S3_URI_L0_DISTILLED = (
    "s3://bucket/cosmos_diffusion_v2/causal_cosmos2/"
    "32n_cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk2_vae_encode_189f_loc6_sft_urban_stationary_mixed_gcp_student_resume/"
    "checkpoints/iter_000000500/model"
)
S3_URI_L1B_TEACHER = (
    "s3://bucket/cosmos_v2_causal_av/cosmos2_gws/"
    "32N@teacher_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m_189frames_1080p@20260309090017/"
    "checkpoints/iter_000007700/model"
)
S3_URI_L2A_STUDENT_INIT = (
    "s3://bucket/cosmos_v2_causal_av/cosmos2_gws/"
    "16N@causal_cosmos2_2B_res720p_30fps_hdmap_hdmap_pretrained_chunk2_vae_mads1m_1080p@20260225100739/"
    "checkpoints/iter_000022000/model"
)


# HF filenames within the repo. Layout: `single_view/<role>/<uuid>_model.pt`.
# The consolidated `.pt` files are not EMA-only (no EMA was enabled during the
# source mid-training runs), so the suffix is the generic `_model.pt`.
_HF_FILENAME_L0 = f"single_view/distilled/{UUID_L0_DISTILLED}_model.pt"
_HF_FILENAME_L1B = f"single_view/teacher/{UUID_L1B_TEACHER}_model.pt"
_HF_FILENAME_L2A = f"single_view/student-init/{UUID_L2A_STUDENT_INIT}_model.pt"


@functools.cache
def register_checkpoints():
    from cosmos_oss.checkpoints_predict2 import register_checkpoints as _register_checkpoints

    _register_checkpoints()

    # L0 — single-view distillation final ckpt.
    # SF i2v + hdmap + 189 frames, urban/stationary mixed SFT, student-resume.
    register_checkpoint(
        CheckpointConfig(
            uuid=UUID_L0_DISTILLED,
            name=f"{_HF_REPOSITORY}/single_view/distilled",
            experiment="32n_cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk2_vae_encode_189f_loc6_sft_urban_stationary_mixed_gcp_student_resume",
            metadata={
                "size": "2B",
                "resolution": "720p",
                "fps": 30,
                "views": 1,
                "frames": 189,
            },
            s3=CheckpointDirS3(uri=S3_URI_L0_DISTILLED),
            hf=CheckpointFileHf(
                repository=_HF_REPOSITORY,
                revision=_HF_REVISION,
                filename=_HF_FILENAME_L0,
            ),
        ),
    )

    # L1b — 189-frame teacher (used as net_real_score_ckpt for L0).
    register_checkpoint(
        CheckpointConfig(
            uuid=UUID_L1B_TEACHER,
            name=f"{_HF_REPOSITORY}/single_view/teacher",
            experiment="32N@teacher_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m_189frames_1080p@20260309090017",
            metadata={
                "size": "2B",
                "resolution": "1080p",
                "fps": 30,
                "views": 1,
                "frames": 189,
                "role": "teacher",
            },
            s3=CheckpointDirS3(uri=S3_URI_L1B_TEACHER),
            hf=CheckpointFileHf(
                repository=_HF_REPOSITORY,
                revision=_HF_REVISION,
                filename=_HF_FILENAME_L1B,
            ),
        ),
    )

    # L2a — 16N causal student-init mid-training ckpt (parent of L1a in the SV ancestry).
    register_checkpoint(
        CheckpointConfig(
            uuid=UUID_L2A_STUDENT_INIT,
            name=f"{_HF_REPOSITORY}/single_view/student-init",
            experiment="16N@causal_cosmos2_2B_res720p_30fps_hdmap_hdmap_pretrained_chunk2_vae_mads1m_1080p@20260225100739",
            metadata={
                "size": "2B",
                "resolution": "1080p",
                "fps": 30,
                "views": 1,
                "role": "student-init",
            },
            s3=CheckpointDirS3(uri=S3_URI_L2A_STUDENT_INIT),
            hf=CheckpointFileHf(
                repository=_HF_REPOSITORY,
                revision=_HF_REVISION,
                filename=_HF_FILENAME_L2A,
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# Helpers used by the offline cache linker (`ckpt_ancestry/link_hf_cache.sh`).
# Keep these in sync with the `register_checkpoint` calls above.
# --------------------------------------------------------------------------- #
HF_CACHE_FILES = {
    UUID_L0_DISTILLED: _HF_FILENAME_L0,
    UUID_L1B_TEACHER: _HF_FILENAME_L1B,
    UUID_L2A_STUDENT_INIT: _HF_FILENAME_L2A,
}
HF_REPOSITORY = _HF_REPOSITORY
HF_REVISION = _HF_REVISION
