# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# PAI-NuRec post-training override.
#
# Inherits from the release-redacted SV-HDMap mid-training student-init base
# (parallels L2a in the cosmos-causal release config matrix; see
# post-training/INTERNAL.md). Two other release bases are available
# for different use cases — swap the import as needed:
#
#   from omnidreams.experiments.causal.release       import COSMOS2_2B_DF_HDMAP_VAE_CHUNK2  # student-init (L2a)
#   from omnidreams.experiments.causal.release       import TEACHER_COSMOS2_2B_HDMAP_VAE     # teacher    (L1b)
#   from omnidreams.experiments.self_forcing.release import COSMOS2_2B_SF_RES720P_FPS30_I2V_HDMAP_CHUNK2_VAE_ENCODE_LOC6  # SF distillation (L0)
from __future__ import annotations

from omnidreams.experiments.causal.release import (
    COSMOS2_2B_DF_HDMAP_VAE_CHUNK2 as _base,
)
from omnidreams._src.imaginaire.lazy_config import LazyDict

# Build via explicit dict merge — `LazyDict(**_base, job=…)` would pass `job`
# twice (once via the splat, once explicitly) and Python raises TypeError.
# The merge form below is safe and equivalent.
#
# `dataloader_train.repeat_factor` is *not* set here — `torchrun_smoke.sh`
# passes it as a Hydra command-line override on each invocation so the
# release-side experiments (which the launcher targets directly) pick it up
# too. Single source of truth.
exp_pai_nurec_sv_hdmap = LazyDict(
    {
        **_base,
        "job": {**_base["job"], "name": "exp_pai_nurec_sv_hdmap"},
    }
)
