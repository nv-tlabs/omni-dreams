#!/usr/bin/env bash
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

# Per-rank Triton/inductor cache wrapper.
#
# Use via torchrun's --no-python:
#   torchrun --nproc_per_node=N --no-python \
#     samples/post-training/triton_per_rank_wrap.sh python -m scripts.train ...
#
# torchrun (and python -m torch.distributed.run) sets LOCAL_RANK per spawned
# process. We require it here and re-derive TRITON_CACHE_DIR per rank so 8
# ranks on one node never write to the same hash dir — eliminating the
# os.replace() ENOTEMPTY race.
#
# TRITON_CACHE_BASE is read from the caller's env. Defaults to
# /tmp/triton_${SLURM_JOB_ID:-$PPID} so the same wrapper works under Slurm,
# plain torchrun on a single multi-GPU box, and containerized launchers
# (Lepton, k8s) without any Slurm-specific assumptions.
set -euo pipefail
: "${TRITON_CACHE_BASE:=/tmp/triton_${SLURM_JOB_ID:-$PPID}}"
if [[ -z "${LOCAL_RANK:-}" ]]; then
  echo "ERROR: LOCAL_RANK is required; run this wrapper via torchrun." >&2
  exit 2
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_BASE}_${LOCAL_RANK}"
mkdir -p "$TRITON_CACHE_DIR"
exec "$@"
