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

# Primary launcher: torchrun on a single compute node. Defaults to 8 GPUs.
#
# Usage (from the repo root):
#   bash samples/post-training/torchrun_smoke.sh <1|2|3> [hydra-overrides...]
#
# The supported minimum is NPROC=8. Trailing positional args are forwarded
# verbatim to scripts.train, so you can override Hydra keys without editing
# this file.
#
# Prereq: setup_env.sh has run (venv built, checkpoints + dataset staged).
# Slurm users invoke smoke_test.slurm, which exec's into this script — one
# script defines the torchrun command, both paths inherit it.

set -euo pipefail

EXPERIMENT="${1:-${EXPERIMENT:-1}}"
[[ $# -gt 0 ]] && shift
EXTRA_ARGS=("$@")
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-12341}"

check_free_disk_gb() {
  local label="$1"
  local path="$2"
  local min_gb="$3"

  if ! [[ "$min_gb" =~ ^[0-9]+$ ]]; then
    echo "ERROR: OMNI_MIN_TRAIN_FREE_GB must be a non-negative integer (got: $min_gb)" >&2
    exit 2
  fi
  if (( min_gb == 0 )); then
    return
  fi

  mkdir -p "$path"

  local available_kb
  available_kb="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  if [[ -z "$available_kb" ]]; then
    echo "ERROR: Could not determine free disk for $label at $path" >&2
    exit 1
  fi

  local min_kb=$((min_gb * 1024 * 1024))
  local available_gb=$((available_kb / 1024 / 1024))
  if (( available_kb < min_kb )); then
    cat >&2 <<EOF
ERROR: Not enough free disk for post-training launch.
  Path:     $path ($label)
  Free:     ${available_gb} GB
  Required: ${min_gb} GB

Set OMNI_CACHE_DIR, IMAGINAIRE_OUTPUT_ROOT, TMPDIR, or TRITON_CACHE_BASE to a filesystem with more space.
Set OMNI_MIN_TRAIN_FREE_GB=0 to skip this preflight check.
EOF
    exit 1
  fi

  echo "Disk check: $label at $path has ${available_gb} GB free (requires ${min_gb} GB)."
}

if ! [[ "$NPROC" =~ ^[0-9]+$ ]]; then
  echo "ERROR: NPROC must be a positive integer (got: $NPROC)" >&2
  exit 2
fi
if (( NPROC < 8 )); then
  echo "ERROR: NPROC=$NPROC is not supported by the post-training configs." >&2
  echo "The supported minimum post-training launch uses NPROC=8." >&2
  exit 2
fi
if (( NPROC != 8 )); then
  echo "NOTE: NPROC=$NPROC selected. Pass parallelism overrides that divide the selected world size." >&2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default: ../../post-training (sibling of samples/). Override with
# OMNI_RELEASE_DIR to point at a different release tree (useful while a
# release-side change — e.g. repeat_factor or HF checkpoint registry — is
# still propagating to your worktree's vendored copy).
REL="${OMNI_RELEASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)/post-training}"
WRAP="$SCRIPT_DIR/triton_per_rank_wrap.sh"

[[ -f "$REL/.venv/bin/activate" ]] || {
  echo "ERROR: venv not built at $REL/.venv. Run setup_env.sh first." >&2; exit 1
}

export REL
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MIN_TRAIN_FREE_GB="${OMNI_MIN_TRAIN_FREE_GB:-20}"
check_free_disk_gb "output root" "$IMAGINAIRE_OUTPUT_ROOT" "$MIN_TRAIN_FREE_GB"
check_free_disk_gb "temporary files" "$TMPDIR" "$MIN_TRAIN_FREE_GB"
check_free_disk_gb "Triton cache parent" "$(dirname "$TRITON_CACHE_BASE")" "$MIN_TRAIN_FREE_GB"

# shellcheck disable=SC1091
source "$REL/.venv/bin/activate"
cd "$REL"

# PYTHONPATH must NOT include $VENV/site-packages — uvx subprocesses inherit
# it and break under mixed Python versions on cluster nodes.
export PYTHONPATH="$REL/packages/cosmos-cuda:$REL/packages/cosmos-oss:$REL${PYTHONPATH:+:$PYTHONPATH}"

run_torchrun() {
  python -m torch.distributed.run \
    --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    --no-python "$WRAP" python -m scripts.train "$@"
}

# dataloader_train.repeat_factor=200 expands the <200-clip sample to ~40k
# effective samples/epoch — without it E3 exhausts at iter ~30. The release
# experiments don't override repeat_factor (only augmentation_config), so we
# pass it on the command line. This launcher is the single source of truth;
# callers invoking experiment= directly (e.g. via a custom Hydra launcher)
# must pass dataloader_train.repeat_factor=200 themselves.
REPEAT="dataloader_train.repeat_factor=200"

# EXTRA_ARGS go LAST so a caller-supplied `model.config.fsdp_shard_size=4`
# wins over the launcher's `=8` default (Hydra takes the last value for a
# repeated key).
case "$EXPERIMENT" in
  1)
    echo "=== Experiment 1: Mid-training student-init (L2a) ==="
    run_torchrun \
      --config=omnidreams/_src/omnidreams/configs/causal_cosmos2/config.py \
      -- experiment=causal_cosmos2_2B_single_view_chunk2_t24_hdmap_vae \
         job.name=causal_cosmos2_2B_single_view_chunk2_t24_hdmap_vae \
         model.config.fsdp_shard_size=8 \
         model_parallel.context_parallel_size=8 \
         "$REPEAT" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    ;;
  2)
    echo "=== Experiment 2: Mid-training teacher (L1b) ==="
    run_torchrun \
      --config=omnidreams/_src/omnidreams/configs/causal_cosmos2/config.py \
      -- experiment=teacher_cosmos2_2B_single_view_t24_hdmap_vae \
         job.name=teacher_cosmos2_2B_single_view_t24_hdmap_vae \
         model.config.fsdp_shard_size=8 \
         model_parallel.context_parallel_size=8 \
         "$REPEAT" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    ;;
  3)
    echo "=== Experiment 3: Self-forcing distillation (L0) ==="
    run_torchrun \
      --config=omnidreams/_src/omnidreams/configs/self_forcing/config.py \
      -- experiment=cosmos_v2_2b_SF_res720p_fps30_i2v_hdmap_chunk2_vae_encode_loc6_release \
         job.wandb_mode=disabled \
         "$REPEAT" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    ;;
  *)
    echo "ERROR: EXPERIMENT must be 1, 2, or 3 (got: $EXPERIMENT)" >&2
    exit 2
    ;;
esac
