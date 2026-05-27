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

# Shared post-training env. Sourced by setup_env.sh, smoke_test.slurm, and
# torchrun_smoke.sh. The single source of truth for cache + runtime paths so
# they don't drift between login-node setup and compute-node launch.
#
# Inputs (all optional — sensible defaults):
#   OMNI_CACHE_DIR  Cache root. Defaults to $HOME/.cache/omni-dreams (warns).
#   REL             Path to the vendored release tree. Defaults to
#                   <repo-root>/post-training, derived from this file's
#                   location (samples/post-training/_env.sh → ../../post-training).
#                   Override if you want CUDA_HOME / LD_LIBRARY_PATH wired
#                   to a different release tree's venv.
#
# Optional pre-source overrides (any of these are honored if already set):
#   TRITON_CACHE_BASE  Per-host default; override on laptops without writable
#                      Lustre, e.g. /tmp/triton_$$ or /dev/shm/triton_$$.
#   HF_HUB_OFFLINE     Default 1 (compute nodes have no internet). setup_env.sh
#                      sets =0 before sourcing so it can stage downloads.
#   WANDB_MODE         Default disabled (E3 self-forcing wires up wandb_dmd
#                      and fails offline without an api_key).
#
# This file does no I/O beyond mkdir -p of the cache subdirs it exports. It
# can be sourced repeatedly without side effects.

# Resolve cache root. Caller-set OMNI_CACHE_DIR wins; otherwise warn and
# fall back to a per-user default.
if [[ -z "${OMNI_CACHE_DIR:-}" ]]; then
  OMNI_CACHE_DIR="$HOME/.cache/omni-dreams"
  echo "WARNING: OMNI_CACHE_DIR not set; defaulting to $OMNI_CACHE_DIR" >&2
fi
export OMNI_CACHE_DIR

# Cache env (uv, pip, hf, xdg, tmp, imaginaire).
export UV_CACHE_DIR="$OMNI_CACHE_DIR/uv"
export UV_PYTHON_INSTALL_DIR="$OMNI_CACHE_DIR/uv-python"
export PIP_CACHE_DIR="$OMNI_CACHE_DIR/pip"
export HF_HOME="$OMNI_CACHE_DIR/huggingface"
export XDG_CACHE_HOME="$OMNI_CACHE_DIR/xdg"
# TMPDIR: only default if unset. AF_UNIX socket paths have a 108-byte limit
# (sun_path), and dataloader workers / NCCL helpers create sockets under
# $TMPDIR; a 100+ char $OMNI_CACHE_DIR (typical on Lustre) blows past this
# and produces "OSError: AF_UNIX path too long" plus a silent hang at
# "Starting training...". Honor a caller-set TMPDIR (e.g. /tmp) instead.
: "${TMPDIR:=$OMNI_CACHE_DIR/tmp}"
export TMPDIR
export IMAGINAIRE_OUTPUT_ROOT="$OMNI_CACHE_DIR/imaginaire4-output"
export IMAGINAIRE_CACHE_DIR="$OMNI_CACHE_DIR/imaginaire4-cache"

# HF offline default (compute nodes / second runs).
: "${HF_HUB_OFFLINE:=1}"
export HF_HUB_OFFLINE

# wandb default disabled (E3 wandb_dmd callback fails offline without api_key).
: "${WANDB_MODE:=disabled}"
export WANDB_MODE

# Triton cache base. Per-host on Lustre by default so jobs on the same node
# hit warm torch.compile / inductor caches. triton_per_rank_wrap.sh appends
# _${LOCAL_RANK} per spawned rank so 8 ranks never share a hash dir (avoids
# the os.replace() ENOTEMPTY race from U-7).
: "${TRITON_CACHE_BASE:=$OMNI_CACHE_DIR/triton/$(hostname -s)}"
export TRITON_CACHE_BASE

mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME" \
         "$IMAGINAIRE_OUTPUT_ROOT" "$IMAGINAIRE_CACHE_DIR" \
         "$(dirname "$TRITON_CACHE_BASE")"

# Default REL to the release tree two levels up from this file
# (samples/post-training/_env.sh → <repo>/post-training/). This lets users
# `source samples/post-training/_env.sh` standalone after `setup_env.sh`
# without exporting REL themselves.
if [[ -z "${REL:-}" ]]; then
  _ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  _REL_DEFAULT="$(cd "$_ENV_DIR/../../post-training" 2>/dev/null && pwd)"
  [[ -n "$_REL_DEFAULT" ]] && REL="$_REL_DEFAULT"
  unset _ENV_DIR _REL_DEFAULT
fi
[[ -n "${REL:-}" ]] && export REL

# Venv-derived paths. Only set when the venv exists, so this file remains
# safe to source on a fresh checkout (before `uv sync` has built the venv).
if [[ -n "${REL:-}" && -d "$REL/.venv/lib/python3.10/site-packages/nvidia" ]]; then
  _VENV="$REL/.venv"
  export CUDA_HOME="$_VENV/lib/python3.10/site-packages/nvidia"
  export LD_LIBRARY_PATH="$(find "$_VENV"/lib/python3.10/site-packages/nvidia -maxdepth 3 -type d -name lib 2>/dev/null | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  unset _VENV
fi
