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

# One-shot setup for omni-dreams post-training (samples/post-training/).
#
# Stages: cache env, HF gated-model checkpoints, HF sample dataset (via
# prepare.py), venv VIRTUAL_ENV patch, transformer_engine TE symlinks.
# Safe to re-run.
#
# Usage:
#   bash samples/post-training/setup_env.sh [--cache-dir DIR]
#
# Env (all optional unless noted):
#   OMNI_CACHE_DIR         Cache root (default: $HOME/.cache/omni-dreams)
#   OMNI_DREAMS_HF_ORG     Override HF org for the omni-dreams-models checkpoint
#                          repo only (default: nvidia)
#   OMNI_HF_CKPT_REVISION  Override checkpoint HF revision (default: main)
#   OMNI_HF_DATA_REPO      Sample dataset repo id, decoupled from the checkpoint
#                          org (default: nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)
#   OMNI_HF_DATA_REVISION  Dataset branch/tag/commit the trainable scenes live on
#   OMNI_HF_DATA_SUBPATH   Override dataset subpath within the repo
#                          (defaults owned by prepare.py; see `prepare.py --help`)
#   OMNI_HF_DATA_INCLUDE   Globs to fetch (default: the per-camera training
#                          media; '**' = whole subpath, incl. the .usdz)
#   OMNI_LOCAL_DATA_SOURCE Fan out the dataset from this already-downloaded
#                          per-scene tree instead of HuggingFace (e.g. an
#                          rclone'd S3 copy); skips the dataset download.
#   OMNI_MIN_SETUP_FREE_GB Minimum free cache disk before staging (default: 150; 0 disables)
#   OMNI_MIN_WORKTREE_FREE_GB Minimum free worktree disk before staging (default: 20; 0 disables)
#   HF_HOME/<token>        HuggingFace token file (REQUIRED for all HF downloads)
set -euo pipefail

check_free_disk_gb() {
  local env_name="$1"
  local label="$2"
  local path="$3"
  local min_gb="$4"

  if ! [[ "$min_gb" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $env_name must be a non-negative integer (got: $min_gb)" >&2
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
ERROR: Not enough free disk for post-training setup.
  Path:     $path ($label)
  Free:     ${available_gb} GB
  Required: ${min_gb} GB

Set OMNI_CACHE_DIR, pass --cache-dir, or use a workspace filesystem with more space.
Set $env_name=0 to skip this preflight check.
EOF
    exit 1
  fi

  echo "Disk check: $label at $path has ${available_gb} GB free (requires ${min_gb} GB)."
}

# ── arg parsing ───────────────────────────────────────────────────────────────
CACHE_DIR_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-dir) CACHE_DIR_ARG="$2"; shift 2 ;;
    -h|--help)   sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REL="$REPO_ROOT/post-training"
if [[ ! -d "$REL" ]]; then
  cat >&2 <<EOF
ERROR: Could not find the post-training release tree at $REL.
Run this script from an omni-dreams checkout or source distribution that keeps
samples/post-training/ and post-training/ in their expected relative locations.
EOF
  exit 1
fi

# ── 0. Resolve CACHE: --cache-dir > $OMNI_CACHE_DIR > $HOME/.cache fallback ───
if [[ -n "$CACHE_DIR_ARG" ]]; then
  CACHE="$CACHE_DIR_ARG"
elif [[ -n "${OMNI_CACHE_DIR:-}" ]]; then
  CACHE="$OMNI_CACHE_DIR"
else
  CACHE="$HOME/.cache/omni-dreams"
  echo "WARNING: OMNI_CACHE_DIR not set; defaulting to $CACHE" >&2
  echo "         For shared/Lustre setups, export OMNI_CACHE_DIR=/path/to/lustre/cache" >&2
fi

echo "Cache dir:         $CACHE"

# ── 1. Cache env (shared with smoke_test.slurm / torchrun_smoke.sh) ──────────
# We need internet on the login node to stage downloads, so flip HF_HUB_OFFLINE
# off for the staging block; _env.sh's default is =1 (compute-node safe).
export OMNI_CACHE_DIR="$CACHE"
export HF_HUB_OFFLINE=0
# _env.sh derives REL from its own path by default; our local $REL matches
# (<repo-root>/post-training) and is exported implicitly via the source.
# shellcheck disable=SC1091  # path resolved at runtime
source "$SCRIPT_DIR/_env.sh"

# ── 2. Disk preflight before large HF downloads ──────────────────────────────
MIN_SETUP_FREE_GB="${OMNI_MIN_SETUP_FREE_GB:-150}"
MIN_WORKTREE_FREE_GB="${OMNI_MIN_WORKTREE_FREE_GB:-20}"
check_free_disk_gb "OMNI_MIN_SETUP_FREE_GB" "cache root" "$OMNI_CACHE_DIR" "$MIN_SETUP_FREE_GB"
check_free_disk_gb "OMNI_MIN_WORKTREE_FREE_GB" "post-training tree" "$REL" "$MIN_WORKTREE_FREE_GB"

# ── 3. HF token pre-check (needed for the two gated NVIDIA models below) ─────
HF_TOKEN_FILE="$HF_HOME/token"
if [[ ! -f "$HF_TOKEN_FILE" ]]; then
  cat >&2 <<EOF
ERROR: HuggingFace token not found at $HF_TOKEN_FILE.
The runtime reads this file directly via huggingface_hub; the HF_TOKEN env var
is NOT a substitute. The two gated NVIDIA models below cannot be downloaded
without it.

Steps:
  1. Get a HF read token: https://huggingface.co/settings/tokens
  2. Accept the NVIDIA Open Model License Agreement on each model page:
       - https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World
       - https://huggingface.co/nvidia/Cosmos-Reason1-7B
  3. Save the token:
       mkdir -p "$HF_HOME"
       printf '%s' '<your-token>' > "$HF_TOKEN_FILE"
EOF
  exit 1
fi

# ── 4. Single-view checkpoints (HuggingFace) ─────────────────────────────────
# These three filenames must match `post-training/omnidreams/checkpoints_omnidreams.py`.
HF_ORG="${OMNI_DREAMS_HF_ORG:-nvidia}"
export OMNI_DREAMS_HF_ORG="$HF_ORG"
HF_CKPT_REPO="$HF_ORG/omni-dreams-models"
HF_CKPT_REVISION="${OMNI_HF_CKPT_REVISION:-main}"
echo "Downloading single-view checkpoints from HuggingFace ($HF_CKPT_REPO@$HF_CKPT_REVISION)..."
for f in \
  single_view/distilled/e5cadda3-8f52-43b2-b621-aa3d4c9f0588_model.pt \
  single_view/teacher/3b4c21d0-7b77-4694-9d9d-6ac9b6dbba51_model.pt \
  single_view/student-init/a12bf26e-8855-4ff0-a651-e7c2a0cb0697_model.pt
do
  uvx --from "huggingface_hub[cli]>=1.3.5" hf download \
    "$HF_CKPT_REPO" --repo-type model --revision "$HF_CKPT_REVISION" "$f"
done

# ── 5. Gated NVIDIA models (always via HF; required for all 3 experiments) ───
# Do NOT pass --cache-dir here. With HF_HOME set, hf download lands files at
# $HF_HOME/hub/models--<org>--<repo>/snapshots/<rev>/...  --cache-dir would
# write to $HF_HOME/models--... (no `hub/`) and the runtime would not find them.
echo "Staging gated NVIDIA models from HuggingFace..."
uvx --from "huggingface_hub[cli]>=1.3.5" hf download \
  nvidia/Cosmos-Predict2-2B-Video2World \
  --repo-type model --revision main tokenizer/tokenizer.pth
uvx --from "huggingface_hub[cli]>=1.3.5" hf download \
  nvidia/Cosmos-Reason1-7B \
  --repo-type model --revision 3210bec0495fdc7a8d3dbb8d58da5711eab4b423

# ── 6. Sample dataset (PAI-NuRec via prepare.py) ──────────────────────────────
# prepare.py owns the dataset logic: snapshot_download from the PAI-NuRec HF
# repo (or fan out OMNI_LOCAL_DATA_SOURCE when set, e.g. an rclone'd S3 copy),
# then map its per-scene layout into the per-camera tree the dataset class
# expects. Repo/revision/subpath come from OMNI_HF_DATA_REPO /
# OMNI_HF_DATA_REVISION / OMNI_HF_DATA_SUBPATH (defaults owned by prepare.py). By
# default it fetches only the per-camera training media (~10 GiB), which
# positively selects the trainable scenes and skips the multi-TB .usdz.
# Idempotent — re-runs replace symlinks only.
# NB: HF_HUB_OFFLINE must still be 0 here (set above); we flip it to 1 *after*
# this block so prepare.py's snapshot_download can hit the network.
DATA_DIR="$REL/data"
if [[ -d "$DATA_DIR/video" && -n "$(ls -A "$DATA_DIR/video" 2>/dev/null)" ]]; then
  echo "Data dir already populated, skipping download."
else
  echo "Staging sample dataset via prepare.py..."
  # huggingface_hub lives in the post-training venv (added by `uv sync`).
  if [[ -x "$REL/.venv/bin/python" ]]; then
    "$REL/.venv/bin/python" "$SCRIPT_DIR/prepare.py" --stage 1 --data-dir "$DATA_DIR"
  else
    echo "ERROR: $REL/.venv/bin/python missing. Run 'uv sync --extra=cu128' from $REL first." >&2
    exit 1
  fi
fi

# Compute nodes have no internet; flip offline mode now that all downloads
# (checkpoints + gated models + sample dataset) are done. smoke_test.slurm and
# torchrun_smoke.sh inherit this from _env.sh's default.
export HF_HUB_OFFLINE=1

# ── 7. Patch venv VIRTUAL_ENV (idempotent) ────────────────────────────────────
# The vendored .venv ships with VIRTUAL_ENV pointing at the original imaginaire4
# build path. Compute nodes may not mount that path, so rewrite to the deployment
# path before any rank sources activate.
VENV="$REL/.venv"
if [[ -f "$VENV/bin/activate" ]]; then
  CURRENT_VENV=$(grep "^VIRTUAL_ENV=" "$VENV/bin/activate" | head -1 | cut -d"'" -f2)
  if [[ "$CURRENT_VENV" != "$VENV" ]]; then
    sed -i "s|VIRTUAL_ENV='.*'|VIRTUAL_ENV='$VENV'|" "$VENV/bin/activate"
    echo "Patched VIRTUAL_ENV in $VENV/bin/activate"
  fi
fi

# ── 8. TE symlinks (idempotent) ───────────────────────────────────────────────
# Re-source _env.sh now that the venv exists so CUDA_HOME / LD_LIBRARY_PATH
# get wired up (skipped earlier on a fresh checkout). Idempotent.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"
for full in "$VENV"/lib/python3.10/site-packages/nvidia/*/lib/lib*.so.*; do
  case "$(basename "$full")" in *.so.*.*) continue ;; esac
  unversioned="${full%.so.*}.so"
  [ -e "$full" ] && [ ! -e "$unversioned" ] && ln -s "$(basename "$full")" "$unversioned"
done

echo ""
echo "Setup complete. Recommended next step (torchrun on a single 8-GPU node):"
echo "  bash samples/post-training/torchrun_smoke.sh 1   # E1 student-init (L2a)"
echo "  bash samples/post-training/torchrun_smoke.sh 2   # E2 teacher (L1b)"
echo "  bash samples/post-training/torchrun_smoke.sh 3   # E3 self-forcing (L0)"
echo "Slurm-managed clusters: sbatch --export=ALL,EXPERIMENT=N samples/post-training/smoke_test.slurm"
