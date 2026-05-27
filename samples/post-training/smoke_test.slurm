#!/bin/bash
#SBATCH --job-name=cosmos-causal-smoke
## REQUIRED: set --account / --partition for your cluster. Either uncomment
## and edit the two lines below, or pass them on the sbatch command line.
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=smoke_%j_%x.out
#SBATCH --error=smoke_%j_%x.err
# Slurm wrapper around torchrun_smoke.sh — same launcher, plus #SBATCH
# directives.
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> \
#     --export=ALL,EXPERIMENT=<1|2|3> samples/post-training/smoke_test.slurm

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"

# Per-job, node-local Triton cache. Each Slurm job gets its own base path;
# the wrapper inside torchrun_smoke.sh further appends _${LOCAL_RANK}.
export TRITON_CACHE_BASE="/tmp/triton_${SLURM_JOB_ID}"

exec bash "$REPO_DIR/samples/post-training/torchrun_smoke.sh" "${EXPERIMENT:-1}"
