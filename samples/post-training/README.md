# post-training (sample)

Post-training fine-tune of the Cosmos2 single-view HDMap world model on the
[PAI-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)
dataset, composed against the `post-training/` release tree.

> **New here?** Read [QUICKSTART.md](./QUICKSTART.md) for a zero-to-training
> walkthrough. This README is the orientation map; QUICKSTART is the recipe.
> The supported minimum post-training launch uses 8 GPUs (`NPROC=8` on a
> single node). Smaller `NPROC` values are unsupported and the launcher fails
> early before starting `torchrun`.
> Plan for at least 150 GB of free disk, with 200 GB or more recommended.
> `setup_env.sh` and `torchrun_smoke.sh` run disk preflights before the
> expensive setup and launch phases.

## Workflow

```text
   ┌──────────────────┐
   │ HF token + EULA  │  — accept gated NVIDIA model licenses
   │  acceptance      │     and drop token at $HF_HOME/token
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │ setup_env.sh     │  — stages HF checkpoints, gated models,
   │                  │     sample dataset, patches venv path, TE symlinks
   └────────┬─────────┘
            ▼
   ┌──────────────────┐    ┌──────────────────┐
   │ torchrun_smoke.sh│ OR │ smoke_test.slurm │
   │ (primary path —  │    │ (Slurm wrapper —  │
   │ single compute   │    │ exec's the same   │
   │ node)            │    │ torchrun launcher)│
   └──────────────────┘    └──────────────────┘
```

## Setup

### 1. Resolve the venv

```bash
cd ../../post-training
uv sync --extra=cu128
```

### 2. Drop a HuggingFace token

The runtime reads the token from a file (not the `HF_TOKEN` env var). Get a
read token at <https://huggingface.co/settings/tokens> and accept the NVIDIA
Open Model License Agreement on each gated model page (see
[QUICKSTART.md](./QUICKSTART.md#3-stage-checkpoints--dataset)).

```bash
export HF_HOME=/path/to/cache/huggingface
mkdir -p "$HF_HOME"
printf '%s' '<your-hf-token>' > "$HF_HOME/token"
```

### 3. Run setup, then source the shared env

This script downloads checkpoints and example data from Hugging Face to your
local device. Checkpoints are downloaded to `$HOME/.cache/omni-dreams` by
default, and data is downloaded to `$REPO_ROOT/post-training/data`. Use
`OMNI_CACHE_DIR` or `--cache-dir` to place caches on a filesystem with enough
space.

```bash
bash   samples/post-training/setup_env.sh   # downloads / dataset / TE symlinks
source samples/post-training/_env.sh        # populate your shell's env
```

`setup_env.sh` requires 150 GB free in the cache and 20 GB free in the
worktree by default before downloading checkpoints and datasets. Override with
`OMNI_MIN_SETUP_FREE_GB` / `OMNI_MIN_WORKTREE_FREE_GB` only if you know your
cache is already populated or you have a smaller validated workflow.

`setup_env.sh` runs in a subshell, so the env vars `_env.sh` exports
(`CUDA_HOME`, `LD_LIBRARY_PATH`, `HF_HOME`, `TMPDIR`, ...) die with it.
`torchrun_smoke.sh` and `smoke_test.slurm` re-source `_env.sh` internally
so the launcher path is self-contained — but for ad-hoc `python -m
scripts.train` invocations, REPL debugging, or any fresh login after
`uv sync`, source `_env.sh` directly. It's idempotent and side-effect-free
beyond `mkdir -p`.

Defaults (override by exporting before the source / setup call):

| Var              | Default                                | |
|------------------|----------------------------------------|---|
| `OMNI_CACHE_DIR` | `$HOME/.cache/omni-dreams` (warns)     | Cache root for HF, uv, pip, xdg, imaginaire, triton. Use a Lustre/shared path on clusters. |
| `REL`            | `<repo-root>/post-training` (auto)     | Derived from `_env.sh`'s own path; override only if you're pointing at a different release tree. |
| `OMNI_MIN_SETUP_FREE_GB` | `150`                          | Minimum free cache disk required by `setup_env.sh`; set `0` to skip. |
| `OMNI_MIN_WORKTREE_FREE_GB` | `20`                       | Minimum free worktree disk required by `setup_env.sh`; set `0` to skip. |
| `OMNI_MIN_TRAIN_FREE_GB` | `20`                           | Minimum free disk required by `torchrun_smoke.sh`; set `0` to skip. |

`setup_env.sh` also accepts `--cache-dir DIR` as a one-shot equivalent to
exporting `OMNI_CACHE_DIR`. See the script header for the rest
(`OMNI_DREAMS_HF_ORG`, `OMNI_HF_CKPT_REVISION`, `OMNI_HF_DATA_SUBPATH`, ...).

### 4. Smoke test

```bash
uv run pytest tests/test_imports_clean.py
```

## Launch training

Three smoke experiments. The torchrun invocations are defined once in
`torchrun_smoke.sh`; `smoke_test.slurm` `exec`s into it after applying
its `#SBATCH` directives.

| ID | Name                         | Config                                          |
|----|------------------------------|-------------------------------------------------|
| 1  | Mid-training student-init (L2a) | `causal_cosmos2/config.py` + chunk2 hdmap exp |
| 2  | Mid-training teacher (L1b)   | `causal_cosmos2/config.py` + teacher exp        |
| 3  | Self-forcing distillation (L0) | `self_forcing/config.py` + release exp        |

```bash
# Direct torchrun on a single rented compute node (Lepton, Lambda, any cloud box).
bash samples/post-training/torchrun_smoke.sh 1
for e in 1 2 3; do bash samples/post-training/torchrun_smoke.sh $e; done

# Slurm cluster. --export=ALL forwards OMNI_CACHE_DIR + HF_HOME state.
# -A / -p are required (the shipped #SBATCH placeholders are commented).
sbatch -A <account> -p <partition> --export=ALL,EXPERIMENT=1 \
  samples/post-training/smoke_test.slurm
```

The Slurm header in `smoke_test.slurm` is 1 node × 8 GPUs × 1 h with
commented placeholders for `--account` and `--partition`. Either pass them
on the `sbatch` command line, or uncomment and edit the `##SBATCH` lines at
the top of the file.

## Required env on compute nodes

Set in `smoke_test.slurm`; documented here so torchrun-only users get them too.

| Var                      | Value                              | Why |
|--------------------------|------------------------------------|-----|
| `HF_HOME`                | `$CACHE/huggingface`               | Where `setup_env.sh` staged models. |
| `HF_HUB_OFFLINE`         | `1`                                | Compute nodes have no internet; force runtime to use local snapshots. |
| `TRITON_CACHE_BASE`      | `/tmp/triton_${SLURM_JOB_ID}` (Slurm) / `/tmp/triton_$$` (torchrun) | Caller-provided base; `triton_per_rank_wrap.sh` appends `_${LOCAL_RANK}` per rank so no two ranks share a hash dir. |
| `WANDB_MODE`             | `disabled`                         | Self-forcing uses wandb; without this, E3 fails with "No API key configured". |
| `IMAGINAIRE_OUTPUT_ROOT` | `$CACHE/imaginaire4-output`        | Run artifacts. |
| `CUDA_HOME`              | `$VENV/lib/python3.10/site-packages/nvidia` | transformer_engine `_load_nvrtc()` glob root. |
| `LD_LIBRARY_PATH`        | union of `nvidia/*/lib`            | `dlopen()` of `libcublas`, `libcudnn`, etc. |
| `PYTHONPATH`             | `$REL/packages/cosmos-cuda:$REL/packages/cosmos-oss:$REL` | Symlink-stable; **must not** include venv site-packages (uvx subprocess inheritance — see comment in `smoke_test.slurm`). |

## Layout

- `_env.sh` — shared cache + runtime env exports. Sourced by `setup_env.sh`,
  `smoke_test.slurm`, and `torchrun_smoke.sh` so cache paths never drift
  between login-node setup and compute-node launch.
- `setup_env.sh` — one-shot stager: cache env, HF checkpoints, gated models,
  sample dataset (via `prepare.py`), venv path patch, TE symlinks. Sources
  `_env.sh` for cache paths.
- `torchrun_smoke.sh` — primary launcher. Runs torchrun on a single 8-GPU node
  (no Slurm). Defines the three experiment invocations exactly once.
- `smoke_test.slurm` — Slurm wrapper. Applies `#SBATCH` directives, scopes
  `TRITON_CACHE_BASE` to `/tmp/triton_$JOB_ID`, then `exec`s `torchrun_smoke.sh`.
- `triton_per_rank_wrap.sh` — torchrun `--no-python` shim that appends
  `_${LOCAL_RANK}` to `TRITON_CACHE_BASE` so 8 ranks never share a hash dir.
- `prepare.py` — `snapshot_download`s the HF sample dataset and symlinks its
  per-scene files into the per-camera layout the dataloader expects. Invoked
  by `setup_env.sh`.
- `configs/` — sample-side experiment + dataset composition (import-and-override
  on top of the vendored tree).
- `tests/` — `test_imports_clean.py` is the redaction smoke (CPU-only, runs
  on every PR; catches a vendored-tree regression that re-introduces
  internal strings, and override mistakes that hardcode them).
- `SKILL.md` — agent-runnable recipe for running E1/E2/E3 at 8 → 256 H100,
  with the FSDP × CP scaling matrix.

## Dataset organization
Omni-dreams post-training expects the following directory structure when loading data. If using our sample data provided through HuggingFace, then `prepare.py` (invoked by `setup_env.sh`) creates this layout by symlinking files from
the HuggingFace snapshot download staging area into the per-camera directories below. Only `camera_front_wide_120fov` is required for single view training:

```text
post-training/data/
├── caption/
│   ├── camera_cross_left_120fov/
│   │   └── <scene-uuid>.txt
│   ├── camera_cross_right_120fov/
│   │   └── <scene-uuid>.txt
│   ├── camera_front_tele_30fov/
│   │   └── <scene-uuid>.txt
│   └── camera_front_wide_120fov/
│       └── <scene-uuid>.txt
├── hdmap/
│   ├── camera_cross_left_120fov/
│   │   └── <scene-uuid>.mp4
│   ├── camera_cross_right_120fov/
│   │   └── <scene-uuid>.mp4
│   ├── camera_front_tele_30fov/
│   │   └── <scene-uuid>.mp4
│   └── camera_front_wide_120fov/
│       └── <scene-uuid>.mp4
└── video/
    ├── camera_cross_left_120fov/
    │   └── <scene-uuid>.mp4
    ├── camera_cross_right_120fov/
    │   └── <scene-uuid>.mp4
    ├── camera_front_tele_30fov/
    │   └── <scene-uuid>.mp4
    └── camera_front_wide_120fov/
        └── <scene-uuid>.mp4
```

Location of the dataset can be configured during training by setting `dataloader_train.data_root=${YOUR_DATA_PATH}`.
Each `mp4` in `video/` and `hdmap/` should be temporally aligned and at least 93 frames long. Each `txt` in `caption/` should be a plain text file containing the caption of each video clip. 

## Troubleshooting

`setup_env.sh` exits with a clear message if the HF token / cache-dir
preconditions aren't met. For runtime errors after launch, see
[QUICKSTART.md § Troubleshooting](./QUICKSTART.md#troubleshooting) and the
upstream [setup.md](../../post-training/docs/setup.md).
