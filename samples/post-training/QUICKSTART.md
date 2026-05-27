# Quickstart: post-training Cosmos2 single-view HDMap

End-to-end on a single 8-GPU node in roughly 30 minutes after the initial
downloads (through the first training step). Designed for a researcher with a
rented Hopper or A100 box.

## 1. Prerequisites

* Supported minimum: 8× Ampere/Hopper NVIDIA GPU. Driver ≥ 570.124.06
  (CUDA 12.8.1 compatible). The flow has been tested on 8x H100 80 GB HBM3
  with driver 570.148.08 and CUDA 12.8. Smaller `NPROC` values are not
  supported by the released configs.
* At least 150 GB of free disk for dependencies, Hugging Face caches, dataset
  staging, Triton caches, and training output. 200 GB or more is recommended.
  `setup_env.sh` enforces a 150 GB cache preflight and a 20 GB worktree
  preflight by default. `torchrun_smoke.sh` enforces a 20 GB launch-time
  preflight for output, temporary files, and Triton caches.
* Linux x86-64, glibc ≥ 2.35 (Ubuntu 22.04+).
* `git` and `uv`:
  ```bash
  sudo apt install -y git
  curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
  ```
* Hugging Face account with the **NVIDIA Open Model License** accepted on:
  * [`nvidia/Cosmos-Predict2-2B-Video2World`](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)
  * [`nvidia/Cosmos-Reason1-7B`](https://huggingface.co/nvidia/Cosmos-Reason1-7B)
  * [`nvidia/omni-dreams-models`](https://huggingface.co/nvidia/omni-dreams-models)
  * [`nvidia/omni-dreams-scenes`](https://huggingface.co/datasets/nvidia/omni-dreams-scenes)
    (sample dataset; <200 clips today — the launcher passes
    `dataloader_train.repeat_factor=200` so all three experiments train
    past their single-epoch ceiling)

## 2. Install

Set caches **before** `uv sync` — uv builds flash-attn and writes to
`UV_CACHE_DIR` (which defaults to `$HOME/.cache/uv`). On HPC head nodes
with a small `$HOME` quota the extraction can fail with "No space left on
device"; keep the cache on a filesystem with at least 150 GB free.

```bash
# Clone the repo URL your NVIDIA contact provided.
git clone <repo-url>
cd <repo-dir>
git checkout <integration-branch>      # e.g. dev/jmccaffrey/post-training-integ

# Point caches at a writable filesystem with at least 150 GB free.
export OMNI_CACHE_DIR=$HOME/.cache/omni-dreams
export UV_CACHE_DIR=$OMNI_CACHE_DIR/uv
export TMPDIR=$OMNI_CACHE_DIR/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

(cd post-training && uv sync --extra=cu128)
```

To validate the lockfile without downloading packages or creating a venv, run:

```bash
(cd post-training && uv lock --check --offline)
(cd post-training && uv sync --extra=cu128 --locked --dry-run --offline)
```

If the cache directory is not writable, set `UV_CACHE_DIR` to a writable path
for the validation command.

## 3. Stage checkpoints + dataset

The runtime reads your HF token from a **file** — `HF_TOKEN` is *not* a
substitute and will be silently ignored.

```bash
mkdir -p "$OMNI_CACHE_DIR/huggingface"
printf '%s' '<your-hf-token>' > "$OMNI_CACHE_DIR/huggingface/token"
chmod 600 "$OMNI_CACHE_DIR/huggingface/token"

# Optional: set this if your assigned repos live under another authorized org.
# export OMNI_DREAMS_HF_ORG=<your-hf-org>

bash samples/post-training/setup_env.sh
```

If you are using another authorized Hugging Face org, export
`OMNI_DREAMS_HF_ORG=<YOUR-HF-ORG>` before running setup and launch commands.

`setup_env.sh` is idempotent. It downloads checkpoints + gated NVIDIA models
into `$OMNI_CACHE_DIR/huggingface/hub/`, populates the sample dataset under
`post-training/data/{video,hdmap,caption}/<camera>/` (by invoking `prepare.py`,
which `snapshot_download`s the HF dataset and symlinks its per-scene layout
into the per-camera layout the dataloader expects), and patches the venv's
`VIRTUAL_ENV` path. Re-run it after a re-vendor or a cache wipe.

Then source `_env.sh` into your shell so subsequent ad-hoc commands inherit
the right `CUDA_HOME`, `LD_LIBRARY_PATH`, `HF_HOME`, `TMPDIR`, etc.:

```bash
source samples/post-training/_env.sh
```

`setup_env.sh` runs in a subshell (`bash`), so the env vars it exports via
`_env.sh` die when it returns — the launchers (`torchrun_smoke.sh`,
`smoke_test.slurm`) re-source `_env.sh` themselves so they're
self-contained, but any direct `python -m scripts.train` invocation, REPL
debugging, or fresh-login session needs the explicit `source`.

`_env.sh` is idempotent and side-effect-free (no `set -e`, no `exit`, no
positional-arg parsing — only `mkdir -p`), so re-sourcing is cheap.

The sample dataset is intentionally small (<200 clips). The launcher
(`torchrun_smoke.sh` / `smoke_test.slurm`) passes
`dataloader_train.repeat_factor=200` as a Hydra CLI override on every
invocation so all three experiments have enough effective samples — without
it E3 self-forcing exhausts at iter ~30. Override the source repo / subpath
with `OMNI_DREAMS_HF_ORG` / `OMNI_HF_DATA_SUBPATH` to point at another
authorized OmniDreams org or dataset slice. Use `prepare.py --repo` only for a
fully custom dataset repo; the layout requirements (per-scene `<uuid>.<camera>_rgb.mp4`,
`_hdmap.mp4`, `.prompt.txt`) are documented at the top of `prepare.py`.

## 4. Train

**HPC / Slurm cluster?** Don't run `torchrun_smoke.sh` directly on a head /
login node — it'll fail with a CUDA error because the head has no GPUs.
Use the Slurm path further down. Before the first `sbatch`, edit
`samples/post-training/smoke_test.slurm` to set the right
`#SBATCH --account=…` and `#SBATCH --partition=…` for your cluster (the
shipped values are placeholders for the maintainer's cluster).

### Direct torchrun (single rented 8-GPU node, no Slurm)

```bash
bash samples/post-training/torchrun_smoke.sh 1   # E1 student-init  (L2a)
bash samples/post-training/torchrun_smoke.sh 2   # E2 teacher       (L1b)
bash samples/post-training/torchrun_smoke.sh 3   # E3 self-forcing  (L0)
```

Iter 1 is ~85–90s on Hopper (checkpoint load + torch.compile graph build).
Iter 2+ settle to ~10s for E1/E2 and ~30s for E3. With `repeat_factor=200`
on the <200-clip sample, all three experiments train past their acceptance
threshold without dataloader exhaustion; E1/E2 run to `max_iter` (10000+),
E3 runs to `max_iter=10000`.

### Slurm

```bash
# -A / -p are required; the shipped #SBATCH lines are commented placeholders.
sbatch -A <account> -p <partition> \
  --export=ALL,EXPERIMENT=1 samples/post-training/smoke_test.slurm
```

You can also bake `--account` / `--partition` into `smoke_test.slurm` once
by uncommenting and editing the two `##SBATCH --account=…` /
`##SBATCH --partition=…` lines at the top.

### Minimum launch shape

The supported minimum is 8 total GPUs. On a single-node box, use the default
launcher shape:

```bash
NPROC=8 bash samples/post-training/torchrun_smoke.sh 1
```

`NPROC<8` is not supported and is rejected before `torchrun` starts.

Trailing positional args after the experiment number are forwarded to
`scripts.train` verbatim; Hydra takes the *last* value when a key is
repeated, so your override wins over the launcher's `=8` defaults.

---

## Reference

### Files in this sample

| File | Role |
|------|------|
| `_env.sh`                | Cache + runtime env (sourced by setup + launcher). |
| `setup_env.sh`           | One-shot stager: checkpoints, dataset, venv path patch. |
| `torchrun_smoke.sh`      | Primary launcher (the three torchrun invocations live here). |
| `smoke_test.slurm`       | Slurm wrapper — `#SBATCH` directives + `exec` into the launcher. |
| `triton_per_rank_wrap.sh`| `torchrun --no-python` shim that scopes `TRITON_CACHE_DIR` per `LOCAL_RANK`. |
| `configs/`, `prepare.py` | Sample-side config overrides + PAI-NuRec stage helper. |

### Environment variables

Set `OMNI_CACHE_DIR` once; everything else derives from it via `_env.sh`.

| Var                      | Default / guidance                         | Notes |
|--------------------------|---------------------------------------------|-------|
| `OMNI_CACHE_DIR`         | `$HOME/.cache/omni-dreams` (warns)          | Cache root. Use a Lustre/shared path on clusters. |
| `REL`                    | `<repo-root>/post-training` (auto)          | Release tree. Derived from `_env.sh`'s own path; override only to point at a different tree. |
| `HF_HOME`                | `$OMNI_CACHE_DIR/huggingface`               | Token at `$HF_HOME/token` (not `HF_TOKEN`). |
| `HF_HUB_OFFLINE`         | `1`                                         | `setup_env.sh` flips to `0` while staging. |
| `IMAGINAIRE_OUTPUT_ROOT` | `$OMNI_CACHE_DIR/imaginaire4-output`        | Run artifacts. |
| `WANDB_MODE`             | `disabled`                                  | E3 wires up `wandb_dmd`; without `disabled` it errors. |
| `TRITON_CACHE_BASE`      | per-host on Lustre; `/tmp/triton_$JOB_ID` under Slurm | Wrapper appends `_${LOCAL_RANK}`. Override to `/dev/shm/triton_$$` on containerized envs without writable `/tmp`. |
| `NPROC`, `MASTER_PORT`   | `8`, `12341`                                | torchrun knobs. `NPROC<8` is not supported. |
| `OMNI_MIN_SETUP_FREE_GB` | `150`                                       | Minimum free cache disk required by `setup_env.sh` before large downloads. Set `0` to skip. |
| `OMNI_MIN_WORKTREE_FREE_GB` | `20`                                    | Minimum free worktree disk required by `setup_env.sh` before dataset staging. Set `0` to skip. |
| `OMNI_MIN_TRAIN_FREE_GB` | `20`                                        | Minimum free disk required by `torchrun_smoke.sh` before launch. Set `0` to skip. |
| `OMNI_DREAMS_HF_ORG`     | `nvidia`                                    | HF org used for `omni-dreams-models` and `omni-dreams-scenes`. Export before setup and launch if using another authorized org. |
| `OMNI_HF_CKPT_REVISION`  | `main`                                      | Pin a release commit SHA once published. |
| `OMNI_HF_DATA_SUBPATH`   | `PAI-900_intersect_PAI-300k`                | Subdir within the dataset repo. |

`CUDA_HOME`, `LD_LIBRARY_PATH`, `PYTHONPATH` are set automatically by
`_env.sh` + the launcher; you don't normally touch them.

---

## Troubleshooting

### `huggingface_hub.utils.GatedRepoError`

You haven't accepted the NVIDIA Open Model License on one of the gated repos.
Visit each model page in [§1](#1-prerequisites) and click "Agree and access repository".

### `ERROR: HuggingFace token not found at $HF_HOME/token`

`setup_env.sh`'s pre-check. The runtime reads the token *file*, not `HF_TOKEN`.

### `ERROR: Not enough free disk`

`setup_env.sh` checks `OMNI_CACHE_DIR` and the `post-training` tree before
large downloads and dataset staging. It requires 150 GB free for the cache and
20 GB free for the worktree by default. `torchrun_smoke.sh` checks
`IMAGINAIRE_OUTPUT_ROOT`, `TMPDIR`, and the Triton cache parent before launch;
it requires 20 GB free by default.

Move caches or output paths to a larger filesystem:

```bash
export OMNI_CACHE_DIR=/path/with/space/omni-dreams
export TMPDIR=/path/with/space/tmp
export TRITON_CACHE_BASE=/path/with/space/triton/$(hostname -s)
```

If you have already staged assets and intentionally want to bypass the checks,
set `OMNI_MIN_SETUP_FREE_GB=0`, `OMNI_MIN_WORKTREE_FREE_GB=0`, or
`OMNI_MIN_TRAIN_FREE_GB=0`.

### `uv.lock` parse failures

If `uv sync --extra=cu128` fails before installing packages with an error such
as `Dependency 'rich' has missing 'source' field`, the checked-in lockfile was
generated by an incompatible `uv` version. First validate the lockfile:

```bash
cd post-training
uv lock --check --offline
uv sync --extra=cu128 --locked --dry-run --offline
```

If validation fails, regenerate the lockfile with the documented `uv` version
and commit the updated `uv.lock`. As a local workaround only, remove
`uv.lock` and run `uv lock` before retrying `uv sync`.

### `world_size (...) is not divisible by ...`

The default launcher uses `NPROC=8` and sets the E1/E2 FSDP and context
parallel sizes to 8. `NPROC<8` is not supported and is rejected before
`torchrun` starts. For larger launch shapes, pass parallelism overrides that
divide the selected world size.

### `RuntimeError: Unsupported Python version: 3.9.23`

A `uvx` subprocess (e.g. `hf download` from `checkpoint_db.py`) inherited a
`PYTHONPATH` that includes the venv `site-packages` built for Python 3.10 but
landed on a node with system Python 3.9. The launchers strip site-packages
from `PYTHONPATH` already; only an issue if you build a custom path.

### `OSError: [Errno 39] Directory not empty: 'tmp_...' -> 'hash_...'`

Two ranks racing to write the same Triton cache file (Lustre exposes a race
that `/tmp` masks). The `triton_per_rank_wrap.sh` wrapper used by both
launchers prevents this by scoping cache dirs per `LOCAL_RANK`. If you see
the error, you've bypassed the wrapper — route torchrun through it:
```bash
torchrun --nproc_per_node=8 --no-python \
  samples/post-training/triton_per_rank_wrap.sh \
  python -m scripts.train ...
```

### `wandb.errors.UsageError: api_key not configured`

E3 wires up `wandb_dmd`. The launchers set `WANDB_MODE=disabled` for you;
only an issue if you override.

### `RuntimeError: Unable to dlopen libcudart.so` / `OSError: libcublas.so.12: cannot open shared object file`

`transformer_engine` `dlopen`s CUDA libs from the venv's `nvidia/*/lib`
directories. `_env.sh` wires `CUDA_HOME` + `LD_LIBRARY_PATH` to those paths
once the venv exists, so this only bites if you skipped `setup_env.sh` or
you're invoking `scripts.train` outside the launcher. Re-run `setup_env.sh`,
or export both yourself:

```bash
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find $VIRTUAL_ENV/lib/python3.10/site-packages/nvidia -maxdepth 3 -type d -name lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### `NCCL WARN NET/FasTrak ... Plug-in core initialization failed, aborting` / `No NCCL_TUNER_CONFIG_PATH provided`

NCCL auto-loads system-installed plugins (`libnccl-net*.so`,
`libnccl-tuner.so`) from standard library paths at init. On GCP A3 H100
images both are present but expect a fully-provisioned TCPDirect setup
(ctrl iface named `eth0`, `NCCL_TUNER_CONFIG_PATH` set). On partially-
configured hosts the FasTrak plugin calls `abort()` instead of falling back
to sockets — the rank takes `SIGABRT` inside the first collective
(typically `dist.barrier()` in `trainer.__init__`). Disable both plugins so
NCCL uses its built-in TCP transport, which is fine for single-node 8-GPU:

```bash
export NCCL_NET_PLUGIN=none
export NCCL_TUNER_PLUGIN=none
```

### `ImportError: ... GLIBC_2.34' not found ... .triton/cache/.../cuda_utils.so`

Stale Triton cache compiled against a newer glibc. Drop it:
```bash
rm -rf ~/.triton "$OMNI_CACHE_DIR/triton/"*
```
The host must still meet the glibc requirement in [§1](#1-prerequisites).

---

* [README.md](./README.md) — integration overview and file layout.
* Upstream [setup.md](../../post-training/docs/setup.md) — vendored-tree reference.
