# Setup Guide

<!--TOC-->

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Downloading Checkpoints](#downloading-checkpoints)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

<!--TOC-->

## System Requirements

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* NVIDIA driver >=570.124.06 compatible with [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)
* Linux x86-64
* glibc>=2.35 (e.g Ubuntu >=22.04)

## Installation

Install [git lfs](https://git-lfs.com/):

```bash
sudo apt install git-lfs
git lfs install
```

Clone the repository:

```bash
git clone git@github.com:nvidia-cosmos/<repository_name>.git
cd <repository_name>
git lfs pull
```

Install one of the following environments:

<details id="virtual-environment"><summary><b>Virtual Environment</b></summary>

Install system dependencies:

```shell
sudo apt update && sudo apt -y install curl ffmpeg libx11-dev tree wget
```

* [uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install the package into a new environment:

```shell
uv python install
uv sync --extra=cu128
source .venv/bin/activate
```

Or, install the package into the active environment (e.g. conda):

```shell
uv sync --extra=cu128 --active --inexact
```

> **Note**: If you copy or extract a pre-built release to a path different
> from the one it was built at, `.venv/bin/activate` still encodes the
> original `VIRTUAL_ENV` path and `pip` / `uv` will operate against the
> wrong prefix. Patch it after relocation:
>
> ```bash
> sed -i "s|VIRTUAL_ENV='.*'|VIRTUAL_ENV='$(pwd)/.venv'|" .venv/bin/activate
> ```
>
> Editable-install `.pth` files inside `.venv` may also encode the canonical
> (realpath) build location. If `import omnidreams` fails after relocation
> with `ModuleNotFoundError`, prepend the workspace packages to `PYTHONPATH`
> before launching:
>
> ```bash
> export PYTHONPATH="$(pwd)/packages/cosmos-cuda:$(pwd)/packages/cosmos-oss:$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
> ```

CUDA Variants:

| CUDA Version | Arguments | Notes |
| --- | --- | --- |
| CUDA 12.8 | `--extra cu128` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |
| CUDA 13.0 | `--extra cu130` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |

For DGX Spark and Jetson AGX, you must use CUDA 13.0.
</details>

<details id="docker-container"><summary><b>Docker Container</b></summary>

Please make sure you have access to Docker on your machine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

Build the container:

```bash
# Ampere - Hopper
image_tag=$(docker build -f Dockerfile -q .)
# Blackwell
image_tag=$(docker build -f docker/nightly.Dockerfile -q .)
```

Run the container:

```bash
docker run -it --runtime=nvidia --ipc=host --rm -v .:/workspace -v /workspace/.venv -v /root/.cache:/root/.cache -e HF_TOKEN="$HF_TOKEN" $image_tag
```

Optional arguments:

* `--ipc=host`: Use host system's shared memory, since parallel torchrun consumes a large amount of shared memory. If not allowed by security policy, increase `--shm-size` ([documentation](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources)).
* `-v /root/.cache:/root/.cache`: Mount host cache to avoid re-downloading cache entries.
* `-e HF_TOKEN="$HF_TOKEN"`: Set Hugging Face token to avoid re-authenticating.

If you get `docker: Error response from daemon: unknown or invalid runtime name: nvidia`, you need to [configure docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker):

```shell
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

</details>

## Downloading Checkpoints

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission
2. Install [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): `uv tool install -U "huggingface_hub[cli]"`
3. Login: `hf auth login`
4. Accept the [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Guardrail1).

Checkpoints are automatically downloaded during inference and post-training. To modify the checkpoint cache location, set the [`HF_HOME`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) environment variable.

### Gated models

The following models are **gated** — accept the NVIDIA Open Model License
Agreement on each model's HF page before downloading:

- [`nvidia/Cosmos-Predict2-2B-Video2World`](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)
  — tokenizer only (`tokenizer/tokenizer.pth`)
- [`nvidia/Cosmos-Reason1-7B`](https://huggingface.co/nvidia/Cosmos-Reason1-7B)
  — full text-encoder model (≈16 GB)

### Pre-staging on offline/HPC nodes

Compute nodes without internet access cannot trigger HF downloads at job
start. Pre-stage on a login or build node, then export `HF_HUB_OFFLINE=1`
on the compute side:

```bash
# Set HF_HOME, not --cache-dir (files must land in $HF_HOME/hub/).
export HF_HOME=/path/to/hf_cache
hf download nvidia/Cosmos-Predict2-2B-Video2World \
  --repo-type model --revision main tokenizer/tokenizer.pth
# Cosmos-Reason1-7B is pinned to a specific revision tested with this release.
hf download nvidia/Cosmos-Reason1-7B \
  --repo-type model --revision 3210bec0495fdc7a8d3dbb8d58da5711eab4b423

# On compute nodes:
export HF_HUB_OFFLINE=1
```

## Environment Variables

Reference for environment variables that affect installation and training.
Set these in your launch script or Slurm prologue as appropriate.

| Variable | When required | Description |
|---|---|---|
| `HF_HOME` | Always | HuggingFace cache root; downloads land in `$HF_HOME/hub/`. Required to share a checkpoint cache across jobs. |
| `HF_HUB_OFFLINE` | On compute nodes without internet | Set to `1` to prevent runtime download attempts; checkpoints must be pre-staged (see [Downloading Checkpoints](#downloading-checkpoints)). |
| `CUDA_HOME` | When relying on the venv-bundled CUDA | Point at `$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia` so `transformer_engine` finds `libnvrtc`. |
| `LD_LIBRARY_PATH` | When relying on the venv-bundled CUDA | Must include every `nvidia/*/lib` dir from the venv (see Troubleshooting). |
| `TRITON_CACHE_DIR` | On shared filesystems (Lustre / NFS) | Set to a node-local **and per-rank** path, e.g. `/tmp/triton_${SLURM_JOB_ID}_${SLURM_LOCALID:-0}`. Per-job alone is insufficient — the 8 ranks within a single job still share the dir and race on `os.replace()`. |
| `WANDB_MODE` | On clusters without wandb auth | Set to `disabled`. Note: some callbacks call `wandb.login()` directly and ignore this — disabling at the experiment level (`job.wandb_mode="disabled"`) is more reliable. |
| `PYTHONPATH` | When editable-install `.pth` files encode the wrong (realpath) build path | Prepend `$REL/packages/cosmos-cuda:$REL/packages/cosmos-oss:$REL`. |
| `IMAGINAIRE_OUTPUT_ROOT` | Optional | Where training outputs are written. |
| `IMAGINAIRE_CACHE_DIR` | Optional | imaginaire4 internal cache. |

## Troubleshooting

These errors surface when running `python scripts/check_environment.py` on a host without CUDA installed system-wide (i.e. relying on the pip-installed `nvidia-*-cu12` wheels that PyTorch pulls in).

### `ldconfig -p | grep 'libnvrtc'` returns non-zero

```
File ".../transformer_engine/common/__init__.py", line 111, in _load_nvrtc
    libs = subprocess.check_output("ldconfig -p | grep 'libnvrtc'", shell=True)
subprocess.CalledProcessError: Command 'ldconfig -p | grep 'libnvrtc'' returned non-zero exit status 1.
```

`transformer_engine._load_nvrtc()` first globs `$CUDA_HOME/**/libnvrtc.so*` and only falls back to `ldconfig` if `CUDA_HOME` is unset. Point `CUDA_HOME` at the venv's bundled CUDA libs:

```bash
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia"
```

### `libcublas.so.12: cannot open shared object file`

```
OSError: libcublas.so.12: cannot open shared object file: No such file or directory
```

The `transformer_engine` native `.so` depends on `libcublas`, `libcudnn`, `libnvJitLink`, etc. at `dlopen()` time. Add every `nvidia/*/lib` directory in the venv to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$(find $VIRTUAL_ENV/lib/python3.10/site-packages/nvidia -maxdepth 3 -type d -name lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### `OSError: [Errno 39] Directory not empty` from `.triton/cache`

```
OSError: [Errno 39] Directory not empty:
  '/path/to/.triton/cache/tmp_abc123' -> '/path/to/.triton/cache/sha256_xyz'
```

When all ranks on a node compile identical Triton kernels concurrently,
they race on `os.replace(tmp_dir, hash_dir)`. On Lustre / NFS this
operation is not atomic when the destination directory already exists,
and one or more ranks fail. Direct the Triton cache to a node-local path
that is also **per-rank** — per-job alone is insufficient because the
8 ranks within a single job still share the dir and still race:

```bash
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID:-$$}_${SLURM_LOCALID:-0}"
```

### `GLIBC_2.34' not found` from `~/.triton/cache/.../cuda_utils.so`

```
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found
  (required by /home/<user>/.triton/cache/.../cuda_utils.so)
```

`~/.triton/cache` contains a `cuda_utils.so` compiled against a newer glibc (e.g. from a previous run inside a container with Ubuntu 22+). On a host with older glibc the cached object is ABI-incompatible. Triton regenerates the cache on demand, so the fix is to drop it:

```bash
rm -rf ~/.triton
# Optional: redirect future Triton compilation cache off $HOME
export TRITON_CACHE_DIR=/path/to/triton_cache
```

Note: this host must still meet the glibc requirement in [System Requirements](#system-requirements) for the recompiled cache to load.
