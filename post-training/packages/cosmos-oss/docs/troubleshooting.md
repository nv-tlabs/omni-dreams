# Troubleshooting

## Issues

Also, check GitHub Issues for the [repository](https://github.com/orgs/nvidia-cosmos/repositories).

### Changing the cache directory
We download packages and checkpoints to the home directory by default. To change this, you can set the following variables.
This is also needed if you get `No space left on device` errors when setting up or downloading checkpoints.
```shell
export UV_CACHE_DIR=<new_dir>/.cache
export PIP_CACHE_DIR=<new_dir>/.cache
export HF_HOME=<new_dir>/checkpoints
```

### Missing Python.h

Error message: `fatal error: Python.h: No such file or directory`

This is fixed by [installing uv managed python](https://docs.astral.sh/uv/guides/install-python/#installing-python):

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install --reinstall
```

Re-install the package:

```shell
rm -rf .venv
uv sync --extra=cu128
```

### CUDA driver version insufficient

**Fix:** Update NVIDIA drivers to latest version compatible with CUDA [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)

Check driver compatibility:

```shell
nvidia-smi | grep "CUDA Version:"
```

### `libcudart.so: cannot open shared object file` from Transformer Engine

```
RuntimeError: Unable to dlopen libcudart.so: libcudart.so: cannot open shared object file: No such file or directory
```

Transformer Engine `dlopen()`s the **unversioned** soname (`libcudart.so`), but the pip-installed `nvidia-*-cu12` wheels only ship the **versioned** file (`libcudart.so.12`). Setting `LD_LIBRARY_PATH` does not help — `dlopen` matches the exact filename and won't auto-resolve `.so` to `.so.N`. Same applies to `libnvrtc.so`, `libcublas.so`, `libcudnn.so`, etc.

**Fix:** create the missing unversioned symlinks once after install. Idempotent and safe to re-run after recreating the venv:

```bash
for full in "$VIRTUAL_ENV"/lib/python3.10/site-packages/nvidia/*/lib/lib*.so.*; do
  case "$(basename "$full")" in *.so.*.*) continue ;; esac  # skip libfoo.so.X.Y
  unversioned="${full%.so.*}.so"
  [ -e "$full" ] && [ ! -e "$unversioned" ] && ln -s "$(basename "$full")" "$unversioned"
done
```

Hosts with system-wide CUDA installed (e.g. inside `nvcr.io/nvidia/pytorch` containers) won't hit this — the symlinks already exist under `/usr/local/cuda/lib64`.

### Triton cache corruption on shared filesystems

```
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found
  (required by /<shared_cache>/triton/.../__triton_launcher.so)
```

Triton keys its compilation cache by **source hash only**, not by host glibc/CPU. If `TRITON_CACHE_DIR` (or `XDG_CACHE_HOME`) points at shared storage in a heterogeneous cluster — nodes with different OS/glibc versions — a `.so` JIT'd on one node will be reused on another and fail to load.

**Fix:** namespace the Triton cache per host:

```bash
export TRITON_CACHE_DIR=<shared_path>/triton/$(hostname -s)
```

Then wipe the contaminated entries:

```bash
rm -rf <shared_path>/triton
```

Each node will rebuild its own cache on first run. Local (per-node) cache dirs don't need this.

### PYTHONPATH conflicts in NVIDIA containers

When using `nvcr.io/nvidia/pytorch:25.xx-py3` containers, you will need to unset `PYTHONPATH` to be compatible with Python 3.10:

```shell
unset PYTHONPATH
```

### Out of Memory (OOM) errors

**Fix:** Use 2B models instead of 14B, multi-GPU, or reduce batch size/resolution

## Guide

### Logs

Logs are saved to `<output_dir>/*.log`.

### Profiling

To profile, pass the `--profile` flag. A [pyinstrument](https://pyinstrument.readthedocs.io/en/latest/guide.html) profile will be exported to `<output_dir>/profile.pyisession`.

View the profile:

```shell
pyinstrument --load=<output_dir>/profile.pyisession
```

Export the profile:

```shell
pyinstrument --load=<output_dir>/profile.pyisession -r html -o <output_dir>/profile.html
```

See [pyinstrument](https://pyinstrument.readthedocs.io/en/latest/guide.html).
