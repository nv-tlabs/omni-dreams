# SKILL.md — running the three post-training experiments

You are a test agent. Your job is to (a) bring up the post-training environment
on a Linux x86-64 box with NVIDIA H100s, (b) run each of the three release
experiments at the GPU count provided to you, and (c) report which iters each
experiment reached and whether the loss stayed finite. Do not attempt to fix
upstream bugs — surface them.

## Authoritative docs (read these first)

* [`samples/post-training/QUICKSTART.md`](./QUICKSTART.md) — the four-step
  out-of-box flow (install, set `OMNI_CACHE_DIR` + HF token, `setup_env.sh`,
  `torchrun_smoke.sh`). **The canonical dataset path lives here**; check the
  Dataset section before staging — that path is being relocated.
* [`samples/post-training/README.md`](./README.md) — file-layout reference
  for this sample's wrappers (`_env.sh`, `setup_env.sh`, `torchrun_smoke.sh`,
  `smoke_test.slurm`, `triton_per_rank_wrap.sh`).
* [`post-training/docs/setup.md`](../../post-training/docs/setup.md) — release
  tree's system requirements, env-variable reference, and troubleshooting.
  When something fails at the framework level, this is where the answer is.
* [`post-training/README.md`](../../post-training/README.md) — release tree
  overview.

The release tree itself lives at `./post-training/`. It is **never edited in
place** — every adaptation goes through this `samples/post-training/`
directory.

## Hardware

* H100 (Hopper) recommended. Ampere (A100) is supported but slower; the
  configs assume a Hopper-class memory profile.
* The supported minimum post-training launch is 8 GPUs. Do not run or report
  smaller launch shapes unless the project owner explicitly requests an
  unsupported experiment.
* Driver ≥ 570.124.06 (CUDA 12.8.1) for the `cu128` extra; ≥ 580 for `cu130`.
* Linux x86-64, glibc ≥ 2.35.

You may be allocated anywhere from 8 to 256 GPUs. **Match the total launch
size to `NPROC × NNODES` and to the FSDP/CP overrides** (table below).

## Step 0 — environment

Follow [QUICKSTART §1–§3](./QUICKSTART.md#1-prerequisites). The order matters
on HPC head nodes with small `$HOME` quotas: **set `OMNI_CACHE_DIR` (and
`UV_CACHE_DIR`) before `uv sync`**, or uv will land flash-attn extraction in
`$HOME/.cache/uv` and may fail with `No space left on device`.

```bash
# 1. Clone the repo URL your NVIDIA contact gave you, then checkout the
#    integration branch (e.g. dev/jmccaffrey/post-training-integ).
git clone <repo-url>; cd <repo-dir>
git checkout <integration-branch>

# 2. Set caches FIRST. OMNI_CACHE_DIR must be on a filesystem with ~50 GB
#    free (HF models + uv build artifacts). Lustre on HPC; any writable
#    path on a rented single box.
export OMNI_CACHE_DIR=/path/to/writable/cache
export UV_CACHE_DIR=$OMNI_CACHE_DIR/uv
export TMPDIR=$OMNI_CACHE_DIR/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

# 3. Build the release-tree venv.
(cd post-training && uv sync --extra=cu128)

# 4. Drop the HF token at $HF_HOME/token. The runtime reads the *file*;
#    the HF_TOKEN env var is NOT a substitute (it will be ignored).
mkdir -p "$OMNI_CACHE_DIR/huggingface"
printf '%s' '<your-hf-token>' > "$OMNI_CACHE_DIR/huggingface/token"
chmod 600 "$OMNI_CACHE_DIR/huggingface/token"

# 5. Stage checkpoints + dataset.
bash samples/post-training/setup_env.sh
```

`setup_env.sh` is idempotent: re-run after a cache wipe and it will skip
what's already staged. If it errors on HF gating, accept the NVIDIA Open
Model License on each gated repo listed in
[QUICKSTART §1](./QUICKSTART.md#1-prerequisites) and re-run.

## Step 1 — pick a launch shape per experiment

The three release experiments and their per-config defaults:

| Exp | Hydra `experiment=` | Default `fsdp_shard_size` | Default `context_parallel_size` | Default world size | Min H100 |
|----:|----------------------|---:|---:|---:|---:|
| **E1** student-init (L2a)         | `causal_cosmos2_2B_single_view_chunk2_t24_hdmap_vae` | 8  | 4  | 32  | **8**  |
| **E2** bidirectional teacher (L1b)| `teacher_cosmos2_2B_single_view_t24_hdmap_vae`        | 8  | 8  | 64  | **8**  |
| **E3** self-forcing distillation (L0) | `cosmos_v2_2b_SF_res720p_fps30_i2v_hdmap_chunk2_vae_encode_loc6_release` | 8  | 1  | 8   | **8**  |

The launcher wrapper `samples/post-training/torchrun_smoke.sh` invokes E1/E2
with `fsdp_shard_size=8 context_parallel_size=8` (single-node 8-GPU smoke
profile) and E3 unmodified. To run at a different shape, **override on the
command line** rather than editing the launcher; the actual constraints:

* `world_size = NPROC_PER_NODE × NNODES` is set by torchrun.
* `data_parallel_size = world_size / (tensor_parallel × pipeline_parallel
  × context_parallel_size)`. The framework computes DP implicitly; it must
  come out a positive integer.
* `fsdp_shard_size` must divide `world_size`. The FSDP and CP groups can
  *share the same ranks* (e.g. on 8 GPUs the launcher uses `fsdp=8 cp=8`,
  both spanning all 8 ranks — DP=1, no data parallelism). Their product
  is not a constraint.
* Increase `context_parallel_size` first when scaling out (it parallelizes
  the long-sequence dimension of the diffusion model). Increase
  `fsdp_shard_size` second.
* `max_latent_frames_per_gpu` (E1) caps the per-GPU latent state; smaller
  than 8-GPU launch shapes are unsupported.

Recommended per-shape overrides (use these as defaults; tune only if memory
or throughput problems surface):

| Total H100 | E1 (FSDP × CP) | E2 (FSDP × CP) | E3 (FSDP × CP) | DP |
|---:|:---:|:---:|:---:|---:|
|   8 | 8×8 (launcher default) | 8×8 (launcher default) | 8×1 (release default) | 1 |
|  16 | 8×8                    | 8×8                    | 8×2 (set CP=2)         | 2 |
|  32 | 8×8 (release default for E1 uses CP=4 → DP=8) | 8×8 | 8×4 | 4 |
|  64 | 8×8                    | 8×8 (release default)  | 8×8                    | 8 |
| 128 | 16×8                   | 16×8                   | 16×8                   | 8 |
| 256 | 32×8                   | 32×8                   | 32×8                   | 8 |

The "DP" column is `world_size / (CP × TP × PP)`. At the 8-GPU smoke shape
DP=1 — every rank holds a shard of the same global batch.

Single-node example (8 H100, E1):

```bash
NPROC=8 bash samples/post-training/torchrun_smoke.sh 1
```

Multi-node example (16 H100 across 2 nodes, E2). Slurm:

```bash
sbatch --nodes=2 --gpus-per-node=8 \
       --export=ALL,EXPERIMENT=2,FSDP=8,CP=2 \
       samples/post-training/smoke_test.slurm
```

…with `FSDP=8 CP=2` plumbed through to the Hydra overrides
(`model.config.fsdp_shard_size=$FSDP model_parallel.context_parallel_size=$CP`).
If `smoke_test.slurm` doesn't already accept these envs, edit it once — keep
the change local to this directory and not in `./post-training/`.

## Step 2 — run

**If you're on a Slurm cluster, use `sbatch` from the start** — running
torchrun directly on a login/head node will fail with a CUDA error. Before
the first `sbatch`, edit `samples/post-training/smoke_test.slurm` to set the
right `#SBATCH --account=…` and `#SBATCH --partition=…` for your cluster
(the shipped values are placeholders).

```bash
# Slurm:
sbatch --export=ALL,EXPERIMENT=1 samples/post-training/smoke_test.slurm

# Direct torchrun (single rented 8-GPU box, no Slurm in the picture):
bash samples/post-training/torchrun_smoke.sh 1   # or 2, or 3
```

What to expect:

* Iter 1 takes ~85–180 s on cold cache (model load + `torch.compile` graph
  build + Triton kernel codegen). Iter 2+ stabilize to ~10 s for E1/E2 and
  ~30 s for E3.
* On a re-run on the same physical node, the per-host Triton cache (default
  `$OMNI_CACHE_DIR/triton/$(hostname -s)`) saves a few seconds — modest, not
  transformative. Don't expect dramatic warm-cache wins.
* `Iteration N: ... Time: Xs` lines roll over to averaged-every-10
  `N : iter_speed Y seconds per iteration` after iter 100 (E1) / 50 (E3).
  Don't read silence between as a hang — `iter_speed` lines are the steady
  state.

## Step 3 — dataset

The training expects a video/caption/(hdmap) tree under a `data_root`
discovered at runtime; the experiment defaults register that as `data` in the
release tree's cwd. **Do not hardcode the dataset path in this file or in the
launcher** — `setup_env.sh` populates the directory and
[QUICKSTART §3](./QUICKSTART.md#3-stage-checkpoints--dataset) names the
canonical location. That path is being relocated; if the QUICKSTART says one
location and `setup_env.sh` reports another, trust QUICKSTART and re-run the
staging step.

### Different dataset sizes

The minimum that lets E1/E2/E3 train is whatever produces ≥ `world_size`
clips per camera. With `batch_size=1` and `drop_last=True` (the release
default), iters per epoch = `floor(num_clips / world_size)`:

| Clips | World size | Iters/epoch |
|--:|--:|--:|
| 256 | 8   | 32  |
| 256 | 32  | 8   |
| 1024 | 8   | 128 |
| 1024 | 32  | 32  |

E1 and E2 carry `max_iter ≥ 150 000`, so the dataset cycles many epochs
during a real run; smoke runs of a few hundred iters are bounded only by
wall clock. E3 carries `max_iter=10 000` and consumes one batch per iter
without resampling, so a small dataset will exhaust it well before
`max_iter`. With the 256-clip smoke set on 8 ranks, E3 cleanly exits at
iter ~30 from a fresh start, or iter ~60 when resuming from an iter-30
checkpoint. **This is expected** — surface it as `COMPLETED 0:0 at iter N`,
not as a failure.

If you are given a larger dataset:

* Drop the new clips into the same camera-keyed directory layout as the
  smoke set (one `*.mp4` per clip; matching `*.txt` captions and, for HDMap
  experiments, matching `*.mp4` hdmap files). The dataset class auto-
  discovers clips by filename intersection across `video/`, `caption/`,
  `hdmap/`.
* The dataloader's `data_root` knob is set per experiment via the
  `data_train` config group. Stick to the default (`data` in the release
  tree's cwd) when at all possible. If you must point elsewhere, use a
  Hydra override (`dataloader_train.data_root=/abs/path`) — do not edit
  the release-side configs.
* Larger sets benefit E3 the most (it lets you push past iter 60). For E1/E2
  the cap is wall clock, not data.

## Step 4 — report

Send back, per experiment:

```
SHA tested:  <git rev-parse HEAD>
GPUs:        <NNODES x NPROC>  (FSDP=<x>, CP=<y>)
Job IDs:     E1=<id>, E2=<id>, E3=<id>
Iters:       E1=<n>, E2=<n>, E3=<n>
Verdict:     E1 PASS/FAIL, E2 PASS/FAIL, E3 PASS/FAIL
Loss range:  E1=<min..max>, E2=<min..max>, E3=<min..max>
Notes:       <anomalies, environmental quirks, anything new>
```

Acceptance:

* **Pass** = no `Traceback` / `OSError` / `BackendCompilerFailed` in the
  `.err`, loss finite (no `inf` / `nan`), at least one checkpoint save
  observed (`Saving` / `Saved` lines), and iter floor met:
  * E1 ≥ 10 iters
  * E2 ≥ 10 iters
  * E3 ≥ 50 iters (the launcher passes `dataloader_train.repeat_factor=200`
    so E3 isn't bounded by single-epoch exhaustion the way it was at iter
    ~30 on the <200-clip sample)
  * Both `TIMEOUT 0:0` (Slurm killed it at wall clock during a clean train
    loop — the train script never exited non-zero) and `COMPLETED 0:0` (the
    train script reached its own `max_iter`) count as PASS. `FAILED 1:0`
    or any non-`0:0` exit code is a fail.
* **Fail** = any of the above. Capture the last 100 lines of `.err`, the
  last 30 `Iteration N:` lines from `.out`, and the
  `sacct -j <jobid> -X -o "JobID,State,ExitCode,Elapsed,NodeList"` row.
  Do not retry blindly.

## Hard rules

1. Never edit anything inside `./post-training/`. If a fix has to land
   there, surface it; the maintainer will push upstream and re-vendor.
2. Don't bypass `triton_per_rank_wrap.sh`. It exists to defeat a real
   `os.replace ENOTEMPTY` race when 8 ranks compile identical Triton
   kernels on a Lustre filesystem.
3. Don't add or modify dataset path strings in this directory unless you
   are intentionally tracking the relocation. The path lives in QUICKSTART;
   `setup_env.sh` reads from it. Reduce duplication, don't add it.
4. Don't `git push --force` on shared branches; don't skip git hooks.

If you hit something not covered above, prefer reading
[`post-training/docs/setup.md`](../../post-training/docs/setup.md) before
asking — it covers `CUDA_HOME`, `LD_LIBRARY_PATH`, `TRITON_CACHE_DIR`,
glibc-on-Triton-cache failures, and the `libcublas.so.12` / `libnvrtc`
loader paths in detail.
