# NVIDIA OmniDreams

NVIDIA OmniDreams is a world model that generates multi-camera photorealistic
video for autonomous-driving simulation in real time.

The model consumes:

- A single input frame
- Initial text prompt
- Per-frame coarse HD map image and trajectory poses

It produces photorealistic video frames in chunks.

## How the pieces fit together

An OmniDreams rollout starts from one real RGB frame. That frame anchors the
appearance of the scene. The text prompt describes the driving context, while
the per-frame HD map image and trajectory poses provide the structured
conditioning for each generated chunk. The world model autoregressively
generates the next video chunk, then feeds that chunk back into the next step
so the scene can continue over time.

Interactive inference and live driving demos now live in the companion
[`flashdreams`](https://github.com/NVIDIA/flashdreams) project. This repo owns
the OmniDreams post-training sample and release tree.

This repository contains the following samples for demonstration:

- **[`samples/post-training`](samples/post-training/README.md)** - launchers
  and configs for fine-tuning the Cosmos2 SV-HDMap world model on a single
  8-GPU node or a Slurm cluster (student-init, bidirectional teacher, and
  self-forcing distillation). See
  [`samples/post-training/QUICKSTART.md`](samples/post-training/QUICKSTART.md)
  for the four-step out-of-box flow.

For interactive driving and offline, reproducible batch `mp4` video
generation, see the companion [`flashdreams`](https://github.com/NVIDIA/flashdreams)
project.

## Resources

- **Research blog** — [research.nvidia.com/labs/sil/projects/omnidreams-blog](https://research.nvidia.com/labs/sil/projects/omnidreams-blog/)
- **White paper** — [*NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation*](https://research.nvidia.com/labs/sil/projects/omnidreams-blog/paper.pdf)
- **Model weights** — [`nvidia/omni-dreams-models`](https://huggingface.co/nvidia/omni-dreams-models) on Hugging Face

## Community

Join us on the NVIDIA Omniverse Discord to share your results, attend office
hours, and take part in technical discussion with the NVIDIA OmniDreams team and
community.

If you are new, start with the
[server invite](https://discord.com/invite/nvidiaomniverse) to on-board.

Channels:

- [`#omnidreams`](https://discord.gg/bsjzh4uZ)
- [`#flashdreams`](https://discord.gg/yTdHDqFP)
- [`#world-model-chit-chat`](https://discord.gg/APbw7EPk)

## Prerequisites

### Hardware and disk

| Workflow | Tested / expected hardware | Disk guidance |
|---|---|---|
| Post-training | Supported minimum is a single 8-GPU Ampere/Hopper node (`NPROC=8`). Smaller `NPROC` values are unsupported. | At least 150 GB free; 200 GB or more is recommended for caches plus training output. |
| Inference / interactive driving | See FlashDreams. | See FlashDreams. |

Use a recent NVIDIA driver compatible with the CUDA stack in the selected
workflow. The post-training quickstart was validated with driver 570.148.08
and CUDA 12.8 on 8x H100 80 GB HBM3.

### Hugging Face access

The post-training sample relies on Hugging Face sample data on first run. The
dataset you will need is:

- [`nvidia/omni-dreams-scenes`](https://huggingface.co/datasets/nvidia/omni-dreams-scenes) — post-training sample scenes

To allow automated download, create a Hugging Face token at https://huggingface.co/settings/tokens/new.

```bash
export HF_TOKEN=<YOUR-HF-TOKEN>    # used for the Hugging Face scenes dataset
```

If any download fails with `401`, `403`, or a gated-repo error, verify both the
token and the repo access above before debugging anything else.

If your environment uses another authorized Hugging Face org, set
`OMNI_DREAMS_HF_ORG=<YOUR-HF-ORG>` once in your shell. Post-training setup and
checkpoint loading use it to derive the OmniDreams model and scene repos.

## Inference And Interactive Driving

Use the FlashDreams OmniDreams runner for inference and interactive driving.
The inputs are the same pieces described above: an initial RGB frame, a text
prompt, and HD-map / trajectory conditioning frames. Batch inference produces
a reproducible `mp4` sequence; the interactive sample provides the live driving
experience.

The interactive driving sample that previously lived in this repository has
moved to FlashDreams:

- [FlashDreams OmniDreams model docs](https://nvidia.github.io/flashdreams/main/models/omnidreams.html)
- [FlashDreams interactive-drive sample](https://github.com/NVIDIA/flashdreams/tree/main/integrations/omnidreams/omnidreams/interactive_drive)

## Quickstart: post-training (fine-tune)

For fine-tuning the Cosmos2 SV-HDMap world model on a single 8-GPU node
(student-init, bidirectional teacher, or self-forcing distillation), see
[`samples/post-training/QUICKSTART.md`](samples/post-training/QUICKSTART.md).
It covers HuggingFace auth setup, checkpoint + sample-dataset staging, and
the launcher invocations for all three experiments. Slurm wrapper included
for multi-tenant clusters.

# License and Attribution

OmniDreams is licensed under the [Apache License, Version 2.0](LICENSE).
Third-party runtime dependencies are fetched by package managers or companion
projects such as `flashdreams`.

- [`LICENSE`](LICENSE) — repository-wide Apache-2.0 grant
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — DCO sign-off and PR conventions
- [`REUSE.toml`](REUSE.toml) — per-path / per-file license metadata
- [`THIRD_PARTY_NOTICES.txt`](./THIRD_PARTY_NOTICES.txt) — upstream
  attribution for vendored third-party code
- [`NOTICE`](./NOTICE) — third-party-fetch notice for runtime / install-time
  downloads
