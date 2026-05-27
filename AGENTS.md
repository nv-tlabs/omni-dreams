# AGENTS.md

This file orients AI coding agents (Claude Code, Codex, Cursor) working on NVIDIA Omniverse Dreams, a multi-camera photorealistic world model for autonomous-driving simulation.

## Directory-level skills

Each `SKILL.md` is the agent-runnable recipe for the directory it sits in.

- [`samples/post-training/SKILL.md`](samples/post-training/SKILL.md) — bring up the post-training environment and run the three release experiments (E1/E2/E3) on 8–256 H100 GPUs, including the FSDP × CP scaling matrix.

> **Maintenance:** when adding a new directory-level `SKILL.md`, append one bullet to the list above with its path and a one-line description of the task it covers.

## Workflow routing

Use the workflow entry point that matches the task:

- **Post-training / fine-tuning:** stay in this repo. Start with [`samples/post-training/README.md`](samples/post-training/README.md), then follow [`samples/post-training/QUICKSTART.md`](samples/post-training/QUICKSTART.md) or the agent-runnable [`samples/post-training/SKILL.md`](samples/post-training/SKILL.md). Do not clone FlashDreams for post-training.
- **Live interactive demo:** use [`samples/interactive-drive`](samples/interactive-drive/README.md). Its `world-model` extra pulls FlashDreams through `uv sync` when that integration is needed.
- **Offline batch inference:** start in FlashDreams instead of this repo. Use this repo when you need the live `interactive-drive` demo or post-training launchers.

Post-training source distributions are supported without git metadata as long
as they preserve the expected relative layout: `samples/post-training/` and
`post-training/` both live under the same repo root.

## Documentation tree

Start at the repo root and follow links into each sample; the READMEs are the source of truth.

- [`README.md`](README.md) — project overview, repo-wide quickstarts, license/notice/contribution links
  - [`CONTRIBUTING.md`](CONTRIBUTING.md) — DCO sign-off and PR conventions
  - Licensed Apache 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for attribution.
- [`samples/interactive-drive/README.md`](samples/interactive-drive/README.md) — interactive driving demo (Vulkan window or MJPEG stream)
- [`samples/post-training/README.md`](samples/post-training/README.md) — fine-tune sample orientation map
  - [`samples/post-training/QUICKSTART.md`](samples/post-training/QUICKSTART.md) — four-step zero-to-training recipe
  - [`samples/post-training/SKILL.md`](samples/post-training/SKILL.md) — agent-runnable E1/E2/E3 procedure
- [`post-training/README.md`](post-training/README.md) — release tree overview (**never edited in place**)
  - [`post-training/docs/setup.md`](post-training/docs/setup.md) — env vars, CUDA / Triton / glibc troubleshooting

## Tool-specific notes

- **Claude Code** reads `AGENTS.md` as a fallback when no `CLAUDE.md` is present in a directory. `SKILL.md` files referenced here are agent-runnable *documentation*, not auto-loaded skills; CC only auto-discovers skills placed under `.claude/skills/<name>/SKILL.md`.
- **Codex / Cursor** read `AGENTS.md` natively. Subdirectory `AGENTS.md` files layer onto this one if added later.
