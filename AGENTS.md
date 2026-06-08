# AGENTS.md

This file orients AI coding agents (Claude Code, Codex, Cursor) working on NVIDIA OmniDreams, a multi-camera photorealistic world model for autonomous-driving simulation.

## Repository skills

Agent Skills live under `skills/<skill-name>/SKILL.md`; each skill directory name matches the YAML `name` field.

- [`skills/run-post-training-sample/SKILL.md`](skills/run-post-training-sample/SKILL.md) — bring up the post-training environment and run the three release experiments (E1/E2/E3) on 8-256 H100 GPUs, including the FSDP x CP scaling matrix.

> **Maintenance:** when adding a new top-level skill, place it at `skills/<skill-name>/SKILL.md`, make the frontmatter `name` match `<skill-name>`, and append one bullet above with a one-line task description.

## Workflow routing

Use the workflow entry point that matches the task:

- **Post-training / fine-tuning:** stay in this repo. Start with [`samples/post-training/README.md`](samples/post-training/README.md), then follow [`samples/post-training/QUICKSTART.md`](samples/post-training/QUICKSTART.md) or the agent-runnable [`skills/run-post-training-sample/SKILL.md`](skills/run-post-training-sample/SKILL.md). Do not clone FlashDreams for post-training.
- **Live interactive demo and offline batch inference:** start in FlashDreams. The interactive driving sample and runtime inference entry points live there.

Post-training source distributions are supported without git metadata as long
as they preserve the expected relative layout: `samples/post-training/` and
`post-training/` both live under the same repo root.

## Documentation tree

Start at the repo root and follow links into each sample; the READMEs are the source of truth.

- [`README.md`](README.md) — project overview, repo-wide quickstarts, license/notice/contribution links
  - [`CONTRIBUTING.md`](CONTRIBUTING.md) — DCO sign-off and PR conventions
  - Licensed Apache 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for attribution.
- [`samples/post-training/README.md`](samples/post-training/README.md) — fine-tune sample orientation map
  - [`samples/post-training/QUICKSTART.md`](samples/post-training/QUICKSTART.md) — four-step zero-to-training recipe
  - [`skills/run-post-training-sample/SKILL.md`](skills/run-post-training-sample/SKILL.md) — agent-runnable E1/E2/E3 procedure
- [`post-training/README.md`](post-training/README.md) — release tree overview (**never edited in place**)
  - [`post-training/docs/setup.md`](post-training/docs/setup.md) — env vars, CUDA / Triton / glibc troubleshooting

## Tool-specific notes

- Top-level `skills/` directories follow the Agent Skills convention: `SKILL.md` has YAML frontmatter with `name` and `description`, and `name` matches the parent directory.
- Codex and Cursor read `AGENTS.md` natively. Subdirectory `AGENTS.md` files may add narrower routing where needed.
- Claude Code reads `AGENTS.md` as a fallback when no `CLAUDE.md` is present. If client auto-discovery is needed, mirror these skills into `.claude/skills/<name>/SKILL.md`.
