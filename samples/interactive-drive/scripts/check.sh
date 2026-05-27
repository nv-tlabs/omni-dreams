#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Run all checks: ruff lint, ruff format, pyright, pytest.
#
# Usage:
#   scripts/check.sh         # check-only; fails on any issue
#   scripts/check.sh --fix   # auto-fix ruff lint + format, then run full check
#
# Runs from the interactive-drive subproject root regardless of cwd. Uses `uv run` so
# dev/ui/world-model extras are ensured.
set -euo pipefail

cd "$(dirname "$0")/.."

UV_RUN=(uv run --extra dev --extra ui --extra world-model)

if [[ "${1:-}" == "--fix" ]]; then
  "${UV_RUN[@]}" ruff check --fix .
  "${UV_RUN[@]}" ruff format .
fi

"${UV_RUN[@]}" ruff check .
"${UV_RUN[@]}" ruff format --check .
"${UV_RUN[@]}" pyright
"${UV_RUN[@]}" pytest --durations=20
