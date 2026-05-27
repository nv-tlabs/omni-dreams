# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared pytest fixtures and constants for the interactive_drive test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SAMPLE_SCENE = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "scenes"
    / "clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz"
)
"""Optional real USDZ scene, downloaded by ``prepare.py``.

Tests that use this path must silently skip when the file is absent so the
suite stays green on machines/CI that haven't fetched the large asset."""

# Captured by test_app_smoke._pump_stream, printed at session end.
captured_presenter_device: str | None = None


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print the Vulkan adapter used by smoke tests."""
    if captured_presenter_device and sys.__stderr__:
        sys.__stderr__.write(f"\n{captured_presenter_device}\n")
        sys.__stderr__.flush()
