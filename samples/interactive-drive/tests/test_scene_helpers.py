# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from interactive_drive.assets.scene_bundle import _discover_first_frames, _discover_prompts


class SceneHelperTest(unittest.TestCase):
    def test_discovers_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "prompt1.txt").write_text("hello", encoding="utf-8")
            (root / "prompt2.txt").write_text("snow", encoding="utf-8")
            (root / "first_image.png").write_bytes(b"")
            (root / "first_image_2.png").write_bytes(b"")
            prompts = _discover_prompts(root)
            first_frames = _discover_first_frames(root)
            self.assertEqual(prompts["1"], "hello")
            self.assertEqual(prompts["default"], "hello")
            self.assertEqual(first_frames["default"].name, "first_image.png")
            self.assertEqual(first_frames["2"].name, "first_image_2.png")


if __name__ == "__main__":
    unittest.main()
