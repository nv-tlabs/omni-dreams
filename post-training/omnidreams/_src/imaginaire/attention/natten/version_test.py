# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for NATTEN version parsing and comparison.
"""

import pytest

from omnidreams._src.imaginaire.attention.natten import _compare_natten_versions, _parse_natten_version


class TestParseNattenVersion:
    """Test _parse_natten_version function."""

    @pytest.mark.L1
    def test_parse_release_version(self):
        """Test parsing release versions."""
        assert _parse_natten_version("0.21.5") == ([0, 21, 5], None)
        assert _parse_natten_version("1.0.0") == ([1, 0, 0], None)
        assert _parse_natten_version("0.0.1") == ([0, 0, 1], None)

    @pytest.mark.L1
    def test_parse_dev_version(self):
        """Test parsing dev versions."""
        assert _parse_natten_version("0.21.5.dev9") == ([0, 21, 5], 9)
        assert _parse_natten_version("0.21.5.dev12") == ([0, 21, 5], 12)
        assert _parse_natten_version("1.0.0.dev1") == ([1, 0, 0], 1)

    @pytest.mark.L1
    def test_parse_invalid_versions(self):
        """Test that invalid versions return None."""
        assert _parse_natten_version("0.21") is None  # Too few components
        assert _parse_natten_version("0.21.5.6.7") is None  # Too many components
        assert _parse_natten_version("abc.def.ghi") is None  # Non-numeric
        assert _parse_natten_version("0.21.x") is None  # Invalid number
        assert _parse_natten_version("0.21.5.beta1") is None  # Non-dev suffix
        assert _parse_natten_version("") is None  # Empty string


class TestCompareNattenVersions:
    """Test _compare_natten_versions function."""

    @pytest.mark.L1
    def test_compare_release_versions(self):
        """Test comparing release versions."""
        assert _compare_natten_versions("0.21.6", "0.21.5") == 1  # Greater
        assert _compare_natten_versions("0.21.5", "0.21.6") == -1  # Less
        assert _compare_natten_versions("0.21.5", "0.21.5") == 0  # Equal
        assert _compare_natten_versions("1.0.0", "0.21.5") == 1  # Major version bump
        assert _compare_natten_versions("0.22.0", "0.21.99") == 1  # Minor version bump

    @pytest.mark.L1
    def test_compare_dev_versions(self):
        """Test comparing dev versions."""
        assert _compare_natten_versions("0.21.5.dev12", "0.21.5.dev9") == 1  # Greater dev
        assert _compare_natten_versions("0.21.5.dev9", "0.21.5.dev12") == -1  # Less dev
        assert _compare_natten_versions("0.21.5.dev9", "0.21.5.dev9") == 0  # Equal dev

    @pytest.mark.L1
    def test_compare_release_vs_dev(self):
        """Test comparing release vs dev versions."""
        # Release version is greater than dev version with same base
        assert _compare_natten_versions("0.21.5", "0.21.5.dev12") == 1
        assert _compare_natten_versions("0.21.5.dev12", "0.21.5") == -1

    @pytest.mark.L1
    def test_compare_different_base_versions(self):
        """Test comparing versions with different base versions."""
        assert _compare_natten_versions("0.21.6.dev1", "0.21.5") == 1
        assert _compare_natten_versions("0.21.5.dev99", "0.21.6.dev1") == -1
        assert _compare_natten_versions("1.0.0.dev1", "0.21.5") == 1

    @pytest.mark.L1
    def test_compare_invalid_versions(self):
        """Test that invalid versions return None."""
        assert _compare_natten_versions("invalid", "0.21.5") is None
        assert _compare_natten_versions("0.21.5", "invalid") is None
        assert _compare_natten_versions("invalid1", "invalid2") is None
        assert _compare_natten_versions("0.21", "0.21.5") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
