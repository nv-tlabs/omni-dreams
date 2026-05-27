# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Redaction smoke for the sample-side experiment composition.

Catches both upstream regressions (a fresh re-vendor reintroducing internal
strings the redaction list missed) and override mistakes (a sample-side
config accidentally hard-coding an internal bucket / lustre path / email).

Runs CPU-only; safe to gate every PR on this even without the release tree
having been synced (the import will fail loudly, which is the correct
signal that the contributor needs to `uv sync --extra=cu128` inside
`../../post-training/` first).
"""

from __future__ import annotations

import re

import pytest

REDACTION_RE = re.compile(
    r"lustre|nv-00-|gitlab-master|@nvidia\.com|s3://(imaginaire4|checkpoints)"
)


def _flatten_to_yaml(obj: object) -> str:
    """Flatten a LazyDict / dict / OmegaConf node to YAML text for regex scanning.

    The plan calls for OmegaConf.to_yaml when the real LazyDict composition
    lands (step 4); until then the placeholder is a plain dict and PyYAML is
    sufficient. Either path produces searchable text — that's all the
    redaction regex cares about.
    """
    try:
        from omegaconf import OmegaConf  # type: ignore[import-not-found]

        return OmegaConf.to_yaml(OmegaConf.create(obj))
    except ImportError:
        import yaml

        return yaml.safe_dump(obj, default_flow_style=False)


def test_exp_pai_nurec_sv_hdmap_no_redaction_targets() -> None:
    from configs.exp_pai_nurec_sv_hdmap import exp_pai_nurec_sv_hdmap

    text = _flatten_to_yaml(exp_pai_nurec_sv_hdmap)
    matches = REDACTION_RE.findall(text)
    assert not matches, (
        f"Redaction-target strings found in resolved experiment config: {matches}. "
        f"If this came from the vendored tree, fix it upstream in "
        f"imaginaire4/projects/cosmos/sil/causal_multiview.toml and re-vendor. "
        f"If it came from this sample's own override, redact it here."
    )


@pytest.mark.parametrize(
    "needle",
    ["lustre", "nv-00-", "gitlab-master", "@nvidia.com", "s3://imaginaire4", "s3://checkpoints"],
)
def test_redaction_regex_actually_matches(needle: str) -> None:
    """Sanity: the regex catches each redaction target. Guards against the
    test silently passing because the regex itself was broken in a refactor."""
    assert REDACTION_RE.search(needle), f"redaction regex failed to match {needle!r}"
