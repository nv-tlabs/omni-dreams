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

# Top-level entry point for run.py --config=.
#
# The vendored train script (`scripts/train.py` inside the vendored tree) loads
# this file as a Python module. Side-effect of import:
#   1. Pulls in the vendored hydra config, which calls
#      `import_all_modules_from_package("omnidreams.experiments", reload=True)`
#      and registers the three release-side experiments (L0/L1b/L2a) with
#      hydra's ConfigStore.
#   2. Imports the sample-side experiment override and registers it under
#      its own name.
#
# After this module is imported, `--experiment=exp_pai_nurec_sv_hdmap` resolves
# to the override defined in exp_pai_nurec_sv_hdmap.py.
from __future__ import annotations

# noqa: F401 — vendored config imports trigger registration as side effects.
from omnidreams._src.omnidreams.configs.causal_cosmos2.config import make_config  # noqa: F401
from hydra.core.config_store import ConfigStore

from configs.exp_pai_nurec_sv_hdmap import exp_pai_nurec_sv_hdmap

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name=exp_pai_nurec_sv_hdmap["job"]["name"],
    node=exp_pai_nurec_sv_hdmap,
)
