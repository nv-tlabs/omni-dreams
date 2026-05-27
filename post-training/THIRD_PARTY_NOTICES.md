# Third-Party Notices

This package is released under the Apache License, Version 2.0 (see `LICENSE`).
It bundles or is partly derived from the open-source components listed below.
Each component retains its own license, and the original copyright notices are
reproduced here per Apache-2.0 §4(b), the MIT License terms, and the
BSD-3-Clause terms, as applicable.

When a derived file is present in the source tree, the in-file header also
preserves the upstream copyright. This file is the canonical, complete list.

---

## Components derived in source

### HuggingFace `diffusers`

* **License:** Apache License, Version 2.0
* **Upstream:** <https://github.com/huggingface/diffusers>
* **Upstream copyright:** Copyright 2024 TSAIL Team and The HuggingFace Team.
  All rights reserved.
* **Used in:** `omnidreams/_src/predict2/models/fm_solvers_unipc.py` is adapted
  from `src/diffusers/schedulers/scheduling_unipc_multistep.py` (v0.31.0),
  converted to support flow matching.

### HuggingFace `transformers`

* **License:** Apache License, Version 2.0
* **Upstream:** <https://github.com/huggingface/transformers>
* **Upstream copyright:** Copyright 2019 Facebook AI Research and the
  HuggingFace Inc. team. Copyright (c) 2018, NVIDIA CORPORATION. All rights
  reserved.
* **Used in:** `omnidreams/_src/predict2/networks/xlm_roberta.py` is modified
  from `transformers.models.xlm_roberta.modeling_xlm_roberta`.

### OpenAI CLIP

* **License:** MIT License
* **Upstream:** <https://github.com/openai/CLIP>
* **Upstream copyright:** Copyright (c) 2021 OpenAI.
* **Used in:** `omnidreams/_src/predict2/networks/clip.py` (text/image encoder
  components are modified from `clip/model.py`).

### `open_clip` (mlfoundations)

* **License:** MIT License
* **Upstream:** <https://github.com/mlfoundations/open_clip>
* **Upstream copyright:** Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell
  Wortsman, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
  John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
  Ludwig Schmidt.
* **Used in:** `omnidreams/_src/predict2/networks/clip.py` (image encoder
  components).

### Self-Forcing

* **License:** Apache License, Version 2.0
* **Upstream:** <https://github.com/guandeh17/Self-Forcing>
* **Upstream copyright:** Copyright the Self-Forcing authors (Guande He et al.,
  NeurIPS 2025).
* **Used in:** `omnidreams/_src/omnidreams/third_party/self_forcing/` (vendored
  pipeline, scheduler, and loss components) and the `omnidreams/_src/omnidreams/
  self_forcing/` training-side adapters.

### nanoGPT

* **License:** MIT License
* **Upstream:** <https://github.com/karpathy/nanoGPT>
* **Upstream copyright:** Copyright (c) 2022 Andrej Karpathy.
* **Used in:** `omnidreams/_src/imaginaire/utils/scheduler.py`
  (`WarmupCosineLR` is adapted from `train.py` cosine LR with warmup).

### `fvcore` (Facebook AI Research)

* **License:** Apache License, Version 2.0
* **Upstream:** <https://github.com/facebookresearch/fvcore>
* **Upstream copyright:** Copyright (c) Facebook, Inc. and its affiliates.
* **Used in:** `omnidreams/_src/imaginaire/lazy_config/` (registry and lazy
  instantiation patterns are derived from `fvcore.common.registry` and
  `fvcore.common.config`; some files also import `fvcore.common.registry`
  directly when available).

### `torchtitan` (PyTorch)

* **License:** BSD 3-Clause License
* **Upstream:** <https://github.com/pytorch/torchtitan>
* **Upstream copyright:** Copyright (c) Meta Platforms, Inc. and affiliates.
* **Used in:** `omnidreams/_src/omnidreams/utils/torch_future.py`
  (`clip_grad_norm_` is adapted from `torchtitan/utils.py`, commit
  `d4c86e3758a84cf23e2e879ab3c995cba9d5e410`).

### Intel Open Image Denoise (OIDN)

* **License:** Apache License, Version 2.0
* **Upstream:** <https://github.com/RenderKit/oidn>
* **Upstream copyright:** Copyright (c) 2018-Present Intel Corporation.
* **Used in:** `omnidreams/_src/imaginaire/utils/tone_curve.py` (color
  conversion utilities are adapted from `training/color.py`).

### IRASim (ByteDance)

* **License:** Apache License, Version 2.0
* **Upstream:** <https://github.com/bytedance/IRASim>
* **Upstream copyright:** Copyright (c) ByteDance, Inc. and its affiliates.
* **Used in:** `omnidreams/_src/predict2/action/datasets/` (dataset utilities
  for action-conditioned video are adapted from IRASim).

### `diffusion_policy` (Columbia AI & Robotics Lab)

* **License:** MIT License
* **Upstream:** <https://github.com/real-stanford/diffusion_policy>
* **Upstream copyright:** Copyright (c) 2023 Columbia Artificial Intelligence
  and Robotics Lab.
* **Used in:** `omnidreams/_src/predict2/action/datasets/gr00t_dreams/data/
  transform/state_action.py` (`RotationTransform` is adapted from
  `diffusion_policy/model/common/rotation_transformer.py`).

### RetinaFace (PyTorch port)

* **License:** MIT License
* **Upstream:** <https://github.com/biubug6/Pytorch_Retinaface>
* **Upstream copyright:** Copyright (c) 2019 biubug6.
* **Used in:** the `retinaface` Python package is consumed at runtime by
  `omnidreams/_src/imaginaire/auxiliary/guardrail/face_blur_filter/`. No source
  is vendored.

---

## License texts

The full MIT, Apache-2.0, and BSD-3-Clause license texts are reproduced below
once each; consult the upstream repositories for any additional notices the
projects ship.

### MIT License

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### Apache License, Version 2.0

The full Apache-2.0 text accompanies this distribution as `LICENSE`. Each
Apache-2.0 component above is licensed under the same terms; their respective
copyright notices are reproduced inline in this document and in the headers
of derived files per §4(b).

### BSD 3-Clause License

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
