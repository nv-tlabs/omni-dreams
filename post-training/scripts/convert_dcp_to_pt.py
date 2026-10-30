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

"""Convert a DCP model checkpoint to a consolidated .pt file, merging LoRA adapters into their base weights.

Training saves the model as a torch.distributed.checkpoint (DCP) directory, `<job>/checkpoints/iter_<N>/model`. The
.pt file written here holds the same flat state dict (`net.*`, and `net_ema.*` or `net_fake_score.*` where the model
has them), so it can be used wherever the experiments take a consolidated checkpoint: `checkpoint.load_path`, or
`net_ckpt` / `net_real_score_ckpt` / `net_fake_score_ckpt` of the self-forcing distillation.

A checkpoint trained with `use_lora` holds, for each adapted Linear `<m>`, its frozen base weight (`<m>.weight`, or
`<m>.base_layer.weight` when the adapters were injected in place) and the adapter weights `<m>.lora_A.<adapter>.weight`
and `<m>.lora_B.<adapter>.weight`. A net without adapters has no parameters for them, so they are merged:
W' = W + (lora_alpha / r) * B @ A, computed in float32 and stored in the dtype of W, with r the adapter rank. The
checkpoint does not record lora_alpha: pass the training config's value (`lora_alpha`, 32 by default) with
--lora-alpha, which is required when the checkpoint has adapters. Other adapter types (DoRA, embedding adapters) are
refused.

The conversion writes a temporary full copy of the checkpoint next to --out, so it needs about twice the checkpoint's
size of free disk space there.

Usage (CPU only):
    python scripts/convert_dcp_to_pt.py --dcp <job>/checkpoints/iter_000001000/model --out student.pt --lora-alpha 32
"""

import argparse
import os
import tempfile

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def merge_lora_weights(
    state_dict: dict[str, torch.Tensor], lora_alpha: float, adapter_name: str = "default"
) -> tuple[dict[str, torch.Tensor], int]:
    """Return state_dict with every LoRA adapter merged into its base weight, and the number of merged adapters."""
    a_suffix = f".lora_A.{adapter_name}.weight"
    b_suffix = f".lora_B.{adapter_name}.weight"
    adapter_keys = {k for k in state_dict if ".lora_" in k}
    merged_weights = {}
    for a_key in sorted(k for k in adapter_keys if k.endswith(a_suffix)):
        prefix = a_key[: -len(a_suffix)]
        b_key = prefix + b_suffix
        base_key = next((k for k in (prefix + ".base_layer.weight", prefix + ".weight") if k in state_dict), None)
        if b_key not in state_dict or base_key is None:
            raise ValueError(f"LoRA adapter {prefix}: expected {b_key} and a base weight in the checkpoint")
        lora_a, lora_b, weight = state_dict[a_key], state_dict[b_key], state_dict[base_key]
        delta = (lora_alpha / lora_a.shape[0]) * (lora_b.float() @ lora_a.float())
        merged_weights[base_key] = (weight.float() + delta).to(weight.dtype)
        adapter_keys -= {a_key, b_key}
    if adapter_keys:
        raise ValueError(
            f"Unsupported LoRA tensors (only lora_A/lora_B of adapter {adapter_name!r} are merged): "
            f"{sorted(adapter_keys)[:5]}"
        )
    merged = {
        k.replace(".base_layer.", "."): merged_weights.get(k, v) for k, v in state_dict.items() if ".lora_" not in k
    }
    return merged, len(merged_weights)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a DCP model checkpoint to a consolidated .pt file.")
    parser.add_argument(
        "--dcp", required=True, help="DCP model checkpoint directory, e.g. <job>/checkpoints/iter_N/model"
    )
    parser.add_argument("--out", required=True, help="output .pt file")
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="lora_alpha of the training config (the config's default is 32); required if the checkpoint has LoRA "
        "adapters",
    )
    parser.add_argument("--lora-adapter-name", default="default", help="name of the LoRA adapter to merge")
    args = parser.parse_args(argv)
    if not args.out.endswith(".pt"):
        parser.error("--out must end with .pt: the checkpoint loaders select the consolidated format by extension")

    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(args.out))) as tmp_dir:
        full_path = os.path.join(tmp_dir, "full.pt")
        dcp_to_torch_save(args.dcp, full_path)
        state_dict = torch.load(full_path, map_location="cpu", mmap=True, weights_only=True)
        if any(".lora_" in k for k in state_dict):
            if args.lora_alpha is None:
                parser.error(f"{args.dcp} has LoRA adapters: pass --lora-alpha (lora_alpha of the training config)")
            state_dict, num_merged = merge_lora_weights(state_dict, args.lora_alpha, args.lora_adapter_name)
            print(f"Merged {num_merged} LoRA adapters into their base weights (lora_alpha {args.lora_alpha}).")
        torch.save(state_dict, args.out)
    print(f"Wrote {len(state_dict)} tensors to {args.out}")


if __name__ == "__main__":
    main()
