# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Memory-bounded compiled FlexAttention for attention-output SAC.

PyTorch's ordinary AOTAutograd wrapper exposes the tensors needed by the
compiled backward as outputs of the compiled forward region. Selective
checkpointing consequently retains several sequence-sized tensors when it
saves that region. This module gives the forward and backward independent
compiled ABIs: the forward exposes only attention output and LSE, while the
backward is exported and compiled explicitly from the same BlockMask graph.
"""

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch._inductor.config as inductor_config
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

from omnidreams._src.predict2.networks.selective_activation_checkpoint import compiled_attention_region


def supports_compiled_flex_attention_output_sac() -> bool:
    """Return whether this PyTorch build exposes the required compiler APIs."""

    return hasattr(inductor_config, "wrap_inductor_compiled_regions") and hasattr(torch, "func")


def _tensor_signature(tensor: Tensor) -> tuple[Any, ...]:
    return (
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


@dataclass
class _CompiledFlexSacArtifacts:
    block_mask: BlockMask
    scale: float | None
    forward: Callable[..., tuple[Tensor, Tensor]]
    raw_output: Callable[[Tensor, Tensor, Tensor], Tensor]
    exported: torch.fx.GraphModule | None = None
    buffer_names: tuple[str, ...] = ()
    buffers: tuple[Tensor, ...] = ()
    backwards: dict[tuple[Any, ...], Callable[..., list[Tensor | None]]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def run_backward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        logsumexp_base2: Tensor,
        output_gradient: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        backward_key = _tensor_signature(output_gradient)
        with self.lock:
            compiled_backward = self.backwards.get(backward_key)
            if compiled_backward is None:
                compiled_backward = self._compile_backward(
                    query,
                    key,
                    value,
                    output,
                    logsumexp_base2,
                    output_gradient,
                )
                self.backwards[backward_key] = compiled_backward

        gradients = compiled_backward(
            *self.buffers,
            query,
            key,
            value,
            output,
            logsumexp_base2,
            output_gradient,
        )
        grad_query, grad_key, grad_value = gradients[-3:]
        return grad_query, grad_key, grad_value

    def _compile_backward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        logsumexp_base2: Tensor,
        output_gradient: Tensor,
    ) -> Callable[..., list[Tensor | None]]:
        # These are deliberately lazy private imports. PyTorch 2.7 remains a
        # supported default environment, but this optimized boundary requires
        # the FlexAttention/AOT APIs provided by the PyTorch 2.10 environment.
        from torch._functorch.aot_autograd import aot_export_joint_simple
        from torch._functorch.partitioners import default_partition
        from torch._inductor.compile_fx import compile_fx

        if self.exported is None:
            self.exported, _ = torch._dynamo.export(self.raw_output)(query, key, value)
            named_buffers = tuple(self.exported.named_buffers())
            self.buffer_names = tuple(name for name, _ in named_buffers)
            self.buffers = tuple(buffer for _, buffer in named_buffers)

        exported = self.exported
        buffer_names = self.buffer_names

        def functional_forward(*args: Tensor) -> tuple[Tensor]:
            *buffer_values, q, k, v = args
            replacements = dict(zip(buffer_names, buffer_values, strict=True))
            return (torch.func.functional_call(exported, replacements, (q, k, v)),)

        primals = (*self.buffers, query, key, value)
        # Custom autograd backward normally runs with grad mode disabled. AOT
        # joint export still needs grad mode enabled to construct the VJP.
        with torch.enable_grad():
            joint_graph = aot_export_joint_simple(
                functional_forward,
                primals,
                trace_joint=True,
                num_params_buffers=len(self.buffers),
            )
            _, backward_graph = default_partition(joint_graph, primals, num_fwd_outputs=1)

        backward_inputs = [
            *self.buffers,
            query,
            key,
            value,
            output,
            logsumexp_base2,
            output_gradient,
        ]
        # compile_fx failures must propagate. Falling back to eager evaluation
        # would materialize the S x S score matrix and can OOM long sequences.
        return compile_fx(backward_graph, backward_inputs)


_SAC_ARTIFACTS: dict[tuple[Any, ...], _CompiledFlexSacArtifacts] = {}
_SAC_ARTIFACTS_LOCK = threading.Lock()


def _get_sac_artifacts(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: BlockMask,
    scale: float | None,
) -> _CompiledFlexSacArtifacts:
    cache_key = (
        id(block_mask),
        _tensor_signature(query),
        _tensor_signature(key),
        _tensor_signature(value),
        scale,
    )
    with _SAC_ARTIFACTS_LOCK:
        artifacts = _SAC_ARTIFACTS.get(cache_key)
        if artifacts is not None:
            return artifacts

        # AuxOutput.lse uses natural logarithms. FlexAttention's lower-level
        # backward consumes log2 LSE, so conversion happens at the ABI boundary.
        from torch.nn.attention.flex_attention import AuxRequest

        def raw_output(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return torch_flex_attention(q, k, v, block_mask=block_mask, scale=scale)

        def forward_only(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
            with torch.no_grad():
                output, aux = torch_flex_attention(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    scale=scale,
                    return_aux=AuxRequest(lse=True),
                )
                return output, aux.lse

        compiled_forward = torch.compile(forward_only, dynamic=False, fullgraph=True)
        compiled_forward = inductor_config.patch(wrap_inductor_compiled_regions=True)(compiled_forward)
        artifacts = _CompiledFlexSacArtifacts(
            block_mask=block_mask,
            scale=scale,
            forward=compiled_forward,
            raw_output=raw_output,
        )
        _SAC_ARTIFACTS[cache_key] = artifacts
        return artifacts


class _CompiledFlexSacFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        artifacts: _CompiledFlexSacArtifacts,
    ) -> Tensor:
        with compiled_attention_region():
            output, logsumexp = artifacts.forward(query, key, value)
        logsumexp_base2 = logsumexp / math.log(2.0)
        ctx.artifacts = artifacts
        ctx.save_for_backward(query, key, value, output, logsumexp_base2)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, output_gradient: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None, None]:
        query, key, value, output, logsumexp_base2 = ctx.saved_tensors
        grad_query, grad_key, grad_value = ctx.artifacts.run_backward(
            query,
            key,
            value,
            output,
            logsumexp_base2,
            output_gradient,
        )
        return grad_query, grad_key, grad_value, None


def compiled_flex_attention_for_output_sac(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Callable[..., Tensor] | None = None,
    block_mask: BlockMask | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: dict[str, Any] | None = None,
    *,
    return_aux: Any = None,
) -> Tensor:
    """Run compiled FlexAttention through its attention-output SAC boundary."""

    if not supports_compiled_flex_attention_output_sac():
        raise RuntimeError("attention-output SAC for compiled FlexAttention requires PyTorch 2.10 or newer")
    if score_mod is not None:
        raise NotImplementedError("the output-only compiled FlexAttention boundary does not support score_mod")
    if block_mask is None:
        raise ValueError("the output-only compiled FlexAttention boundary requires a BlockMask")
    if enable_gqa:
        raise NotImplementedError("the output-only compiled FlexAttention boundary does not support GQA")
    if return_lse or return_aux is not None:
        raise NotImplementedError("the output-only compiled FlexAttention boundary returns only attention output")
    if kernel_options is not None:
        raise NotImplementedError("the output-only compiled FlexAttention boundary does not accept kernel_options")

    artifacts = _get_sac_artifacts(query, key, value, block_mask, scale)
    return _CompiledFlexSacFunction.apply(query, key, value, artifacts)


def clear_compiled_flex_attention_sac_cache() -> None:
    """Clear process-local compiler artifacts. Intended for isolated tests."""

    with _SAC_ARTIFACTS_LOCK:
        _SAC_ARTIFACTS.clear()
