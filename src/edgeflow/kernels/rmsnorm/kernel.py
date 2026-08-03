from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_residual_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    hidden: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < hidden
    base = row * hidden + offsets
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    combined = x + residual
    sum_squares = tl.sum(combined * combined, axis=0)
    reciprocal_rms = tl.rsqrt(sum_squares / hidden + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output = combined * reciprocal_rms * weight
    tl.store(output_ptr + base, output, mask=mask)


def triton_residual_rmsnorm(x: Any, residual: Any, weight: Any, eps: float = 1e-6) -> Any:
    if not all(tensor.is_cuda for tensor in (x, residual, weight)):
        raise ValueError("Triton path requires CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in (x, residual, weight)):
        raise ValueError("Triton path requires contiguous tensors")
    if x.shape != residual.shape or x.ndim != 2 or weight.shape != (x.shape[1],):
        raise ValueError("contract is x/residual [rows, hidden], weight [hidden]")
    rows, hidden = x.shape
    if hidden > 65_536:
        raise ValueError("hidden dimension exceeds the validated single-program reduction contract")
    output = torch.empty_like(x)
    block = triton.next_power_of_2(hidden)
    num_warps = 4 if block <= 2048 else 8
    _fused_residual_rmsnorm_kernel[(rows,)](
        x, residual, weight, output, hidden=hidden, eps=eps, BLOCK=block, num_warps=num_warps
    )
    return output
