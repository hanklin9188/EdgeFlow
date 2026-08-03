from __future__ import annotations

from typing import Any


def reference_residual_rmsnorm(x: Any, residual: Any, weight: Any, eps: float = 1e-6) -> Any:
    """Pure PyTorch reference with FP32 accumulation and output-dtype preservation."""

    import torch

    if x.shape != residual.shape:
        raise ValueError("x and residual must have identical shapes")
    if x.ndim != 2 or weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("contract is x/residual [rows, hidden], weight [hidden]")
    combined = x.float() + residual.float()
    reciprocal_rms = torch.rsqrt(combined.square().mean(dim=-1, keepdim=True) + eps)
    return (combined * reciprocal_rms * weight.float()).to(dtype=x.dtype)
