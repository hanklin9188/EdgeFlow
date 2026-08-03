from __future__ import annotations

from typing import Any


def compare_tensors(reference: Any, candidate: Any, *, dtype: str) -> dict[str, Any]:
    import torch

    tolerances = {
        "fp32": (1e-4, 1e-4),
        "fp16": (2e-2, 2e-2),
        "bf16": (5e-2, 5e-2),
    }
    if dtype not in tolerances:
        raise ValueError(f"unsupported correctness dtype: {dtype}")
    atol, rtol = tolerances[dtype]
    finite = bool(torch.isfinite(candidate).all())
    difference = (reference.float() - candidate.float()).abs()
    passed = finite and bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol))
    denominator = reference.float().abs().clamp_min(1e-8)
    cosine = float(
        torch.nn.functional.cosine_similarity(
            reference.float().flatten(), candidate.float().flatten(), dim=0
        ).item()
    )
    return {
        "pass": passed,
        "dtype": dtype,
        "atol": atol,
        "rtol": rtol,
        "nan_count": int(torch.isnan(candidate).sum().item()),
        "inf_count": int(torch.isinf(candidate).sum().item()),
        "max_abs_error": float(difference.max().item()),
        "max_relative_error": float((difference / denominator).max().item()),
        "cosine_similarity": cosine,
        "summary": "Tensor parity passed." if passed else "Tensor parity failed.",
    }


def compare_token_sequences(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    first_divergence = next(
        (index for index, (left, right) in enumerate(zip(reference, candidate, strict=False)) if left != right),
        None,
    )
    if first_divergence is None and len(reference) != len(candidate):
        first_divergence = min(len(reference), len(candidate))
    return {
        "pass": first_divergence is None,
        "reference_length": len(reference),
        "candidate_length": len(candidate),
        "first_divergent_token": first_divergence,
        "top1_agreement": sum(a == b for a, b in zip(reference, candidate, strict=False)) / max(len(reference), len(candidate), 1),
    }
