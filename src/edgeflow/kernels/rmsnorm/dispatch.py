from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from edgeflow.core.serialization import project_root, write_json
from edgeflow.kernels.rmsnorm.reference import reference_residual_rmsnorm

KERNEL_VERSION = "fused-residual-rmsnorm-v1"
MIN_PRACTICAL_SPEEDUP = 1.05


def _cache_path() -> Path:
    return project_root() / ".cache" / "edgeflow" / "kernel_validation.json"


def _performance_cache_path() -> Path:
    return project_root() / ".cache" / "edgeflow" / "kernel_performance.json"


def _key(x: Any) -> str:
    import torch

    gpu = torch.cuda.get_device_name(x.device) if x.is_cuda else "cpu"
    return f"{KERNEL_VERSION}|{gpu}|{x.dtype!s}|{x.shape[0]}x{x.shape[1]}"


def dispatch_decision(x: Any, residual: Any, weight: Any) -> dict[str, Any]:
    """Explain whether this exact tensor scope can use the measured Triton path."""

    if not getattr(x, "is_cuda", False):
        return {"use_triton": False, "reason": "cpu_reference", "key": None}
    supported = (
        x.ndim == 2
        and x.shape == residual.shape
        and tuple(weight.shape) == (x.shape[1],)
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and x.shape[1] <= 65_536
    )
    if not supported:
        return {"use_triton": False, "reason": "unsupported_contract", "key": None}
    key = _key(x)
    if _load_cache().get(key, {}).get("status") != "PASS":
        return {"use_triton": False, "reason": "correctness_unvalidated", "key": key}
    performance = _load_performance_cache().get(key, {})
    if not performance.get("enabled"):
        return {"use_triton": False, "reason": "no_measured_practical_speedup", "key": key}
    return {
        "use_triton": True,
        "reason": "validated_measured_winner",
        "key": key,
        "speedup_vs_eager": performance.get("speedup_vs_eager"),
    }


@lru_cache(maxsize=8)
def _load_json_cache(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_cache() -> dict[str, Any]:
    return _load_json_cache(str(_cache_path()))


def _load_performance_cache() -> dict[str, Any]:
    return _load_json_cache(str(_performance_cache_path()))


def clear_dispatch_caches() -> None:
    """Invalidate process-local cache after a calibration artifact changes."""

    _load_json_cache.cache_clear()


def record_performance_decisions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist measured dispatch decisions; unmeasured shapes remain on the reference path."""

    cache = dict(_load_performance_cache())
    for row in rows:
        shape = row.get("shape", [])
        dtype = row.get("dtype")
        gpu = row.get("gpu")
        speedup = float(row.get("speedup", 0.0))
        if len(shape) != 2 or not dtype or not gpu:
            continue
        key = f"{KERNEL_VERSION}|{gpu}|{dtype}|{shape[0]}x{shape[1]}"
        cache[key] = {
            "enabled": bool(
                row.get("correctness", {}).get("status") == "PASS"
                and speedup >= MIN_PRACTICAL_SPEEDUP
            ),
            "speedup_vs_eager": speedup,
            "minimum_practical_speedup": MIN_PRACTICAL_SPEEDUP,
        }
    write_json(_performance_cache_path(), cache)
    clear_dispatch_caches()
    return cache


def validate_shape(x: Any, residual: Any, weight: Any, eps: float = 1e-6) -> dict[str, Any]:
    import torch

    from edgeflow.kernels.rmsnorm.kernel import triton_residual_rmsnorm

    reference = reference_residual_rmsnorm(x, residual, weight, eps)
    actual = triton_residual_rmsnorm(x, residual, weight, eps)
    tolerance = {
        torch.float32: (1e-4, 1e-4),
        torch.float16: (2e-2, 2e-2),
        torch.bfloat16: (5e-2, 5e-2),
    }
    atol, rtol = tolerance.get(x.dtype, (1e-4, 1e-4))
    passed = True
    error: str | None = None
    try:
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    except AssertionError as exc:
        passed = False
        error = str(exc)
    maximum = float((actual.float() - reference.float()).abs().max().item())
    result = {
        "status": "PASS" if passed else "FAIL",
        "kernel_version": KERNEL_VERSION,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": maximum,
        "error": error,
    }
    cache = dict(_load_cache())
    cache[_key(x)] = result
    write_json(_cache_path(), cache)
    clear_dispatch_caches()
    return result


def fused_residual_rmsnorm(
    x: Any,
    residual: Any,
    weight: Any,
    eps: float = 1e-6,
    *,
    force_reference: bool = False,
    _decision: dict[str, Any] | None = None,
) -> Any:
    """Use Triton only for a shape that passed validation on this exact GPU."""

    if force_reference or not getattr(x, "is_cuda", False):
        return reference_residual_rmsnorm(x, residual, weight, eps)
    decision = _decision or dispatch_decision(x, residual, weight)
    if not decision["use_triton"]:
        return reference_residual_rmsnorm(x, residual, weight, eps)
    try:
        from edgeflow.kernels.rmsnorm.kernel import triton_residual_rmsnorm

        return triton_residual_rmsnorm(x, residual, weight, eps)
    except (ImportError, RuntimeError, ValueError):
        return reference_residual_rmsnorm(x, residual, weight, eps)
