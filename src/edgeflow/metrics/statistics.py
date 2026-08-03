from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    return float(np.percentile(np.asarray(values, dtype=float), quantile, method="linear"))


def robust_cv(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot calculate robust CV of an empty sample")
    median = float(np.median(array))
    if median == 0:
        return math.inf
    mad = float(np.median(np.abs(array - median)))
    return 1.4826 * mad / abs(median)


def describe(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot describe an empty sample")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": median,
        "mad": mad,
        "robust_cv": robust_cv(array),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
    statistic: str = "median",
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError("bootstrap requires at least two observations")
    if resamples < 1_000:
        raise ValueError("EdgeFlow requires at least 1,000 bootstrap resamples")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(resamples, array.size))
    samples = array[indices]
    if statistic == "median":
        estimates = np.median(samples, axis=1)
    elif statistic == "mean":
        estimates = np.mean(samples, axis=1)
    else:
        raise ValueError(f"unsupported statistic: {statistic}")
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def paired_bootstrap(
    baseline: Sequence[float],
    intervention: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired latency comparison where positive improvement means intervention is faster."""

    a = np.asarray(baseline, dtype=float)
    b = np.asarray(intervention, dtype=float)
    if a.size != b.size or a.size < 2:
        raise ValueError("paired comparison requires equal samples with at least two pairs")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("paired performance values must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, a.size, size=(resamples, a.size))
    differences = a - b
    boot_difference = np.median(differences[indices], axis=1)
    log_ratios = np.log(a / b)
    boot_log_ratio = np.mean(log_ratios[indices], axis=1)
    delta = float(np.median(differences))
    ratio = float(np.median(a) / np.median(b))
    geometric_speedup = float(np.exp(np.mean(log_ratios)))
    ci_delta = [float(item) for item in np.quantile(boot_difference, [0.025, 0.975])]
    ci_speedup = [float(item) for item in np.exp(np.quantile(boot_log_ratio, [0.025, 0.975]))]
    practical_improvement = (ratio - 1.0) * 100.0
    supported = ci_delta[0] > 0 and practical_improvement >= 2.0
    return {
        "pairs": int(a.size),
        "paired_median_difference_ms": delta,
        "paired_median_difference_ci95_ms": ci_delta,
        "ratio_of_medians": ratio,
        "paired_geometric_speedup": geometric_speedup,
        "paired_geometric_speedup_ci95": ci_speedup,
        "practical_improvement_pct": practical_improvement,
        "claim_direction_supported": supported,
        "resamples": resamples,
        "seed": seed,
    }


def third_drift(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size < 6:
        return math.inf
    size = max(1, array.size // 3)
    first = float(np.median(array[:size]))
    last = float(np.median(array[-size:]))
    if first == 0:
        return math.inf
    return abs(last - first) / abs(first)
