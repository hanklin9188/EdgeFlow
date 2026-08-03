from __future__ import annotations

import pytest

from edgeflow.metrics.statistics import bootstrap_ci, describe, paired_bootstrap, third_drift


def test_describe_and_bootstrap_are_deterministic() -> None:
    values = [10.0, 10.2, 9.9, 10.1, 10.0]
    summary = describe(values)
    assert summary["median"] == 10.0
    assert bootstrap_ci(values, resamples=1000, seed=7) == bootstrap_ci(values, resamples=1000, seed=7)


def test_paired_bootstrap_supports_practical_improvement() -> None:
    baseline = [10.0 + index * 0.01 for index in range(30)]
    intervention = [value * 0.9 for value in baseline]
    result = paired_bootstrap(baseline, intervention, resamples=10_000)
    assert result["claim_direction_supported"] is True
    assert result["practical_improvement_pct"] == pytest.approx(11.111111, rel=1e-5)


def test_third_drift_rejects_small_sample() -> None:
    assert third_drift([1.0, 1.0]) == float("inf")
