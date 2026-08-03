from __future__ import annotations

from edgeflow.profiler import diagnose_profile


def test_launch_hypothesis_requires_multiple_signals() -> None:
    result = diagnose_profile(
        {
            "run_id": "profile-1", "source_type": "measured",
            "diagnostic_features": {
                "kernel_gap_ratio": 0.25, "median_kernel_us": 7.0, "gpu_active_ratio": 0.45,
                "dram_throughput_pct_peak": 20.0, "tensor_core_utilization_pct": 20.0,
            },
        }
    )
    launch = next(item for item in result["hypotheses"] if item["label"] == "launch_overhead_bound")
    assert launch["confidence"] == "high"
    assert result["claim_status"] == "HYPOTHESIS_ONLY_UNTIL_MATCHED_VALIDATION"


def test_empty_profile_reports_insufficient_evidence() -> None:
    result = diagnose_profile({"run_id": "profile-2", "diagnostic_features": {}})
    assert result["hypotheses"][0]["label"] == "insufficient_evidence"
