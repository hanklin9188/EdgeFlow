from __future__ import annotations

from edgeflow.core.models import CapabilityReport
from edgeflow.optimizer import build_candidates, score_candidates, session_break_even
from edgeflow.policy import build_policy, select_plan


def eligible_row(plan_id: str, prompt: int, request_ms: float, startup_ms: float) -> dict:
    return {
        "plan_id": plan_id,
        "source_type": "measured",
        "workload": {
            "model_id": "model", "prompt_tokens": prompt, "concurrency": 1,
            "session_requests": 20,
        },
        "session_requests": 20,
        "metrics": {"request_latency_ms": request_ms, "startup_ms": startup_ms},
        "validation": {"policy_eligible": True, "quality_pass": True},
        "evidence_ids": [f"evidence-{plan_id}-{prompt}"],
    }


def test_candidate_generation_prunes_unavailable_backends() -> None:
    result = build_candidates(
        model_id="model",
        capabilities=[CapabilityReport(backend="pytorch_eager", available=True)],
        vram_bytes=16 * 1024**3,
        parameter_count=1_000_000_000,
    )
    assert result["candidate_count"] == 2
    assert any(item["reason"] == "backend_unavailable" for item in result["pruned"])


def test_score_excludes_quality_failure() -> None:
    good = eligible_row("good", 128, 10, 0)
    bad = eligible_row("bad", 128, 1, 0)
    bad["validation"]["quality_pass"] = False
    assert [item["plan_id"] for item in score_candidates([good, bad], objective="session")] == ["good"]


def test_policy_is_workload_conditioned_and_drift_safe() -> None:
    rows = [
        eligible_row("fast-start", 128, 10, 1), eligible_row("compiled", 128, 8, 100),
        eligible_row("fast-start", 1024, 20, 1), eligible_row("compiled", 1024, 12, 100),
    ]
    policy = build_policy(rows, hardware_sha256="hash", model_id="model", holdout_run_ids=["holdout"])
    selected = select_plan(
        policy,
        {"prompt_tokens": 128, "concurrency": 1, "session_requests": 20},
        hardware_sha256="hash",
    )
    assert selected["reason"] == "matched_rule"
    assert select_plan(policy, {"prompt_tokens": 128, "concurrency": 1, "session_requests": 20}, hardware_sha256="other")["policy_status"] == "STALE"


def test_break_even() -> None:
    result = session_break_even(startup_a_ms=100, request_a_ms=5, startup_b_ms=0, request_b_ms=10)
    assert result["requests"] == 20
