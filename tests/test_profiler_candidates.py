from edgeflow.profiler import rank_kernel_candidates


def test_kernel_candidate_ranking_requires_every_fusion_pattern() -> None:
    operations = [
        {"name": "aten::silu", "calls": 20, "self_device_time_us": 100.0},
        {"name": "aten::mul", "calls": 20, "self_device_time_us": 80.0},
        {"name": "aten::softmax", "calls": 5, "self_device_time_us": 20.0},
    ]
    ranked = rank_kernel_candidates(operations)

    assert ranked[0]["candidate_id"] == "fused-swiglu-v1"
    assert {row["candidate_id"] for row in ranked} == {
        "fused-swiglu-v1",
        "fused-scaled-softmax-v1",
    }


def test_kernel_candidate_ranking_does_not_invent_missing_ops() -> None:
    assert rank_kernel_candidates([{"name": "aten::matmul", "calls": 10}]) == []
