from edgeflow.learned import build_cost_dataset, build_intervention_dataset


def _row(run_id: str, plan: str = "plan", workload: str = "workload") -> dict[str, object]:
    return {
        "run_id": run_id,
        "created_at": run_id,
        "source_type": "measured",
        "hardware_fingerprint_sha256": "hardware",
        "plan_id": plan,
        "plan_sha256": plan,
        "workload_id": workload,
        "workload_sha256": workload,
        "backend": "vllm",
        "workload": {
            "model_id": "model",
            "prompt_source": "synthetic",
            "prompt_tokens": 128,
            "output_tokens": 32,
        },
        "metrics": {"request_latency_ms": 10.0},
        "validation": {"policy_eligible": True},
    }


def test_cost_dataset_never_counts_reruns_as_independent_points() -> None:
    dataset = build_cost_dataset([_row("run-1"), _row("run-2")], required_unique_points=2)

    assert dataset["unique_validated_points"] == 1
    assert dataset["deduplicated_eligible_runs"] == 1
    assert dataset["pass"] is False


def test_intervention_dataset_counts_evidence_records_not_pairs() -> None:
    record = {
        "evidence_id": "evidence-1",
        "hypothesis": "launch_bound",
        "status": "SUPPORTED",
        "evidence_level": "E4",
        "paired_samples": list(range(30)),
    }
    dataset = build_intervention_dataset([record, record], required_per_class=2)

    assert dataset["unique_evidence_records"] == 1
    assert dataset["label_counts"] == {"launch_bound": 1}
    assert dataset["status"] == "BLOCKED_PREREQUISITE"
