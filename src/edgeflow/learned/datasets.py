from __future__ import annotations

from collections import Counter
from typing import Any

from edgeflow.core.models import utc_now
from edgeflow.core.serialization import sha256_value


def _point_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("hardware_fingerprint_sha256")),
        str(row.get("plan_sha256") or row.get("plan_id")),
        str(row.get("workload_sha256") or row.get("workload_id")),
    )


def _split(workload_group: str) -> str:
    # All plans/reruns for one workload group stay in one split.
    bucket = int(sha256_value(workload_group)[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "holdout"


def build_cost_dataset(
    rows: list[dict[str, Any]], *, required_unique_points: int = 2_000
) -> dict[str, Any]:
    """Create one aggregated row per hardware-plan-workload point, never per repetition."""

    eligible = [
        row
        for row in rows
        if row.get("source_type") == "measured"
        and row.get("validation", {}).get("policy_eligible") is True
    ]
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicates = 0
    for row in sorted(
        eligible, key=lambda item: (str(item.get("created_at")), str(item["run_id"]))
    ):
        key = _point_key(row)
        if key in unique:
            duplicates += 1
        workload = row["workload"]
        workload_group = sha256_value(
            {
                "model_id": workload.get("model_id"),
                "prompt_source": workload.get("prompt_source"),
                "prompt_tokens": workload.get("prompt_tokens"),
                "output_tokens": workload.get("output_tokens"),
                "batch_size": workload.get("batch_size", 1),
                "concurrency": workload.get("concurrency", 1),
                "arrival_pattern": workload.get("arrival_pattern"),
            }
        )
        unique[key] = {
            "point_id": sha256_value(key),
            "run_id": row["run_id"],
            "created_at": row.get("created_at"),
            "hardware_fingerprint_sha256": key[0],
            "plan_sha256": key[1],
            "workload_sha256": key[2],
            "workload_group_sha256": workload_group,
            "split": _split(workload_group),
            "features": {
                "backend": row.get("backend"),
                "prompt_tokens": workload.get("prompt_tokens"),
                "output_tokens": workload.get("output_tokens"),
                "batch_size": workload.get("batch_size", 1),
                "concurrency": workload.get("concurrency", 1),
                "session_requests": workload.get("session_requests", 20),
            },
            "targets": row["metrics"],
        }
    points = sorted(unique.values(), key=lambda item: item["point_id"])
    split_counts = Counter(str(point["split"]) for point in points)
    ready = len(points) >= required_unique_points and all(
        split_counts.get(name, 0) > 0 for name in ("train", "validation", "holdout")
    )
    identity = {
        "points": points,
        "required_unique_points": required_unique_points,
    }
    return {
        "schema_version": "1.0",
        "experiment_id": "E25",
        "created_at": utc_now(),
        "source_type": "measured",
        "status": "PASS" if ready else "BLOCKED_PREREQUISITE",
        "pass": ready,
        "unique_validated_points": len(points),
        "required_unique_points": required_unique_points,
        "deduplicated_eligible_runs": duplicates,
        "split_strategy": "workload-group hash; no workload group crosses train/validation/holdout",
        "split_counts": dict(split_counts),
        "snapshot_sha256": sha256_value(identity),
        "points": points,
        "claim_scope": "Dataset readiness only; repetitions are aggregated and never independent rows.",
    }


def build_intervention_dataset(
    evidence_records: list[dict[str, Any]], *, required_per_class: int = 50
) -> dict[str, Any]:
    eligible = [
        record
        for record in evidence_records
        if record.get("status") in {"SUPPORTED", "REJECTED"}
        and record.get("evidence_level") in {"E3", "E4", "E5"}
    ]
    # Evidence IDs, not individual paired samples, are classifier rows.
    unique = {str(record["evidence_id"]): record for record in eligible}
    labels = Counter(str(record.get("hypothesis")) for record in unique.values())
    ready = bool(labels) and all(count >= required_per_class for count in labels.values())
    return {
        "schema_version": "1.0",
        "experiment_id": "E27",
        "created_at": utc_now(),
        "source_type": "measured",
        "status": "READY_FOR_EVALUATION" if ready else "BLOCKED_PREREQUISITE",
        "pass": False,
        "unique_evidence_records": len(unique),
        "label_counts": dict(labels),
        "required_per_available_class": required_per_class,
        "rows": [
            {
                "evidence_id": evidence_id,
                "hypothesis": record.get("hypothesis"),
                "status": record.get("status"),
                "evidence_level": record.get("evidence_level"),
            }
            for evidence_id, record in sorted(unique.items())
        ],
        "claim_scope": "Classifier dataset readiness only; no classifier claim before holdout evaluation.",
    }
