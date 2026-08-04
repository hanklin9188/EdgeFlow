from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from edgeflow.core.models import utc_now
from edgeflow.core.serialization import read_json, sha256_value

_SCOPE_FIELDS = (
    "model_id",
    "prompt_source",
    "prompt_tokens",
    "output_tokens",
    "batch_size",
    "concurrency",
    "arrival_pattern",
    "request_rate",
    "sampling",
    "seed",
    "streaming",
    "session_requests",
    "quality_profile",
)

_GPU_IDENTITY_FIELDS = (
    "vendor",
    "name",
    "uuid",
    "vram_bytes",
    "compute_capability",
    "driver_version",
    "power_limit_w",
)
_HOST_IDENTITY_FIELDS = (
    "cpu",
    "logical_cores",
    "physical_cores",
    "ram_bytes",
    "execution_mode",
    "kernel",
    "os",
    "wsl_version",
)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get("metrics", {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _scope(workload: dict[str, Any]) -> dict[str, Any]:
    return {key: workload.get(key) for key in _SCOPE_FIELDS}


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _hardware_identity(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Exclude capture time, git state, and the selected runtime service.

    Those values must remain in each provenance artifact, but they are expected to
    differ across runtime runs and cannot identify whether the physical test host
    changed.
    """

    gpu = fingerprint.get("gpu", {})
    host = fingerprint.get("host", {})
    return {
        "gpu": {key: gpu.get(key) for key in _GPU_IDENTITY_FIELDS},
        "host": {key: host.get(key) for key in _HOST_IDENTITY_FIELDS},
    }


def _plan_scope(plan: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    backend_args = plan.get("backend_args", {})
    return {
        "backend": plan.get("backend"),
        "model_revision": manifest.get("model_revision"),
        "model_format": plan.get("model_format"),
        "dtype": plan.get("dtype"),
        "quantization": plan.get("quantization"),
        "execution_mode": backend_args.get("execution_mode"),
        "cuda_graph": plan.get("cuda_graph"),
        "max_num_batched_tokens": plan.get("max_num_batched_tokens"),
        "max_num_seqs": plan.get("max_num_seqs"),
    }


def audit_runtime_fairness(
    run_dirs: list[Path], *, minimum_repetitions: int = 30
) -> dict[str, Any]:
    """Audit whether measured runtime runs are eligible for a matched comparison."""

    if len(run_dirs) < 2:
        raise ValueError("fairness audit requires at least two run directories")
    records: list[dict[str, Any]] = []
    reference_scope: dict[str, Any] | None = None
    reference_hardware: dict[str, Any] | None = None
    global_issues: list[dict[str, str]] = []

    for raw_path in run_dirs:
        path = raw_path.resolve()
        required = (
            "run_manifest.json",
            "workload.json",
            "execution_plan.json",
            "hardware_fingerprint.json",
            "metrics.jsonl",
            "validation_verdict.json",
        )
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            records.append(
                {
                    "run_id": path.name,
                    "artifact_path": str(path),
                    "eligible": False,
                    "issues": [f"missing artifact: {name}" for name in missing],
                }
            )
            continue

        manifest = read_json(path / "run_manifest.json")
        workload = read_json(path / "workload.json")
        plan = read_json(path / "execution_plan.json")
        validation = read_json(path / "validation_verdict.json")
        fingerprint = read_json(path / "hardware_fingerprint.json")
        measured = [
            row
            for row in _read_metric_rows(path / "metrics.jsonl")
            if row.get("phase") == "end_to_end"
        ]
        request_latencies = _metric_values(measured, "request_latency_ms")
        issues: list[str] = []
        current_scope = _scope(workload)
        hardware = _hardware_identity(fingerprint)

        if reference_scope is None:
            reference_scope = current_scope
            reference_hardware = hardware
        else:
            if current_scope != reference_scope:
                issues.append("workload scope differs from the first run")
            if hardware != reference_hardware:
                issues.append("stable hardware identity differs from the first run")
        if manifest.get("source_type") != "measured":
            issues.append("source_type is not measured")
        if manifest.get("profiler_level") != "none":
            issues.append("profiled timing cannot enter a fairness comparison")
        if validation.get("policy_eligible") is not True:
            issues.append("run did not pass all policy-eligibility gates")
        if validation.get("quality_pass") is not True:
            issues.append("quality equivalence is not established")
        group_size = max(int(workload.get("batch_size", 1)), int(workload.get("concurrency", 1)))
        repetition_count = len(measured) // group_size
        if repetition_count < minimum_repetitions:
            issues.append(
                f"only {repetition_count} request groups; {minimum_repetitions} are required"
            )
        if not request_latencies:
            issues.append("no measured request latency rows")

        records.append(
            {
                "run_id": manifest.get("run_id", path.name),
                "artifact_path": str(path),
                "backend": plan.get("backend"),
                "plan_id": plan.get("plan_id"),
                "model_revision": manifest.get("model_revision"),
                "hardware_capture_sha256": manifest.get("hardware_fingerprint_sha256"),
                "plan_scope": _plan_scope(plan, manifest),
                "eligible": not issues,
                "issues": issues,
                "measured_request_groups": repetition_count,
                "measured_request_rows": len(measured),
                "metrics": {
                    "median_request_latency_ms": (
                        statistics.median(request_latencies) if request_latencies else None
                    ),
                    "p95_request_latency_ms": _percentile(request_latencies, 95),
                    "median_ttft_ms": (
                        statistics.median(values)
                        if (values := _metric_values(measured, "ttft_ms"))
                        else None
                    ),
                    "median_tpot_ms": (
                        statistics.median(values)
                        if (values := _metric_values(measured, "tpot_ms"))
                        else None
                    ),
                    "median_generation_tokens_per_s": (
                        statistics.median(values)
                        if (values := _metric_values(measured, "generation_tokens_per_s"))
                        else None
                    ),
                },
            }
        )

    eligible = [row for row in records if row.get("eligible")]
    eligible_backends = {row.get("backend") for row in eligible}
    if len(eligible_backends) < 2:
        global_issues.append(
            {
                "code": "INSUFFICIENT_RUNTIME_SCOPES",
                "message": "At least two distinct eligible runtime backends are required.",
            }
        )
    invalid = any(
        "differs from the first run" in issue for row in records for issue in row["issues"]
    )
    status = (
        "INVALID"
        if invalid
        else "PASS"
        if not global_issues and len(eligible) == len(records)
        else "INCOMPLETE"
    )
    descriptive_order = []
    if status == "PASS":
        descriptive_order = [
            row["run_id"]
            for row in sorted(
                eligible,
                key=lambda item: item["metrics"]["median_request_latency_ms"],
            )
        ]
    plan_scopes = [row["plan_scope"] for row in eligible]
    differing_plan_fields = sorted(
        key
        for key in (plan_scopes[0] if plan_scopes else {})
        if key != "backend" and len({json.dumps(scope.get(key), sort_keys=True) for scope in plan_scopes}) > 1
    )
    comparison_caveats = []
    if differing_plan_fields:
        comparison_caveats.append(
            "Plans differ in "
            + ", ".join(differing_plan_fields)
            + "; the result is a quality-gated plan comparison, not a pure runtime causal effect."
        )
    identity = {
        "run_ids": sorted(str(row.get("run_id")) for row in records),
        "scope": reference_scope,
        "hardware": reference_hardware,
    }
    return {
        "schema_version": "1.0",
        "experiment_id": "E09",
        "audit_id": f"fairness-{sha256_value(identity)[:12]}",
        "source_type": "measured",
        "status": status,
        "pass": status == "PASS",
        "minimum_repetitions": minimum_repetitions,
        "stable_hardware_identity": reference_hardware,
        "stable_hardware_identity_sha256": (
            sha256_value(reference_hardware) if reference_hardware is not None else None
        ),
        "comparison_scope": reference_scope,
        "runs": records,
        "descriptive_latency_order": descriptive_order,
        "issues": global_issues,
        "comparison_caveats": comparison_caveats,
        "causal_runtime_isolation": status == "PASS" and not differing_plan_fields,
        "created_at": utc_now(),
        "claim_scope": (
            "Descriptive ordering of independently quality-gated plans within the exact audited "
            "workload only; no extrapolation and no pure-runtime causal claim when plan scopes differ."
            if status == "PASS"
            else "No cross-runtime ordering is permitted until every fairness gate passes."
        ),
    }
