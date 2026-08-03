from __future__ import annotations

import itertools
import statistics
from collections import Counter, defaultdict
from typing import Any

from edgeflow.optimizer import session_break_even


def _bucket(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    workload = row["workload"]
    prompt = workload["prompt_tokens"]
    if not isinstance(prompt, int):
        raise ValueError("formal policy analysis requires explicit prompt buckets")
    return (
        prompt,
        int(workload["output_tokens"]),
        int(workload.get("batch_size", 1)),
        int(workload.get("concurrency", 1)),
        int(workload.get("session_requests", row.get("session_requests", 20))),
    )


def _session_cost(row: dict[str, Any]) -> float:
    metrics = row["metrics"]
    requests = _bucket(row)[-1]
    return float(metrics.get("startup_ms", 0.0)) + requests * float(
        metrics["request_latency_ms"]
    )


def _plan_identity(row: dict[str, Any]) -> str:
    """Prefer a workload-independent plan signature when the artifact supplies one."""

    return str(row.get("plan_signature") or row["plan_id"])


def fixed_plan_dominance(
    rows: list[dict[str, Any]],
    *,
    expected_bucket_count: int = 45,
    practical_threshold_pct: float = 2.0,
) -> dict[str, Any]:
    """Evaluate E10 on only the complete, eligible plan-by-bucket rectangle."""

    eligible = [
        row
        for row in rows
        if row.get("source_type") == "measured"
        and row.get("paired_prompt_ids") is True
        and row.get("validation", {}).get("policy_eligible") is True
        and row.get("validation", {}).get("quality_pass") is True
    ]
    by_plan: dict[str, dict[tuple[int, int, int, int, int], dict[str, Any]]] = defaultdict(dict)
    for row in eligible:
        by_plan[_plan_identity(row)][_bucket(row)] = row
    plan_ids = sorted(by_plan)
    common_buckets = (
        set.intersection(*(set(by_plan[plan_id]) for plan_id in plan_ids)) if plan_ids else set()
    )
    coverage = len(common_buckets) / expected_bucket_count if expected_bucket_count else 1.0
    if len(plan_ids) < 2 or not common_buckets:
        return {
            "schema_version": "1.0",
            "experiment_id": "E10",
            "status": "INCOMPLETE",
            "pass": False,
            "eligible_plan_count": len(plan_ids),
            "common_bucket_count": len(common_buckets),
            "expected_bucket_count": expected_bucket_count,
            "coverage": coverage,
            "issues": ["At least two quality-passing plans on common workload buckets are required."],
            "claim_scope": "No fixed-plan or conditioned-policy conclusion is permitted.",
        }

    costs = {
        plan_id: {
            bucket: _session_cost(by_plan[plan_id][bucket]) for bucket in common_buckets
        }
        for plan_id in plan_ids
    }
    fixed_expected = {
        plan_id: statistics.fmean(bucket_costs.values())
        for plan_id, bucket_costs in costs.items()
    }
    global_winner = min(fixed_expected, key=fixed_expected.__getitem__)
    per_bucket_winners: list[dict[str, Any]] = []
    oracle_costs: list[float] = []
    for bucket in sorted(common_buckets):
        winner = min(plan_ids, key=lambda plan_id: costs[plan_id][bucket])
        winner_cost = costs[winner][bucket]
        oracle_costs.append(winner_cost)
        per_bucket_winners.append(
            {
                "bucket": {
                    "prompt_tokens": bucket[0],
                    "output_tokens": bucket[1],
                    "batch_size": bucket[2],
                    "concurrency": bucket[3],
                    "session_requests": bucket[4],
                },
                "plan_id": winner,
                "session_cost_ms": winner_cost,
            }
        )
    global_cost = fixed_expected[global_winner]
    oracle_cost = statistics.fmean(oracle_costs)
    oracle_gain = (global_cost - oracle_cost) / global_cost * 100.0 if global_cost else 0.0
    full_coverage = len(common_buckets) >= expected_bucket_count
    return {
        "schema_version": "1.0",
        "experiment_id": "E10",
        "status": "PASS" if full_coverage else "INCOMPLETE",
        "pass": full_coverage,
        "eligible_plan_count": len(plan_ids),
        "common_bucket_count": len(common_buckets),
        "expected_bucket_count": expected_bucket_count,
        "coverage": coverage,
        "global_winner": global_winner,
        "fixed_plan_expected_cost_ms": fixed_expected,
        "oracle_expected_cost_ms": oracle_cost,
        "oracle_gain_pct": oracle_gain,
        "conditioned_policy_motivated": bool(
            full_coverage and oracle_gain >= practical_threshold_pct
        ),
        "practical_threshold_pct": practical_threshold_pct,
        "per_bucket_winners": per_bucket_winners,
        "issues": [] if full_coverage else ["The preregistered 45-bucket grid is incomplete."],
        "claim_scope": (
            "Equal-weight expected session cost over the complete preregistered grid."
            if full_coverage
            else "Descriptive partial-grid result only; no E10 conclusion is permitted."
        ),
    }


def session_break_even_study(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate E19 intersections only between exact-scope eligible plans."""

    eligible = [
        row
        for row in rows
        if row.get("source_type") == "measured"
        and row.get("paired_prompt_ids") is True
        and row.get("validation", {}).get("policy_eligible") is True
        and row.get("validation", {}).get("quality_pass") is True
    ]
    grouped: dict[tuple[int, int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[_bucket(row)].append(row)
    comparisons: list[dict[str, Any]] = []
    for bucket, bucket_rows in sorted(grouped.items()):
        unique = {_plan_identity(row): row for row in bucket_rows}
        for plan_a, plan_b in itertools.combinations(sorted(unique), 2):
            row_a, row_b = unique[plan_a], unique[plan_b]
            result = session_break_even(
                startup_a_ms=float(row_a["metrics"].get("startup_ms", 0.0)),
                request_a_ms=float(row_a["metrics"]["request_latency_ms"]),
                startup_b_ms=float(row_b["metrics"].get("startup_ms", 0.0)),
                request_b_ms=float(row_b["metrics"]["request_latency_ms"]),
            )
            comparisons.append(
                {
                    "bucket": list(bucket),
                    "plan_a": plan_a,
                    "plan_b": plan_b,
                    **result,
                }
            )
    return {
        "schema_version": "1.0",
        "experiment_id": "E19",
        "status": "PASS" if comparisons else "INCOMPLETE",
        "pass": bool(comparisons),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "claim_scope": (
            "Analytical break-even within exact measured workload scopes."
            if comparisons
            else "No exact-scope eligible plan pair is available."
        ),
    }


def audit_learned_prerequisites(
    rows: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]],
    *,
    grounded_question_count: int,
) -> dict[str, Any]:
    unique_points = {
        (
            str(row.get("hardware_fingerprint_sha256")),
            str(row.get("plan_sha256") or row.get("plan_id")),
            str(row.get("workload_sha256") or row.get("workload_id")),
        )
        for row in rows
        if row.get("source_type") == "measured"
        and row.get("validation", {}).get("policy_eligible") is True
    }
    supported_labels = Counter(
        str(record.get("hypothesis"))
        for record in evidence_records
        if record.get("status") in {"SUPPORTED", "REJECTED"}
        and record.get("evidence_level") in {"E3", "E4", "E5"}
    )
    per_class_minimum_met = bool(supported_labels) and all(
        count >= 50 for count in supported_labels.values()
    )
    e25_ready = len(unique_points) >= 2_000
    return {
        "schema_version": "1.0",
        "experiments": {
            "E25": {
                "status": "READY" if e25_ready else "BLOCKED_PREREQUISITE",
                "unique_validated_points": len(unique_points),
                "required": 2_000,
            },
            "E26": {
                "status": "READY" if e25_ready else "BLOCKED_PREREQUISITE",
                "prerequisite": "E25 validated dataset",
            },
            "E27": {
                "status": "READY" if per_class_minimum_met else "BLOCKED_PREREQUISITE",
                "intervention_label_counts": dict(supported_labels),
                "required_per_available_class": 50,
            },
            "E28": {
                "status": "READY" if grounded_question_count > 0 else "BLOCKED_PREREQUISITE",
                "grounded_question_count": grounded_question_count,
                "prerequisite": "fixed grounded question set with expected citations and answers",
            },
        },
        "claim_scope": "Readiness audit only; READY does not substitute for experiment validation.",
    }
