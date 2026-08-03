from __future__ import annotations

from collections import defaultdict
from typing import Any

from edgeflow.core.models import utc_now
from edgeflow.core.serialization import sha256_value
from edgeflow.optimizer import score_candidates


def _scope_key(row: dict[str, Any]) -> tuple[int, int, int]:
    workload = row["workload"]
    prompt = workload["prompt_tokens"]
    if not isinstance(prompt, int):
        raise ValueError("policy synthesis requires explicit prompt buckets")
    return prompt, int(workload["concurrency"]), int(workload["session_requests"])


def build_policy(
    rows: list[dict[str, Any]],
    *,
    hardware_sha256: str,
    model_id: str,
    quality_profile: str = "balanced",
    objective: str = "session",
    holdout_run_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build an explainable decision list exclusively from eligible measured rows."""

    compatible = [
        row for row in rows
        if row.get("workload", {}).get("model_id") == model_id
        and row.get("source_type") == "measured"
    ]
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in compatible:
        grouped[_scope_key(row)].append(row)
    rules: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ranked = score_candidates(grouped[key], objective=objective)
        if not ranked:
            continue
        winner = ranked[0]
        winners.append(winner)
        prompt, concurrency, session_requests = key
        evidence_ids = sorted(set(winner.get("evidence_ids", [])))
        if not evidence_ids:
            continue
        rules.append(
            {
                "priority": len(rules),
                "predicate": {
                    "prompt_tokens_lte": prompt,
                    "concurrency_eq": concurrency,
                    "session_requests_lte": session_requests,
                },
                "plan_id": winner["plan_id"],
                "evidence_ids": evidence_ids,
                "expected_metrics": winner["metrics"],
                "scope_notes": "No extrapolation outside the measured bucket; fallback applies.",
            }
        )
    if not rules:
        raise ValueError("no policy rules have eligible measured runs with evidence")
    support_count: dict[str, int] = defaultdict(int)
    objective_sum: dict[str, float] = defaultdict(float)
    for winner in winners:
        plan_id = winner["plan_id"]
        support_count[plan_id] += 1
        objective_sum[plan_id] += float(winner["objective_score"])
    fallback_plan_id = min(
        support_count,
        key=lambda plan_id: (-support_count[plan_id], objective_sum[plan_id] / support_count[plan_id]),
    )
    holdouts = holdout_run_ids or []
    identity = {"hardware": hardware_sha256, "model": model_id, "rules": rules}
    return {
        "schema_version": "1.0",
        "policy_id": f"policy-{sha256_value(identity)[:12]}",
        "hardware_fingerprint_sha256": hardware_sha256,
        "model_id": model_id,
        "objective": {"name": objective},
        "quality_constraint": {"profile": quality_profile, "hard_gate": True},
        "rules": rules,
        "fallback_plan_id": fallback_plan_id,
        "holdout_validation": {"status": "PASS" if holdouts else "NOT_RUN", "run_ids": holdouts},
        "created_at": utc_now(),
        "expires_on_environment_change": True,
    }


def _matches(predicate: dict[str, Any], workload: dict[str, Any]) -> bool:
    checks = {
        "prompt_tokens_lte": lambda value: int(workload["prompt_tokens"]) <= int(value),
        "prompt_tokens_gt": lambda value: int(workload["prompt_tokens"]) > int(value),
        "concurrency_eq": lambda value: int(workload["concurrency"]) == int(value),
        "concurrency_gte": lambda value: int(workload["concurrency"]) >= int(value),
        "session_requests_lte": lambda value: int(workload["session_requests"]) <= int(value),
        "session_requests_gte": lambda value: int(workload["session_requests"]) >= int(value),
    }
    return all(checks[key](value) for key, value in predicate.items() if key in checks)


def select_plan(policy: dict[str, Any], workload: dict[str, Any], *, hardware_sha256: str) -> dict[str, Any]:
    if hardware_sha256 != policy["hardware_fingerprint_sha256"]:
        return {"plan_id": policy["fallback_plan_id"], "reason": "environment_drift", "policy_status": "STALE"}
    for rule in sorted(policy["rules"], key=lambda item: item["priority"]):
        if _matches(rule["predicate"], workload):
            return {"plan_id": rule["plan_id"], "reason": "matched_rule", "rule": rule, "policy_status": "VALID"}
    return {"plan_id": policy["fallback_plan_id"], "reason": "sparse_scope_fallback", "policy_status": "VALID"}
