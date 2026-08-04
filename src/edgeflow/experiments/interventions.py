from __future__ import annotations

import statistics
from typing import Any

from edgeflow.metrics.statistics import paired_bootstrap


def _paired_result(
    baseline: list[float],
    intervention: list[float],
    *,
    lower_is_better: bool,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if lower_is_better:
        result = paired_bootstrap(
            baseline,
            intervention,
            resamples=resamples,
            seed=seed,
        )
    else:
        result = paired_bootstrap(
            intervention,
            baseline,
            resamples=resamples,
            seed=seed,
        )
    result["lower_is_better"] = lower_is_better
    result["baseline_median"] = float(statistics.median(baseline))
    result["intervention_median"] = float(statistics.median(intervention))
    return result


def build_intervention_evidence(
    *,
    evidence_id: str,
    hypothesis: str,
    observation: dict[str, Any],
    intervention_name: str,
    changed_variables: dict[str, Any],
    controlled_variables: dict[str, Any],
    mediator_name: str,
    baseline_mediator: list[float],
    intervention_mediator: list[float],
    baseline_outcome_ms: list[float],
    intervention_outcome_ms: list[float],
    supporting_run_ids: list[str],
    scope: dict[str, Any],
    negative_control: str | None = None,
    negative_control_pass: bool | None = None,
    holdout_confirmed: bool = False,
    correctness_pass: bool = True,
    controls_match: bool = True,
    mediator_lower_is_better: bool = True,
    resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a schema-valid causal record without promoting incomplete evidence.

    A matched pair can reach E3. It reaches E4/SUPPORTED only when the mediator,
    unprofiled outcome, correctness, negative control, and holdout all pass.
    """

    if len(supporting_run_ids) < 2:
        raise ValueError("a matched intervention requires baseline and intervention run IDs")
    sample_sizes = {
        len(baseline_mediator),
        len(intervention_mediator),
        len(baseline_outcome_ms),
        len(intervention_outcome_ms),
    }
    if len(sample_sizes) != 1 or next(iter(sample_sizes)) < 30:
        raise ValueError("formal intervention evidence requires at least 30 complete paired samples")

    mediator = _paired_result(
        baseline_mediator,
        intervention_mediator,
        lower_is_better=mediator_lower_is_better,
        resamples=resamples,
        seed=seed,
    )
    outcome = _paired_result(
        baseline_outcome_ms,
        intervention_outcome_ms,
        lower_is_better=True,
        resamples=resamples,
        seed=seed + 1,
    )
    matched_valid = correctness_pass and controls_match
    matched_support = bool(
        matched_valid
        and mediator["claim_direction_supported"]
        and outcome["claim_direction_supported"]
    )
    controls_complete = negative_control_pass is True and holdout_confirmed
    if matched_support and controls_complete:
        status = "SUPPORTED"
        evidence_level = "E4"
    elif matched_support:
        status = "PENDING"
        evidence_level = "E3"
    elif matched_valid and (
        mediator["paired_geometric_speedup_ci95"][1] <= 1.0
        or outcome["paired_geometric_speedup_ci95"][1] <= 1.0
    ):
        status = "REJECTED"
        evidence_level = "E3"
    else:
        status = "INCONCLUSIVE"
        evidence_level = "E3" if matched_valid else "E2"

    alternatives: list[str] = []
    if not controls_match:
        alternatives.append("Matched controls differ; the observed effect is confounded.")
    if not correctness_pass:
        alternatives.append("Correctness failed; performance observations are not deployable.")
    if negative_control_pass is not True:
        alternatives.append("The preregistered negative control has not passed.")
    if not holdout_confirmed:
        alternatives.append("An untouched workload-bucket holdout has not confirmed the effect.")

    return {
        "schema_version": "1.0",
        "evidence_id": evidence_id,
        "hypothesis": hypothesis,
        "status": status,
        "observations": [observation],
        "intervention": {
            "name": intervention_name,
            "changed_variables": changed_variables,
            "negative_control": negative_control,
        },
        "controlled_variables": controlled_variables,
        "mediator_results": [{"metric": mediator_name, **mediator}],
        "outcome_results": [{"metric": "request_latency_ms", **outcome}],
        "supporting_run_ids": supporting_run_ids,
        "scope": {
            **scope,
            "correctness_pass": correctness_pass,
            "controls_match": controls_match,
            "negative_control_pass": negative_control_pass,
            "holdout_confirmed": holdout_confirmed,
        },
        "evidence_level": evidence_level,
        "alternative_explanations": alternatives,
        "reviewed_by": None,
    }
