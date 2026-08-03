from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from edgeflow.experiments import (
    E06_MODES,
    E06_SEQUENCE,
    audit_learned_prerequisites,
    build_intervention_evidence,
    fixed_plan_dominance,
    session_break_even_study,
    summarize_dynamic_shape_study,
)


def _row(
    plan_id: str,
    prompt_tokens: int,
    request_latency_ms: float,
    *,
    startup_ms: float = 0.0,
) -> dict:
    return {
        "plan_id": plan_id,
        "plan_sha256": f"sha-{plan_id}",
        "workload_id": f"p{prompt_tokens}",
        "workload_sha256": f"workload-{prompt_tokens}",
        "hardware_fingerprint_sha256": "hardware",
        "source_type": "measured",
        "paired_prompt_ids": True,
        "workload": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": 32,
            "batch_size": 1,
            "concurrency": 1,
            "session_requests": 20,
        },
        "metrics": {
            "request_latency_ms": request_latency_ms,
            "startup_ms": startup_ms,
        },
        "validation": {"policy_eligible": True, "quality_pass": True},
    }


def test_intervention_requires_complete_formal_pairs() -> None:
    with pytest.raises(ValueError, match="at least 30"):
        build_intervention_evidence(
            evidence_id="evidence-short",
            hypothesis="launch overhead causes decode latency",
            observation={"kernel_gap_ratio": 0.2},
            intervention_name="fuse_operator",
            changed_variables={"fusion": [False, True]},
            controlled_variables={"prompt_ids": "same"},
            mediator_name="kernel_gap_ratio",
            baseline_mediator=[0.2] * 29,
            intervention_mediator=[0.1] * 29,
            baseline_outcome_ms=[100.0] * 29,
            intervention_outcome_ms=[90.0] * 29,
            supporting_run_ids=["baseline", "intervention"],
            scope={"prompt_tokens": 128},
        )


def test_intervention_only_promotes_after_negative_control_and_holdout() -> None:
    arguments = {
        "evidence_id": "evidence-launch-001",
        "hypothesis": "launch overhead causes decode latency",
        "observation": {"kernel_gap_ratio": 0.2},
        "intervention_name": "fuse_operator",
        "changed_variables": {"fusion": [False, True]},
        "controlled_variables": {"prompt_ids": "same", "model_revision": "same"},
        "mediator_name": "kernel_gap_ratio",
        "baseline_mediator": [0.20 + index * 0.0001 for index in range(30)],
        "intervention_mediator": [0.12 + index * 0.0001 for index in range(30)],
        "baseline_outcome_ms": [100.0 + index * 0.01 for index in range(30)],
        "intervention_outcome_ms": [90.0 + index * 0.01 for index in range(30)],
        "supporting_run_ids": ["baseline", "intervention"],
        "scope": {"prompt_tokens": 128, "phase": "decode"},
        "negative_control": "repeat baseline in ABBA block",
        "negative_control_pass": True,
    }
    pending = build_intervention_evidence(**arguments)
    supported = build_intervention_evidence(**arguments, holdout_confirmed=True)

    assert pending["status"] == "PENDING"
    assert pending["evidence_level"] == "E3"
    assert supported["status"] == "SUPPORTED"
    assert supported["evidence_level"] == "E4"
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "specs" / "evidence_record.schema.json").read_text())
    jsonschema.validate(supported, schema)


def test_fixed_plan_dominance_requires_full_registered_grid() -> None:
    rows = [
        _row("a", 128, 10.0),
        _row("b", 128, 20.0),
        _row("a", 1024, 30.0),
        _row("b", 1024, 10.0),
    ]
    formal = fixed_plan_dominance(rows, expected_bucket_count=2)
    partial = fixed_plan_dominance(rows)

    assert formal["status"] == "PASS"
    assert formal["conditioned_policy_motivated"] is True
    assert {row["plan_id"] for row in formal["per_bucket_winners"]} == {"a", "b"}
    assert partial["status"] == "INCOMPLETE"
    assert partial["conditioned_policy_motivated"] is False


def test_break_even_and_learned_prerequisites_remain_scope_gated() -> None:
    rows = [_row("a", 128, 10.0, startup_ms=100.0), _row("b", 128, 20.0)]
    study = session_break_even_study(rows)
    audit = audit_learned_prerequisites(rows, [], grounded_question_count=0)

    assert study["status"] == "PASS"
    assert study["comparisons"][0]["requests"] == 10.0
    assert audit["experiments"]["E25"]["status"] == "BLOCKED_PREREQUISITE"
    assert audit["experiments"]["E27"]["status"] == "BLOCKED_PREREQUISITE"
    assert audit["experiments"]["E28"]["status"] == "BLOCKED_PREREQUISITE"


def test_dynamic_shape_summary_requires_every_formal_mode() -> None:
    output_hashes = {str(prompt): f"hash-{prompt}" for prompt in set(E06_SEQUENCE)}
    cases = []
    for mode_index, mode in enumerate(E06_MODES):
        observations = []
        for block in range(30):
            for sequence_index, prompt in enumerate(E06_SEQUENCE):
                observations.append(
                    {
                        "block": block,
                        "sequence_index": sequence_index,
                        "prompt_tokens": prompt,
                        "latency_ms": 10.0 + mode_index,
                        "peak_vram_bytes": 1024,
                    }
                )
        cases.append(
            {
                "dynamic_mode": mode,
                "status": "COMPLETED",
                "observations": observations,
                "output_hashes": output_hashes,
                "final_counters": {"stats.unique_graphs": mode_index + 1},
            }
        )

    complete = summarize_dynamic_shape_study(cases, repetitions=30)
    incomplete = summarize_dynamic_shape_study(cases[:-1], repetitions=30)

    assert complete["status"] == "PASS"
    assert complete["shape_bucket_rule"]["strategy"] == "static_exact_buckets"
    assert incomplete["status"] == "INCOMPLETE"
