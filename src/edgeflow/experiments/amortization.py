from __future__ import annotations

from typing import Any

from edgeflow.metrics.statistics import describe, paired_bootstrap


def summarize_cold_warm_study(
    samples: list[dict[str, Any]],
    *,
    repetitions: int,
    resamples: int = 10_000,
) -> dict[str, Any]:
    """Summarize E20 without conflating a cached host with an OS-cache-cold boot.

    Each pair comes from one fresh Python process: the first usable response is
    compared with a later request after the model and CUDA context are warm.  The
    scope is deliberately narrower than a machine reboot or dropped filesystem
    cache, neither of which this runner can safely manufacture.
    """

    completed = [row for row in samples if row.get("status") == "COMPLETED"]
    protocol_complete = repetitions >= 30 and len(completed) == repetitions
    errors = [
        {
            "sample_index": row.get("sample_index"),
            "error_type": row.get("error_type"),
            "error": row.get("error"),
        }
        for row in samples
        if row.get("status") != "COMPLETED"
    ]
    if not completed:
        return {
            "experiment_id": "E20",
            "status": "INCOMPLETE",
            "pass": False,
            "protocol_complete": False,
            "completed_pairs": 0,
            "required_pairs": max(30, repetitions),
            "errors": errors,
        }

    first_usable = [float(row["first_usable_ms"]) for row in completed]
    warmed_response = [float(row["warmed_response_host_ms"]) for row in completed]
    load = [float(row["load_ms"]) for row in completed]
    first_engine = [float(row["first_response_engine_ms"]) for row in completed]
    warmed_engine = [float(row["warmed_response_engine_ms"]) for row in completed]
    correctness_pass = all(
        row.get("first_output_sha256") == row.get("warmed_output_sha256")
        and bool(row.get("output_length_matches"))
        for row in completed
    )
    fingerprint_ids = {
        str(row.get("hardware_fingerprint_sha256"))
        for row in completed
        if row.get("hardware_fingerprint_sha256")
    }
    fingerprint_consistent = len(fingerprint_ids) == 1
    comparison = (
        paired_bootstrap(
            first_usable,
            warmed_response,
            resamples=resamples,
            seed=42,
        )
        if len(completed) >= 2
        else None
    )
    accepted = bool(
        protocol_complete
        and correctness_pass
        and fingerprint_consistent
        and comparison
        and comparison["claim_direction_supported"]
    )
    if not protocol_complete:
        status = "INCOMPLETE"
    elif not correctness_pass or not fingerprint_consistent:
        status = "FAILED"
    elif accepted:
        status = "PASS"
    else:
        status = "VALIDATED_NEUTRAL"

    return {
        "experiment_id": "E20",
        "status": status,
        "pass": accepted,
        "protocol_complete": protocol_complete,
        "completed_pairs": len(completed),
        "required_pairs": max(30, repetitions),
        "correctness_pass": correctness_pass,
        "hardware_fingerprint_consistent": fingerprint_consistent,
        "hardware_fingerprint_sha256": next(iter(fingerprint_ids), None),
        "statistics": {
            "fresh_process_time_to_first_usable_ms": describe(first_usable),
            "model_and_tokenizer_load_ms": describe(load),
            "first_response_engine_ms": describe(first_engine),
            "warmed_response_engine_ms": describe(warmed_engine),
            "warmed_response_host_ms": describe(warmed_response),
        },
        "comparison": comparison,
        "scope": {
            "process": "fresh Python process per pair",
            "model_files": "local pinned artifacts; OS filesystem cache warm or uncontrolled",
            "warm_arm": "same process, loaded model and initialized CUDA context",
            "timer": "host time-to-first-usable plus CUDA-event engine timings",
        },
        "unvalidated_scopes": [
            "machine reboot",
            "explicitly dropped OS filesystem cache",
            "persisted torch.compile cache across processes",
            "external llama.cpp or vLLM service restart",
        ],
        "errors": errors,
    }
