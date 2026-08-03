from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from edgeflow.metrics.statistics import robust_cv

E06_SEQUENCE = (128, 128, 1024, 128, 2048, 1024, 4096)
E06_MODES = ("false", "auto", "true")


def summarize_dynamic_shape_study(
    cases: list[dict[str, Any]],
    *,
    repetitions: int,
    sequence: tuple[int, ...] = E06_SEQUENCE,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    issues: list[str] = []
    warnings: list[str] = []
    expected_observations = repetitions * len(sequence)
    completed = {str(case.get("dynamic_mode")): case for case in cases if case.get("status") == "COMPLETED"}
    hashes_by_mode = {
        mode: dict(case.get("output_hashes", {})) for mode, case in completed.items()
    }
    prompts = sorted({prompt for hashes in hashes_by_mode.values() for prompt in hashes})
    consensus = {
        prompt: Counter(hashes[prompt] for hashes in hashes_by_mode.values() if prompt in hashes)
        .most_common(1)[0][0]
        for prompt in prompts
    }
    for mode in E06_MODES:
        case = completed.get(mode)
        if case is None:
            issues.append(f"dynamic={mode} did not complete")
            continue
        observations = list(case.get("observations", []))
        if len(observations) != expected_observations:
            issues.append(
                f"dynamic={mode} has {len(observations)} observations; expected {expected_observations}"
            )
            continue
        block_totals: list[float] = []
        for block in range(repetitions):
            rows = [row for row in observations if int(row["block"]) == block]
            block_totals.append(sum(float(row["latency_ms"]) for row in rows))
        first_occurrence: dict[int, float] = {}
        later: dict[int, list[float]] = {}
        for row in observations:
            prompt = int(row["prompt_tokens"])
            latency = float(row["latency_ms"])
            if prompt not in first_occurrence:
                first_occurrence[prompt] = latency
            else:
                later.setdefault(prompt, []).append(latency)
        spike_ratios = {
            str(prompt): first_occurrence[prompt] / statistics.median(later[prompt])
            for prompt in first_occurrence
            if later.get(prompt) and statistics.median(later[prompt]) > 0
        }
        steady_cv = robust_cv(block_totals[1:])
        stable = steady_cv <= 0.10
        if not stable:
            issues.append(
                f"dynamic={mode} steady mixed-sequence robust CV is {steady_cv:.3%}; maximum is 10%"
            )
        mode_hashes = hashes_by_mode.get(mode, {})
        matches = sum(mode_hashes.get(prompt) == expected for prompt, expected in consensus.items())
        agreement = matches / len(consensus) if consensus else 0.0
        correctness_pass = agreement == 1.0 and len(mode_hashes) == len(consensus)
        if not correctness_pass:
            warnings.append(
                f"dynamic={mode} exact greedy output agreement is {agreement:.1%}; mode is excluded"
            )
        summaries.append(
            {
                "dynamic_mode": mode,
                "observation_count": len(observations),
                "unique_graphs": int(case.get("final_counters", {}).get("stats.unique_graphs", 0)),
                "graph_breaks": int(case.get("final_counters", {}).get("graph_break.total", 0)),
                "cold_sequence_ms": block_totals[0],
                "steady_sequence_median_ms": float(statistics.median(block_totals[1:])),
                "steady_sequence_robust_cv": steady_cv,
                "stability_pass": stable,
                "correctness_pass": correctness_pass,
                "exact_output_scope_agreement": agreement,
                "maximum_first_shape_spike_ratio": max(spike_ratios.values(), default=1.0),
                "first_shape_spike_ratios": spike_ratios,
                "peak_vram_bytes": max(
                    (int(row.get("peak_vram_bytes") or 0) for row in observations),
                    default=0,
                ),
            }
        )
    output_sets = {
        tuple(sorted((str(key), str(value)) for key, value in case.get("output_hashes", {}).items()))
        for case in completed.values()
    }
    cross_mode_correctness = len(output_sets) == 1 and len(completed) == len(E06_MODES)

    shape_bucket_rule: dict[str, Any] | None = None
    eligible_summaries = [
        row for row in summaries if row["correctness_pass"] and row["stability_pass"]
    ]
    if len(eligible_summaries) < 2:
        issues.append("fewer than two dynamic modes passed correctness and stability gates")
    if eligible_summaries:
        winner = min(
            eligible_summaries,
            key=lambda row: (
                float(row["steady_sequence_median_ms"]),
                int(row["unique_graphs"]),
            ),
        )
        if winner["dynamic_mode"] == "false":
            shape_bucket_rule = {
                "strategy": "static_exact_buckets",
                "buckets": sorted(set(sequence)),
                "compile_once_per_bucket": True,
                "fallback": "dynamic_auto_for_unseen_shape",
                "selected_from": "minimum measured steady mixed-sequence latency",
            }
        else:
            shape_bucket_rule = {
                "strategy": "dynamic_graph",
                "dynamic_mode": winner["dynamic_mode"],
                "observed_shapes": sorted(set(sequence)),
                "fallback": "pytorch_eager_on_compile_failure",
                "selected_from": "minimum measured steady mixed-sequence latency",
            }
    passed = not issues and shape_bucket_rule is not None and repetitions >= 30
    return {
        "schema_version": "1.0",
        "experiment_id": "E06",
        "status": "PASS" if passed else "INCOMPLETE",
        "pass": passed,
        "protocol_status": "FORMAL" if repetitions >= 30 else "DEVELOPMENT",
        "repetitions": repetitions,
        "sequence": list(sequence),
        "cross_mode_correctness": cross_mode_correctness,
        "correctness_consensus_hashes": consensus,
        "mode_summaries": summaries,
        "excluded_modes": [
            row["dynamic_mode"]
            for row in summaries
            if not row["correctness_pass"] or not row["stability_pass"]
        ],
        "shape_bucket_rule": shape_bucket_rule,
        "issues": issues,
        "warnings": warnings,
        "claim_scope": (
            "Registered mixed-shape sequence on the pinned local model and hardware only."
            if passed
            else "No dynamic-shape recommendation is eligible until every registered mode completes."
        ),
    }
