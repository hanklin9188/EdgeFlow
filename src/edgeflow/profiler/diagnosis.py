from __future__ import annotations

from typing import Any


def _numeric(features: dict[str, Any], key: str) -> float | None:
    value = features.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def diagnose_profile(summary: dict[str, Any]) -> dict[str, Any]:
    """Map literal profiler observations to bounded, testable hypotheses."""

    features = summary.get("diagnostic_features", {})
    observations: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    missing: list[str] = []

    for key in (
        "kernel_gap_ratio", "median_kernel_us", "gpu_active_ratio",
        "dram_throughput_pct_peak", "tensor_core_utilization_pct",
        "queue_delay_ratio", "peak_vram_ratio", "compile_share",
    ):
        value = _numeric(features, key)
        if value is None:
            missing.append(key)
        else:
            observations.append({"metric": key, "value": value, "source": "profiler_summary"})

    gap = _numeric(features, "kernel_gap_ratio")
    kernel_us = _numeric(features, "median_kernel_us")
    gpu_active = _numeric(features, "gpu_active_ratio")
    dram = _numeric(features, "dram_throughput_pct_peak")
    tensor = _numeric(features, "tensor_core_utilization_pct")
    queue = _numeric(features, "queue_delay_ratio")
    vram = _numeric(features, "peak_vram_ratio")
    compile_share = _numeric(features, "compile_share")

    def add(
        label: str,
        evidence: list[str],
        intervention: str,
        changed: dict[str, Any],
        mediator: str,
        negative_control: str,
    ) -> None:
        confidence = "high" if len(evidence) >= 3 else "medium" if len(evidence) == 2 else "low"
        hypotheses.append(
            {
                "label": label,
                "confidence": confidence,
                "evidence": evidence,
                "status": "HYPOTHESIS",
                "wording": f"The trace is consistent with {label}; causality requires the proposed matched intervention.",
            }
        )
        interventions.append(
            {
                "target_hypothesis": label,
                "name": intervention,
                "changed_variables": changed,
                "controlled_variables": [
                    "model_revision", "prompt_ids", "output_tokens", "dtype", "sampling", "thermal_block"
                ],
                "expected_mediator": mediator,
                "expected_outcome": "unprofiled request metric improves by at least the preregistered practical threshold",
                "negative_control": negative_control,
                "minimum_repetitions": 30,
                "rollback": "restore the baseline plan and keep the fallback enabled",
            }
        )

    launch_evidence: list[str] = []
    if gap is not None and gap >= 0.15:
        launch_evidence.append(f"kernel_gap_ratio={gap:.3f}")
    if kernel_us is not None and kernel_us <= 15:
        launch_evidence.append(f"median_kernel_us={kernel_us:.3f}")
    if gpu_active is not None and gpu_active <= 0.60:
        launch_evidence.append(f"gpu_active_ratio={gpu_active:.3f}")
    if len(launch_evidence) >= 2:
        add(
            "launch_overhead_bound", launch_evidence, "enable_cuda_graph",
            {"cuda_graph": [False, True]}, "kernel_gap_ratio decreases",
            "repeat baseline in the same randomized ABBA block",
        )

    memory_evidence: list[str] = []
    if dram is not None and dram >= 70:
        memory_evidence.append(f"dram_throughput_pct_peak={dram:.2f}")
    if tensor is not None and tensor <= 40:
        memory_evidence.append(f"tensor_core_utilization_pct={tensor:.2f}")
    if gpu_active is not None and gpu_active >= 0.60:
        memory_evidence.append(f"gpu_active_ratio={gpu_active:.3f}")
    if len(memory_evidence) >= 2:
        add(
            "memory_bandwidth_bound", memory_evidence, "precision_or_batch_intervention",
            {"quantization_or_batch": ["baseline", "one preregistered alternative"]},
            "absolute DRAM bytes or duration decreases", "same-format no-op configuration",
        )

    compute_evidence: list[str] = []
    if tensor is not None and tensor >= 70:
        compute_evidence.append(f"tensor_core_utilization_pct={tensor:.2f}")
    if gap is not None and gap <= 0.08:
        compute_evidence.append(f"kernel_gap_ratio={gap:.3f}")
    if gpu_active is not None and gpu_active >= 0.80:
        compute_evidence.append(f"gpu_active_ratio={gpu_active:.3f}")
    if len(compute_evidence) >= 2:
        add(
            "compute_bound", compute_evidence, "compile_mode_shape_intervention",
            {"compile_mode": ["default", "max-autotune"]}, "target GEMM/attention kernel duration decreases",
            "repeat default compile in matched cache state",
        )

    if vram is not None and vram >= 0.90:
        add(
            "kv_capacity_bound", [f"peak_vram_ratio={vram:.3f}"], "kv_cache_dtype_intervention",
            {"kv_cache_dtype": ["auto", "validated lower precision"]}, "KV allocation decreases",
            "same dtype with identical context/concurrency",
        )
    if queue is not None and queue >= 0.20:
        add(
            "scheduler_bound", [f"queue_delay_ratio={queue:.3f}"], "token_budget_intervention",
            {"max_num_batched_tokens": ["baseline", "neighbor"]}, "queue delay p95 decreases",
            "replay identical arrival trace using baseline token budget",
        )
    if compile_share is not None and compile_share >= 0.30:
        add(
            "compile_bound", [f"compile_share={compile_share:.3f}"], "shape_bucket_intervention",
            {"dynamic_shapes": [True, False]}, "recompile count decreases",
            "repeat identical shape sequence with original settings",
        )

    if not hypotheses:
        hypotheses.append(
            {
                "label": "insufficient_evidence",
                "confidence": "insufficient",
                "evidence": [],
                "status": "HYPOTHESIS",
                "wording": "The available profiler fields do not support a bounded bottleneck hypothesis.",
            }
        )
    return {
        "run_id": summary.get("run_id"),
        "source_type": summary.get("source_type"),
        "observations": observations,
        "hypotheses": hypotheses,
        "recommended_interventions": interventions,
        "insufficient_evidence": missing,
        "claim_status": "HYPOTHESIS_ONLY_UNTIL_MATCHED_VALIDATION",
    }
