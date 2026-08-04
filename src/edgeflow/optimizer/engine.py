from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from edgeflow.core.models import CapabilityReport, ExecutionPlan


def estimate_memory_bytes(
    *,
    parameter_count: int,
    bytes_per_parameter: float,
    prompt_tokens: int,
    concurrency: int,
    hidden_size: int = 3072,
) -> int:
    model = parameter_count * bytes_per_parameter
    # Conservative two-buffer KV proxy. Model-specific plugins may override it.
    kv = prompt_tokens * concurrency * hidden_size * 2 * 2
    workspace = max(512 * 1024**2, int(model * 0.12))
    return int(model + kv + workspace)


def build_candidates(
    *,
    model_id: str,
    capabilities: list[CapabilityReport],
    vram_bytes: int,
    parameter_count: int = 3_000_000_000,
    prompt_tokens: int = 1024,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Generate the finite MVP candidate space and record every deterministic prune."""

    available = {item.backend: item for item in capabilities if item.available}
    candidates: list[ExecutionPlan] = []
    pruned: list[dict[str, str]] = []

    def add(plan: ExecutionPlan, bytes_per_parameter: float) -> None:
        memory = estimate_memory_bytes(
            parameter_count=parameter_count,
            bytes_per_parameter=bytes_per_parameter,
            prompt_tokens=prompt_tokens,
            concurrency=concurrency,
        )
        if plan.backend not in available:
            pruned.append({"plan_id": plan.plan_id, "reason": "backend_unavailable"})
        elif memory > int(vram_bytes * 0.92):
            pruned.append(
                {"plan_id": plan.plan_id, "reason": f"estimated_memory_{memory}_exceeds_safe_vram"}
            )
        else:
            candidates.append(plan.with_hash())

    for dtype in ("bf16", "fp16"):
        add(
            ExecutionPlan(
                plan_id=f"pytorch-eager-{dtype}",
                model_id=model_id,
                backend="pytorch_eager",
                model_format="safetensors",
                dtype=dtype,
                custom_kernels=(),
                backend_args={},
            ),
            2.0,
        )
    for mode, dynamic in itertools.product(
        ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        (False, True),
    ):
        plan_id = f"torch-compile-{mode}-{'dynamic' if dynamic else 'static'}-bf16"
        if mode in {"reduce-overhead", "max-autotune"}:
            pruned.append(
                {"plan_id": plan_id, "reason": "internal_cudagraph_mutable_kv_unsupported"}
            )
            continue
        add(
            ExecutionPlan(
                plan_id=plan_id,
                model_id=model_id,
                backend="torch_compile",
                model_format="safetensors",
                dtype="bf16",
                compile_mode=mode,
                dynamic_shapes=dynamic,
                fullgraph=False,
                cuda_graph=mode == "reduce-overhead" and not dynamic,
                custom_kernels=(),
                backend_args={},
            ),
            2.0,
        )
    quant_bytes = {"Q8_0": 1.1, "Q6_K": 0.8, "Q5_K_M": 0.7, "Q4_K_M": 0.6}
    for quantization, bpp in quant_bytes.items():
        add(
            ExecutionPlan(
                plan_id=f"llama-cpp-{quantization.lower()}",
                model_id=model_id,
                backend="llama_cpp",
                model_format="gguf",
                quantization=quantization,
                flash_attention=True,
                custom_kernels=(),
                backend_args={"n_gpu_layers": -1},
            ),
            bpp,
        )
    for token_budget, sequences in itertools.product((1024, 2048, 4096, 8192), (1, 4, 8)):
        add(
            ExecutionPlan(
                plan_id=f"vllm-b{token_budget}-s{sequences}-bf16",
                model_id=model_id,
                backend="vllm",
                model_format="safetensors",
                dtype="bf16",
                max_num_batched_tokens=token_budget,
                max_num_seqs=sequences,
                kv_cache_dtype="auto",
                custom_kernels=(),
                backend_args={},
            ),
            2.0,
        )
    unique: dict[str, ExecutionPlan] = {}
    for candidate in candidates:
        if candidate.content_sha256 in unique:
            pruned.append({"plan_id": candidate.plan_id, "reason": "duplicate_canonical_plan"})
        else:
            unique[candidate.content_sha256] = candidate
    return {
        "candidates": [item.model_dump(mode="json") for item in unique.values()],
        "pruned": pruned,
        "candidate_count": len(unique),
        "pruned_count": len(pruned),
    }


def _minmax(values: np.ndarray) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def score_candidates(
    rows: list[dict[str, Any]], *, objective: str = "interactive"
) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in rows
        if row.get("validation", {}).get("policy_eligible") is True
        and row.get("validation", {}).get("quality_pass") is True
    ]
    if not eligible:
        return []
    if objective == "interactive":
        weights = {
            "ttft_ms": 0.35,
            "tpot_ms": 0.35,
            "p95_itl_ms": 0.15,
            "peak_vram_bytes": 0.10,
            "startup_ms": 0.05,
        }
        directions = {key: 1.0 for key in weights}
    elif objective == "throughput":
        weights = {
            "throughput_tokens_s": 0.55,
            "p95_latency_ms": 0.20,
            "peak_vram_bytes": 0.15,
            "failure_rate": 0.10,
        }
        directions = {
            "throughput_tokens_s": -1.0,
            "p95_latency_ms": 1.0,
            "peak_vram_bytes": 1.0,
            "failure_rate": 1.0,
        }
    elif objective == "session":
        for row in eligible:
            metrics = row["metrics"]
            requests = int(row.get("session_requests", 20))
            row["objective_score"] = float(metrics.get("startup_ms", 0)) + requests * float(
                metrics["request_latency_ms"]
            )
        return sorted(eligible, key=lambda item: item["objective_score"])
    else:
        raise ValueError(f"unsupported objective: {objective}")
    for metric in weights:
        if any(metric not in row.get("metrics", {}) for row in eligible):
            raise ValueError(f"all candidate rows must include {metric}")
        values = np.asarray([float(row["metrics"][metric]) for row in eligible])
        normalized = _minmax(values) * directions[metric]
        for row, value in zip(eligible, normalized, strict=True):
            row["objective_score"] = row.get("objective_score", 0.0) + weights[metric] * float(
                value
            )
    return sorted(eligible, key=lambda item: item["objective_score"])


def session_break_even(
    *,
    startup_a_ms: float,
    request_a_ms: float,
    startup_b_ms: float,
    request_b_ms: float,
) -> dict[str, Any]:
    denominator = request_b_ms - request_a_ms
    if denominator == 0:
        return {"requests": None, "reason": "equal steady-state request cost"}
    requests = (startup_a_ms - startup_b_ms) / denominator
    if requests <= 0:
        return {"requests": 0, "reason": "one plan dominates startup and steady-state cost"}
    return {"requests": float(requests), "reason": "cost curves intersect"}
