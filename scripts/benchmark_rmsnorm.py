#!/usr/bin/env python3
"""Measured RMSNorm correctness + microbenchmark with all rows preserved."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from edgeflow.core.models import utc_now  # noqa: E402
from edgeflow.core.serialization import write_json  # noqa: E402
from edgeflow.hardware import inspect_hardware  # noqa: E402
from edgeflow.kernels.rmsnorm.dispatch import validate_shape  # noqa: E402
from edgeflow.kernels.rmsnorm.kernel import triton_residual_rmsnorm  # noqa: E402
from edgeflow.kernels.rmsnorm.reference import reference_residual_rmsnorm  # noqa: E402


def time_cuda(function, iterations: int) -> list[float]:
    values: list[float] = []
    for _ in range(25):
        function()
    torch.cuda.synchronize()
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000.0)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "kernel" / "rmsnorm-benchmark.json")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA unavailable", file=sys.stderr)
        return 2
    shapes = [(1, 768), (8, 1000), (32, 4096)] if arguments.quick else [
        (row, hidden) for row in (1, 2, 4, 8, 16, 32, 128, 512)
        for hidden in (768, 1000, 1024, 1536, 2048, 3072, 3073, 4095, 4096)
    ]
    dtypes = (torch.float16,) if arguments.quick else (torch.float32, torch.float16, torch.bfloat16)
    iterations = 100
    rows = []
    compiled_reference = torch.compile(
        reference_residual_rmsnorm,
        fullgraph=True,
        dynamic=False,
        mode="reduce-overhead",
    )
    for dtype in dtypes:
        for row_count, hidden in shapes:
            generator = torch.Generator(device="cuda").manual_seed(42 + row_count + hidden)
            x = torch.randn((row_count, hidden), dtype=dtype, device="cuda", generator=generator)
            residual = torch.randn_like(x)
            weight = torch.randn((hidden,), dtype=dtype, device="cuda", generator=generator)
            correctness = validate_shape(x, residual, weight)
            compile_started = time.perf_counter_ns()
            compiled_reference(x, residual, weight)
            torch.cuda.synchronize()
            compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000
            functions = {
                "pytorch": lambda x=x, residual=residual, weight=weight: reference_residual_rmsnorm(
                    x, residual, weight
                ),
                "triton": lambda x=x, residual=residual, weight=weight: triton_residual_rmsnorm(
                    x, residual, weight
                ),
                "torch_compile": lambda x=x, residual=residual, weight=weight: compiled_reference(
                    x, residual, weight
                ),
            }
            order = list(functions)
            random.Random(42 + row_count + hidden).shuffle(order)
            timings = {name: time_cuda(functions[name], iterations) for name in order}
            pytorch_median = statistics.median(timings["pytorch"])
            triton_median = statistics.median(timings["triton"])
            compiled_median = statistics.median(timings["torch_compile"])
            rows.append(
                {
                    "shape": [row_count, hidden], "dtype": str(dtype), "correctness": correctness,
                    "iterations": iterations, "warmup": 25,
                    "torch_compile_first_call_ms": compile_ms,
                    "pytorch_median_us": pytorch_median,
                    "torch_compile_median_us": compiled_median,
                    "triton_median_us": triton_median,
                    "speedup": pytorch_median / triton_median,
                    "speedup_vs_torch_compile": compiled_median / triton_median,
                    "pytorch_p95_us": sorted(timings["pytorch"])[94],
                    "torch_compile_p95_us": sorted(timings["torch_compile"])[94],
                    "triton_p95_us": sorted(timings["triton"])[94],
                    "implementation_order": order,
                }
            )
    payload = {
        "schema_version": "1.0", "source_type": "measured", "created_at": utc_now(),
        "hardware": inspect_hardware(ROOT), "kernel": "fused_residual_rmsnorm",
        "protocol": {"warmup": 25, "iterations": iterations, "timer": "cuda_event"},
        "rows": rows, "all_correct": all(row["correctness"]["status"] == "PASS" for row in rows),
        "claim_scope": "microbenchmark only; not an end-to-end LLM claim",
    }
    write_json(arguments.output, payload)
    print(json.dumps({"output": str(arguments.output), "rows": len(rows), "all_correct": payload["all_correct"]}))
    return 0 if payload["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
