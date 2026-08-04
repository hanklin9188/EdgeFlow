from __future__ import annotations

import random
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from edgeflow.core.models import utc_now
from edgeflow.core.serialization import write_json
from edgeflow.hardware import inspect_hardware
from edgeflow.kernels.rmsnorm.dispatch import validate_shape
from edgeflow.metrics.statistics import robust_cv
from edgeflow.runtimes import LlamaCppAdapter, PytorchAdapter, VllmAdapter


def _percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), value))


def _telemetry() -> dict[str, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
        values = [float(item.strip()) for item in result.stdout.splitlines()[0].split(",")]
        return {
            "temperature_c": values[0],
            "sm_clock_mhz": values[1],
            "power_w": values[2],
            "utilization_pct": values[3],
        }
    except (FileNotFoundError, IndexError, ValueError, subprocess.TimeoutExpired):
        return {
            "temperature_c": None,
            "sm_clock_mhz": None,
            "power_w": None,
            "utilization_pct": None,
        }


def _cuda_event_us(function: Callable[[], Any], iterations: int) -> list[float]:
    import torch

    values: list[float] = []
    for _ in range(10):
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


def _wall_sync_us(function: Callable[[], Any], iterations: int) -> list[float]:
    import torch

    values: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - started) / 1000.0)
    return values


def evaluate_thermal_block(
    *,
    stabilized: list[float],
    background: list[float],
    stabilized_before: dict[str, float | None],
    stabilized_after: dict[str, float | None],
    pre_run: dict[str, float | None],
    required_iterations: int,
) -> dict[str, Any]:
    """Apply the registered E02 thresholds to a completed thermal block."""

    midpoint = len(stabilized) // 2
    first_half = statistics.median(stabilized[:midpoint])
    second_half = statistics.median(stabilized[midpoint:])
    latency_drift = abs(second_half - first_half) / first_half if first_half > 0 else float("inf")
    stable_cv = robust_cv(stabilized)
    background_cv = robust_cv(background)
    background_effect = statistics.median(background) / statistics.median(stabilized)

    before_temperature = stabilized_before.get("temperature_c")
    after_temperature = stabilized_after.get("temperature_c")
    temperature_delta = (
        abs(float(after_temperature) - float(before_temperature))
        if before_temperature is not None and after_temperature is not None
        else None
    )
    before_clock = stabilized_before.get("sm_clock_mhz")
    after_clock = stabilized_after.get("sm_clock_mhz")
    clock_drift = (
        abs(float(after_clock) - float(before_clock)) / float(before_clock)
        if before_clock not in {None, 0} and after_clock is not None
        else None
    )
    pre_run_utilization = pre_run.get("utilization_pct")
    checks = {
        "protocol_iterations": len(stabilized) >= required_iterations
        and len(background) >= required_iterations,
        "pre_run_gpu_idle": pre_run_utilization is not None
        and float(pre_run_utilization) < 5.0,
        "stabilized_latency_drift": latency_drift <= 0.03,
        "stabilized_robust_cv": stable_cv <= 0.10,
        "matched_temperature": temperature_delta is not None and temperature_delta <= 5.0,
        "active_clock_drift": clock_drift is not None and clock_drift <= 0.03,
        "negative_control_detected": background_effect > 1.05 or background_cv > stable_cv,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "stabilized_latency_drift": latency_drift,
        "stabilized_robust_cv": stable_cv,
        "background_robust_cv": background_cv,
        "background_latency_ratio": background_effect,
        "matched_temperature_delta_c": temperature_delta,
        "active_clock_drift": clock_drift,
        "pre_run_gpu_utilization_pct": pre_run_utilization,
    }


def _wall_after_us(function: Callable[[], Any], iterations: int) -> list[float]:
    import torch

    values: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - started) / 1000.0)
    return values


def _cpu_wall_us(function: Callable[[], Any], iterations: int) -> list[float]:
    values: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - started) / 1000.0)
    return values


def _enqueue_us(function: Callable[[], Any], iterations: int) -> list[float]:
    import torch

    values: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        output = function()
        values.append((time.perf_counter_ns() - started) / 1000.0)
        del output
    torch.cuda.synchronize()
    return values


def run_e00(root: Path, artifact_root: Path) -> dict[str, Any]:
    import torch

    hardware = inspect_hardware(root)
    tensor = torch.arange(4096, device="cuda", dtype=torch.float32)
    cuda_sum = float(tensor.sum().item())
    x = torch.randn((8, 1024), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn((1024,), device="cuda", dtype=torch.float16)
    kernel = validate_shape(x, residual, weight)
    capabilities = [
        PytorchAdapter(compiled=False).probe().model_dump(mode="json"),
        PytorchAdapter(compiled=True).probe().model_dump(mode="json"),
        LlamaCppAdapter().probe().model_dump(mode="json"),
        VllmAdapter().probe().model_dump(mode="json"),
    ]
    passed = bool(
        torch.cuda.is_available()
        and "RTX 4080 SUPER" in str(hardware.get("gpu", {}).get("name"))
        and cuda_sum == 8_386_560.0
        and kernel["status"] == "PASS"
        and all(item["available"] for item in capabilities)
    )
    result = {
        "schema_version": "1.0",
        "experiment_id": "E00",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "source_type": "measured",
        "created_at": utc_now(),
        "hardware_fingerprint_sha256": hardware["sha256"],
        "checks": {
            "target_gpu": "RTX 4080 SUPER" in str(hardware.get("gpu", {}).get("name")),
            "cuda_tensor": cuda_sum == 8_386_560.0,
            "triton_kernel": kernel,
            "runtime_capabilities": capabilities,
        },
        "claim_scope": "Environment and capability validation only.",
    }
    destination = artifact_root / "experiments" / "E00"
    write_json(destination / "hardware_fingerprint.json", hardware)
    write_json(destination / "capabilities.json", {"capabilities": capabilities})
    write_json(destination / "result.json", result)
    return result


def run_e01(root: Path, artifact_root: Path, *, quick: bool = False) -> dict[str, Any]:
    import torch

    iterations = 80 if quick else 1000
    vector_a = torch.randn((1_048_576,), device="cuda")
    vector_b = torch.randn_like(vector_a)
    matrix_a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    matrix_b = torch.randn_like(matrix_a)
    workloads: dict[str, Callable[[], Any]] = {
        "vector_add": lambda: vector_a + vector_b,
        "gemm_1024": lambda: matrix_a @ matrix_b,
    }
    rows: list[dict[str, Any]] = []
    for name, function in workloads.items():
        event = _cuda_event_us(function, iterations)
        wall = _wall_sync_us(function, iterations)
        after_only = _wall_after_us(function, iterations)
        enqueue = _enqueue_us(function, iterations)
        rows.append(
            {
                "workload": name,
                "iterations": iterations,
                "cuda_event_median_us": statistics.median(event),
                "wall_sync_median_us": statistics.median(wall),
                "wall_after_only_median_us": statistics.median(after_only),
                "enqueue_no_sync_median_us": statistics.median(enqueue),
                "cuda_event_p95_us": _percentile(event, 95),
                "wall_sync_p95_us": _percentile(wall, 95),
                "enqueue_underestimate_ratio": statistics.median(enqueue)
                / statistics.median(wall),
            }
        )
    cpu_iterations = 20 if quick else 100
    cpu_sleep = _cpu_wall_us(lambda: time.sleep(0.001), cpu_iterations)
    event_order = [row["workload"] for row in sorted(rows, key=lambda row: row["cuda_event_median_us"])]
    wall_order = [row["workload"] for row in sorted(rows, key=lambda row: row["wall_sync_median_us"])]
    negative_control_detected = any(row["enqueue_underestimate_ratio"] < 0.8 for row in rows)
    passed = event_order == wall_order and negative_control_detected
    result = {
        "schema_version": "1.0",
        "experiment_id": "E01",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "source_type": "measured",
        "created_at": utc_now(),
        "hardware_fingerprint_sha256": inspect_hardware(root)["sha256"],
        "timer_boundary": "CUDA Event for GPU work; synchronized perf_counter for local end-to-end wall time.",
        "negative_control": "Unsynchronized enqueue timing is prohibited for performance claims.",
        "ranking_matches": event_order == wall_order,
        "negative_control_detected": negative_control_detected,
        "cpu_wall_calibration": {
            "workload": "sleep_1ms",
            "iterations": cpu_iterations,
            "wall_median_us": statistics.median(cpu_sleep),
            "timer": "perf_counter_ns",
        },
        "rows": rows,
    }
    write_json(artifact_root / "experiments" / "E01" / "result.json", result)
    return result


def run_e02(root: Path, artifact_root: Path, *, quick: bool = False) -> dict[str, Any]:
    import torch

    iterations = 20 if quick else 1_000
    warmup_seconds = 1.0 if quick else 30.0
    pre_run = _telemetry()
    matrix_a = torch.randn((1536, 1536), device="cuda", dtype=torch.float16)
    matrix_b = torch.randn_like(matrix_a)

    def function() -> Any:
        return matrix_a @ matrix_b

    cold = _cuda_event_us(function, iterations)
    warmup_started = time.monotonic()
    while time.monotonic() - warmup_started < warmup_seconds:
        function()
    torch.cuda.synchronize()
    stable_before = _telemetry()
    stable = _cuda_event_us(function, iterations)
    stable_after = _telemetry()

    stop = threading.Event()

    def background_load() -> None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            while not stop.is_set():
                function()
        stream.synchronize()

    thread = threading.Thread(target=background_load, name="edgeflow-e02-negative-control")
    thread.start()
    time.sleep(0.5)
    background = _cuda_event_us(function, iterations)
    background_telemetry = _telemetry()
    stop.set()
    thread.join(timeout=30)
    torch.cuda.synchronize()

    def summary(values: list[float]) -> dict[str, float]:
        return {
            "median_us": statistics.median(values),
            "p95_us": _percentile(values, 95),
            "robust_cv": robust_cv(values),
        }

    cold_summary = summary(cold)
    stable_summary = summary(stable)
    background_summary = summary(background)
    required_iterations = 20 if quick else 1_000
    gate = evaluate_thermal_block(
        stabilized=stable,
        background=background,
        stabilized_before=stable_before,
        stabilized_after=stable_after,
        pre_run=pre_run,
        required_iterations=required_iterations,
    )
    finite_positive = all(np.isfinite(value) and value > 0 for value in cold + stable + background)
    passed = bool(finite_positive and gate["pass"])
    measurement_policy = {
        "schema_version": "1.0",
        "pre_run_gpu_utilization_pct_max": 5.0,
        "matched_temperature_delta_c_max": 5.0,
        "active_clock_drift_max": 0.03,
        "latency_drift_max": 0.03,
        "robust_cv_max": 0.10,
        "clock_samples_only_when_gpu_active": True,
    }
    result = {
        "schema_version": "1.0",
        "experiment_id": "E02",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "source_type": "measured",
        "created_at": utc_now(),
        "hardware_fingerprint_sha256": inspect_hardware(root)["sha256"],
        "protocol": {
            "iterations_per_condition": iterations,
            "stabilization_seconds": warmup_seconds,
            "timer": "cuda_event",
            "negative_control": "concurrent CUDA GEMM stream",
            "protocol_status": "DEVELOPMENT" if quick else "FORMAL",
        },
        "conditions": {
            "cold": cold_summary,
            "stabilized": stable_summary,
            "background_cuda": background_summary,
        },
        "telemetry": {
            "pre_run": pre_run,
            "stabilized_before": stable_before,
            "stabilized_after": stable_after,
            "background": background_telemetry,
        },
        "background_latency_ratio": gate["background_latency_ratio"],
        "stabilized_latency_drift": gate["stabilized_latency_drift"],
        "matched_temperature_delta_c": gate["matched_temperature_delta_c"],
        "active_clock_drift": gate["active_clock_drift"],
        "gate_checks": gate["checks"],
        "hypothesis_supported": gate["checks"]["negative_control_detected"],
        "measurement_policy": measurement_policy,
        "samples_us": {
            "cold": cold,
            "stabilized": stable,
            "background_cuda": background,
        },
    }
    destination = artifact_root / "experiments" / "E02"
    write_json(destination / "result.json", result)
    (destination / "measurement_policy.yaml").write_text(
        yaml.safe_dump(measurement_policy, sort_keys=False), encoding="utf-8"
    )
    return result


def run_e03(root: Path, artifact_root: Path, *, quick: bool = False) -> dict[str, Any]:
    import torch

    reference_iterations = 80 if quick else 200
    trials = 120 if quick else 500
    matrix_a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    matrix_b = torch.randn_like(matrix_a)
    baseline = _cuda_event_us(lambda: matrix_a @ matrix_b, reference_iterations)
    intervention = _cuda_event_us(
        lambda: torch.relu(matrix_a @ matrix_b), reference_iterations
    )
    full_winner = "baseline" if statistics.median(baseline) <= statistics.median(intervention) else "intervention"
    generator = random.Random(42)
    sizes = [10, 20, 30, 50]
    if reference_iterations < 50:
        sizes = [10, 20, 30]
    rows: list[dict[str, Any]] = []
    full_median = statistics.median(baseline)
    for size in sizes:
        errors: list[float] = []
        reversals = 0
        ci_widths: list[float] = []
        for _ in range(trials):
            a = generator.sample(baseline, size)
            b = generator.sample(intervention, size)
            errors.append(abs(statistics.median(a) - full_median) / full_median)
            winner = "baseline" if statistics.median(a) <= statistics.median(b) else "intervention"
            reversals += winner != full_winner
            bootstrap = [statistics.median(generator.choices(a, k=size)) for _ in range(200)]
            ci_widths.append((_percentile(bootstrap, 97.5) - _percentile(bootstrap, 2.5)) / full_median)
        rows.append(
            {
                "repetitions": size,
                "median_error_p95": _percentile(errors, 95),
                "relative_ci_width_median": statistics.median(ci_widths),
                "winner_reversal_rate": reversals / trials,
                "trials": trials,
            }
        )
    eligible = [
        row
        for row in rows
        if row["winner_reversal_rate"] < 0.05 and row["median_error_p95"] < 0.05
    ]
    recommended = min((row["repetitions"] for row in eligible), default=50)
    passed = reference_iterations >= 50 and all(np.isfinite(value) for value in baseline + intervention)
    result = {
        "schema_version": "1.0",
        "experiment_id": "E03",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "source_type": "measured",
        "created_at": utc_now(),
        "hardware_fingerprint_sha256": inspect_hardware(root)["sha256"],
        "reference_iterations": reference_iterations,
        "full_reference_winner": full_winner,
        "baseline_median_us": statistics.median(baseline),
        "intervention_median_us": statistics.median(intervention),
        "rows": rows,
        "recommended_repetitions": recommended,
        "acceptance": "winner reversal <5% and p95 median error <5%",
    }
    write_json(artifact_root / "experiments" / "E03" / "result.json", result)
    return result


def run_foundation_experiments(
    *,
    root: Path,
    artifact_root: Path,
    experiment_ids: list[str] | None = None,
    quick: bool = False,
) -> list[dict[str, Any]]:
    selected = experiment_ids or ["E00", "E01", "E02", "E03"]
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "E00": lambda: run_e00(root, artifact_root),
        "E01": lambda: run_e01(root, artifact_root, quick=quick),
        "E02": lambda: run_e02(root, artifact_root, quick=quick),
        "E03": lambda: run_e03(root, artifact_root, quick=quick),
    }
    unknown = set(selected) - runners.keys()
    if unknown:
        raise ValueError(f"unknown foundation experiment(s): {sorted(unknown)}")
    return [runners[experiment_id]() for experiment_id in selected]
