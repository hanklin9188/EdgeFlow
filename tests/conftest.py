from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from edgeflow.core.models import ExecutionPlan, MetricRecord, MetricValues, SourceType
from edgeflow.core.serialization import write_json
from edgeflow.workloads import create_workload


@pytest.fixture
def valid_run_dir(tmp_path: Path) -> Path:
    workload = create_workload(
        workload_id="test-p32-o4-c1",
        model_id="smoke-small",
        prompt_distribution="32",
        output_tokens=4,
        session_requests=30,
    )
    plan = ExecutionPlan(
        plan_id="pytorch-eager-smoke-bf16",
        model_id="smoke-small",
        backend="pytorch_eager",
        model_format="safetensors",
        dtype="bf16",
        custom_kernels=(),
        backend_args={"revision": "0123456789abcdef"},
    ).with_hash()
    hardware: dict[str, Any] = {
        "schema_version": "1.0",
        "fingerprint_id": "hw-test",
        "captured_at": "2026-08-04T00:00:00Z",
        "host": {
            "os": "test-linux", "execution_mode": "linux_native", "wsl_version": None,
            "kernel": "test", "cpu": "test", "physical_cores": 8, "logical_cores": 16,
            "ram_bytes": 32 * 1024**3,
        },
        "gpu": {
            "vendor": "NVIDIA", "name": "NVIDIA GeForce RTX 4080 SUPER", "uuid": "GPU-test",
            "compute_capability": "8.9", "vram_bytes": 16 * 1024**3, "driver_version": "test",
            "power_limit_w": 320.0, "persistence_mode": False,
        },
        "software": {
            "python": "3.12", "pytorch": "test", "cuda_runtime": "test", "cuda_toolkit": None,
            "transformers": "test", "triton": "test", "vllm": None,
            "llama_cpp_commit": None, "nsight_systems": None, "nsight_compute": None,
        },
        "measurement_controls": {
            "gpu_idle_threshold_pct": 5.0, "temperature_ceiling_c": 80.0,
            "background_gpu_processes": [],
        },
        "git": {"commit": "abc", "dirty": False},
        "sha256": "a" * 64,
    }
    manifest = {
        "schema_version": "1.0", "run_id": "run-test", "experiment_id": "E04",
        "block_id": "block-test", "paired_group_id": None,
        "run_type": "performance_unprofiled", "status": "PASSED",
        "created_at": "2026-08-04T00:00:00Z", "completed_at": "2026-08-04T00:01:00Z",
        "hardware_fingerprint_sha256": hardware["sha256"], "model_id": "smoke-small",
        "model_revision": "0123456789abcdef", "model_files_sha256": [],
        "tokenizer_revision": "0123456789abcdef", "workload_id": workload.workload_id,
        "workload_sha256": workload.content_sha256, "plan_id": plan.plan_id,
        "plan_sha256": plan.content_sha256, "protocol_version": "edgeflow-bench-1.0",
        "git_commit": "abc", "git_dirty": False,
        "command": ["edgeflow", "benchmark", "run"], "seed": 42, "profiler_level": "none",
        "source_type": "measured", "artifact_files": ["metrics.jsonl"],
        "supersedes_run_id": None, "notes": "test fixture",
    }
    write_json(tmp_path / "workload.json", workload.model_dump(mode="json"))
    write_json(tmp_path / "execution_plan.json", plan.model_dump(mode="json"))
    write_json(tmp_path / "hardware_fingerprint.json", hardware)
    write_json(tmp_path / "run_manifest.json", manifest)
    write_json(
        tmp_path / "correctness.json",
        {"pass": True, "nan_count": 0, "summary": "reference parity passed"},
    )
    write_json(tmp_path / "quality.json", {"pass": True, "summary": "strict quality fixture"})
    rows = []
    for index in range(30):
        wall = 10.0 + ((index % 3) - 1) * 0.02
        rows.append(
            MetricRecord(
                run_id="run-test", request_id=f"request-{index}", source_type=SourceType.MEASURED,
                phase="end_to_end", iteration=index, prompt_tokens=32, output_tokens=4,
                metrics=MetricValues(
                    wall_ms=wall, ttft_ms=4.0, tpot_ms=2.0, request_latency_ms=wall,
                    generation_tokens_per_s=400.0, requests_per_s=100.0,
                    peak_vram_bytes=1024, temperature_c=50.0, sm_clock_mhz=2000,
                    memory_clock_mhz=10000,
                ),
                token_timestamps_ms=(4.0, 6.0, 8.0, 10.0),
            ).model_dump(mode="json")
        )
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return tmp_path
