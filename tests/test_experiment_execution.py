from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edgeflow.core.models import CapabilityReport, ExecutionPlan
from edgeflow.experiments import BenchmarkConfig, RunOrchestrator
from edgeflow.kernels.rmsnorm import dispatch
from edgeflow.runtimes.base import GenerationResult
from edgeflow.workloads import create_workload


class FakeTokenizer:
    bos_token_id = 1

    def encode(self, _text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(2, 34))


class FakeRuntime:
    load_ms = 3.0
    compile_ms = 0.0

    def __init__(self) -> None:
        self.group_sizes: list[int] = []

    def generate_batch(
        self, token_id_batches: list[list[int]], output_tokens: int
    ) -> list[GenerationResult]:
        self.group_sizes.append(len(token_id_batches))
        timestamps = tuple(float(index + 1) for index in range(output_tokens))
        return [
            GenerationResult(
                output_token_ids=tuple([batch[-1]] * output_tokens),
                token_timestamps_ms=timestamps,
                wall_ms=10.0,
                ttft_ms=1.0,
                tpot_ms=1.0 if output_tokens > 1 else None,
                peak_vram_bytes=1024,
                native_metrics={},
            )
            for batch in token_id_batches
        ]

    def shutdown(self) -> None:
        return None


class FakeAdapter:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime

    def probe(self) -> CapabilityReport:
        return CapabilityReport(backend="pytorch_eager", available=True)

    def prepare(self, *_args: Any, **_kwargs: Any) -> tuple[FakeRuntime, FakeTokenizer]:
        return self.runtime, FakeTokenizer()


def test_orchestrator_preserves_true_batch_groups(tmp_path: Path, monkeypatch: Any) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(
        RunOrchestrator,
        "adapter_for",
        staticmethod(lambda _plan: FakeAdapter(runtime)),
    )
    monkeypatch.setattr("edgeflow.experiments.orchestrator._gpu_utilization", lambda: 0.0)
    monkeypatch.setattr(
        "edgeflow.experiments.orchestrator._gpu_telemetry",
        lambda: {
            "temperature_c": 50.0,
            "sm_clock_mhz": 2000.0,
            "memory_clock_mhz": 10000.0,
            "gpu_utilization_pct": 90.0,
            "power_w": 200.0,
        },
    )
    monkeypatch.setattr(
        "edgeflow.experiments.orchestrator.inspect_hardware",
        lambda _root: {
            "fingerprint_id": "hw-test",
            "captured_at": "2026-08-04T00:00:00Z",
            "sha256": "a" * 64,
            "git": {"commit": "abc", "dirty": False},
        },
    )
    monkeypatch.setattr(
        "edgeflow.experiments.orchestrator.validate_run",
        lambda run_dir, **_kwargs: {
            "run_id": json.loads((run_dir / "run_manifest.json").read_text())["run_id"],
            "verdict": "CONDITIONAL_PASS",
            "policy_eligible": False,
            "public_claim_eligible": False,
        },
    )
    workload = create_workload(
        workload_id="batch-test",
        model_id="model",
        prompt_distribution="16",
        output_tokens=2,
        batch_size=3,
        concurrency=1,
        session_requests=3,
    )
    plan = ExecutionPlan(
        plan_id="pytorch-batch-test",
        model_id="model",
        backend="pytorch_eager",
        model_format="safetensors",
        dtype="bf16",
        backend_args={"revision": "abc"},
    )
    output = RunOrchestrator(root=tmp_path, artifact_root=tmp_path / "artifacts").run(
        model_ref="model",
        workload=workload,
        plan=plan,
        config=BenchmarkConfig(repetitions=3, warmup_requests=1),
    )
    correctness = json.loads((output / "correctness.json").read_text())
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    measured = [row for row in rows if row["phase"] == "end_to_end"]
    assert correctness["request_group_size"] == 3
    assert correctness["grouping_kind"] == "batch"
    assert len(measured) == 9
    assert set(runtime.group_sizes) == {3}


def test_kernel_dispatch_cache_only_enables_measured_winners(
    tmp_path: Path, monkeypatch: Any
) -> None:
    validation_path = tmp_path / "validation.json"
    performance_path = tmp_path / "performance.json"
    monkeypatch.setattr(dispatch, "_cache_path", lambda: validation_path)
    monkeypatch.setattr(dispatch, "_performance_cache_path", lambda: performance_path)
    rows = [
        {
            "shape": [1, 1024],
            "dtype": "torch.float16",
            "gpu": "GPU-test",
            "correctness": {"status": "PASS"},
            "speedup": 1.10,
        },
        {
            "shape": [4, 1024],
            "dtype": "torch.float16",
            "gpu": "GPU-test",
            "correctness": {"status": "PASS"},
            "speedup": 1.01,
        },
    ]
    cache = dispatch.record_performance_decisions(rows)
    assert cache[f"{dispatch.KERNEL_VERSION}|GPU-test|torch.float16|1x1024"]["enabled"] is True
    assert cache[f"{dispatch.KERNEL_VERSION}|GPU-test|torch.float16|4x1024"]["enabled"] is False
