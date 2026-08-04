from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from edgeflow.core.models import CapabilityReport, ExecutionPlan
from edgeflow.experiments import BenchmarkConfig, RunOrchestrator
from edgeflow.experiments.matrix import (
    matrix_case_label,
    matrix_progress_status,
    pytorch_matrix_cases,
)
from edgeflow.experiments.orchestrator import TelemetryMonitor, _prompt_seed
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
        self.group_prompt_lengths: list[tuple[int, ...]] = []

    def generate_batch(
        self, token_id_batches: list[list[int]], output_tokens: int
    ) -> list[GenerationResult]:
        self.group_sizes.append(len(token_id_batches))
        self.group_prompt_lengths.append(tuple(len(batch) for batch in token_id_batches))
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

    def generate(self, token_ids: list[int], output_tokens: int) -> GenerationResult:
        return self.generate_batch([token_ids], output_tokens)[0]

    def shutdown(self) -> None:
        return None


class FakeAdapter:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime

    def probe(self) -> CapabilityReport:
        return CapabilityReport(backend="pytorch_eager", available=True)

    def prepare(self, *_args: Any, **_kwargs: Any) -> tuple[FakeRuntime, FakeTokenizer]:
        return self.runtime, FakeTokenizer()


def test_fixed_prompt_scope_reuses_exact_token_seed() -> None:
    fixed = create_workload(
        workload_id="fixed-prompt",
        model_id="model",
        prompt_distribution="128",
        output_tokens=2,
        seed=73,
    )
    mixed = create_workload(
        workload_id="mixed-prompts",
        model_id="model",
        prompt_distribution="128:0.5,1024:0.5",
        output_tokens=2,
        seed=73,
    )

    assert _prompt_seed(
        fixed, prompt_index=0, request_group_size=1, member=0
    ) == _prompt_seed(fixed, prompt_index=29, request_group_size=1, member=0)
    assert _prompt_seed(
        fixed, prompt_index=0, request_group_size=4, member=1
    ) != _prompt_seed(fixed, prompt_index=0, request_group_size=4, member=2)
    assert _prompt_seed(
        mixed, prompt_index=0, request_group_size=1, member=0
    ) != _prompt_seed(mixed, prompt_index=29, request_group_size=1, member=0)


def test_telemetry_defaults_to_post_request_sampling(monkeypatch: Any) -> None:
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
    monitor = TelemetryMonitor()
    monitor.start()
    assert monitor.samples == []
    assert monitor.sample()["temperature_c"] == 50.0
    assert len(monitor.samples) == 1
    monitor.stop()


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
    assert all(len(set(lengths)) == 1 for lengths in runtime.group_prompt_lengths)


def test_orchestrator_preserves_mixed_http_concurrency(tmp_path: Path, monkeypatch: Any) -> None:
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
        workload_id="concurrency-mix-test",
        model_id="model",
        prompt_distribution="8:0.5,16:0.5",
        output_tokens=2,
        batch_size=1,
        concurrency=4,
        session_requests=3,
    )
    plan = ExecutionPlan(
        plan_id="vllm-concurrency-test",
        model_id="model",
        backend="vllm",
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

    assert correctness["grouping_kind"] == "concurrency"
    assert correctness["prompt_token_counts"] != [correctness["prompt_tokens"]] * 4
    assert len(measured) == 12
    assert {row["prompt_tokens"] for row in measured} == {8, 16}
    assert any(len(set(lengths)) > 1 for lengths in runtime.group_prompt_lengths)


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


def test_dispatch_cache_reads_calibration_once(tmp_path: Path, monkeypatch: Any) -> None:
    performance_path = tmp_path / "performance.json"
    validation_path = tmp_path / "validation.json"
    performance_path.write_text("{}", encoding="utf-8")
    validation_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(dispatch, "_cache_path", lambda: validation_path)
    monkeypatch.setattr(dispatch, "_performance_cache_path", lambda: performance_path)
    dispatch.clear_dispatch_caches()

    first = dispatch._load_performance_cache()
    performance_path.write_text('{"changed": true}', encoding="utf-8")

    assert dispatch._load_performance_cache() is first
    assert "changed" not in dispatch._load_performance_cache()
    dispatch.clear_dispatch_caches()
    assert dispatch._load_performance_cache()["changed"] is True


def test_registered_pytorch_matrices_expand_without_duplicates() -> None:
    e04 = pytorch_matrix_cases("E04")
    e05 = pytorch_matrix_cases("E05")
    assert len(e04) == 12
    assert len(e05) == 32
    assert len({row["case_id"] for row in e04 + e05}) == 44
    labels = [matrix_case_label(row["case_id"]) for row in e04 + e05]
    assert all(len(label) <= 48 for label in labels)
    assert len(labels) == len(set(labels))


def test_matrix_progress_distinguishes_partial_from_running() -> None:
    partial = [{"case_id": "one", "status": "COMPLETED"}]
    running = [*partial, {"case_id": "two", "status": "RUNNING"}]
    pending = [*partial, {"case_id": "two", "status": "PENDING"}]
    failed = [*partial, {"case_id": "two", "status": "FAILED"}]
    pruned = [*partial, {"case_id": "two", "status": "PRUNED"}]
    passed = [*partial, {"case_id": "two", "status": "COMPLETED"}]

    assert matrix_progress_status(partial, total_case_count=2) == ("PARTIAL", False)
    assert matrix_progress_status(running, total_case_count=2) == ("RUNNING", False)
    assert matrix_progress_status(pending, total_case_count=2) == ("PARTIAL", False)
    assert matrix_progress_status(failed, total_case_count=2) == (
        "COMPLETE_WITH_FAILURES",
        False,
    )
    assert matrix_progress_status(pruned, total_case_count=2) == (
        "COMPLETE_WITH_PRUNES",
        True,
    )
    assert matrix_progress_status(passed, total_case_count=2) == ("PASS", True)


def test_orchestrator_recovers_native_worker_partial(tmp_path: Path, valid_run_dir: Path) -> None:
    artifact_root = tmp_path / "recovered-artifacts"
    partial = artifact_root / ".run-test.partial"
    shutil.copytree(valid_run_dir, partial)
    manifest_path = partial / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "WARMING"
    manifest["completed_at"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    orchestrator = RunOrchestrator(artifact_root=artifact_root)

    recovered = orchestrator.recover_interrupted_run(partial, reason="SIGSEGV test")

    recovered_manifest = json.loads((recovered / "run_manifest.json").read_text())
    verdict = json.loads((recovered / "validation_verdict.json").read_text())
    assert recovered == artifact_root / "run-test"
    assert not partial.exists()
    assert recovered_manifest["status"] == "INVALID"
    assert verdict["policy_eligible"] is False
    assert "SIGSEGV test" in (recovered / "stderr.log").read_text()
