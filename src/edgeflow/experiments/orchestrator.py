from __future__ import annotations

import json
import os
import random
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edgeflow.core.models import (
    ExecutionPlan,
    MetricRecord,
    MetricValues,
    SourceType,
    WorkloadSpec,
    utc_now,
)
from edgeflow.core.serialization import project_root, write_json
from edgeflow.hardware.inspector import inspect_hardware
from edgeflow.quality import find_compatible_quality_report
from edgeflow.runtimes import (
    LlamaCppAdapter,
    PytorchAdapter,
    RuntimeAdapter,
    RuntimeUnavailable,
    VllmAdapter,
)
from edgeflow.storage import EdgeFlowDB
from edgeflow.validation import validate_run
from edgeflow.workloads.builder import build_exact_token_ids, choose_prompt_tokens


def _gpu_telemetry() -> dict[str, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
        values = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        return {
            "temperature_c": float(values[0]),
            "sm_clock_mhz": float(values[1]),
            "memory_clock_mhz": float(values[2]),
            "gpu_utilization_pct": float(values[3]),
            "power_w": float(values[4]),
        }
    except (FileNotFoundError, IndexError, ValueError, subprocess.TimeoutExpired):
        return {
            "temperature_c": None,
            "sm_clock_mhz": None,
            "memory_clock_mhz": None,
            "gpu_utilization_pct": None,
            "power_w": None,
        }


def _gpu_utilization() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
        return float(result.stdout.splitlines()[0].strip())
    except (FileNotFoundError, IndexError, ValueError, subprocess.TimeoutExpired):
        return None


class TelemetryMonitor:
    """Sample nvidia-smi off the latency-critical request path."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="edgeflow-gpu-monitor", daemon=True)
        self._started = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append({"monotonic_ns": time.monotonic_ns(), **_gpu_telemetry()})
            self._stop.wait(self.interval_seconds)

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._thread.join(timeout=3)

    def latest(self) -> dict[str, float | None]:
        if not self.samples:
            return {
                "temperature_c": None,
                "sm_clock_mhz": None,
                "memory_clock_mhz": None,
                "gpu_utilization_pct": None,
                "power_w": None,
            }
        return self.samples[-1]


@dataclass(frozen=True)
class BenchmarkConfig:
    experiment_id: str = "E04"
    repetitions: int = 30
    warmup_requests: int = 5
    local_files_only: bool = True
    source_type: SourceType = SourceType.MEASURED
    timeout_seconds: int = 1800
    enforce_idle: bool = True


class RunOrchestrator:
    def __init__(self, *, root: Path | None = None, artifact_root: Path | None = None) -> None:
        self.root = (root or project_root()).resolve()
        self.artifact_root = (artifact_root or self.root / "artifacts").resolve()
        self.db = EdgeFlowDB(self.artifact_root / "runs.sqlite")

    @staticmethod
    def adapter_for(plan: ExecutionPlan) -> RuntimeAdapter:
        adapters: dict[str, RuntimeAdapter] = {
            "pytorch_eager": PytorchAdapter(compiled=False),
            "torch_compile": PytorchAdapter(compiled=True),
            "llama_cpp": LlamaCppAdapter(),
            "vllm": VllmAdapter(),
        }
        return adapters[plan.backend]

    def _base_manifest(
        self,
        *,
        run_id: str,
        config: BenchmarkConfig,
        workload: WorkloadSpec,
        plan: ExecutionPlan,
        hardware: dict[str, Any],
        command: list[str],
    ) -> dict[str, Any]:
        revision = str(plan.backend_args.get("revision") or "unknown")
        git = hardware.get("git") or {}
        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "experiment_id": config.experiment_id,
            "block_id": f"block-{time.strftime('%Y%m%d')}",
            "paired_group_id": None,
            "run_type": "performance_unprofiled",
            "status": "PLANNED",
            "created_at": utc_now(),
            "completed_at": None,
            "hardware_fingerprint_sha256": hardware["sha256"],
            "model_id": workload.model_id,
            "model_revision": revision,
            "model_files_sha256": list(plan.backend_args.get("model_files_sha256", [])),
            "tokenizer_revision": revision,
            "workload_id": workload.workload_id,
            "workload_sha256": workload.content_sha256,
            "plan_id": plan.plan_id,
            "plan_sha256": plan.content_sha256,
            "protocol_version": "edgeflow-bench-1.0",
            "git_commit": git.get("commit"),
            "git_dirty": bool(git.get("dirty", True)),
            "command": command,
            "seed": workload.seed,
            "profiler_level": "none",
            "source_type": config.source_type.value,
            "artifact_files": [
                "run_manifest.json", "workload.json", "execution_plan.json",
                "hardware_fingerprint.json", "metrics.jsonl", "stdout.log", "stderr.log",
                "monitor.jsonl", "validation_verdict.json", "VALIDATION.md",
                "correctness.json",
                "quality.json",
            ],
            "supersedes_run_id": None,
            "notes": "Production timing uses synchronized engine boundaries; warmup is stored separately.",
        }

    def run(
        self,
        *,
        model_ref: str,
        workload: WorkloadSpec,
        plan: ExecutionPlan,
        config: BenchmarkConfig | None = None,
        command: list[str] | None = None,
    ) -> Path:
        config = config or BenchmarkConfig()
        if config.repetitions < 1 or config.warmup_requests < 1:
            raise ValueError("repetitions and warmup_requests must be positive")
        if plan.backend in {"pytorch_eager", "torch_compile"}:
            if workload.concurrency != 1:
                raise ValueError("PyTorch runtimes use batch_size; concurrency must be 1")
            request_group_size = workload.batch_size
            grouping_kind = "batch"
        else:
            if workload.batch_size != 1:
                raise ValueError("HTTP runtimes use concurrency; batch_size must be 1")
            request_group_size = workload.concurrency
            grouping_kind = "concurrency"
        plan = plan.with_hash()
        adapter = self.adapter_for(plan)
        report = adapter.probe()
        if not report.available:
            raise RuntimeUnavailable("; ".join(report.reasons))
        utilization = _gpu_utilization()
        if config.enforce_idle and utilization is not None and utilization >= 5.0:
            raise RuntimeError(f"GPU precheck failed: utilization is {utilization:.1f}% (must be <5%)")
        hardware = inspect_hardware(self.root)
        run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        temporary = self.artifact_root / f".{run_id}.partial"
        final = self.artifact_root / run_id
        temporary.mkdir(parents=False, exist_ok=False)
        manifest = self._base_manifest(
            run_id=run_id,
            config=config,
            workload=workload,
            plan=plan,
            hardware=hardware,
            command=command or ["edgeflow", "benchmark", "run"],
        )
        write_json(temporary / "workload.json", workload.model_dump(mode="json"))
        write_json(temporary / "execution_plan.json", plan.model_dump(mode="json"))
        write_json(temporary / "hardware_fingerprint.json", hardware)
        (temporary / "stdout.log").write_text("", encoding="utf-8")
        (temporary / "stderr.log").write_text("", encoding="utf-8")
        raw_path = temporary / "metrics.jsonl"
        records: list[dict[str, Any]] = []

        def append(record: MetricRecord) -> None:
            row = record.model_dump(mode="json")
            records.append(row)
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()

        runtime = None
        monitor = TelemetryMonitor()
        monitor_written = False
        try:
            manifest["status"] = "PREPARING"
            write_json(temporary / "run_manifest.json", manifest)
            runtime, tokenizer = adapter.prepare(
                model_ref,
                plan,
                workload,
                local_files_only=config.local_files_only,
            )
            append(
                MetricRecord(
                    run_id=run_id, request_id=None, source_type=config.source_type, phase="load", iteration=0,
                    metrics=MetricValues(wall_ms=runtime.load_ms), notes="Model and tokenizer preparation.",
                )
            )
            if runtime.compile_ms > 0:
                append(
                    MetricRecord(
                        run_id=run_id, request_id=None, source_type=config.source_type, phase="compile", iteration=0,
                        metrics=MetricValues(wall_ms=runtime.compile_ms), notes="torch.compile wrapper construction; first graph compile appears in warmup.",
                    )
                )
            monitor.start()
            manifest["status"] = "WARMING"
            write_json(temporary / "run_manifest.json", manifest)
            warmup_latencies: list[float] = []
            if plan.backend == "torch_compile":
                count = choose_prompt_tokens(workload, -10_000)
                token_groups = [
                    build_exact_token_ids(tokenizer, count, seed=workload.seed + member)
                    for member in range(request_group_size)
                ]
                first_compiled = runtime.generate_batch(token_groups, workload.output_tokens)[0]
                append(
                    MetricRecord(
                        run_id=run_id,
                        request_id="compile-first-execution",
                        source_type=config.source_type,
                        phase="compile",
                        iteration=1,
                        prompt_tokens=count,
                        output_tokens=len(first_compiled.output_token_ids),
                        metrics=MetricValues(wall_ms=first_compiled.wall_ms),
                        token_timestamps_ms=first_compiled.token_timestamps_ms,
                        notes="First graph compile/autotune execution; excluded from warmup and steady-state.",
                    )
                )
            maximum_warmup = max(config.warmup_requests, 20)
            for iteration in range(maximum_warmup):
                count = choose_prompt_tokens(workload, -iteration - 1)
                token_groups = [
                    build_exact_token_ids(
                        tokenizer,
                        count,
                        seed=workload.seed + iteration * request_group_size + member,
                    )
                    for member in range(request_group_size)
                ]
                warmup_started = time.perf_counter_ns()
                warmup_results = runtime.generate_batch(token_groups, workload.output_tokens)
                warmup_wall_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000
                result = warmup_results[0]
                warmup_latencies.append(warmup_wall_ms)
                append(
                    MetricRecord(
                        run_id=run_id, request_id=f"warmup-{iteration:04d}", source_type=config.source_type,
                        phase="warmup", iteration=iteration, prompt_tokens=count, output_tokens=len(result.output_token_ids),
                        metrics=MetricValues(wall_ms=warmup_wall_ms, ttft_ms=result.ttft_ms, tpot_ms=result.tpot_ms),
                        token_timestamps_ms=result.token_timestamps_ms,
                        notes=f"Excluded from steady-state summary; {grouping_kind}={request_group_size}.",
                    )
                )
                if iteration + 1 >= max(config.warmup_requests, 10):
                    previous = sorted(warmup_latencies[-10:-5])[2]
                    recent = sorted(warmup_latencies[-5:])[2]
                    if previous > 0 and abs(recent - previous) / previous < 0.02:
                        break
            correctness_count = choose_prompt_tokens(workload, -20_000)
            correctness_ids = [
                build_exact_token_ids(
                    tokenizer,
                    correctness_count,
                    seed=workload.seed + 20_000 + member,
                )
                for member in range(request_group_size)
            ]
            correctness_a = runtime.generate_batch(correctness_ids, workload.output_tokens)
            correctness_b = runtime.generate_batch(correctness_ids, workload.output_tokens)
            deterministic = [item.output_token_ids for item in correctness_a] == [
                item.output_token_ids for item in correctness_b
            ]
            timestamps_valid = all(
                len(result.token_timestamps_ms) in {0, len(result.output_token_ids)}
                for result in (*correctness_a, *correctness_b)
            )
            correctness_pass = bool(
                deterministic
                and all(result.output_token_ids for result in (*correctness_a, *correctness_b))
                and timestamps_valid
            )
            write_json(
                temporary / "correctness.json",
                {
                    "schema_version": "1.0",
                    "pass": correctness_pass,
                    "nan_count": 0,
                    "reference_type": "pinned_runtime_deterministic_repeat",
                    "prompt_tokens": correctness_count,
                    "requested_output_tokens": workload.output_tokens,
                    "request_group_size": request_group_size,
                    "grouping_kind": grouping_kind,
                    "observed_output_tokens": [
                        len(result.output_token_ids) for result in correctness_a
                    ],
                    "exact_output_match": deterministic,
                    "timestamp_cardinality_valid": timestamps_valid,
                    "summary": (
                        "Pinned runtime produced identical greedy outputs for a repeated exact-token prompt."
                        if correctness_pass
                        else "Pinned runtime repeatability or token-timestamp integrity failed."
                    ),
                },
            )
            if not correctness_pass:
                raise RuntimeError("runtime correctness repeat failed before measured timing")
            manifest["status"] = "RUNNING"
            write_json(temporary / "run_manifest.json", manifest)
            order = list(range(config.repetitions))
            random.Random(workload.seed).shuffle(order)
            for execution_index, prompt_index in enumerate(order):
                count = choose_prompt_tokens(workload, prompt_index)
                token_groups = [
                    build_exact_token_ids(
                        tokenizer,
                        count,
                        seed=workload.seed + prompt_index * request_group_size + member,
                    )
                    for member in range(request_group_size)
                ]
                group_started = time.perf_counter_ns()
                results = runtime.generate_batch(token_groups, workload.output_tokens)
                group_wall_ms = (time.perf_counter_ns() - group_started) / 1_000_000
                telemetry = monitor.latest()
                group_output_tokens = sum(len(result.output_token_ids) for result in results)
                generation_rate = (
                    group_output_tokens / (group_wall_ms / 1000.0) if group_wall_ms > 0 else None
                )
                for member, result in enumerate(results):
                    append(
                        MetricRecord(
                            run_id=run_id,
                            request_id=f"request-{prompt_index:06d}-{member:02d}",
                            source_type=config.source_type,
                            phase="end_to_end",
                            iteration=execution_index * request_group_size + member,
                            prompt_tokens=count,
                            output_tokens=len(result.output_token_ids),
                            metrics=MetricValues(
                                wall_ms=result.wall_ms,
                                ttft_ms=result.ttft_ms,
                                tpot_ms=result.tpot_ms,
                                request_latency_ms=result.wall_ms,
                                generation_tokens_per_s=generation_rate,
                                requests_per_s=request_group_size * 1000.0 / group_wall_ms,
                                peak_vram_bytes=result.peak_vram_bytes,
                                temperature_c=telemetry["temperature_c"],
                                sm_clock_mhz=telemetry["sm_clock_mhz"],
                                memory_clock_mhz=telemetry["memory_clock_mhz"],
                                gpu_utilization_pct=telemetry["gpu_utilization_pct"],
                                power_w=telemetry["power_w"],
                            ),
                            token_timestamps_ms=result.token_timestamps_ms,
                            notes=(
                                "Engine-only greedy generation; randomized prompt order; "
                                f"{grouping_kind}={request_group_size}."
                            ),
                        ),
                    )
            quality_report = find_compatible_quality_report(
                artifact_root=self.artifact_root,
                model_id=workload.model_id,
                model_revision=str(plan.backend_args.get("revision") or "unknown"),
                plan=plan,
            )
            if quality_report is not None:
                write_json(temporary / "quality.json", quality_report)
            manifest["status"] = "PASSED"
            manifest["completed_at"] = utc_now()
            write_json(temporary / "run_manifest.json", manifest)
        except Exception as exc:
            manifest["status"] = "FAILED"
            manifest["completed_at"] = utc_now()
            manifest["notes"] = f"Run failed and raw artifacts were preserved: {type(exc).__name__}."
            (temporary / "stderr.log").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            write_json(temporary / "run_manifest.json", manifest)
            monitor.stop()
            (temporary / "monitor.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in monitor.samples),
                encoding="utf-8",
            )
            monitor_written = True
            os.replace(temporary, final)
            verdict = validate_run(final, root=self.root, write=True)
            manifest["status"] = verdict["verdict"]
            write_json(final / "run_manifest.json", manifest)
            self.db.save_hardware(hardware)
            self.db.save_workload(workload.model_dump(mode="json"), workload.content_sha256)
            self.db.save_plan(plan.model_dump(mode="json"), plan.content_sha256)
            self.db.save_run(manifest, final)
            self.db.save_metrics(run_id, records)
            self.db.save_validation(verdict)
            raise
        finally:
            if not monitor_written:
                monitor.stop()
                (temporary / "monitor.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in monitor.samples),
                    encoding="utf-8",
                )
            if runtime is not None:
                runtime.shutdown()
        os.replace(temporary, final)
        verdict = validate_run(final, root=self.root, write=True)
        manifest["status"] = verdict["verdict"] if verdict["verdict"] != "PASS" else "PASSED"
        write_json(final / "run_manifest.json", manifest)
        self.db.save_hardware(hardware)
        self.db.save_workload(workload.model_dump(mode="json"), workload.content_sha256)
        self.db.save_plan(plan.model_dump(mode="json"), plan.content_sha256)
        self.db.save_run(manifest, final)
        self.db.save_metrics(run_id, records)
        self.db.save_validation(verdict)
        return final
