from __future__ import annotations

import json
import os
import random
import statistics
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
from edgeflow.metrics.statistics import robust_cv
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

_NVML_HANDLE: Any | None = None
_NVML_UNAVAILABLE = False


def _prompt_counts_for_group(
    workload: WorkloadSpec,
    *,
    iteration: int,
    request_group_size: int,
    grouping_kind: str,
) -> list[int]:
    """Choose exact prompt sizes while preserving each runtime's grouping semantics."""

    if grouping_kind == "batch":
        # The PyTorch adapter stacks inputs into a dense tensor, so all members of
        # one true batch must share a sequence length.
        count = choose_prompt_tokens(workload, iteration)
        return [count] * request_group_size
    # An HTTP concurrency group represents independent requests and must preserve
    # the registered distribution within a group instead of cloning one prompt.
    return [
        choose_prompt_tokens(workload, iteration * request_group_size + member)
        for member in range(request_group_size)
    ]


def _prompt_seed(
    workload: WorkloadSpec,
    *,
    prompt_index: int,
    request_group_size: int,
    member: int,
) -> int:
    """Keep fixed-bucket benchmarks paired while varying distribution replays.

    A scalar prompt-token workload represents one registered exact prompt. Reusing
    its token IDs across warmup, correctness, and measured repetitions is required
    for a matched stability estimate. Distribution workloads deliberately retain a
    deterministic per-request seed so replay diversity is not collapsed.
    """

    if isinstance(workload.prompt_tokens, int):
        return workload.seed + member
    return workload.seed + prompt_index * request_group_size + member


def _gpu_telemetry() -> dict[str, float | None]:
    global _NVML_HANDLE, _NVML_UNAVAILABLE
    if not _NVML_UNAVAILABLE:
        try:
            import pynvml
        except ImportError:
            _NVML_UNAVAILABLE = True
        else:
            try:
                if _NVML_HANDLE is None:
                    pynvml.nvmlInit()
                    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                return {
                    "temperature_c": float(
                        pynvml.nvmlDeviceGetTemperature(
                            _NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                    "sm_clock_mhz": float(
                        pynvml.nvmlDeviceGetClockInfo(_NVML_HANDLE, pynvml.NVML_CLOCK_SM)
                    ),
                    "memory_clock_mhz": float(
                        pynvml.nvmlDeviceGetClockInfo(_NVML_HANDLE, pynvml.NVML_CLOCK_MEM)
                    ),
                    "gpu_utilization_pct": float(utilization.gpu),
                    "power_w": float(pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE)) / 1000.0,
                }
            except pynvml.NVMLError:
                _NVML_UNAVAILABLE = True
        # NVML is optional in the lightweight control-plane environment. A
        # runtime environment that supplies it avoids spawning nvidia-smi
        # between short timed requests; otherwise preserve the old fallback.
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
    """Collect telemetry without querying the driver during timed GPU work."""

    def __init__(self, interval_seconds: float = 1.0, *, background: bool = False) -> None:
        self.interval_seconds = interval_seconds
        self.background = background
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="edgeflow-gpu-monitor", daemon=True)
        self._started = False

    def start(self) -> None:
        self._started = True
        if self.background:
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval_seconds)

    def sample(self) -> dict[str, float | None]:
        values = _gpu_telemetry()
        self.samples.append({"monotonic_ns": time.monotonic_ns(), **values})
        return values

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self.background:
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

    @staticmethod
    def _output_tokens(result: Any) -> int:
        return int(result.reported_output_tokens or len(result.output_token_ids))

    @staticmethod
    def _output_identity(result: Any) -> str | tuple[int, ...]:
        return result.output_digest or result.output_token_ids

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
                "run_manifest.json",
                "workload.json",
                "execution_plan.json",
                "hardware_fingerprint.json",
                "metrics.jsonl",
                "stdout.log",
                "stderr.log",
                "monitor.jsonl",
                "validation_verdict.json",
                "VALIDATION.md",
                "correctness.json",
                "quality.json",
            ],
            "supersedes_run_id": None,
            "notes": "Production timing uses synchronized engine boundaries; warmup is stored separately.",
        }

    def recover_interrupted_run(self, partial: Path, *, reason: str) -> Path:
        """Promote an orphaned partial run into an auditable failed run.

        Native CUDA/PyTorch failures can terminate a worker before Python's exception
        handlers run. The supervising matrix process uses this method after a non-zero
        worker exit so the raw observations are retained and indexed as ineligible.
        """

        partial = partial.resolve()
        if partial.parent != self.artifact_root or not partial.name.endswith(".partial"):
            raise ValueError("partial run must be an immediate artifact-root child")
        manifest_path = partial / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("partial run has no manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest["run_id"])
        expected_name = f".{run_id}.partial"
        if partial.name != expected_name:
            raise ValueError("partial directory name does not match its run manifest")
        manifest["status"] = "FAILED"
        manifest["completed_at"] = utc_now()
        manifest["notes"] = f"Supervising process recovered a native worker failure: {reason}"
        write_json(manifest_path, manifest)
        stderr_path = partial / "stderr.log"
        prior_stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.is_file() else ""
        stderr_path.write_text(f"{prior_stderr}{reason}\n", encoding="utf-8")
        monitor_path = partial / "monitor.jsonl"
        monitor_path.touch(exist_ok=True)
        final = self.artifact_root / run_id
        if final.exists():
            raise FileExistsError(f"cannot recover over existing artifact {final}")
        os.replace(partial, final)
        verdict = validate_run(final, root=self.root, write=True)
        manifest["status"] = verdict["verdict"]
        write_json(final / "run_manifest.json", manifest)

        hardware = json.loads((final / "hardware_fingerprint.json").read_text(encoding="utf-8"))
        workload = json.loads((final / "workload.json").read_text(encoding="utf-8"))
        plan = json.loads((final / "execution_plan.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (final / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.db.save_hardware(hardware)
        self.db.save_workload(workload, manifest["workload_sha256"])
        self.db.save_plan(plan, manifest["plan_sha256"])
        self.db.save_run(manifest, final)
        self.db.save_metrics(run_id, records)
        self.db.save_validation(verdict)
        return final

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
            raise RuntimeError(
                f"GPU precheck failed: utilization is {utilization:.1f}% (must be <5%)"
            )
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
                    run_id=run_id,
                    request_id=None,
                    source_type=config.source_type,
                    phase="load",
                    iteration=0,
                    metrics=MetricValues(wall_ms=runtime.load_ms),
                    notes="Model and tokenizer preparation.",
                )
            )
            if runtime.compile_ms > 0:
                append(
                    MetricRecord(
                        run_id=run_id,
                        request_id=None,
                        source_type=config.source_type,
                        phase="compile",
                        iteration=0,
                        metrics=MetricValues(wall_ms=runtime.compile_ms),
                        notes="torch.compile wrapper construction; first graph compile appears in warmup.",
                    )
                )
            monitor.start()
            monitor.sample()
            manifest["status"] = "WARMING"
            write_json(temporary / "run_manifest.json", manifest)
            warmup_latencies: list[float] = []
            warmup_temperatures: list[float] = []
            warmup_clocks: list[float] = []
            if plan.backend == "torch_compile":
                prompt_counts = _prompt_counts_for_group(
                    workload,
                    iteration=-10_000,
                    request_group_size=request_group_size,
                    grouping_kind=grouping_kind,
                )
                token_groups = [
                    build_exact_token_ids(
                        tokenizer,
                        count,
                        seed=_prompt_seed(
                            workload,
                            prompt_index=-10_000,
                            request_group_size=request_group_size,
                            member=member,
                        ),
                    )
                    for member, count in enumerate(prompt_counts)
                ]
                first_compiled = runtime.generate_batch(token_groups, workload.output_tokens)[0]
                append(
                    MetricRecord(
                        run_id=run_id,
                        request_id="compile-first-execution",
                        source_type=config.source_type,
                        phase="compile",
                        iteration=1,
                        prompt_tokens=prompt_counts[0],
                        output_tokens=self._output_tokens(first_compiled),
                        metrics=MetricValues(wall_ms=first_compiled.wall_ms),
                        token_timestamps_ms=first_compiled.token_timestamps_ms,
                        notes="First graph compile/autotune execution; excluded from warmup and steady-state.",
                    )
                )
            maximum_warmup = max(config.warmup_requests, 300)
            warmup_converged = False
            for iteration in range(maximum_warmup):
                prompt_counts = _prompt_counts_for_group(
                    workload,
                    iteration=-iteration - 1,
                    request_group_size=request_group_size,
                    grouping_kind=grouping_kind,
                )
                token_groups = [
                    build_exact_token_ids(
                        tokenizer,
                        count,
                        seed=_prompt_seed(
                            workload,
                            prompt_index=-iteration - 1,
                            request_group_size=request_group_size,
                            member=member,
                        ),
                    )
                    for member, count in enumerate(prompt_counts)
                ]
                warmup_started = time.perf_counter_ns()
                warmup_results = runtime.generate_batch(token_groups, workload.output_tokens)
                warmup_wall_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000
                result = warmup_results[0]
                warmup_engine_ms = sorted(item.wall_ms for item in warmup_results)[
                    len(warmup_results) // 2
                ]
                warmup_latencies.append(warmup_engine_ms)
                warmup_telemetry = monitor.sample()
                if warmup_telemetry["temperature_c"] is not None:
                    warmup_temperatures.append(float(warmup_telemetry["temperature_c"]))
                if warmup_telemetry["sm_clock_mhz"] is not None:
                    warmup_clocks.append(float(warmup_telemetry["sm_clock_mhz"]))
                append(
                    MetricRecord(
                        run_id=run_id,
                        request_id=f"warmup-{iteration:04d}",
                        source_type=config.source_type,
                        phase="warmup",
                        iteration=iteration,
                        prompt_tokens=prompt_counts[0],
                        output_tokens=self._output_tokens(result),
                        metrics=MetricValues(
                            wall_ms=warmup_wall_ms,
                            ttft_ms=result.ttft_ms,
                            tpot_ms=result.tpot_ms,
                            temperature_c=warmup_telemetry["temperature_c"],
                            sm_clock_mhz=warmup_telemetry["sm_clock_mhz"],
                            memory_clock_mhz=warmup_telemetry["memory_clock_mhz"],
                            gpu_utilization_pct=warmup_telemetry["gpu_utilization_pct"],
                            power_w=warmup_telemetry["power_w"],
                        ),
                        token_timestamps_ms=result.token_timestamps_ms,
                        notes=f"Excluded from steady-state summary; {grouping_kind}={request_group_size}.",
                    )
                )
                if iteration + 1 >= max(config.warmup_requests, 10):
                    previous = sorted(warmup_latencies[-10:-5])[2]
                    recent = sorted(warmup_latencies[-5:])[2]
                    drift = abs(recent - previous) / previous if previous > 0 else float("inf")
                    temperature_stable = (
                        len(warmup_temperatures) >= 10
                        and abs(
                            statistics.median(warmup_temperatures[-10:-5])
                            - statistics.median(warmup_temperatures[-5:])
                        )
                        <= 1.0
                    )
                    clock_stable = (
                        len(warmup_clocks) >= 10
                        and min(warmup_clocks[-10:]) / max(warmup_clocks[-10:]) >= 0.97
                    )
                    if (
                        drift < 0.02
                        and robust_cv(warmup_latencies[-10:]) <= 0.10
                        and temperature_stable
                        and clock_stable
                    ):
                        warmup_converged = True
                        break
            if not warmup_converged:
                manifest["notes"] += (
                    " Warmup did not satisfy the registered 2% median-drift, 10% robust-CV, "
                    "1°C recent median-temperature-drift, and 0.97 active-clock-ratio thresholds "
                    f"within {maximum_warmup} request groups; validation must decide eligibility."
                )
            correctness_counts = _prompt_counts_for_group(
                workload,
                iteration=-20_000,
                request_group_size=request_group_size,
                grouping_kind=grouping_kind,
            )
            correctness_ids = [
                build_exact_token_ids(
                    tokenizer,
                    count,
                    seed=_prompt_seed(
                        workload,
                        prompt_index=-20_000,
                        request_group_size=request_group_size,
                        member=member,
                    ),
                )
                for member, count in enumerate(correctness_counts)
            ]
            if grouping_kind == "concurrency" and request_group_size > 1:
                # HTTP arrival order can change scheduler batch slots between
                # otherwise identical concurrent repeats. Verify every fixed
                # prompt independently so G2 measures runtime/model correctness;
                # the measured block below still exercises true concurrency.
                correctness_a = [
                    runtime.generate(token_ids, workload.output_tokens)
                    for token_ids in correctness_ids
                ]
                correctness_b = [
                    runtime.generate(token_ids, workload.output_tokens)
                    for token_ids in correctness_ids
                ]
                correctness_reference = "pinned_runtime_sequential_prompt_repeat"
            else:
                correctness_a = runtime.generate_batch(correctness_ids, workload.output_tokens)
                correctness_b = runtime.generate_batch(correctness_ids, workload.output_tokens)
                correctness_reference = "pinned_runtime_deterministic_repeat"
            deterministic = [self._output_identity(item) for item in correctness_a] == [
                self._output_identity(item) for item in correctness_b
            ]
            timestamps_valid = all(
                len(result.token_timestamps_ms) in {0, self._output_tokens(result)}
                for result in (*correctness_a, *correctness_b)
            )
            correctness_pass = bool(
                deterministic
                and all(self._output_tokens(result) > 0 for result in (*correctness_a, *correctness_b))
                and timestamps_valid
            )
            write_json(
                temporary / "correctness.json",
                {
                    "schema_version": "1.0",
                    "pass": correctness_pass,
                    "nan_count": 0,
                    "reference_type": correctness_reference,
                    "prompt_tokens": correctness_counts[0],
                    "prompt_token_counts": correctness_counts,
                    "requested_output_tokens": workload.output_tokens,
                    "request_group_size": request_group_size,
                    "grouping_kind": grouping_kind,
                    "observed_output_tokens": [
                        self._output_tokens(result) for result in correctness_a
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
                prompt_counts = _prompt_counts_for_group(
                    workload,
                    iteration=prompt_index,
                    request_group_size=request_group_size,
                    grouping_kind=grouping_kind,
                )
                token_groups = [
                    build_exact_token_ids(
                        tokenizer,
                        count,
                        seed=_prompt_seed(
                            workload,
                            prompt_index=prompt_index,
                            request_group_size=request_group_size,
                            member=member,
                        ),
                    )
                    for member, count in enumerate(prompt_counts)
                ]
                group_started = time.perf_counter_ns()
                results = runtime.generate_batch(token_groups, workload.output_tokens)
                group_wall_ms = (time.perf_counter_ns() - group_started) / 1_000_000
                # Querying nvidia-smi can serialize driver work on WSL/consumer GPUs.
                # Sample only after the synchronized engine timer has closed.
                telemetry = monitor.sample()
                group_output_tokens = sum(self._output_tokens(result) for result in results)
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
                            prompt_tokens=prompt_counts[member],
                            output_tokens=self._output_tokens(result),
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
                                "Engine-only greedy generation; deterministic paired prompts; "
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
            manifest["notes"] = (
                f"Run failed and raw artifacts were preserved: {type(exc).__name__}."
            )
            (temporary / "stderr.log").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
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
