#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROCESS_STARTED_NS = time.perf_counter_ns()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.core.models import ExecutionPlan, utc_now  # noqa: E402
from edgeflow.core.serialization import read_json, sha256_value, write_json  # noqa: E402
from edgeflow.experiments import summarize_cold_warm_study  # noqa: E402
from edgeflow.experiments.orchestrator import _gpu_telemetry, _gpu_utilization  # noqa: E402
from edgeflow.hardware import inspect_hardware  # noqa: E402
from edgeflow.models import ModelRegistry  # noqa: E402
from edgeflow.quality import find_compatible_quality_report  # noqa: E402
from edgeflow.runtimes import PytorchAdapter  # noqa: E402
from edgeflow.workloads import create_workload  # noqa: E402
from edgeflow.workloads.builder import build_exact_token_ids  # noqa: E402


def _git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.returncode == 0 and not result.stdout.strip()


def _worker(
    *,
    sample_index: int,
    model_id: str,
    prompt_tokens: int,
    output_tokens: int,
    warmup: int,
) -> dict[str, Any]:
    utilization = _gpu_utilization()
    if utilization is not None and utilization >= 5.0:
        raise RuntimeError(f"GPU precheck failed: utilization is {utilization:.1f}%")
    hardware = inspect_hardware(ROOT)
    registry = ModelRegistry(ROOT / "specs" / "model_registry.yaml")
    model_ref, revision = registry.resolve_source(model_id, "safetensors")
    workload = create_workload(
        workload_id="e20-cached-process-cold-vs-warm",
        model_id=model_id,
        prompt_distribution=str(prompt_tokens),
        output_tokens=output_tokens,
        seed=42,
    )
    plan = ExecutionPlan(
        plan_id="e20-pytorch-eager-bf16",
        model_id=model_id,
        backend="pytorch_eager",
        model_format="safetensors",
        dtype="bf16",
        backend_args={"revision": revision, "trust_remote_code": False},
    ).with_hash()
    quality = find_compatible_quality_report(
        artifact_root=ROOT / "artifacts",
        model_id=model_id,
        model_revision=revision,
        plan=plan,
    )
    if quality is None:
        raise RuntimeError("E20 requires the pinned exact-scope BF16 quality report")

    prepare_started = time.perf_counter_ns()
    runtime, tokenizer = PytorchAdapter(compiled=False).prepare(
        model_ref,
        plan,
        workload,
        local_files_only=True,
    )
    prepare_wall_ms = (time.perf_counter_ns() - prepare_started) / 1_000_000
    try:
        token_ids = build_exact_token_ids(tokenizer, prompt_tokens, seed=42)
        first_host_started = time.perf_counter_ns()
        first = runtime.generate(token_ids, output_tokens)
        first_host_ms = (time.perf_counter_ns() - first_host_started) / 1_000_000
        first_usable_ms = (time.perf_counter_ns() - PROCESS_STARTED_NS) / 1_000_000
        warmed = first
        warmed_host_ms = first_host_ms
        for _ in range(warmup):
            warmed_host_started = time.perf_counter_ns()
            warmed = runtime.generate(token_ids, output_tokens)
            warmed_host_ms = (time.perf_counter_ns() - warmed_host_started) / 1_000_000
        return {
            "sample_index": sample_index,
            "status": "COMPLETED",
            "model_id": model_id,
            "model_revision": revision,
            "plan_sha256": plan.canonical_sha256,
            "quality_report_id": quality.get("report_id"),
            "hardware_fingerprint_sha256": hardware["sha256"],
            "process_bootstrap_before_prepare_ms": (
                prepare_started - PROCESS_STARTED_NS
            )
            / 1_000_000,
            "prepare_wall_ms": prepare_wall_ms,
            "load_ms": runtime.load_ms,
            "first_usable_ms": first_usable_ms,
            "first_response_host_ms": first_host_ms,
            "first_response_engine_ms": first.wall_ms,
            "warmed_response_host_ms": warmed_host_ms,
            "warmed_response_engine_ms": warmed.wall_ms,
            "first_output_sha256": sha256_value(list(first.output_token_ids)),
            "warmed_output_sha256": sha256_value(list(warmed.output_token_ids)),
            "output_length_matches": (
                len(first.output_token_ids) == output_tokens
                and len(warmed.output_token_ids) == output_tokens
            ),
            "telemetry_after_timing": _gpu_telemetry(),
            "completed_at": utc_now(),
        }
    finally:
        runtime.shutdown()


def _worker_command(arguments: argparse.Namespace, index: int, result_path: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-id",
        arguments.model_id,
        "--repetitions",
        str(arguments.repetitions),
        "--prompt-tokens",
        str(arguments.prompt_tokens),
        "--output-tokens",
        str(arguments.output_tokens),
        "--warmup",
        str(arguments.warmup),
        "--worker-index",
        str(index),
        "--worker-result",
        str(result_path),
    ]


def _summarize(
    output_root: Path,
    samples: list[dict[str, Any]],
    *,
    repetitions: int,
    model_id: str,
    prompt_tokens: int,
    output_tokens: int,
    warmup: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        **summarize_cold_warm_study(samples, repetitions=repetitions),
        "source_type": "measured",
        "protocol": {
            "design": "fresh-process paired first-versus-warmed response",
            "repetitions": repetitions,
            "warmup_requests_before_warm_arm": warmup,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "bootstrap_resamples": 10_000,
            "filesystem_cache": "warm_or_uncontrolled",
        },
        "model_id": model_id,
        "samples": samples,
        "hardware": inspect_hardware(ROOT),
        "created_at": utc_now(),
        "artifact_root": str(output_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E20 cached-process cold/warm study")
    parser.add_argument("--model-id", default="llama-3.2-3b-instruct")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if min(
        arguments.repetitions,
        arguments.prompt_tokens,
        arguments.output_tokens,
        arguments.warmup,
    ) < 1:
        parser.error("repetitions, token counts, and warmup must be positive")
    if arguments.repetitions >= 30 and not _git_clean():
        print("Formal E20 requires a clean git checkout.", file=sys.stderr)
        return 2
    if arguments.worker_index is not None:
        if arguments.worker_result is None:
            parser.error("--worker-result is required with --worker-index")
        try:
            result = _worker(
                sample_index=arguments.worker_index,
                model_id=arguments.model_id,
                prompt_tokens=arguments.prompt_tokens,
                output_tokens=arguments.output_tokens,
                warmup=arguments.warmup,
            )
        except Exception as exc:
            result = {
                "sample_index": arguments.worker_index,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "completed_at": utc_now(),
            }
        write_json(arguments.worker_result, result)
        return 0 if result["status"] == "COMPLETED" else 1

    output_root = ROOT / "artifacts" / "experiments" / "E20"
    worker_root = output_root / "workers"
    log_root = output_root / "logs"
    worker_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    if arguments.summarize_existing:
        samples = [
            read_json(path)
            for path in sorted(worker_root.glob("sample-*.json"))
        ][: arguments.repetitions]
    else:
        samples: list[dict[str, Any]] = []
        for index in range(arguments.repetitions):
            result_path = worker_root / f"sample-{index:03d}.json"
            log_path = log_root / f"sample-{index:03d}.log"
            result_path.unlink(missing_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    _worker_command(arguments, index, result_path),
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            if result_path.is_file():
                samples.append(read_json(result_path))
            else:
                samples.append(
                    {
                        "sample_index": index,
                        "status": "FAILED",
                        "error_type": "WorkerExit",
                        "error": f"worker exited with code {process.returncode}",
                        "completed_at": utc_now(),
                    }
                )
            progress = _summarize(
                output_root,
                samples,
                repetitions=arguments.repetitions,
                model_id=arguments.model_id,
                prompt_tokens=arguments.prompt_tokens,
                output_tokens=arguments.output_tokens,
                warmup=arguments.warmup,
            )
            write_json(output_root / "progress.json", progress)

    result = _summarize(
        output_root,
        samples,
        repetitions=arguments.repetitions,
        model_id=arguments.model_id,
        prompt_tokens=arguments.prompt_tokens,
        output_tokens=arguments.output_tokens,
        warmup=arguments.warmup,
    )
    write_json(output_root / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_pairs": result["completed_pairs"],
                "output": str(output_root / "result.json"),
            }
        )
    )
    return 0 if result["protocol_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
