#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.core.models import ExecutionPlan, utc_now  # noqa: E402
from edgeflow.core.serialization import read_json, sha256_value, write_json  # noqa: E402
from edgeflow.experiments import (  # noqa: E402
    E06_MODES,
    E06_SEQUENCE,
    summarize_dynamic_shape_study,
)
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


def _counter_snapshot() -> dict[str, int]:
    import torch

    counters = getattr(getattr(torch, "_dynamo", None), "utils", None)
    values = getattr(counters, "counters", {})
    flattened: dict[str, int] = {}
    for category, entries in values.items():
        for name, value in entries.items():
            if isinstance(value, int):
                flattened[f"{category}.{name}"] = value
    return flattened


def _worker(mode: str, model_id: str, repetitions: int, output_tokens: int) -> dict[str, Any]:
    import torch

    registry = ModelRegistry(ROOT / "specs" / "model_registry.yaml")
    model_ref, revision = registry.resolve_source(model_id, "safetensors")
    dynamic = {"false": False, "auto": None, "true": True}[mode]
    workload = create_workload(
        workload_id=f"e06-dynamic-{mode}",
        model_id=model_id,
        prompt_distribution="128",
        output_tokens=output_tokens,
        seed=42,
    )
    plan = ExecutionPlan(
        plan_id=f"e06-torch-compile-dynamic-{mode}-bf16",
        model_id=model_id,
        backend="torch_compile",
        model_format="safetensors",
        dtype="bf16",
        compile_mode="default",
        dynamic_shapes=dynamic,
        fullgraph=False,
        cuda_graph=False,
        backend_args={"revision": revision, "trust_remote_code": False},
    ).with_hash()
    quality = find_compatible_quality_report(
        artifact_root=ROOT / "artifacts",
        model_id=model_id,
        model_revision=revision,
        plan=plan,
    )
    if quality is None:
        raise RuntimeError("E06 requires a formal exact-scope BF16 quality report")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    runtime, tokenizer = PytorchAdapter(compiled=True).prepare(
        model_ref,
        plan,
        workload,
        local_files_only=True,
    )
    observations: list[dict[str, Any]] = []
    output_hashes: dict[str, str] = {}
    output_token_ids: dict[str, list[int]] = {}
    try:
        token_ids = {
            prompt: build_exact_token_ids(tokenizer, prompt, seed=workload.seed + prompt)
            for prompt in sorted(set(E06_SEQUENCE))
        }
        for block in range(repetitions):
            for sequence_index, prompt in enumerate(E06_SEQUENCE):
                started = time.perf_counter_ns()
                result = runtime.generate(token_ids[prompt], output_tokens)
                host_ms = (time.perf_counter_ns() - started) / 1_000_000
                if block == 0 and str(prompt) not in output_hashes:
                    output_hashes[str(prompt)] = sha256_value(list(result.output_token_ids))
                    output_token_ids[str(prompt)] = list(result.output_token_ids)
                observations.append(
                    {
                        "block": block,
                        "sequence_index": sequence_index,
                        "prompt_tokens": prompt,
                        "output_tokens": len(result.output_token_ids),
                        "latency_ms": result.wall_ms,
                        "host_latency_ms": host_ms,
                        "ttft_ms": result.ttft_ms,
                        "tpot_ms": result.tpot_ms,
                        "peak_vram_bytes": result.peak_vram_bytes,
                        "counters": _counter_snapshot(),
                    }
                )
        return {
            "dynamic_mode": mode,
            "status": "COMPLETED",
            "model_id": model_id,
            "model_revision": revision,
            "plan": plan.model_dump(mode="json"),
            "quality_report_id": quality.get("report_id"),
            "repetitions": repetitions,
            "sequence": list(E06_SEQUENCE),
            "observations": observations,
            "output_hashes": output_hashes,
            "output_token_ids": output_token_ids,
            "final_counters": _counter_snapshot(),
            "completed_at": utc_now(),
        }
    finally:
        runtime.shutdown()


def _worker_command(arguments: argparse.Namespace, mode: str, result_path: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-id",
        arguments.model_id,
        "--repetitions",
        str(arguments.repetitions),
        "--output-tokens",
        str(arguments.output_tokens),
        "--worker-mode",
        mode,
        "--worker-result",
        str(result_path),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E06 dynamic-shape/recompilation study")
    parser.add_argument("--model-id", default="llama-3.2-3b-instruct")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--worker-mode", choices=E06_MODES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.repetitions < 1 or arguments.output_tokens < 1:
        parser.error("repetitions and output tokens must be positive")
    if arguments.repetitions >= 30 and not _git_clean():
        print("Formal E06 requires a clean git checkout.", file=sys.stderr)
        return 2
    if arguments.worker_mode:
        if arguments.worker_result is None:
            parser.error("--worker-result is required with --worker-mode")
        try:
            result = _worker(
                arguments.worker_mode,
                arguments.model_id,
                arguments.repetitions,
                arguments.output_tokens,
            )
        except Exception as exc:
            result = {
                "dynamic_mode": arguments.worker_mode,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "completed_at": utc_now(),
            }
        write_json(arguments.worker_result, result)
        return 0 if result["status"] == "COMPLETED" else 1

    output_root = ROOT / "artifacts" / "experiments" / "E06"
    worker_root = output_root / "workers"
    log_root = output_root / "logs"
    if arguments.summarize_existing:
        cases = [
            read_json(worker_root / f"dynamic-{mode}.json")
            for mode in E06_MODES
            if (worker_root / f"dynamic-{mode}.json").is_file()
        ]
        previous = read_json(output_root / "result.json")
        result = {
            **previous,
            **summarize_dynamic_shape_study(cases, repetitions=arguments.repetitions),
            "cases": cases,
            "created_at": utc_now(),
        }
        write_json(output_root / "result.json", result)
        write_json(output_root / "progress.json", result)
        print(json.dumps({"status": result["status"], "output": str(output_root / "result.json")}))
        return 0 if result["pass"] else 1
    worker_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for mode in E06_MODES:
        result_path = worker_root / f"dynamic-{mode}.json"
        log_path = log_root / f"dynamic-{mode}.log"
        result_path.unlink(missing_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                _worker_command(arguments, mode, result_path),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if result_path.is_file():
            cases.append(read_json(result_path))
        else:
            cases.append(
                {
                    "dynamic_mode": mode,
                    "status": "FAILED",
                    "error_type": "WorkerExit",
                    "error": f"worker exited with code {process.returncode}",
                }
            )
        write_json(
            output_root / "progress.json",
            {
                "schema_version": "1.0",
                "experiment_id": "E06",
                "status": "RUNNING",
                "cases": [
                    {
                        key: case.get(key)
                        for key in ("dynamic_mode", "status", "error_type", "error")
                        if case.get(key) is not None
                    }
                    for case in cases
                ],
                "updated_at": utc_now(),
            },
        )
    summary = summarize_dynamic_shape_study(cases, repetitions=arguments.repetitions)
    result = {
        **summary,
        "model_id": arguments.model_id,
        "source_type": "measured",
        "output_tokens": arguments.output_tokens,
        "hardware": inspect_hardware(ROOT),
        "cases": cases,
        "created_at": utc_now(),
    }
    write_json(output_root / "result.json", result)
    write_json(output_root / "progress.json", result)
    print(json.dumps({"status": result["status"], "output": str(output_root / "result.json")}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
