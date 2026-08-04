#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.core.models import ExecutionPlan, utc_now  # noqa: E402
from edgeflow.core.serialization import write_json  # noqa: E402
from edgeflow.experiments.orchestrator import _gpu_telemetry, _gpu_utilization  # noqa: E402
from edgeflow.hardware import inspect_hardware  # noqa: E402
from edgeflow.kernels.rmsnorm import LlamaRMSNormIntegration  # noqa: E402
from edgeflow.metrics.statistics import describe, paired_bootstrap  # noqa: E402
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


def _scope_result(
    name: str,
    prompt_tokens: int,
    baseline: list[float],
    intervention: list[float],
    *,
    correctness_pass: bool,
) -> dict[str, Any]:
    comparison = paired_bootstrap(baseline, intervention, resamples=10_000, seed=42)
    return {
        "scope": name,
        "prompt_tokens": prompt_tokens,
        "correctness_pass": correctness_pass,
        "baseline": describe(baseline),
        "intervention": describe(intervention),
        "comparison": comparison,
        "accepted": bool(correctness_pass and comparison["claim_direction_supported"]),
    }


def _run_worker(
    *,
    model_id: str,
    repetitions: int,
    output_tokens: int,
    warmup: int,
) -> dict[str, Any]:
    utilization = _gpu_utilization()
    if utilization is not None and utilization >= 5.0:
        raise RuntimeError(f"GPU precheck failed: utilization is {utilization:.1f}%")
    registry = ModelRegistry(ROOT / "specs" / "model_registry.yaml")
    model_ref, revision = registry.resolve_source(model_id, "safetensors")
    workload = create_workload(
        workload_id="e24-llama-rmsnorm-integration",
        model_id=model_id,
        prompt_distribution="128",
        output_tokens=output_tokens,
        seed=42,
    )
    plan = ExecutionPlan(
        plan_id="e24-pytorch-eager-bf16-rmsnorm",
        model_id=model_id,
        backend="pytorch_eager",
        model_format="safetensors",
        dtype="bf16",
        custom_kernels=("fused-residual-rmsnorm-v1",),
        backend_args={"revision": revision, "trust_remote_code": False},
    ).with_hash()
    quality_plan = plan.model_copy(update={"custom_kernels": ()}).with_hash()
    quality = find_compatible_quality_report(
        artifact_root=ROOT / "artifacts",
        model_id=model_id,
        model_revision=revision,
        plan=quality_plan,
    )
    if quality is None:
        raise RuntimeError("E24 requires the pinned BF16 model quality report")
    runtime, tokenizer = PytorchAdapter(compiled=False).prepare(
        model_ref,
        plan,
        workload,
        local_files_only=True,
    )
    integration = LlamaRMSNormIntegration(runtime.model)
    scopes = (("search", 128, 42), ("holdout", 257, 314159))
    results: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    try:
        prompts = {
            name: build_exact_token_ids(tokenizer, tokens, seed=seed)
            for name, tokens, seed in scopes
        }
        for _ in range(warmup):
            runtime.generate(prompts["search"], output_tokens)
        integration.enable()
        for _ in range(warmup):
            runtime.generate(prompts["search"], output_tokens)
        integration.disable()
        for name, prompt_tokens, _seed in scopes:
            baseline_correctness = runtime.generate(prompts[name], output_tokens)
            integration.enable()
            intervention_correctness = runtime.generate(prompts[name], output_tokens)
            integration.disable()
            correctness_pass = (
                baseline_correctness.output_token_ids
                == intervention_correctness.output_token_ids
                and len(baseline_correctness.output_token_ids) == output_tokens
            )
            baseline: list[float] = []
            intervention: list[float] = []
            for pair_index in range(repetitions):
                order = (False, True) if pair_index % 2 == 0 else (True, False)
                for enabled in order:
                    if enabled:
                        integration.enable()
                    else:
                        integration.disable()
                    generated = runtime.generate(prompts[name], output_tokens)
                    (intervention if enabled else baseline).append(generated.wall_ms)
                    telemetry.append(
                        {
                            "scope": name,
                            "pair_index": pair_index,
                            "kernel_enabled": enabled,
                            **_gpu_telemetry(),
                        }
                    )
            integration.disable()
            results.append(
                _scope_result(
                    name,
                    prompt_tokens,
                    baseline,
                    intervention,
                    correctness_pass=correctness_pass,
                )
            )
        integration.enable()
        integration_summary = integration.summary()
        integration.disable()
    finally:
        integration.disable()
        runtime.shutdown()
    protocol_complete = (
        repetitions >= 30
        and len(results) == 2
        and all(row["correctness_pass"] for row in results)
        and integration_summary["triton_calls"] > 0
    )
    accepted = protocol_complete and all(row["accepted"] for row in results)
    return {
        "schema_version": "1.0",
        "experiment_id": "E24",
        "status": "PASS" if accepted else "VALIDATED_NEUTRAL" if protocol_complete else "FAILED",
        "pass": accepted,
        "protocol_complete": protocol_complete,
        "claim_status": "END_TO_END_SUPPORTED" if accepted else "MICRO_ONLY",
        "source_type": "measured",
        "model_id": model_id,
        "model_revision": revision,
        "plan": plan.model_dump(mode="json"),
        "quality_report_id": quality.get("report_id"),
        "protocol": {
            "design": "paired ABBA",
            "repetitions_per_scope": repetitions,
            "warmup_per_arm": warmup,
            "output_tokens": output_tokens,
            "bootstrap_resamples": 10_000,
            "acceptance": "search and untouched holdout lower CI with >=2% practical gain",
            "timing": "unprofiled CUDA-event engine boundary",
        },
        "scopes": results,
        "integration": integration_summary,
        "telemetry": telemetry,
        "hardware": inspect_hardware(ROOT),
        "created_at": utc_now(),
        "rollback": "LlamaRMSNormIntegration.disable restores every original decoder forward method.",
        "claim_scope": (
            "End-to-end Llama 3.2 3B BF16 result on the exact search and holdout scopes."
            if accepted
            else "The Triton contribution remains microbenchmark-only; no model-level speedup claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E24 end-to-end Triton integration study")
    parser.add_argument("--model-id", default="llama-3.2-3b-instruct")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if min(arguments.repetitions, arguments.output_tokens, arguments.warmup) < 1:
        parser.error("repetitions, output tokens, and warmup must be positive")
    if arguments.repetitions >= 30 and not _git_clean():
        print("Formal E24 requires a clean git checkout.", file=sys.stderr)
        return 2
    if arguments.worker_result:
        try:
            result = _run_worker(
                model_id=arguments.model_id,
                repetitions=arguments.repetitions,
                output_tokens=arguments.output_tokens,
                warmup=arguments.warmup,
            )
        except Exception as exc:
            result = {
                "schema_version": "1.0",
                "experiment_id": "E24",
                "status": "FAILED",
                "pass": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "created_at": utc_now(),
            }
        write_json(arguments.worker_result, result)
        return 0 if result.get("protocol_complete") else 1

    output_root = ROOT / "artifacts" / "experiments" / "E24"
    output_root.mkdir(parents=True, exist_ok=True)
    worker_result = output_root / "worker-result.json"
    log_path = output_root / "worker.log"
    worker_result.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-id",
        arguments.model_id,
        "--repetitions",
        str(arguments.repetitions),
        "--output-tokens",
        str(arguments.output_tokens),
        "--warmup",
        str(arguments.warmup),
        "--worker-result",
        str(worker_result),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if worker_result.is_file():
        result = json.loads(worker_result.read_text(encoding="utf-8"))
    else:
        result = {
            "schema_version": "1.0",
            "experiment_id": "E24",
            "status": "FAILED",
            "pass": False,
            "error_type": "WorkerExit",
            "error": f"worker exited with code {process.returncode}",
            "created_at": utc_now(),
        }
    write_json(output_root / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "claim_status": result.get("claim_status"),
                "output": str(output_root / "result.json"),
            }
        )
    )
    return 0 if result.get("protocol_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
