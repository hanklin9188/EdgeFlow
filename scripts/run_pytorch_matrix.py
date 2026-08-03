#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.api.schemas import BenchmarkSubmission  # noqa: E402
from edgeflow.core.models import utc_now  # noqa: E402
from edgeflow.core.serialization import read_json, write_json  # noqa: E402
from edgeflow.experiments import BenchmarkConfig, RunOrchestrator  # noqa: E402
from edgeflow.experiments.matrix import (  # noqa: E402
    matrix_case_label,
    matrix_progress_status,
    pytorch_matrix_cases,
)
from edgeflow.models import ModelRegistry  # noqa: E402
from edgeflow.quality import find_compatible_quality_report  # noqa: E402


def _git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.returncode == 0 and not result.stdout.strip()


def _save_progress(
    path: Path,
    *,
    experiment_id: str,
    model_id: str,
    quick: bool,
    cases: list[dict[str, Any]],
    total_case_count: int,
) -> None:
    failed = sum(row["status"] == "FAILED" for row in cases)
    status, passed = matrix_progress_status(cases, total_case_count=total_case_count)
    payload = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "model_id": model_id,
        "source_type": "measured",
        "protocol_status": "DEVELOPMENT" if quick else "FORMAL",
        "status": status,
        "pass": passed,
        "updated_at": utc_now(),
        "case_count": len(cases),
        "failed_case_count": failed,
        "cases": cases,
        "claim_scope": "Matrix execution status only; individual validation verdicts govern claims.",
    }
    write_json(path, payload)


def _execute_case(
    arguments: argparse.Namespace,
    case: dict[str, Any],
    *,
    registry: ModelRegistry,
    revision: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {**case, "status": "RUNNING", "started_at": utc_now()}
    if case["compile_mode"] in {"reduce-overhead", "max-autotune"}:
        record.update(
            {
                "status": "FAILED",
                "error_type": "UnsupportedCompileMode",
                "error": (
                    f"{case['compile_mode']} is capability-pruned: its internal CUDA Graph "
                    "path is incompatible with the mutable token-by-token KV cache"
                ),
                "completed_at": utc_now(),
            }
        )
        return record
    submission = BenchmarkSubmission(
        label=matrix_case_label(case["case_id"]),
        model_id=arguments.model_id,
        model_format="safetensors",
        backend=case["backend"],
        prompt_tokens=case["prompt_tokens"],
        output_tokens=case["output_tokens"],
        batch_size=case["batch_size"],
        concurrency=case["concurrency"],
        session_requests=20,
        quality_profile="balanced",
        dtype="bf16",
        compile_mode=case["compile_mode"],
        dynamic_shapes=case["dynamic_shapes"],
        fullgraph=False,
        cuda_graph=False,
        repetitions=arguments.repetitions or (3 if arguments.quick else 30),
        warmup_requests=arguments.warmup,
        experiment_id=arguments.experiment_id,
        allow_download=arguments.allow_download,
        allow_busy_gpu=arguments.allow_busy_gpu,
    )
    resolved_ref, _, plan = submission.resolve(registry)
    quality = find_compatible_quality_report(
        artifact_root=ROOT / "artifacts",
        model_id=arguments.model_id,
        model_revision=revision,
        plan=plan,
    )
    if not arguments.quick and quality is None and not arguments.allow_missing_quality:
        record.update(
            {
                "status": "FAILED",
                "error_type": "MissingQualityReport",
                "error": f"No formal exact-scope quality report for {arguments.model_id}/{plan.dtype}.",
                "completed_at": utc_now(),
            }
        )
        return record
    orchestrator = RunOrchestrator(root=ROOT, artifact_root=ROOT / "artifacts")
    try:
        run_dir = orchestrator.run(
            model_ref=resolved_ref,
            workload=submission.workload(),
            plan=plan,
            config=BenchmarkConfig(
                experiment_id=arguments.experiment_id,
                repetitions=submission.repetitions,
                warmup_requests=submission.warmup_requests,
                local_files_only=not submission.allow_download,
                enforce_idle=not submission.allow_busy_gpu,
            ),
            command=[
                sys.executable,
                "scripts/run_pytorch_matrix.py",
                arguments.experiment_id,
                "--model-id",
                arguments.model_id,
                "--worker-case",
                case["case_id"],
            ],
        )
        verdict = read_json(run_dir / "validation_verdict.json")
        record.update(
            {
                "status": "COMPLETED",
                "run_id": run_dir.name,
                "verdict": verdict["verdict"],
                "policy_eligible": verdict["policy_eligible"],
                "completed_at": utc_now(),
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "completed_at": utc_now(),
            }
        )
    return record


def _worker_command(arguments: argparse.Namespace, case_id: str, result_file: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        arguments.experiment_id,
        "--model-id",
        arguments.model_id,
        "--warmup",
        str(arguments.warmup),
        "--worker-case",
        case_id,
        "--worker-result",
        str(result_file),
    ]
    if arguments.quick:
        command.append("--quick")
    if arguments.repetitions is not None:
        command.extend(("--repetitions", str(arguments.repetitions)))
    if arguments.allow_download:
        command.append("--allow-download")
    if arguments.allow_busy_gpu:
        command.append("--allow-busy-gpu")
    if arguments.allow_missing_quality:
        command.append("--allow-missing-quality")
    return command


def _native_exit_description(returncode: int) -> str:
    if returncode >= 0:
        return f"worker exited with code {returncode}"
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        name = f"signal {-returncode}"
    return f"worker terminated by {name} ({returncode})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or resume the registered PyTorch matrix")
    parser.add_argument("experiment_id", choices=["E04", "E05"])
    parser.add_argument("--model-id", default="llama-3.2-3b-instruct")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--allow-missing-quality", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--max-new-cases", type=int)
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.max_new_cases is not None and arguments.max_new_cases < 1:
        parser.error("--max-new-cases must be positive")
    if not arguments.quick and not _git_clean():
        print("Formal matrix requires a clean git checkout.", file=sys.stderr)
        return 2
    registry = ModelRegistry(ROOT / "specs" / "model_registry.yaml")
    _model_ref, revision = registry.resolve_source(arguments.model_id, "safetensors")
    cases = pytorch_matrix_cases(arguments.experiment_id, quick=arguments.quick)
    output = (
        ROOT
        / "artifacts"
        / "experiments"
        / arguments.experiment_id
        / ("matrix-quick.json" if arguments.quick else "matrix.json")
    )
    if arguments.worker_case:
        if arguments.worker_result is None:
            parser.error("--worker-result is required with --worker-case")
        case = next((row for row in cases if row["case_id"] == arguments.worker_case), None)
        if case is None:
            parser.error(f"unknown worker case {arguments.worker_case}")
        try:
            record = _execute_case(arguments, case, registry=registry, revision=revision)
        except Exception as exc:
            record = {
                **case,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "completed_at": utc_now(),
            }
        write_json(arguments.worker_result, record)
        return 0 if record["status"] == "COMPLETED" else 1

    previous: dict[str, Any] = read_json(output) if output.is_file() else {"cases": []}
    orchestrator = RunOrchestrator(root=ROOT, artifact_root=ROOT / "artifacts")

    # Recover a native-crash orphan before deciding which cases need to resume.
    previous_cases = list(previous.get("cases", []))
    for row in previous_cases:
        if row.get("status") != "RUNNING":
            continue
        for partial in (ROOT / "artifacts").glob(".*.partial"):
            manifest_path = partial / "run_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = read_json(manifest_path)
            if row["case_id"].lower() not in str(manifest.get("plan_id", "")):
                continue
            reason = "orphaned matrix worker detected during resume"
            recovered = orchestrator.recover_interrupted_run(partial, reason=reason)
            row.update(
                {
                    "status": "FAILED",
                    "error_type": "InterruptedWorker",
                    "error": reason,
                    "run_id": recovered.name,
                    "completed_at": utc_now(),
                }
            )

    settled = {
        row["case_id"]: row
        for row in previous_cases
        if (row.get("status") == "COMPLETED" and not arguments.rerun_completed)
        or (row.get("status") == "FAILED" and not arguments.rerun_failed)
    }
    results = list(settled.values())
    new_cases = 0
    for case in cases:
        if case["case_id"] in settled:
            continue
        if arguments.max_new_cases is not None and new_cases >= arguments.max_new_cases:
            break
        record: dict[str, Any] = {**case, "status": "RUNNING", "started_at": utc_now()}
        results.append(record)
        new_cases += 1
        _save_progress(
            output,
            experiment_id=arguments.experiment_id,
            model_id=arguments.model_id,
            quick=arguments.quick,
            cases=results,
            total_case_count=len(cases),
        )
        result_dir = output.parent / "worker-results"
        log_dir = output.parent / "worker-logs"
        result_file = result_dir / f"{case['case_id']}.json"
        log_file = log_dir / f"{case['case_id']}.log"
        result_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        result_file.unlink(missing_ok=True)
        partials_before = set((ROOT / "artifacts").glob(".*.partial"))
        with log_file.open("w", encoding="utf-8") as worker_log:
            process = subprocess.run(
                _worker_command(arguments, case["case_id"], result_file),
                cwd=ROOT,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if result_file.is_file():
            record.update(read_json(result_file))
            result_file.unlink()
        else:
            description = _native_exit_description(process.returncode)
            recovered_ids: list[str] = []
            for partial in set((ROOT / "artifacts").glob(".*.partial")) - partials_before:
                try:
                    recovered_ids.append(
                        orchestrator.recover_interrupted_run(partial, reason=description).name
                    )
                except Exception as recovery_error:
                    description += f"; recovery failed: {recovery_error}"
            record.update(
                {
                    "status": "FAILED",
                    "error_type": "NativeWorkerFailure",
                    "error": description[:1000],
                    "run_id": recovered_ids[0] if len(recovered_ids) == 1 else None,
                    "completed_at": utc_now(),
                }
            )
        _save_progress(
            output,
            experiment_id=arguments.experiment_id,
            model_id=arguments.model_id,
            quick=arguments.quick,
            cases=results,
            total_case_count=len(cases),
        )
    # Always normalize the aggregate state. This also repairs a stale RUNNING marker
    # after an interrupted process when every recorded case has already settled.
    _save_progress(
        output,
        experiment_id=arguments.experiment_id,
        model_id=arguments.model_id,
        quick=arguments.quick,
        cases=results,
        total_case_count=len(cases),
    )
    print(json.dumps({"output": str(output), "completed": len(results), "total": len(cases)}))
    return 1 if any(row["status"] == "FAILED" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
