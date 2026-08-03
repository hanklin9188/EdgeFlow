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

from edgeflow.api.schemas import BenchmarkSubmission  # noqa: E402
from edgeflow.core.models import utc_now  # noqa: E402
from edgeflow.core.serialization import read_json, write_json  # noqa: E402
from edgeflow.experiments import BenchmarkConfig, RunOrchestrator  # noqa: E402
from edgeflow.experiments.matrix import matrix_progress_status, pytorch_matrix_cases  # noqa: E402
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
    parser.add_argument("--max-new-cases", type=int)
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
    previous: dict[str, Any] = read_json(output) if output.is_file() else {"cases": []}
    completed: dict[str, dict[str, Any]] = (
        {}
        if arguments.rerun_completed
        else {
            row["case_id"]: row
            for row in previous.get("cases", [])
            if row.get("status") == "COMPLETED"
        }
    )
    orchestrator = RunOrchestrator(root=ROOT, artifact_root=ROOT / "artifacts")
    results = list(completed.values())
    new_cases = 0
    for case in cases:
        if case["case_id"] in completed:
            continue
        if arguments.max_new_cases is not None and new_cases >= arguments.max_new_cases:
            break
        submission = BenchmarkSubmission(
            label=case["case_id"].lower(),
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
            print(
                f"No formal exact-scope quality report for {arguments.model_id}/{plan.dtype}.",
                file=sys.stderr,
            )
            return 2
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
                    "python",
                    "scripts/run_pytorch_matrix.py",
                    arguments.experiment_id,
                    "--model-id",
                    arguments.model_id,
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
