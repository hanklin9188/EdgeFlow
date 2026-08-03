from __future__ import annotations

import argparse
import os
from pathlib import Path

from edgeflow.api.schemas import BenchmarkSubmission
from edgeflow.core.serialization import project_root, read_json, write_json
from edgeflow.experiments import BenchmarkConfig, RunOrchestrator
from edgeflow.models import ModelRegistry


def execute(spec_path: Path, result_path: Path) -> None:
    root = project_root()
    artifact_root = Path(os.environ.get("EDGEFLOW_ARTIFACT_ROOT", root / "artifacts")).resolve()
    submission = BenchmarkSubmission.model_validate(read_json(spec_path))
    registry = ModelRegistry(root / "specs" / "model_registry.yaml")
    workload = submission.workload()
    model_ref, revision, plan = submission.resolve(registry)
    command = [
        "edgeflow",
        "benchmark",
        "run",
        "--registered-model",
        submission.model_id,
        "--revision",
        revision,
        "--backend",
        submission.backend,
        "--experiment-id",
        submission.experiment_id,
    ]
    orchestrator = RunOrchestrator(root=root, artifact_root=artifact_root)
    try:
        run_dir = orchestrator.run(
            model_ref=model_ref,
            workload=workload,
            plan=plan,
            config=BenchmarkConfig(
                experiment_id=submission.experiment_id,
                repetitions=submission.repetitions,
                warmup_requests=submission.warmup_requests,
                local_files_only=not submission.allow_download,
                enforce_idle=not submission.allow_busy_gpu,
            ),
            command=command,
        )
        verdict = read_json(run_dir / "validation_verdict.json")
        write_json(
            result_path,
            {
                "run_id": verdict["run_id"],
                "artifact_name": run_dir.name,
                "verdict": verdict["verdict"],
                "policy_eligible": verdict["policy_eligible"],
                "public_claim_eligible": verdict["public_claim_eligible"],
            },
        )
    except Exception as exc:
        write_json(
            result_path,
            {
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "policy_eligible": False,
                "public_claim_eligible": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeFlow isolated local benchmark worker")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    arguments = parser.parse_args()
    execute(arguments.spec.resolve(), arguments.result.resolve())


if __name__ == "__main__":
    main()
