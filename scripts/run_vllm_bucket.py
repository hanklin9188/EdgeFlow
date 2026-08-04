#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.api.schemas import BenchmarkSubmission  # noqa: E402
from edgeflow.core.serialization import read_json  # noqa: E402
from edgeflow.experiments import BenchmarkConfig, RunOrchestrator  # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the registered vLLM exact-token bucket")
    parser.add_argument("--model-id", default="llama-3.2-3b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    arguments = parser.parse_args()
    if min(
        arguments.prompt_tokens,
        arguments.output_tokens,
        arguments.repetitions,
        arguments.warmup,
    ) < 1:
        parser.error("token counts, repetitions, and warmup must be positive")
    if arguments.repetitions >= 30 and not _git_clean():
        print("Formal vLLM bucket requires a clean git checkout.", file=sys.stderr)
        return 2
    registry = ModelRegistry(ROOT / "specs" / "model_registry.yaml")
    submission = BenchmarkSubmission(
        label="e08-vllm-llama32-p1024-o128",
        model_id=arguments.model_id,
        model_format="safetensors",
        backend="vllm",
        prompt_tokens=arguments.prompt_tokens,
        output_tokens=arguments.output_tokens,
        batch_size=1,
        concurrency=1,
        session_requests=20,
        quality_profile="balanced",
        dtype="bf16",
        external_base_url=arguments.base_url,
        repetitions=arguments.repetitions,
        warmup_requests=arguments.warmup,
        experiment_id="E08",
    )
    model_ref, revision, plan = submission.resolve(registry)
    quality = find_compatible_quality_report(
        artifact_root=ROOT / "artifacts",
        model_id=arguments.model_id,
        model_revision=revision,
        plan=plan,
    )
    if quality is None:
        print("Formal vLLM bucket requires its runtime-specific quality report.", file=sys.stderr)
        return 2
    run_dir = RunOrchestrator(root=ROOT, artifact_root=ROOT / "artifacts").run(
        model_ref=model_ref,
        workload=submission.workload(),
        plan=plan,
        config=BenchmarkConfig(
            experiment_id="E08",
            repetitions=arguments.repetitions,
            warmup_requests=arguments.warmup,
            local_files_only=True,
            enforce_idle=True,
        ),
        command=[
            sys.executable,
            "scripts/run_vllm_bucket.py",
            "--model-id",
            arguments.model_id,
            "--prompt-tokens",
            str(arguments.prompt_tokens),
            "--output-tokens",
            str(arguments.output_tokens),
            "--repetitions",
            str(arguments.repetitions),
            "--warmup",
            str(arguments.warmup),
        ],
    )
    verdict = read_json(run_dir / "validation_verdict.json")
    print(
        json.dumps(
            {
                "run_id": run_dir.name,
                "verdict": verdict["verdict"],
                "policy_eligible": verdict["policy_eligible"],
            }
        )
    )
    return 0 if verdict["policy_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
