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
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, check=False, text=True
    )
    return result.returncode == 0 and not result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one registered external-runtime bucket")
    parser.add_argument("--backend", choices=["llama_cpp", "vllm"], required=True)
    parser.add_argument("--model-id", default="ministral3-3b-instruct-2512")
    parser.add_argument("--base-url")
    parser.add_argument("--served-model")
    parser.add_argument("--server-profile", required=True)
    parser.add_argument("--execution-mode", choices=["eager", "graph"], default="eager")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--experiment-id", default=None)
    arguments = parser.parse_args()
    if (
        min(
            arguments.prompt_tokens,
            arguments.output_tokens,
            arguments.concurrency,
            arguments.repetitions,
            arguments.warmup,
        )
        < 1
    ):
        parser.error("token counts, concurrency, repetitions, and warmup must be positive")
    if arguments.repetitions >= 30 and not _git_clean():
        print("Formal external-runtime buckets require a clean git checkout.", file=sys.stderr)
        return 2

    backend = arguments.backend
    model_format = "gguf" if backend == "llama_cpp" else "safetensors"
    base_url = arguments.base_url or (
        "http://127.0.0.1:8001" if backend == "llama_cpp" else "http://127.0.0.1:8002"
    )
    experiment_id = arguments.experiment_id or ("E07" if backend == "llama_cpp" else "E08")
    label = (
        f"{experiment_id.lower()}-{backend.replace('_', '-')}-"
        f"p{arguments.prompt_tokens}-o{arguments.output_tokens}-c{arguments.concurrency}"
    )
    registry = ModelRegistry(ROOT / "specs" / "model_registry.yaml")
    submission = BenchmarkSubmission(
        label=label,
        model_id=arguments.model_id,
        model_format=model_format,
        backend=backend,
        prompt_tokens=arguments.prompt_tokens,
        output_tokens=arguments.output_tokens,
        batch_size=1,
        concurrency=arguments.concurrency,
        session_requests=20,
        quality_profile="balanced",
        dtype="bf16",
        quantization=arguments.quantization if backend == "llama_cpp" else None,
        external_base_url=base_url,
        repetitions=arguments.repetitions,
        warmup_requests=arguments.warmup,
        experiment_id=experiment_id,
    )
    model_ref, revision, plan = submission.resolve(registry)
    backend_args = {
        **plan.backend_args,
        "server_profile": arguments.server_profile,
        "exact_token_prompts": True,
        "execution_mode": arguments.execution_mode,
        "enforce_eager": arguments.execution_mode == "eager",
    }
    if arguments.served_model:
        backend_args["served_model_name"] = arguments.served_model
    update: dict[str, object] = {"backend_args": backend_args}
    if backend == "llama_cpp":
        backend_args.update(
            {
                "enable_prompt_caching": False,
                "slot_prompt_similarity": 0.0,
            }
        )
        update.update({"dtype": None, "flash_attention": True})
    else:
        if arguments.model_id == "ministral3-3b-instruct-2512":
            backend_args.update(
                {
                    "language_model_only": True,
                    "skip_mm_profiling": True,
                    "mm_processor_cache_gb": 0,
                }
            )
        update.update(
            {
                "cuda_graph": arguments.execution_mode == "graph",
                "max_num_batched_tokens": arguments.max_num_batched_tokens,
                "max_num_seqs": arguments.max_num_seqs,
            }
        )
    plan = plan.model_copy(update=update).with_hash()
    quality = find_compatible_quality_report(
        artifact_root=ROOT / "artifacts",
        model_id=arguments.model_id,
        model_revision=revision,
        plan=plan,
    )
    if quality is None:
        print(
            "Formal bucket requires an exact runtime/format/quantization quality report.",
            file=sys.stderr,
        )
        return 2
    command = [
        sys.executable,
        "scripts/run_external_bucket.py",
        "--backend",
        backend,
        "--model-id",
        arguments.model_id,
        "--server-profile",
        arguments.server_profile,
        "--execution-mode",
        arguments.execution_mode,
        "--prompt-tokens",
        str(arguments.prompt_tokens),
        "--output-tokens",
        str(arguments.output_tokens),
        "--concurrency",
        str(arguments.concurrency),
        "--repetitions",
        str(arguments.repetitions),
        "--warmup",
        str(arguments.warmup),
    ]
    run_dir = RunOrchestrator(root=ROOT, artifact_root=ROOT / "artifacts").run(
        model_ref=model_ref,
        workload=submission.workload(),
        plan=plan,
        config=BenchmarkConfig(
            experiment_id=experiment_id,
            repetitions=arguments.repetitions,
            warmup_requests=arguments.warmup,
            local_files_only=True,
            enforce_idle=True,
        ),
        command=command,
    )
    verdict = read_json(run_dir / "validation_verdict.json")
    print(
        json.dumps(
            {
                "run_id": run_dir.name,
                "verdict": verdict["verdict"],
                "policy_eligible": verdict["policy_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0 if verdict["policy_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
