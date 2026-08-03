from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from edgeflow import __version__
from edgeflow.core.models import ExecutionPlan, WorkloadSpec
from edgeflow.core.serialization import project_root, read_json, write_json
from edgeflow.experiments import BenchmarkConfig, RunOrchestrator
from edgeflow.hardware import inspect_hardware, run_doctor
from edgeflow.models import ModelRegistry
from edgeflow.optimizer import build_candidates
from edgeflow.policy import build_policy
from edgeflow.profiler import diagnose_profile
from edgeflow.quality import evaluate_quality
from edgeflow.reports import render_run_report
from edgeflow.runtimes import LlamaCppAdapter, PytorchAdapter, VllmAdapter
from edgeflow.storage import EdgeFlowDB
from edgeflow.validation import validate_run
from edgeflow.workloads import create_workload

app = typer.Typer(
    name="edgeflow",
    help="Evidence-backed, workload-conditioned local LLM inference autotuning.",
    no_args_is_help=True,
)
workload_app = typer.Typer(help="Create and inspect versioned workload specifications.")
tune_app = typer.Typer(help="Generate and screen capability-compatible execution plans.")
benchmark_app = typer.Typer(help="Run isolated, raw-artifact-first benchmarks.")
policy_app = typer.Typer(help="Build and inspect validated deployment policies.")
kernel_app = typer.Typer(help="Validate correctness-gated Triton kernels.")
experiment_app = typer.Typer(help="Plan or run catalog experiments.")
model_app = typer.Typer(help="Inspect pinned model/runtime support metadata.")
quality_app = typer.Typer(help="Evaluate quality constraints as hard gates.")
app.add_typer(workload_app, name="workload")
app.add_typer(tune_app, name="tune")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(policy_app, name="policy")
app.add_typer(kernel_app, name="kernel")
app.add_typer(experiment_app, name="experiment")
app.add_typer(model_app, name="model")
app.add_typer(quality_app, name="quality")


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True, help="Show the EdgeFlow version."),
    ] = None,
) -> None:
    pass


@app.command("inspect")
def inspect_command(
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Save the fingerprint JSON.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    fingerprint = inspect_hardware()
    if output:
        write_json(output, fingerprint)
    if json_output:
        _print_json(fingerprint)
        return
    gpu = fingerprint["gpu"]
    software = fingerprint["software"]
    typer.echo(f"GPU       {gpu['name']} · {gpu['vram_bytes'] / 1024**3:.1f} GB · SM {gpu['compute_capability']}")
    typer.echo(f"Runtime   CUDA {software['cuda_runtime']} · PyTorch {software['pytorch']} · Triton {software['triton']}")
    typer.echo(f"Host      {fingerprint['host']['execution_mode']} · Python {software['python']}")
    typer.echo(f"Identity  {fingerprint['fingerprint_id']} · {fingerprint['sha256'][:16]}")


@app.command("doctor")
def doctor_command(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    strict_optional: Annotated[bool, typer.Option(help="Treat missing optional backends/profilers as failure.")] = False,
) -> None:
    report = run_doctor()
    if json_output:
        _print_json(report.as_dict())
    else:
        for check in report.checks:
            typer.echo(f"[{check.status:4}] {check.name:18} {check.message}")
        typer.echo(f"\nCore status: {'READY' if report.ready else 'NOT READY'}")
    optional_failed = any(check.status != "PASS" for check in report.checks if not check.required)
    if not report.ready or (strict_optional and optional_failed):
        raise typer.Exit(code=1)


@workload_app.command("create")
def workload_create(
    model: Annotated[str, typer.Option("--model")],
    profile: Annotated[str, typer.Option("--profile")] = "local-agent",
    prompt_distribution: Annotated[str, typer.Option("--prompt-distribution")] = "1024",
    output_tokens: Annotated[int, typer.Option("--output", min=1)] = 128,
    concurrency: Annotated[int, typer.Option(min=1)] = 1,
    batch_size: Annotated[int, typer.Option(min=1)] = 1,
    session_requests: Annotated[int, typer.Option(min=1)] = 20,
    quality_profile: Annotated[str, typer.Option()] = "balanced",
    seed: Annotated[int, typer.Option(min=0)] = 42,
    destination: Annotated[Path | None, typer.Option("--save", "-o")] = None,
) -> None:
    workload_id = f"{profile}-p{prompt_distribution.replace(':', '-').replace(',', '_')}-o{output_tokens}-c{concurrency}"
    workload = create_workload(
        workload_id=workload_id,
        model_id=model,
        prompt_distribution=prompt_distribution,
        output_tokens=output_tokens,
        concurrency=concurrency,
        batch_size=batch_size,
        session_requests=session_requests,
        quality_profile=quality_profile,
        seed=seed,
    )
    payload = workload.model_dump(mode="json")
    if destination:
        write_json(destination, payload)
        typer.echo(str(destination.resolve()))
    else:
        _print_json(payload)


@tune_app.command("screen")
def tune_screen(
    workload_path: Annotated[Path, typer.Option("--workload", exists=True, dir_okay=False)],
    parameter_count: Annotated[int, typer.Option(min=1)] = 3_000_000_000,
    destination: Annotated[Path | None, typer.Option("--save", "-o")] = None,
) -> None:
    workload = WorkloadSpec.model_validate(read_json(workload_path))
    fingerprint = inspect_hardware()
    capabilities = [
        PytorchAdapter(compiled=False).probe(),
        PytorchAdapter(compiled=True).probe(),
        LlamaCppAdapter().probe(),
        VllmAdapter().probe(),
    ]
    prompt_tokens = workload.prompt_tokens if isinstance(workload.prompt_tokens, int) else max(item.tokens for item in workload.prompt_tokens)
    result = build_candidates(
        model_id=workload.model_id,
        capabilities=capabilities,
        vram_bytes=int(fingerprint["gpu"]["vram_bytes"]),
        parameter_count=parameter_count,
        prompt_tokens=prompt_tokens,
        concurrency=workload.concurrency,
    )
    result["capabilities"] = [item.model_dump(mode="json") for item in capabilities]
    result["warning"] = "Candidates are unmeasured until benchmark + validation; this is not a recommendation."
    if destination:
        write_json(destination, result)
        typer.echo(str(destination.resolve()))
    else:
        _print_json(result)


@benchmark_app.command("run")
def benchmark_run(
    model_ref: Annotated[str, typer.Option("--model-ref", help="Pinned HF ID or local snapshot path.")],
    workload_path: Annotated[Path, typer.Option("--workload", exists=True, dir_okay=False)],
    plan_path: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    repetitions: Annotated[int, typer.Option(min=1)] = 30,
    warmup: Annotated[int, typer.Option(min=1)] = 5,
    experiment_id: Annotated[str, typer.Option()] = "E04",
    allow_download: Annotated[bool, typer.Option(help="Allow model/tokenizer network access.")] = False,
    allow_busy_gpu: Annotated[bool, typer.Option(help="Disable the <5% pre-run utilization gate.")] = False,
    artifact_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    workload = WorkloadSpec.model_validate(read_json(workload_path))
    plan = ExecutionPlan.model_validate(read_json(plan_path))
    orchestrator = RunOrchestrator(artifact_root=artifact_root)
    command = [
        "edgeflow", "benchmark", "run", "--model-ref", model_ref,
        "--workload", str(workload_path), "--plan", str(plan_path),
        "--repetitions", str(repetitions), "--warmup", str(warmup),
        "--experiment-id", experiment_id,
    ]
    result = orchestrator.run(
        model_ref=model_ref,
        workload=workload,
        plan=plan,
        config=BenchmarkConfig(
            experiment_id=experiment_id,
            repetitions=repetitions,
            warmup_requests=warmup,
            local_files_only=not allow_download,
            enforce_idle=not allow_busy_gpu,
        ),
        command=command,
    )
    verdict = read_json(result / "validation_verdict.json")
    typer.echo(f"Artifact: {result}")
    typer.echo(f"Verdict: {verdict['verdict']} · policy eligible={verdict['policy_eligible']}")


@app.command("validate")
def validate_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    verdict = validate_run(run_dir)
    if json_output:
        _print_json(verdict)
    else:
        typer.echo(f"{verdict['verdict']} · policy eligible={verdict['policy_eligible']}")
        for issue in verdict["issues"]:
            typer.echo(f"- {issue['code']}: {issue['message']}")
    if verdict["verdict"] in {"FAIL", "INVALID"}:
        raise typer.Exit(code=1)


@app.command("report")
def report_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    destination: Annotated[Path | None, typer.Option("--save", "-o")] = None,
) -> None:
    report = render_run_report(run_dir)
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
        typer.echo(str(destination.resolve()))
    else:
        typer.echo(report)


@app.command("diagnose")
def diagnose_command(
    profile: Annotated[Path, typer.Option("--profile", exists=True, dir_okay=False)],
    destination: Annotated[Path | None, typer.Option("--save", "-o")] = None,
) -> None:
    result = diagnose_profile(read_json(profile))
    if destination:
        write_json(destination, result)
        typer.echo(str(destination.resolve()))
    else:
        _print_json(result)


@app.command("profile")
def profile_command(
    run_id: Annotated[str, typer.Option("--run")],
    level: Annotated[str, typer.Option()] = "nsys",
    artifact_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    root = (artifact_root or project_root() / "artifacts").resolve()
    db = EdgeFlowDB(root / "runs.sqlite")
    run = db.get_run(run_id)
    if run is None:
        raise typer.BadParameter(f"unknown run: {run_id}")
    command = run["manifest"]["command"]
    if level == "nsys":
        profile_command_line = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--output", f"artifacts/{run_id}/trace/{run_id}", *command]
    elif level == "ncu":
        profile_command_line = ["ncu", "--set", "basic", "--target-processes", "all", *command]
    elif level == "torch":
        profile_command_line = [*command, "--profiler-level", "torch"]
    else:
        raise typer.BadParameter("level must be torch, nsys, or ncu")
    _print_json(
        {
            "run_id": run_id,
            "level": level,
            "status": "PLANNED",
            "command": profile_command_line,
            "warning": "Profiled latency is diagnostic and must never replace unprofiled production timing.",
        }
    )


@policy_app.command("build")
def policy_build(
    results: Annotated[Path, typer.Option("--results", exists=True, dir_okay=False)],
    hardware_sha256: Annotated[str, typer.Option()],
    model_id: Annotated[str, typer.Option()],
    objective: Annotated[str, typer.Option()] = "session",
    quality_profile: Annotated[str, typer.Option()] = "balanced",
    destination: Annotated[Path, typer.Option("--save", "-o")] = Path("policy.json"),
) -> None:
    payload = json.loads(results.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    policy = build_policy(
        rows,
        hardware_sha256=hardware_sha256,
        model_id=model_id,
        objective=objective,
        quality_profile=quality_profile,
        holdout_run_ids=payload.get("holdout_run_ids", []) if isinstance(payload, dict) else [],
    )
    write_json(destination, policy)
    typer.echo(str(destination.resolve()))


@policy_app.command("show")
def policy_show(policy_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    policy = read_json(policy_path)
    typer.echo(f"Policy {policy['policy_id']} · model {policy['model_id']}")
    for rule in sorted(policy["rules"], key=lambda item: item["priority"]):
        typer.echo(f"[{rule['priority']}] {rule['predicate']} -> {rule['plan_id']} · evidence={rule['evidence_ids']}")
    typer.echo(f"Fallback: {policy['fallback_plan_id']}")
    typer.echo(f"Holdout: {policy['holdout_validation'].get('status')}")


@kernel_app.command("validate")
def kernel_validate(
    full: Annotated[bool, typer.Option(help="Run the full registered shape/dtype sweep.")] = False,
    destination: Annotated[Path, typer.Option("--save", "-o")] = Path("artifacts/kernel/rmsnorm-correctness.json"),
) -> None:
    import torch

    from edgeflow.kernels.rmsnorm.dispatch import validate_shape

    if not torch.cuda.is_available():
        raise typer.BadParameter("CUDA is required")
    rows = [1, 2, 4, 8, 16, 32, 128, 512] if full else [1, 8]
    hidden_sizes = [768, 1000, 1024, 1536, 2048, 3072, 3073, 4095, 4096] if full else [768, 1000, 4096]
    dtypes = [torch.float32, torch.float16, torch.bfloat16] if full else [torch.float32, torch.float16]
    results: list[dict[str, Any]] = []
    for dtype in dtypes:
        for row in rows:
            for hidden in hidden_sizes:
                generator = torch.Generator(device="cuda").manual_seed(42 + row + hidden)
                x = torch.randn((row, hidden), device="cuda", dtype=dtype, generator=generator)
                residual = torch.randn((row, hidden), device="cuda", dtype=dtype, generator=generator)
                weight = torch.randn((hidden,), device="cuda", dtype=dtype, generator=generator)
                results.append(validate_shape(x, residual, weight))
    payload = {
        "kernel": "fused_residual_rmsnorm",
        "source_type": "measured",
        "results": results,
        "pass": all(item["status"] == "PASS" for item in results),
    }
    write_json(destination, payload)
    typer.echo(f"{sum(item['status'] == 'PASS' for item in results)}/{len(results)} passed · {destination}")
    if not payload["pass"]:
        raise typer.Exit(code=1)


@experiment_app.command("plan")
def experiment_plan(
    experiment_id: Annotated[str, typer.Argument()],
    destination: Annotated[Path, typer.Option("--save", "-o")] = Path("planned_matrix.json"),
) -> None:
    import yaml

    matrix = yaml.safe_load((project_root() / "specs" / "experiment_matrix.yaml").read_text(encoding="utf-8"))
    experiment = matrix.get("experiments", {}).get(experiment_id)
    if experiment is None:
        raise typer.BadParameter(f"experiment {experiment_id} is not registered")
    output = {
        "experiment_id": experiment_id,
        "protocol_version": matrix["schema_version"],
        "status": "PLANNED",
        "definition": experiment,
        "strategy": matrix["strategy"],
        "note": "Expand only after runtime capability and model-license prechecks.",
    }
    write_json(destination, output)
    typer.echo(str(destination.resolve()))


@model_app.command("list")
def model_list() -> None:
    registry = ModelRegistry()
    _print_json(registry.list())


@model_app.command("resolve")
def model_resolve(
    model_id: Annotated[str, typer.Argument()],
    model_format: Annotated[str, typer.Option()] = "safetensors",
) -> None:
    repository, revision = ModelRegistry().resolve_source(model_id, model_format)
    _print_json({"model_id": model_id, "format": model_format, "repository": repository, "revision": revision})


@quality_app.command("evaluate")
def quality_evaluate(
    reference_path: Annotated[Path, typer.Option("--reference", exists=True, dir_okay=False)],
    candidate_path: Annotated[Path, typer.Option("--candidate", exists=True, dir_okay=False)],
    profile: Annotated[str, typer.Option()] = "balanced",
    destination: Annotated[Path, typer.Option("--save", "-o")] = Path("quality.json"),
    protocol_match: Annotated[bool, typer.Option("--protocol-match/--protocol-mismatch")] = True,
) -> None:
    result = evaluate_quality(
        reference=read_json(reference_path),
        candidate=read_json(candidate_path),
        profile=profile,
        protocol_match=protocol_match,
    )
    write_json(destination, result)
    typer.echo(f"{'PASS' if result['pass'] else 'FAIL'} · {destination.resolve()}")
    if not result["pass"]:
        raise typer.Exit(code=1)


@app.command("serve")
def serve_command(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8787,
    artifact_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    import os

    import uvicorn

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise typer.BadParameter(
            "EdgeFlow is local-first: --host must be 127.0.0.1, localhost, or ::1"
        )
    if artifact_root:
        os.environ["EDGEFLOW_ARTIFACT_ROOT"] = str(artifact_root.resolve())
    typer.echo(f"EdgeFlow local console: http://{host}:{port}")
    typer.echo("GPU jobs and artifacts remain on this machine; press Ctrl+C to stop.")
    uvicorn.run("edgeflow.api.app:app", host=host, port=port, reload=False)
