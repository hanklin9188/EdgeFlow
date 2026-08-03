from __future__ import annotations

import json
import os
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from edgeflow import __version__
from edgeflow.api.schemas import BenchmarkSubmission
from edgeflow.core.models import CapabilityReport, WorkloadSpec
from edgeflow.core.serialization import project_root
from edgeflow.hardware import inspect_hardware
from edgeflow.local import LocalJobManager, LocalRuntimeServiceManager
from edgeflow.metrics.statistics import paired_bootstrap
from edgeflow.models import ModelRegistry
from edgeflow.optimizer import build_candidates
from edgeflow.runtimes import LlamaCppAdapter, PytorchAdapter, VllmAdapter
from edgeflow.storage import EdgeFlowDB

MAX_REQUEST_BYTES = 1024 * 1024
SAFE_ARTIFACT_NAMES = {
    "VALIDATION.md",
    "checksums.sha256",
    "execution_plan.json",
    "hardware_fingerprint.json",
    "metrics.jsonl",
    "monitor.jsonl",
    "run_manifest.json",
    "summary.json",
    "validation_verdict.json",
    "workload.json",
}
SAFE_ARTIFACT_SUFFIXES = {".ncu-rep", ".nsys-rep"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _capabilities() -> list[CapabilityReport]:
    return [
        PytorchAdapter(compiled=False).probe(),
        PytorchAdapter(compiled=True).probe(),
        LlamaCppAdapter().probe(),
        VllmAdapter().probe(),
    ]


def _safe_origin(origin: str) -> bool:
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in LOOPBACK_HOSTS


def create_app(
    *,
    root: Path | None = None,
    artifact_root: Path | None = None,
    job_manager: LocalJobManager | None = None,
    runtime_manager: LocalRuntimeServiceManager | None = None,
) -> FastAPI:
    project_path = (root or project_root()).resolve()
    artifacts = (
        artifact_root
        or Path(os.environ.get("EDGEFLOW_ARTIFACT_ROOT", project_path / "artifacts"))
    ).resolve()
    db = EdgeFlowDB(artifacts / "runs.sqlite")
    registry = ModelRegistry(project_path / "specs" / "model_registry.yaml")
    manager = job_manager or LocalJobManager(project_root=project_path, artifact_root=artifacts)
    service_manager = runtime_manager or LocalRuntimeServiceManager(
        project_root=project_path, artifact_root=artifacts
    )
    control_token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        yield
        service_manager.shutdown()

    application = FastAPI(
        title="EdgeFlow Local Control API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
        description=(
            "Local-first control plane for registered EdgeFlow experiments. "
            "GPU execution stays on this machine and every result passes the evidence gates."
        ),
    )
    application.state.project_root = project_path
    application.state.artifact_root = artifacts
    application.state.db = db
    application.state.job_manager = manager
    application.state.runtime_manager = service_manager
    application.state.control_token = control_token
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @application.middleware("http")
    async def local_security(request: Request, call_next: Callable[..., Any]) -> Any:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_BYTES:
                    return JSONResponse(status_code=413, content={"detail": "request body exceeds 1 MiB"})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "invalid Content-Length"})
        elif request.method not in {"GET", "HEAD", "OPTIONS"}:
            body = await request.body()
            if len(body) > MAX_REQUEST_BYTES:
                return JSONResponse(status_code=413, content={"detail": "request body exceeds 1 MiB"})
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and not _safe_origin(origin):
                return JSONResponse(status_code=403, content={"detail": "cross-origin control request rejected"})
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
            "form-action 'self'"
        )
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def require_control(request: Request) -> None:
        supplied = request.headers.get("x-edgeflow-token", "")
        if not supplied or not secrets.compare_digest(supplied, control_token):
            raise HTTPException(status_code=403, detail="valid local control token required")

    def registered_experiments() -> dict[str, Any]:
        matrix = yaml.safe_load(
            (project_path / "specs" / "experiment_matrix.yaml").read_text(encoding="utf-8")
        )
        return matrix

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "mode": "local-first",
            "control_scope": "loopback",
            "artifact_storage": "local",
            "managed_runtime": service_manager.status()["state"],
        }

    @application.get("/api/v1/session")
    def local_session() -> dict[str, Any]:
        return {
            "mode": "local-first",
            "control_token": control_token,
            "max_request_bytes": MAX_REQUEST_BYTES,
            "max_parallel_gpu_jobs": 1,
            "public_export": "read-only validated artifacts only",
        }

    @application.post("/api/v1/inspect")
    def inspect_endpoint() -> dict[str, Any]:
        return inspect_hardware(project_path)

    @application.get("/api/v1/runtime-capabilities")
    def runtime_capabilities() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in _capabilities():
            row = item.model_dump(mode="json")
            version = row.get("version")
            if isinstance(version, str) and version.startswith("isolated:"):
                row["version"] = "isolated local executable"
            features = row.get("features")
            if isinstance(features, dict) and features.get("isolated_executable"):
                features["isolated_executable"] = True
            rows.append(row)
        return rows

    @application.get("/api/v1/runtime-services")
    def runtime_services() -> list[dict[str, Any]]:
        return service_manager.services()

    def active_gpu_job() -> bool:
        return any(
            item.get("status") in {"QUEUED", "RUNNING", "CANCELLING"}
            for item in manager.list_jobs(limit=100)
        )

    @application.post("/api/v1/runtime-services/{backend}/start", status_code=202)
    def start_runtime_service(
        backend: str,
        _control: None = Depends(require_control),
    ) -> dict[str, Any]:
        if active_gpu_job():
            raise HTTPException(status_code=409, detail="wait for the active GPU job to finish")
        try:
            return service_manager.start(backend)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/api/v1/runtime-services/{backend}/stop")
    def stop_runtime_service(
        backend: str,
        _control: None = Depends(require_control),
    ) -> dict[str, Any]:
        if active_gpu_job():
            raise HTTPException(status_code=409, detail="cannot stop a runtime used by an active job")
        try:
            return service_manager.stop(backend)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/api/v1/models")
    def list_models() -> list[dict[str, Any]]:
        return registry.list()

    @application.get("/api/v1/experiments")
    def list_experiments() -> dict[str, Any]:
        matrix = registered_experiments()
        return {
            "schema_version": matrix["schema_version"],
            "strategy": matrix["strategy"],
            "experiments": matrix["experiments"],
        }

    @application.get("/api/v1/runs")
    def list_runs(eligible_only: bool = False, limit: int = 100) -> list[dict[str, Any]]:
        return db.list_runs(eligible_only=eligible_only, limit=min(max(limit, 1), 1000))

    @application.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        result = db.get_run(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="run not found")
        artifact = Path(result["artifact_path"])
        validation_path = artifact / "validation_verdict.json"
        return {
            "manifest": result["manifest"],
            "validation": (
                json.loads(validation_path.read_text(encoding="utf-8"))
                if validation_path.exists()
                else None
            ),
            "artifact_uri": f"/api/v1/runs/{run_id}/artifacts",
        }

    @application.get("/api/v1/runs/{run_id}/metrics")
    def get_run_metrics(run_id: str, limit: int = 2000) -> list[dict[str, Any]]:
        if db.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return db.list_metrics(run_id, limit=min(max(limit, 1), 5000))

    @application.get("/api/v1/runs/{run_id}/artifacts")
    def list_run_artifacts(run_id: str) -> list[dict[str, Any]]:
        result = db.get_run(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="run not found")
        run_root = Path(result["artifact_path"]).resolve()
        return [
            {
                "name": path.relative_to(run_root).as_posix(),
                "bytes": path.stat().st_size,
                "href": f"/api/v1/runs/{run_id}/artifacts/{path.relative_to(run_root).as_posix()}",
            }
            for path in sorted(run_root.rglob("*"))
            if path.is_file()
            and (path.name in SAFE_ARTIFACT_NAMES or path.suffix in SAFE_ARTIFACT_SUFFIXES)
        ]

    @application.get("/api/v1/runs/{run_id}/artifacts/{artifact_path:path}")
    def get_run_artifact(run_id: str, artifact_path: str) -> FileResponse:
        result = db.get_run(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="run not found")
        run_root = Path(result["artifact_path"]).resolve()
        target = (run_root / artifact_path).resolve()
        try:
            target.relative_to(run_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="artifact path escapes run directory") from exc
        if not target.is_file() or (
            target.name not in SAFE_ARTIFACT_NAMES and target.suffix not in SAFE_ARTIFACT_SUFFIXES
        ):
            raise HTTPException(status_code=404, detail="artifact is not on the readable allowlist")
        return FileResponse(target)

    @application.post("/api/v1/tune")
    def tune(workload: WorkloadSpec, parameter_count: int = 3_000_000_000) -> dict[str, Any]:
        fingerprint = inspect_hardware(project_path)
        capabilities = _capabilities()
        prompt = (
            workload.prompt_tokens
            if isinstance(workload.prompt_tokens, int)
            else max(item.tokens for item in workload.prompt_tokens)
        )
        result = build_candidates(
            model_id=workload.model_id,
            capabilities=capabilities,
            vram_bytes=int(fingerprint["gpu"].get("vram_bytes") or 0),
            parameter_count=parameter_count,
            prompt_tokens=prompt,
            concurrency=workload.concurrency,
        )
        result["source_type"] = "estimated"
        result["policy_eligible"] = False
        result["warning"] = "Screening candidates are not recommendations until G0-G8 pass."
        return result

    @application.post("/api/v1/compare")
    def compare(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("protocol_a") != payload.get("protocol_b"):
            raise HTTPException(status_code=422, detail="incompatible timing protocols")
        if payload.get("source_type_a") == "demo" or payload.get("source_type_b") == "demo":
            raise HTTPException(status_code=422, detail="demo rows cannot support measured comparisons")
        try:
            return paired_bootstrap(payload["baseline"], payload["intervention"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/v1/jobs")
    def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
        return manager.list_jobs(limit=min(max(limit, 1), 500))

    @application.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            result = manager.get_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="job not found")
        return result

    @application.post("/api/v1/jobs/benchmark", status_code=202)
    def submit_benchmark(
        submission: BenchmarkSubmission,
        _control: None = Depends(require_control),
    ) -> dict[str, Any]:
        try:
            _model_ref, _revision, resolved_plan = submission.resolve(registry)
            service = service_manager.status()
            if service["state"] in {"STARTING", "RUNNING", "STOPPING"}:
                if service["state"] != "RUNNING":
                    raise RuntimeError(
                        f"managed {service['backend']} runtime is {str(service['state']).lower()}"
                    )
                if submission.backend != service["backend"]:
                    raise RuntimeError(
                        f"stop managed {service['backend']} before running {submission.backend}"
                    )
                if resolved_plan.backend_args.get("base_url") != service["base_url"]:
                    raise RuntimeError(
                        f"managed {service['backend']} is available at {service['base_url']}"
                    )
            matrix = registered_experiments()
            experiment = matrix["experiments"].get(submission.experiment_id)
            if experiment is None:
                raise ValueError(f"experiment {submission.experiment_id} is not registered")
            allowed_backends = experiment.get("backends")
            if allowed_backends and submission.backend not in allowed_backends:
                raise ValueError(
                    f"{submission.experiment_id} does not permit backend {submission.backend}"
                )
            return manager.submit_benchmark(submission)
        except (KeyError, RuntimeError, ValueError) as exc:
            status_code = 409 if isinstance(exc, RuntimeError) else 422
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    @application.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(
        job_id: str,
        _control: None = Depends(require_control),
    ) -> dict[str, Any]:
        try:
            return manager.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @application.get("/api/v1/policies")
    def list_policies() -> list[dict[str, Any]]:
        return db.list_policies()

    @application.get("/api/v1/policies/{policy_id}")
    def get_policy(policy_id: str) -> dict[str, Any]:
        result = db.get_policy(policy_id)
        if result is None:
            raise HTTPException(status_code=404, detail="policy not found")
        return result

    @application.get("/api/v1/evidence")
    def list_evidence(limit: int = 200) -> list[dict[str, Any]]:
        return db.list_evidence(limit=min(max(limit, 1), 1000))

    @application.get("/api/v1/evidence/{evidence_id}")
    def get_evidence(evidence_id: str) -> dict[str, Any]:
        result = db.get_evidence_chain(evidence_id)
        if result is None:
            raise HTTPException(status_code=404, detail="evidence not found")
        return result

    @application.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        runs = db.list_runs(limit=1000)
        jobs = manager.list_jobs(limit=1000)
        validated = sum(item["validation"] is not None for item in runs)
        eligible = sum(bool((item["validation"] or {}).get("policy_eligible")) for item in runs)
        active = sum(item.get("status") in {"QUEUED", "RUNNING", "CANCELLING"} for item in jobs)
        runtime_active = int(service_manager.status()["state"] in {"STARTING", "RUNNING"})
        return (
            "# HELP edgeflow_runs_total Indexed EdgeFlow runs\n"
            "# TYPE edgeflow_runs_total gauge\n"
            f"edgeflow_runs_total {len(runs)}\n"
            f"edgeflow_validated_runs_total {validated}\n"
            f"edgeflow_policy_eligible_runs_total {eligible}\n"
            f"edgeflow_active_local_jobs {active}\n"
            f"edgeflow_managed_runtime_active {runtime_active}\n"
        )

    dashboard_root = project_path / "dashboard"
    if dashboard_root.exists():
        application.mount(
            "/dashboard/assets", StaticFiles(directory=dashboard_root), name="dashboard-assets"
        )

        @application.get("/", include_in_schema=False)
        def dashboard() -> FileResponse:
            return FileResponse(dashboard_root / "index.html")

    return application


PROJECT_ROOT = project_root()
ARTIFACT_ROOT = Path(
    os.environ.get("EDGEFLOW_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts")
).resolve()
app = create_app(root=PROJECT_ROOT, artifact_root=ARTIFACT_ROOT)
