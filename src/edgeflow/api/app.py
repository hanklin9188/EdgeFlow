from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from edgeflow import __version__
from edgeflow.core.models import CapabilityReport, WorkloadSpec
from edgeflow.core.serialization import project_root
from edgeflow.hardware import inspect_hardware
from edgeflow.metrics.statistics import paired_bootstrap
from edgeflow.optimizer import build_candidates
from edgeflow.runtimes import LlamaCppAdapter, PytorchAdapter, VllmAdapter
from edgeflow.storage import EdgeFlowDB

PROJECT_ROOT = project_root()
ARTIFACT_ROOT = Path(os.environ.get("EDGEFLOW_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts")).resolve()
DB = EdgeFlowDB(ARTIFACT_ROOT / "runs.sqlite")

app = FastAPI(
    title="EdgeFlow API",
    version=__version__,
    description="Read-first API over validated EdgeFlow artifacts and bounded planning operations.",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "artifact_root": str(ARTIFACT_ROOT)}


@app.post("/api/v1/inspect")
def inspect_endpoint() -> dict[str, Any]:
    return inspect_hardware(PROJECT_ROOT)


@app.get("/api/v1/runs")
def list_runs(eligible_only: bool = False, limit: int = 100) -> list[dict[str, Any]]:
    return DB.list_runs(eligible_only=eligible_only, limit=min(max(limit, 1), 1000))


@app.get("/api/v1/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    result = DB.get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    artifact = Path(result["artifact_path"])
    validation_path = artifact / "validation_verdict.json"
    if validation_path.exists():
        import json

        result["validation"] = json.loads(validation_path.read_text(encoding="utf-8"))
    return result


@app.get("/api/v1/runs/{run_id}/artifacts")
def list_run_artifacts(run_id: str) -> list[dict[str, Any]]:
    result = DB.get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    root = Path(result["artifact_path"]).resolve()
    return [
        {"name": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"stdout.log", "stderr.log"}
    ]


@app.get("/api/v1/runs/{run_id}/artifacts/{artifact_path:path}")
def get_run_artifact(run_id: str, artifact_path: str) -> FileResponse:
    result = DB.get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    root = Path(result["artifact_path"]).resolve()
    target = (root / artifact_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="artifact path escapes run directory") from exc
    if not target.is_file() or target.name in {"stdout.log", "stderr.log"}:
        raise HTTPException(status_code=404, detail="artifact not found or not publicly readable")
    return FileResponse(target)


@app.post("/api/v1/tune")
def tune(workload: WorkloadSpec, parameter_count: int = 3_000_000_000) -> dict[str, Any]:
    fingerprint = inspect_hardware(PROJECT_ROOT)
    capabilities: list[CapabilityReport] = [
        PytorchAdapter(compiled=False).probe(), PytorchAdapter(compiled=True).probe(),
        LlamaCppAdapter().probe(), VllmAdapter().probe(),
    ]
    prompt = workload.prompt_tokens if isinstance(workload.prompt_tokens, int) else max(item.tokens for item in workload.prompt_tokens)
    result = build_candidates(
        model_id=workload.model_id,
        capabilities=capabilities,
        vram_bytes=fingerprint["gpu"]["vram_bytes"],
        parameter_count=parameter_count,
        prompt_tokens=prompt,
        concurrency=workload.concurrency,
    )
    result["source_type"] = "estimated"
    result["policy_eligible"] = False
    return result


@app.post("/api/v1/compare")
def compare(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("protocol_a") != payload.get("protocol_b"):
        raise HTTPException(status_code=422, detail="incompatible timing protocols")
    if payload.get("source_type_a") == "demo" or payload.get("source_type_b") == "demo":
        raise HTTPException(status_code=422, detail="demo rows cannot support measured comparisons")
    try:
        return paired_bootstrap(payload["baseline"], payload["intervention"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/policies")
def list_policies() -> list[dict[str, Any]]:
    return DB.list_policies()


@app.get("/api/v1/policies/{policy_id}")
def get_policy(policy_id: str) -> dict[str, Any]:
    result = DB.get_policy(policy_id)
    if result is None:
        raise HTTPException(status_code=404, detail="policy not found")
    return result


@app.get("/api/v1/evidence/{evidence_id}")
def get_evidence(evidence_id: str) -> dict[str, Any]:
    result = DB.get_evidence_chain(evidence_id)
    if result is None:
        raise HTTPException(status_code=404, detail="evidence not found")
    return result


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    runs = DB.list_runs(limit=1000)
    validated = sum(item["validation"] is not None for item in runs)
    eligible = sum(bool((item["validation"] or {}).get("policy_eligible")) for item in runs)
    return (
        "# HELP edgeflow_runs_total Indexed EdgeFlow runs\n"
        "# TYPE edgeflow_runs_total gauge\n"
        f"edgeflow_runs_total {len(runs)}\n"
        f"edgeflow_validated_runs_total {validated}\n"
        f"edgeflow_policy_eligible_runs_total {eligible}\n"
    )


DASHBOARD = PROJECT_ROOT / "dashboard"
if DASHBOARD.exists():
    app.mount("/dashboard/assets", StaticFiles(directory=DASHBOARD), name="dashboard-assets")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(DASHBOARD / "index.html")
