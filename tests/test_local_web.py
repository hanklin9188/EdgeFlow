from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from edgeflow.api.app import create_app
from edgeflow.api.schemas import BenchmarkSubmission
from edgeflow.core.serialization import write_json
from edgeflow.local import LocalJobManager, LocalRuntimeServiceManager


class FakeJobManager:
    def __init__(self) -> None:
        self.submissions: list[BenchmarkSubmission] = []

    def list_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return []

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return None

    def submit_benchmark(self, submission: BenchmarkSubmission) -> dict[str, Any]:
        self.submissions.append(submission)
        return {"job_id": "job-20260804-120000-abcdef", "status": "QUEUED"}

    def cancel(self, job_id: str) -> dict[str, Any]:
        raise KeyError(job_id)


class FakeRuntimeManager:
    def __init__(self) -> None:
        self.current = {
            "state": "STOPPED",
            "backend": None,
            "base_url": None,
            "message": "stopped",
        }

    def status(self) -> dict[str, Any]:
        return dict(self.current)

    def services(self) -> list[dict[str, Any]]:
        return [
            {
                "backend": backend,
                "state": self.current["state"] if self.current["backend"] == backend else "STOPPED",
                "base_url": f"http://127.0.0.1:{port}",
                "model_id": (
                    "ministral3-3b-instruct-2512"
                    if backend == "llama_cpp"
                    else "smollm2-360m-instruct"
                ),
                "installed": True,
                "message": "test",
            }
            for backend, port in (("llama_cpp", 8001), ("vllm", 8002))
        ]

    def start(self, backend: str) -> dict[str, Any]:
        if backend not in {"llama_cpp", "vllm"}:
            raise ValueError("not allowlisted")
        port = 8001 if backend == "llama_cpp" else 8002
        self.current = {
            "state": "RUNNING",
            "backend": backend,
            "base_url": f"http://127.0.0.1:{port}",
            "model_id": (
                "ministral3-3b-instruct-2512" if backend == "llama_cpp" else "smollm2-360m-instruct"
            ),
            "message": "ready",
        }
        return self.status()

    def stop(self, backend: str | None = None) -> dict[str, Any]:
        self.current = {
            "state": "STOPPED",
            "backend": None,
            "base_url": None,
            "message": "stopped",
        }
        return self.status()

    def shutdown(self) -> None:
        self.stop()


def valid_submission() -> dict[str, Any]:
    return {
        "model_id": "smollm2-360m-instruct",
        "backend": "pytorch_eager",
        "model_format": "safetensors",
        "experiment_id": "E04",
    }


def test_benchmark_submission_rejects_unsafe_or_invalid_runtime_input() -> None:
    with pytest.raises(ValidationError):
        BenchmarkSubmission.model_validate({**valid_submission(), "command": "rm -rf /"})
    with pytest.raises(ValidationError):
        BenchmarkSubmission.model_validate(
            {
                **valid_submission(),
                "backend": "llama_cpp",
                "model_format": "gguf",
                "external_base_url": "http://example.com:8000",
            }
        )
    with pytest.raises(ValidationError):
        BenchmarkSubmission.model_validate(
            {
                **valid_submission(),
                "backend": "torch_compile",
                "compile_mode": "max-autotune",
                "experiment_id": "E05",
            }
        )
    with pytest.raises(ValidationError):
        BenchmarkSubmission.model_validate(
            {
                **valid_submission(),
                "backend": "torch_compile",
                "compile_mode": "reduce-overhead",
                "experiment_id": "E05",
            }
        )
    with pytest.raises(ValidationError):
        BenchmarkSubmission.model_validate({**valid_submission(), "concurrency": 2})
    with pytest.raises(ValidationError):
        BenchmarkSubmission.model_validate(
            {
                **valid_submission(),
                "backend": "llama_cpp",
                "model_format": "gguf",
                "experiment_id": "E07",
                "batch_size": 2,
            }
        )


def test_local_control_requires_token_and_rejects_cross_origin(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manager = FakeJobManager()
    runtimes = FakeRuntimeManager()
    application = create_app(
        root=root,
        artifact_root=tmp_path / "artifacts",
        job_manager=manager,
        runtime_manager=runtimes,  # type: ignore[arg-type]
    )
    client = TestClient(application)

    session = client.get("/api/v1/session").json()
    assert session["mode"] == "local-first"
    assert session["max_parallel_gpu_jobs"] == 1

    assert client.post("/api/v1/jobs/benchmark", json=valid_submission()).status_code == 403
    response = client.post(
        "/api/v1/jobs/benchmark",
        json=valid_submission(),
        headers={"X-EdgeFlow-Token": session["control_token"]},
    )
    assert response.status_code == 202
    assert manager.submissions[0].model_id == "smollm2-360m-instruct"

    assert client.post("/api/v1/runtime-services/llama_cpp/start").status_code == 403
    started = client.post(
        "/api/v1/runtime-services/llama_cpp/start",
        headers={"X-EdgeFlow-Token": session["control_token"]},
    )
    assert started.status_code == 202
    assert started.json()["base_url"] == "http://127.0.0.1:8001"
    blocked = client.post(
        "/api/v1/jobs/benchmark",
        json=valid_submission(),
        headers={"X-EdgeFlow-Token": session["control_token"]},
    )
    assert blocked.status_code == 409

    rejected = client.post(
        "/api/v1/inspect",
        headers={"Origin": "https://attacker.invalid"},
    )
    assert rejected.status_code == 403


def test_local_control_security_headers_host_and_size_limit(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    application = create_app(
        root=root,
        artifact_root=tmp_path / "artifacts",
        job_manager=FakeJobManager(),
    )
    client = TestClient(application)
    response = client.get("/health")
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert client.get("/health", headers={"Host": "remote.example"}).status_code == 400
    assert (
        client.post(
            "/api/v1/inspect",
            content=b"{}",
            headers={"Content-Length": str(1024 * 1024 + 1)},
        ).status_code
        == 413
    )
    chunked = client.post(
        "/api/v1/inspect",
        content=iter([b"x" * (1024 * 1024), b"x"]),
    )
    assert chunked.status_code == 413
    capabilities = client.get("/api/v1/runtime-capabilities")
    assert capabilities.status_code == 200
    assert str(root) not in capabilities.text
    for capability in capabilities.json():
        executable = capability.get("features", {}).get("isolated_executable")
        assert executable in {True, False, None}
    progress = client.get("/api/v1/experiment-progress")
    assert progress.status_code == 200
    assert progress.json()["total"] == 31
    assert progress.json()["external_required"] == 2
    readiness = client.get("/api/v1/formal-readiness")
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "NOT_RUN"


def test_job_manager_marks_stale_worker_interrupted_and_rejects_bad_id(tmp_path: Path) -> None:
    jobs = tmp_path / "artifacts" / "jobs" / "job-20260804-120000-abcdef"
    write_json(
        jobs / "status.json",
        {"job_id": jobs.name, "status": "RUNNING", "message": "old"},
    )
    manager = LocalJobManager(project_root=tmp_path, artifact_root=tmp_path / "artifacts")
    assert manager.get_job(jobs.name)["status"] == "INTERRUPTED"
    with pytest.raises(ValueError):
        manager.get_job("../../escape")


def test_runtime_manager_exposes_only_allowlisted_launchers(tmp_path: Path) -> None:
    manager = LocalRuntimeServiceManager(
        project_root=tmp_path, artifact_root=tmp_path / "artifacts"
    )
    assert {item["backend"] for item in manager.services()} == {"llama_cpp", "vllm"}
    assert all(not item["installed"] for item in manager.services())
    with pytest.raises(ValueError):
        manager.start("arbitrary-command")


def test_dashboard_is_local_first_and_has_required_workflows() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "dashboard" / "index.html").read_text(encoding="utf-8")
    script = (root / "dashboard" / "app.js").read_text(encoding="utf-8")
    style = (root / "dashboard" / "styles.css").read_text(encoding="utf-8")
    assert "Local Control Console" in html
    assert all(
        anchor in html
        for anchor in [
            'id="progress"',
            'id="tune"',
            'id="jobs"',
            'id="runs"',
            'id="evidence"',
            'id="serviceList"',
        ]
    )
    assert "X-EdgeFlow-Token" in script
    assert "jobs/benchmark" in script
    assert "runtime-services" in script
    assert "experiment-progress" in script
    assert 'id="batchSize"' in html
    assert 'data-source-type="demo"' not in html
    assert "prefers-reduced-motion" in style
    assert "http://127.0.0.1:8001" in html
    assert "http://127.0.0.1:8002" in script
    assert "https://" not in html
