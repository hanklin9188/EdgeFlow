from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from edgeflow.api.schemas import BenchmarkSubmission
from edgeflow.core.models import utc_now
from edgeflow.core.serialization import read_json, write_json

TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED", "TIMED_OUT"}


class LocalJobManager:
    """Single-GPU, subprocess-backed local job queue.

    Only typed benchmark specifications are accepted. Workers inherit the current
    Python environment and never invoke a shell.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        artifact_root: Path,
        max_parallel_jobs: int = 1,
        timeout_seconds: int = 6 * 60 * 60,
    ) -> None:
        if max_parallel_jobs != 1:
            raise ValueError("EdgeFlow currently permits exactly one GPU benchmark at a time")
        self.project_root = project_root.resolve()
        self.artifact_root = artifact_root.resolve()
        self.jobs_root = self.artifact_root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._recover_interrupted_jobs()

    def _job_dir(self, job_id: str) -> Path:
        if not job_id.startswith("job-") or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in job_id
        ):
            raise ValueError("invalid job id")
        return self.jobs_root / job_id

    def _status_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "status.json"

    def _recover_interrupted_jobs(self) -> None:
        for path in self.jobs_root.glob("job-*/status.json"):
            try:
                status = read_json(path)
            except (OSError, ValueError):
                continue
            if status.get("status") in {"QUEUED", "RUNNING", "CANCELLING"}:
                status.update(
                    {
                        "status": "INTERRUPTED",
                        "completed_at": utc_now(),
                        "message": "The local web process stopped before this job reported completion.",
                    }
                )
                write_json(path, status)

    def _active_count(self) -> int:
        return sum(thread.is_alive() for thread in self._threads.values())

    def submit_benchmark(self, submission: BenchmarkSubmission) -> dict[str, Any]:
        with self._lock:
            if self._active_count() >= 1:
                raise RuntimeError("a GPU benchmark is already running")
            job_id = f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            job_dir = self._job_dir(job_id)
            job_dir.mkdir(parents=False, exist_ok=False)
            write_json(job_dir / "spec.json", submission.model_dump(mode="json"))
            status = {
                "schema_version": "1.0",
                "job_id": job_id,
                "kind": "benchmark",
                "status": "QUEUED",
                "created_at": utc_now(),
                "started_at": None,
                "completed_at": None,
                "pid": None,
                "model_id": submission.model_id,
                "backend": submission.backend,
                "experiment_id": submission.experiment_id,
                "message": "Waiting for the isolated local worker.",
                "result_available": False,
            }
            write_json(self._status_path(job_id), status)
            thread = threading.Thread(
                target=self._execute,
                args=(job_id,),
                name=f"edgeflow-{job_id}",
                daemon=True,
            )
            self._threads[job_id] = thread
            thread.start()
            return status

    def _execute(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        status_path = self._status_path(job_id)
        status = read_json(status_path)
        if status.get("status") == "CANCELLING":
            status.update(
                {
                    "status": "CANCELLED",
                    "completed_at": utc_now(),
                    "message": "Cancelled before the isolated worker started.",
                }
            )
            write_json(status_path, status)
            return
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        command = [
            sys.executable,
            "-m",
            "edgeflow.local.worker",
            "--spec",
            str(job_dir / "spec.json"),
            "--result",
            str(job_dir / "result.json"),
        ]
        environment = os.environ.copy()
        environment["EDGEFLOW_PROJECT_ROOT"] = str(self.project_root)
        environment["EDGEFLOW_ARTIFACT_ROOT"] = str(self.artifact_root)
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    shell=False,
                    start_new_session=True,
                )
                with self._lock:
                    self._processes[job_id] = process
                    status.update(
                        {
                            "status": "RUNNING",
                            "started_at": utc_now(),
                            "pid": process.pid,
                            "message": "Benchmark worker is running locally.",
                        }
                    )
                    write_json(status_path, status)
                try:
                    return_code = process.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    status.update(
                        {
                            "status": "TIMED_OUT",
                            "completed_at": utc_now(),
                            "message": f"Worker exceeded the {self.timeout_seconds}-second safety limit.",
                        }
                    )
                    write_json(status_path, status)
                    return
            current = read_json(status_path)
            if current.get("status") == "CANCELLING":
                final_status = "CANCELLED"
                message = "Cancelled by the local operator. Partial run artifacts remain auditable."
            elif return_code == 0 and (job_dir / "result.json").is_file():
                final_status = "SUCCEEDED"
                message = (
                    "Worker completed; inspect the validation verdict before using the result."
                )
            else:
                final_status = "FAILED"
                message = "Worker failed. The failure log is local-only and is not exposed as a run artifact."
            current.update(
                {
                    "status": final_status,
                    "completed_at": utc_now(),
                    "message": message,
                    "result_available": (job_dir / "result.json").is_file(),
                }
            )
            write_json(status_path, current)
        except Exception as exc:
            status.update(
                {
                    "status": "FAILED",
                    "completed_at": utc_now(),
                    "message": f"Job manager failure: {type(exc).__name__}",
                }
            )
            write_json(status_path, status)
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def list_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.jobs_root.glob("job-*/status.json"), reverse=True)[:limit]:
            try:
                status = read_json(path)
                result_path = path.parent / "result.json"
                if result_path.is_file():
                    status["result"] = read_json(result_path)
                rows.append(status)
            except (OSError, ValueError):
                continue
        return rows

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        path = self._status_path(job_id)
        if not path.is_file():
            return None
        status = read_json(path)
        result_path = path.parent / "result.json"
        if result_path.is_file():
            status["result"] = read_json(result_path)
        return status

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            status = self.get_job(job_id)
            if status is None:
                raise KeyError(job_id)
            if status["status"] in TERMINAL_STATES:
                return status
            process = self._processes.get(job_id)
            status.update({"status": "CANCELLING", "message": "Stopping the isolated worker."})
            status.pop("result", None)
            write_json(self._status_path(job_id), status)
            if process is not None and process.poll() is None:
                process.terminate()
            return status
