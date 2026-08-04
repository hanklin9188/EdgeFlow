from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any, ClassVar

from edgeflow.core.models import utc_now


class LocalRuntimeServiceManager:
    """Own one allowlisted loopback inference service at a time."""

    _definitions: ClassVar[dict[str, tuple[str, int, str]]] = {
        "llama_cpp": (
            "scripts/start_llama_cpp_server.sh",
            8001,
            "ministral3-3b-instruct-2512",
        ),
        "vllm": ("scripts/start_vllm_server.sh", 8002, "smollm2-360m-instruct"),
    }

    def __init__(self, *, project_root: Path, artifact_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.log_root = (artifact_root / "runtime-services").resolve()
        self.log_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._backend: str | None = None
        self._api_key: str | None = None
        self._status: dict[str, Any] = self._stopped_status()

    @staticmethod
    def _stopped_status(message: str = "No managed runtime service is active.") -> dict[str, Any]:
        return {
            "state": "STOPPED",
            "backend": None,
            "base_url": None,
            "pid": None,
            "started_at": None,
            "ready_at": None,
            "message": message,
            "managed": True,
        }

    @staticmethod
    def _request(base_url: str, api_key: str | None) -> urllib.request.Request:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        return urllib.request.Request(base_url.rstrip("/") + "/health", headers=headers)

    def _healthy(self, base_url: str, api_key: str | None) -> bool:
        try:
            with urllib.request.urlopen(self._request(base_url, api_key), timeout=1) as response:
                return response.status < 500
        except (urllib.error.URLError, TimeoutError):
            return False

    @staticmethod
    def _port_open(port: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            return False

    def _runtime_installed(self, backend: str) -> bool:
        relative = (
            ".runtime/llama.cpp/build/bin/llama-server"
            if backend == "llama_cpp"
            else ".runtime/vllm/.venv/bin/vllm"
        )
        target = self.project_root / relative
        return target.is_file() and os.access(target, os.X_OK)

    def status(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is not None and self._status["state"] not in {
                "FAILED",
                "STOPPED",
            }:
                self._status.update(
                    {
                        "state": "FAILED",
                        "message": f"Managed runtime exited with code {process.returncode}.",
                    }
                )
                self._clear_secret()
            return dict(self._status)

    def services(self) -> list[dict[str, Any]]:
        current = self.status()
        rows: list[dict[str, Any]] = []
        for backend, (_script_name, port, model_id) in self._definitions.items():
            selected = current.get("backend") == backend
            rows.append(
                {
                    "backend": backend,
                    "state": current["state"] if selected else "STOPPED",
                    "base_url": f"http://127.0.0.1:{port}",
                    "model_id": model_id,
                    "pid": current.get("pid") if selected else None,
                    "started_at": current.get("started_at") if selected else None,
                    "ready_at": current.get("ready_at") if selected else None,
                    "message": (
                        current["message"]
                        if selected
                        else "Available through the pinned local launcher."
                    ),
                    "installed": self._runtime_installed(backend),
                    "managed": True,
                }
            )
        return rows

    def start(self, backend: str) -> dict[str, Any]:
        if backend not in self._definitions:
            raise ValueError(f"runtime service is not allowlisted: {backend}")
        with self._lock:
            current = self.status()
            if current["state"] in {"STARTING", "RUNNING", "STOPPING"}:
                raise RuntimeError(
                    f"{current['backend']} is already {str(current['state']).lower()}"
                )
            script_name, port, model_id = self._definitions[backend]
            script = (self.project_root / script_name).resolve()
            if not script.is_file() or not os.access(script, os.X_OK):
                raise RuntimeError(f"pinned launcher is unavailable: {script_name}")
            base_url = f"http://127.0.0.1:{port}"
            if self._port_open(port):
                raise RuntimeError(
                    f"{base_url} already has an unmanaged service; stop it before using app control"
                )
            api_key = secrets.token_urlsafe(32)
            environment = os.environ.copy()
            environment["EDGEFLOW_RUNTIME_API_KEY"] = api_key
            log_path = self.log_root / f"{backend}.log"
            log_handle = log_path.open("ab")
            try:
                process = subprocess.Popen(
                    [str(script)],
                    cwd=self.project_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    start_new_session=True,
                )
            finally:
                log_handle.close()
            self._process = process
            self._backend = backend
            self._api_key = api_key
            os.environ["EDGEFLOW_RUNTIME_API_KEY"] = api_key
            self._status = {
                "state": "STARTING",
                "backend": backend,
                "base_url": base_url,
                "model_id": model_id,
                "pid": process.pid,
                "started_at": utc_now(),
                "ready_at": None,
                "message": "Loading the pinned model into the local GPU service.",
                "managed": True,
            }
            threading.Thread(
                target=self._wait_until_ready,
                args=(process, base_url, api_key),
                name=f"edgeflow-runtime-{backend}",
                daemon=True,
            ).start()
            return dict(self._status)

    def _wait_until_ready(
        self, process: subprocess.Popen[bytes], base_url: str, api_key: str
    ) -> None:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline and process.poll() is None:
            if self._healthy(base_url, api_key):
                with self._lock:
                    if self._process is process:
                        self._status.update(
                            {
                                "state": "RUNNING",
                                "ready_at": utc_now(),
                                "message": "Pinned runtime is ready on loopback.",
                            }
                        )
                return
            time.sleep(1)
        with self._lock:
            if self._process is process:
                return_code = process.poll()
                self._status.update(
                    {
                        "state": "FAILED",
                        "message": (
                            f"Runtime exited with code {return_code}."
                            if return_code is not None
                            else "Runtime did not become healthy within 300 seconds."
                        ),
                    }
                )
                self._clear_secret()
        if process.poll() is None:
            self._terminate(process)

    def stop(self, backend: str | None = None) -> dict[str, Any]:
        with self._lock:
            if backend is not None and backend not in self._definitions:
                raise ValueError(f"runtime service is not allowlisted: {backend}")
            process = self._process
            if process is None or process.poll() is not None:
                self._process = None
                self._backend = None
                self._clear_secret()
                self._status = self._stopped_status()
                return dict(self._status)
            if backend is not None and backend != self._backend:
                raise RuntimeError(f"{backend} is not the active managed runtime")
            self._status.update(
                {"state": "STOPPING", "message": "Stopping the managed runtime service."}
            )
        self._terminate(process)
        with self._lock:
            self._process = None
            stopped_backend = self._backend
            self._backend = None
            self._clear_secret()
            self._status = self._stopped_status(f"Managed {stopped_backend} service stopped.")
            return dict(self._status)

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)

    def _clear_secret(self) -> None:
        current = os.environ.get("EDGEFLOW_RUNTIME_API_KEY")
        if self._api_key and current == self._api_key:
            os.environ.pop("EDGEFLOW_RUNTIME_API_KEY", None)
        self._api_key = None

    def shutdown(self) -> None:
        self.stop()

    def diagnostics(self) -> dict[str, Any]:
        """Path-free status suitable for a local API response."""
        return json.loads(json.dumps(self.status()))
