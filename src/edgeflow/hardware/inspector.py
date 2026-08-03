from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from edgeflow.core.models import utc_now
from edgeflow.core.serialization import project_root, sha256_value


def _run(arguments: list[str], timeout: float = 8.0) -> str | None:
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _ram_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return max(1, int(getattr(os, "sysconf", lambda _key: 1)("SC_PAGE_SIZE")))


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _execution_mode() -> tuple[str, str | None]:
    release = platform.release().lower()
    if os.name == "nt":
        return "windows_native", None
    if "microsoft" in release or os.environ.get("WSL_DISTRO_NAME"):
        return "wsl2", os.environ.get("WSL_DISTRO_NAME") or platform.release()
    return "linux_native", None


def _git_state(root: Path) -> dict[str, Any] | None:
    commit = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    if not commit:
        return None
    status = _run(["git", "-C", str(root), "status", "--porcelain"])
    remote = _run(["git", "-C", str(root), "remote", "get-url", "origin"])
    return {"commit": commit, "dirty": bool(status), "remote": remote}


def _gpu_processes() -> list[str]:
    output = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return [line.strip() for line in (output or "").splitlines() if line.strip()]


def _gpu_record() -> dict[str, Any]:
    query = (
        "name,uuid,memory.total,driver_version,power.limit,persistence_mode"
    )
    output = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if not output:
        raise RuntimeError("nvidia-smi could not find an NVIDIA GPU")
    row = [item.strip() for item in output.splitlines()[0].split(",")]
    if len(row) != 6:
        raise RuntimeError(f"Unexpected nvidia-smi output: {output}")
    compute_capability = "unknown"
    persistence: bool | None = None
    cuda_runtime = "unavailable"
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"{major}.{minor}"
            cuda_runtime = str(torch.version.cuda or "unavailable")
    except ImportError:
        pass
    if row[5].lower() in {"enabled", "disabled"}:
        persistence = row[5].lower() == "enabled"
    power = None if row[4] in {"N/A", "[N/A]"} else float(row[4])
    return {
        "vendor": "NVIDIA",
        "name": row[0],
        "uuid": row[1],
        "compute_capability": compute_capability,
        "vram_bytes": int(float(row[2]) * 1024 * 1024),
        "driver_version": row[3],
        "power_limit_w": power,
        "persistence_mode": persistence,
        "_cuda_runtime": cuda_runtime,
    }


def inspect_hardware(root: Path | None = None) -> dict[str, Any]:
    """Capture a schema-compatible, content-addressed environment fingerprint."""

    root = (root or project_root()).resolve()
    execution_mode, wsl_version = _execution_mode()
    gpu = _gpu_record()
    cuda_runtime = gpu.pop("_cuda_runtime")
    toolkit = _run(["nvcc", "--version"])
    nsys = _run(["nsys", "--version"])
    ncu = _run(["ncu", "--version"])
    llama_commit = _run(["llama-cli", "--version"])
    fingerprint: dict[str, Any] = {
        "schema_version": "1.0",
        "fingerprint_id": "pending",
        "captured_at": utc_now(),
        "host": {
            "os": platform.platform(),
            "execution_mode": execution_mode,
            "wsl_version": wsl_version,
            "kernel": platform.release(),
            "cpu": _cpu_name(),
            "physical_cores": None,
            "logical_cores": os.cpu_count(),
            "ram_bytes": _ram_bytes(),
        },
        "gpu": gpu,
        "software": {
            "python": platform.python_version(),
            "pytorch": _package_version("torch"),
            "cuda_runtime": cuda_runtime,
            "cuda_toolkit": toolkit,
            "transformers": _package_version("transformers"),
            "triton": _package_version("triton"),
            "vllm": None if _package_version("vllm") == "unavailable" else _package_version("vllm"),
            "llama_cpp_commit": llama_commit,
            "nsight_systems": nsys,
            "nsight_compute": ncu,
        },
        "measurement_controls": {
            "gpu_idle_threshold_pct": 5.0,
            "temperature_ceiling_c": 80.0,
            "background_gpu_processes": _gpu_processes(),
        },
        "git": _git_state(root),
        "sha256": "pending",
    }
    # Capture time and transient process state are validation evidence, not policy identity.
    digest_payload = json.loads(json.dumps(fingerprint))
    for key in ("sha256", "fingerprint_id", "captured_at"):
        digest_payload.pop(key, None)
    digest_payload["measurement_controls"]["background_gpu_processes"] = []
    digest = sha256_value(digest_payload)
    fingerprint["fingerprint_id"] = f"hw-{digest[:12]}"
    fingerprint["sha256"] = digest
    return fingerprint


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    required: bool
    message: str


@dataclass(frozen=True)
class DoctorReport:
    ready: bool
    checks: tuple[DoctorCheck, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "checks": [asdict(check) for check in self.checks]}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


def run_doctor(root: Path | None = None) -> DoctorReport:
    checks: list[DoctorCheck] = []

    def add(name: str, ok: bool, message: str, *, required: bool = True) -> None:
        checks.append(DoctorCheck(name, "PASS" if ok else ("FAIL" if required else "WARN"), required, message))

    add("python", sys.version_info >= (3, 11), platform.python_version())
    add("nvidia_smi", shutil.which("nvidia-smi") is not None, shutil.which("nvidia-smi") or "not found")
    try:
        fingerprint = inspect_hardware(root)
        gpu = fingerprint["gpu"]
        add("gpu", "RTX 4080 SUPER" in gpu["name"], f"{gpu['name']} · SM {gpu['compute_capability']}")
    except RuntimeError as exc:
        add("gpu", False, str(exc))
    try:
        import torch

        add("pytorch_cuda", torch.cuda.is_available(), f"torch {torch.__version__} · CUDA {torch.version.cuda}")
    except ImportError:
        add("pytorch_cuda", False, "PyTorch is not installed")
    add("triton", _package_version("triton") != "unavailable", _package_version("triton"))
    add("transformers", _package_version("transformers") != "unavailable", _package_version("transformers"))
    add("llama_cpp", shutil.which("llama-cli") is not None, "optional backend", required=False)
    add("vllm", _package_version("vllm") != "unavailable", "optional backend", required=False)
    add("nsys", shutil.which("nsys") is not None, shutil.which("nsys") or "optional", required=False)
    add("ncu", shutil.which("ncu") is not None, shutil.which("ncu") or "optional", required=False)
    root_path = (root or project_root()).resolve()
    usage = shutil.disk_usage(root_path)
    add("disk", usage.free >= 5 * 1024**3, f"{usage.free / 1024**3:.1f} GiB free")
    ready = all(check.status == "PASS" for check in checks if check.required)
    return DoctorReport(ready=ready, checks=tuple(checks))
