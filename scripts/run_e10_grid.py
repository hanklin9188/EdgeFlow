#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
VLLM_PYTHON = ROOT / ".runtime" / "vllm" / ".venv" / "bin" / "python"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _ready() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8002/health", timeout=2) as response:
            return response.status < 500
    except (urllib.error.URLError, TimeoutError):
        return False


def _eligible(run_id: str) -> bool:
    path = ROOT / "artifacts" / run_id / "validation_verdict.json"
    return path.is_file() and _read_json(path).get("policy_eligible") is True


def _wait_for_server(process: subprocess.Popen[str], timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited during startup with code {process.returncode}")
        if _ready():
            return
        time.sleep(2)
    raise TimeoutError("vLLM server did not become ready within 15 minutes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume one preregistered E10 vLLM plan grid")
    parser.add_argument(
        "--profile",
        choices=["ministral3-3b-grid-mbt32768", "ministral3-3b-grid-mbt65536"],
        required=True,
    )
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    arguments = parser.parse_args()
    spec = yaml.safe_load((ROOT / "specs" / "e10_grid.yaml").read_text(encoding="utf-8"))
    plan = next(item for item in spec["plans"] if item["profile"] == arguments.profile)
    progress_path = ROOT / "artifacts" / "experiments" / "E10" / f"{arguments.profile}.json"
    progress = (
        _read_json(progress_path)
        if progress_path.is_file()
        else {
            "schema_version": "1.0",
            "experiment_id": "E10",
            "model_id": spec["model_id"],
            "profile": arguments.profile,
            "status": "RUNNING",
            "cases": {},
        }
    )
    cases: dict[str, Any] = progress["cases"]
    server: subprocess.Popen[str] | None = None
    log_handle = None
    try:
        if arguments.no_start_server:
            if not _ready():
                raise RuntimeError("--no-start-server was used but port 8002 is not ready")
        else:
            if _ready():
                raise RuntimeError("port 8002 is already occupied; stop the existing service first")
            log_path = (
                ROOT / "artifacts" / "experiments" / "E10" / f"{arguments.profile}-server.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
            environment = {**os.environ, "EDGEFLOW_VLLM_PROFILE": arguments.profile}
            server = subprocess.Popen(
                [str(ROOT / "scripts" / "start_vllm_server.sh")],
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _wait_for_server(server)

        for prompt_tokens in spec["prompt_tokens"]:
            for output_tokens in spec["output_tokens"]:
                for concurrency in spec["concurrency"]:
                    case_id = f"p{prompt_tokens}-o{output_tokens}-c{concurrency}"
                    prior = cases.get(case_id, {})
                    if prior.get("status") == "PASS" and _eligible(str(prior.get("run_id"))):
                        continue
                    command = [
                        str(VLLM_PYTHON),
                        str(ROOT / "scripts" / "run_external_bucket.py"),
                        "--backend",
                        "vllm",
                        "--model-id",
                        spec["model_id"],
                        "--served-model",
                        "ministral3-3b-edgeflow",
                        "--server-profile",
                        arguments.profile,
                        "--prompt-tokens",
                        str(prompt_tokens),
                        "--output-tokens",
                        str(output_tokens),
                        "--concurrency",
                        str(concurrency),
                        "--repetitions",
                        str(spec["repetitions"]),
                        "--warmup",
                        str(spec["warmup_requests"]),
                        "--max-num-batched-tokens",
                        str(plan["max_num_batched_tokens"]),
                        "--max-num-seqs",
                        str(plan["max_num_seqs"]),
                        "--experiment-id",
                        "E10",
                    ]
                    completed = subprocess.run(
                        command, cwd=ROOT, capture_output=True, check=False, text=True, timeout=3600
                    )
                    parsed: dict[str, Any] = {}
                    for line in reversed(completed.stdout.splitlines()):
                        try:
                            parsed = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                    cases[case_id] = {
                        "status": "PASS" if completed.returncode == 0 else "FAIL",
                        "run_id": parsed.get("run_id"),
                        "verdict": parsed.get("verdict"),
                        "returncode": completed.returncode,
                        "stderr_tail": completed.stderr.splitlines()[-20:],
                        "updated_at_unix": time.time(),
                    }
                    passed = sum(item.get("status") == "PASS" for item in cases.values())
                    progress.update(
                        {
                            "status": "RUNNING",
                            "completed_eligible_cases": passed,
                            "required_cases": 45,
                            "coverage": passed / 45,
                        }
                    )
                    _write_json(progress_path, progress)
                    print(
                        json.dumps({"case": case_id, **cases[case_id]}, sort_keys=True), flush=True
                    )
                    if completed.returncode != 0 and arguments.stop_on_failure:
                        return completed.returncode
        passed = sum(item.get("status") == "PASS" for item in cases.values())
        progress.update(
            {
                "status": "PASS" if passed == 45 else "INCOMPLETE",
                "pass": passed == 45,
                "completed_eligible_cases": passed,
                "required_cases": 45,
                "coverage": passed / 45,
            }
        )
        _write_json(progress_path, progress)
        return 0 if passed == 45 else 1
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
