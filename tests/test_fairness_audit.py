from __future__ import annotations

import json
from pathlib import Path

from edgeflow.experiments import audit_runtime_fairness


def _write_run(
    root: Path,
    *,
    run_id: str,
    backend: str,
    hardware: str = "hardware-a",
    prompt_tokens: int = 128,
    eligible: bool = True,
) -> Path:
    path = root / run_id
    path.mkdir()
    workload = {
        "schema_version": "1.0",
        "workload_id": "matched-workload",
        "model_id": "model",
        "prompt_source": {"type": "synthetic", "name": "test", "revision": "1"},
        "prompt_tokens": prompt_tokens,
        "output_tokens": 32,
        "batch_size": 1,
        "concurrency": 1,
        "arrival_pattern": "closed_loop",
        "request_rate": None,
        "sampling": {
            "strategy": "greedy",
            "temperature": 0,
            "top_p": 1,
            "top_k": None,
            "ignore_eos": True,
        },
        "seed": 42,
        "streaming": True,
        "session_requests": 20,
        "quality_profile": "balanced",
        "notes": "test",
    }
    manifest = {
        "run_id": run_id,
        "hardware_fingerprint_sha256": hardware,
        "model_revision": "revision",
        "source_type": "measured",
        "profiler_level": "none",
    }
    plan = {
        "plan_id": f"{backend}-plan",
        "backend": backend,
        "model_format": "safetensors",
        "dtype": "bf16",
        "quantization": None,
        "cuda_graph": backend == "torch_compile",
        "max_num_batched_tokens": None,
        "max_num_seqs": None,
        "backend_args": {"execution_mode": "graph" if backend == "torch_compile" else "eager"},
    }
    fingerprint = {
        "gpu": {
            "vendor": "NVIDIA",
            "name": "test-gpu",
            "uuid": hardware,
            "vram_bytes": 1,
            "compute_capability": "8.9",
            "driver_version": "1",
            "power_limit_w": 1,
        },
        "host": {
            "cpu": "test-cpu",
            "logical_cores": 1,
            "physical_cores": 1,
            "ram_bytes": 1,
            "execution_mode": "test",
            "kernel": "test",
            "os": "test",
            "wsl_version": None,
        },
    }
    validation = {"policy_eligible": eligible, "quality_pass": eligible}
    (path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "workload.json").write_text(json.dumps(workload), encoding="utf-8")
    (path / "execution_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (path / "hardware_fingerprint.json").write_text(json.dumps(fingerprint), encoding="utf-8")
    (path / "validation_verdict.json").write_text(json.dumps(validation), encoding="utf-8")
    rows = [
        {
            "phase": "end_to_end",
            "metrics": {
                "request_latency_ms": 10 + index / 10,
                "ttft_ms": 2,
                "tpot_ms": 1,
                "generation_tokens_per_s": 100,
            },
        }
        for index in range(30)
    ]
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_fairness_audit_passes_only_matched_eligible_runtimes(tmp_path: Path) -> None:
    eager = _write_run(tmp_path, run_id="eager", backend="pytorch_eager")
    compiled = _write_run(tmp_path, run_id="compiled", backend="torch_compile")

    report = audit_runtime_fairness([eager, compiled])

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert set(report["descriptive_latency_order"]) == {"eager", "compiled"}
    assert report["causal_runtime_isolation"] is False
    assert report["comparison_caveats"]


def test_fairness_audit_refuses_mismatched_scope(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, run_id="baseline", backend="pytorch_eager")
    mismatch = _write_run(
        tmp_path,
        run_id="mismatch",
        backend="torch_compile",
        prompt_tokens=1024,
    )

    report = audit_runtime_fairness([baseline, mismatch])

    assert report["status"] == "INVALID"
    assert report["pass"] is False
    assert report["descriptive_latency_order"] == []


def test_fairness_audit_refuses_ineligible_run(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, run_id="baseline", backend="pytorch_eager")
    ineligible = _write_run(
        tmp_path,
        run_id="ineligible",
        backend="torch_compile",
        eligible=False,
    )

    report = audit_runtime_fairness([baseline, ineligible])

    assert report["status"] == "INCOMPLETE"
    assert report["pass"] is False
    assert any("policy-eligibility" in issue for issue in report["runs"][1]["issues"])
