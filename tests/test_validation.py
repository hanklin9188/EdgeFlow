from __future__ import annotations

import json
from pathlib import Path

from edgeflow.validation import ValidationEngine


def test_complete_run_passes_all_policy_gates(valid_run_dir: Path) -> None:
    verdict = ValidationEngine().validate(valid_run_dir, write=True)
    assert verdict["verdict"] == "PASS"
    assert verdict["policy_eligible"] is True
    assert (valid_run_dir / "VALIDATION.md").exists()


def test_demo_source_is_never_policy_eligible(valid_run_dir: Path) -> None:
    manifest_path = valid_run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_type"] = "demo"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verdict = ValidationEngine().validate(valid_run_dir, write=False)
    assert verdict["policy_eligible"] is False
    assert verdict["public_claim_eligible"] is False


def test_profiled_latency_is_invalid(valid_run_dir: Path) -> None:
    manifest_path = valid_run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profiler_level"] = "nsys"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verdict = ValidationEngine().validate(valid_run_dir, write=False)
    assert verdict["verdict"] == "INVALID"


def test_failed_execution_cannot_validate_from_partial_metrics(valid_run_dir: Path) -> None:
    manifest_path = valid_run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "FAILED"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    verdict = ValidationEngine().validate(valid_run_dir, write=False)

    assert verdict["verdict"] == "INVALID"
    assert verdict["policy_eligible"] is False
