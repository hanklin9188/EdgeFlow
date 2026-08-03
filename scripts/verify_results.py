#!/usr/bin/env python3
"""Revalidate public artifacts and reject unsupported claims or demo leakage."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.validation import ValidationEngine  # noqa: E402

SECRET_PATTERNS = [
    re.compile(r"gh[opusr]_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
EXCLUDED_PARTS = {".git", ".venv", ".cache", ".pytest_cache", ".ruff_cache", "__pycache__"}


def validate_json(path: Path, schema_name: str) -> list[str]:
    schema = json.loads((ROOT / "specs" / schema_name).read_text(encoding="utf-8"))
    instance = json.loads(path.read_text(encoding="utf-8"))
    errors = Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(instance)
    return [f"{path}: {error.message}" for error in errors]


def main() -> int:
    errors: list[str] = []
    checked_runs = 0
    artifact_root = ROOT / "artifacts"
    if artifact_root.exists():
        validator = ValidationEngine(root=ROOT)
        for manifest_path in sorted(artifact_root.glob("run-*/run_manifest.json")):
            run_dir = manifest_path.parent
            stored_path = run_dir / "validation_verdict.json"
            if not stored_path.exists():
                errors.append(f"{run_dir}: missing validation_verdict.json")
                continue
            regenerated = validator.validate(run_dir, write=False)
            stored = json.loads(stored_path.read_text(encoding="utf-8"))
            for key in ("verdict", "policy_eligible", "public_claim_eligible"):
                if stored.get(key) != regenerated.get(key):
                    errors.append(f"{run_dir}: stored {key} differs from regenerated verdict")
            if stored.get("policy_eligible") and manifest_path.read_text(encoding="utf-8").find('"source_type": "demo"') >= 0:
                errors.append(f"{run_dir}: demo run is marked policy eligible")
            checked_runs += 1
    for directory, schema in (("policies", "deployment_policy.schema.json"), ("evidence", "evidence_record.schema.json")):
        target = ROOT / "results" / directory
        if target.exists():
            for path in target.glob("*.json"):
                errors.extend(validate_json(path, schema))
                value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
                if value.get("source_type") == "demo":
                    errors.append(f"{path}: demo artifact in public result directory")
    scanned = 0
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
            or path.suffix in {".png", ".pdf", ".lock"}
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"{path}: possible secret matching {pattern.pattern}")
    print(f"EdgeFlow result audit: {checked_runs} run(s), {scanned} text file(s), {len(errors)} error(s)")
    for error in errors:
        print(f"[FAIL] {error}")
    if not checked_runs:
        print("[INFO] No formal run artifacts are committed; no performance claim is eligible.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
