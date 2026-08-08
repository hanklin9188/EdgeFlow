#!/usr/bin/env python3
"""Audit EdgeFlow validation specification/implementation parity.

The default mode validates structure and checks paths for implemented rules.
Use --strict-release to require every hard rule blocking a release profile to
be implemented. Use --schema-only when validating an overlay package without
the complete EdgeFlow checkout.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ALLOWED_STATUS = {"implemented", "partial", "planned", "deferred"}
ALLOWED_KIND = {"hard", "advisory"}
ALLOWED_GATES = {f"G{i}" for i in range(9)}
REQUIRED_FIELDS = {
    "id",
    "title",
    "gate",
    "kind",
    "status",
    "rule",
    "risk",
    "implementation",
    "acceptance",
    "evidence",
    "release_gates",
    "owner",
    "backlog_ticket",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--matrix", type=Path, default=Path("specs/validation_requirements.yaml"))
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--strict-release", action="store_true")
    parser.add_argument("--release-profile", default="first_evidence")
    return parser.parse_args()


def nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def audit(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    root = args.root.resolve()
    matrix_path = args.matrix if args.matrix.is_absolute() else root / args.matrix
    errors: list[str] = []
    warnings: list[str] = []

    if not matrix_path.exists():
        return [f"matrix missing: {matrix_path}"], []

    try:
        data = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - parser reports details
        return [f"matrix parse failed: {exc}"], []

    if not isinstance(data, dict):
        return ["matrix root must be a mapping"], []
    if data.get("schema_version") != "1.0":
        errors.append("schema_version must be '1.0'")

    release_profiles = data.get("release_profiles")
    if not isinstance(release_profiles, dict) or not release_profiles:
        errors.append("release_profiles must be a non-empty mapping")
        release_profiles = {}

    requirements = data.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        errors.append("requirements must be a non-empty list")
        return errors, warnings

    ids: list[str] = []
    for index, requirement in enumerate(requirements, start=1):
        prefix = f"requirement[{index}]"
        if not isinstance(requirement, dict):
            errors.append(f"{prefix} must be a mapping")
            continue

        missing = sorted(REQUIRED_FIELDS - requirement.keys())
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(missing)}")

        req_id = requirement.get("id")
        if not nonempty_text(req_id):
            errors.append(f"{prefix} id must be non-empty text")
            req_id = prefix
        else:
            ids.append(req_id)
            prefix = req_id

        status = requirement.get("status")
        kind = requirement.get("kind")
        gate = requirement.get("gate")
        if status not in ALLOWED_STATUS:
            errors.append(f"{prefix}: invalid status {status!r}")
        if kind not in ALLOWED_KIND:
            errors.append(f"{prefix}: invalid kind {kind!r}")
        if gate not in ALLOWED_GATES:
            errors.append(f"{prefix}: invalid gate {gate!r}")

        for text_field in ["title", "rule", "risk", "evidence", "owner"]:
            if not nonempty_text(requirement.get(text_field)):
                errors.append(f"{prefix}: {text_field} must be non-empty text")

        implementation = requirement.get("implementation")
        if not isinstance(implementation, dict):
            errors.append(f"{prefix}: implementation must be a mapping")
            implementation = {}
        enforcement_paths = implementation.get("enforcement_paths", [])
        test_paths = implementation.get("test_paths", [])
        if not isinstance(enforcement_paths, list) or not all(nonempty_text(p) for p in enforcement_paths):
            errors.append(f"{prefix}: enforcement_paths must be a list of paths")
            enforcement_paths = []
        if not isinstance(test_paths, list) or not all(nonempty_text(p) for p in test_paths):
            errors.append(f"{prefix}: test_paths must be a list of paths")
            test_paths = []

        acceptance = requirement.get("acceptance")
        if not isinstance(acceptance, dict):
            errors.append(f"{prefix}: acceptance must be a mapping")
            acceptance = {}
        if not isinstance(acceptance.get("executable"), bool):
            errors.append(f"{prefix}: acceptance.executable must be boolean")
        if not nonempty_text(acceptance.get("criterion")):
            errors.append(f"{prefix}: acceptance.criterion must be non-empty")

        release_gates = requirement.get("release_gates")
        if not isinstance(release_gates, list) or not release_gates:
            errors.append(f"{prefix}: release_gates must be a non-empty list")
            release_gates = []
        unknown_profiles = sorted(set(release_gates) - set(release_profiles))
        if unknown_profiles:
            errors.append(f"{prefix}: unknown release profiles {unknown_profiles}")

        backlog = requirement.get("backlog_ticket")
        if status == "implemented":
            if not enforcement_paths:
                errors.append(f"{prefix}: implemented rule needs enforcement_paths")
            if not test_paths:
                errors.append(f"{prefix}: implemented rule needs test_paths")
            if acceptance.get("executable") is not True:
                errors.append(f"{prefix}: implemented rule needs executable acceptance")
        elif status in {"partial", "planned"} and not nonempty_text(backlog):
            errors.append(f"{prefix}: {status} rule needs backlog_ticket")

        if not args.schema_only:
            for path_text in enforcement_paths + test_paths:
                path = root / path_text
                if status == "implemented" and not path.exists():
                    errors.append(f"{prefix}: implemented path missing: {path_text}")
                elif status == "partial" and not path.exists():
                    warnings.append(f"{prefix}: partial path missing: {path_text}")

        if args.strict_release and kind == "hard" and args.release_profile in release_gates:
            if status != "implemented":
                errors.append(
                    f"{prefix}: hard requirement for {args.release_profile} is {status}, not implemented"
                )

    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate requirement IDs: {', '.join(sorted(duplicates))}")

    if args.strict_release and args.release_profile not in release_profiles:
        errors.append(f"unknown --release-profile {args.release_profile!r}")

    return errors, warnings


def main() -> int:
    args = parse_args()
    errors, warnings = audit(args)
    print("EdgeFlow validation parity audit")
    print("=" * 36)
    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[FAIL] {error}")
    if not errors:
        print("[PASS] validation requirements are structurally consistent")
    print(f"{len(errors)} failed · {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
