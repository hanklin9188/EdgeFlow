#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.core.serialization import write_json  # noqa: E402
from edgeflow.experiments.fairness import audit_runtime_fairness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit exact-scope runtime runs before comparing their performance"
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--minimum-repetitions", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "experiments" / "E09" / "fairness_audit.json",
    )
    arguments = parser.parse_args()
    if arguments.minimum_repetitions < 1:
        parser.error("--minimum-repetitions must be positive")
    report = audit_runtime_fairness(
        arguments.run_dirs, minimum_repetitions=arguments.minimum_repetitions
    )
    write_json(arguments.output, report)
    print(json.dumps({"output": str(arguments.output), "status": report["status"]}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
