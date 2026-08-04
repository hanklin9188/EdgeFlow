#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.experiments.foundation import run_foundation_experiments  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EdgeFlow E00-E03 measurement foundations")
    parser.add_argument("--quick", action="store_true", help="Use reduced iterations for development")
    parser.add_argument("--only", action="append", choices=["E00", "E01", "E02", "E03"])
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    arguments = parser.parse_args()
    results = run_foundation_experiments(
        root=ROOT,
        artifact_root=arguments.artifact_root.resolve(),
        experiment_ids=arguments.only,
        quick=arguments.quick,
    )
    print(
        json.dumps(
            {
                "experiments": [result["experiment_id"] for result in results],
                "passed": sum(result["pass"] for result in results),
                "total": len(results),
            }
        )
    )
    return 0 if all(result["pass"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
