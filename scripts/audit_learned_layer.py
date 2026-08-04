#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_formal_readiness import collect_analysis_rows  # noqa: E402

from edgeflow.core.models import utc_now  # noqa: E402
from edgeflow.core.serialization import write_json  # noqa: E402
from edgeflow.learned import build_cost_dataset, build_intervention_dataset  # noqa: E402
from edgeflow.storage import EdgeFlowDB  # noqa: E402


def main() -> int:
    artifact_root = ROOT / "artifacts"
    rows = collect_analysis_rows(artifact_root)
    evidence = EdgeFlowDB(artifact_root / "runs.sqlite").list_evidence(limit=100_000)
    e25 = build_cost_dataset(rows)
    e27 = build_intervention_dataset(evidence)
    e26 = {
        "schema_version": "1.0",
        "experiment_id": "E26",
        "created_at": utc_now(),
        "status": "BLOCKED_PREREQUISITE" if not e25["pass"] else "READY_FOR_EVALUATION",
        "pass": False,
        "prerequisite_snapshot_sha256": e25["snapshot_sha256"],
        "prerequisite_unique_points": e25["unique_validated_points"],
        "required_unique_points": e25["required_unique_points"],
        "evaluation_protocol": "grouped holdout plus temporal holdout; measured selection confirms every recommendation",
        "claim_scope": "No cost-model claim until E25 passes and both holdouts are evaluated.",
    }
    write_json(artifact_root / "learned" / "e25-cost-dataset.json", e25)
    write_json(artifact_root / "learned" / "e27-intervention-dataset.json", e27)
    for experiment_id, report in (("E25", e25), ("E26", e26), ("E27", e27)):
        write_json(artifact_root / "experiments" / experiment_id / "result.json", report)
    print(
        json.dumps(
            {
                "E25": e25["status"],
                "E26": e26["status"],
                "E27": e27["status"],
                "unique_points": e25["unique_validated_points"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
