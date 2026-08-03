#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.core.models import utc_now  # noqa: E402
from edgeflow.core.serialization import read_json, sha256_value, write_json  # noqa: E402
from edgeflow.experiments import (  # noqa: E402
    audit_learned_prerequisites,
    fixed_plan_dominance,
    session_break_even_study,
)
from edgeflow.storage import EdgeFlowDB  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _median(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        float(value)
        for row in rows
        if isinstance((value := row.get("metrics", {}).get(metric)), (int, float))
    ]
    return float(statistics.median(values)) if values else None


def _plan_signature(plan: dict[str, Any]) -> str:
    excluded = {"plan_id", "canonical_sha256", "support_status"}
    return sha256_value({key: value for key, value in plan.items() if key not in excluded})


def collect_analysis_rows(artifact_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(artifact_root.glob("run-*")):
        required = (
            "run_manifest.json",
            "workload.json",
            "execution_plan.json",
            "validation_verdict.json",
            "metrics.jsonl",
        )
        if not all((run_dir / name).is_file() for name in required):
            continue
        manifest = read_json(run_dir / "run_manifest.json")
        workload = read_json(run_dir / "workload.json")
        plan = read_json(run_dir / "execution_plan.json")
        validation = read_json(run_dir / "validation_verdict.json")
        metrics = _rows(run_dir / "metrics.jsonl")
        measured = [row for row in metrics if row.get("phase") == "end_to_end"]
        request_latency = _median(measured, "request_latency_ms")
        if request_latency is None:
            continue
        startup = sum(
            float(value)
            for phase in ("load", "compile", "capture", "autotune")
            if (value := _median([row for row in metrics if row.get("phase") == phase], "wall_ms"))
            is not None
        )
        rows.append(
            {
                "run_id": manifest["run_id"],
                "experiment_id": manifest["experiment_id"],
                "hardware_fingerprint_sha256": manifest["hardware_fingerprint_sha256"],
                "plan_id": plan["plan_id"],
                "plan_sha256": manifest.get("plan_sha256"),
                "plan_signature": _plan_signature(plan),
                "workload_id": workload["workload_id"],
                "workload_sha256": manifest.get("workload_sha256"),
                "source_type": manifest["source_type"],
                "paired_prompt_ids": bool(measured)
                and all(
                    "deterministic paired prompts" in str(row.get("notes", ""))
                    for row in measured
                ),
                "workload": workload,
                "metrics": {
                    "request_latency_ms": request_latency,
                    "startup_ms": startup,
                    "ttft_ms": _median(measured, "ttft_ms"),
                    "tpot_ms": _median(measured, "tpot_ms"),
                    "peak_vram_bytes": _median(measured, "peak_vram_bytes"),
                },
                "validation": validation,
            }
        )
    return rows


def main() -> int:
    artifact_root = ROOT / "artifacts"
    rows = collect_analysis_rows(artifact_root)
    database = EdgeFlowDB(artifact_root / "runs.sqlite")
    evidence = database.list_evidence(limit=100_000)
    question_path = ROOT / "datasets" / "grounded_questions.jsonl"
    grounded_questions = len(_rows(question_path)) if question_path.is_file() else 0
    e10 = fixed_plan_dominance(rows)
    e19 = session_break_even_study(rows)
    prerequisites = audit_learned_prerequisites(
        rows,
        evidence,
        grounded_question_count=grounded_questions,
    )
    report = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "run_count": len(rows),
        "E10": e10,
        "E19": e19,
        "E25_E28": prerequisites,
        "claim_scope": "Readiness and registered analysis only; incomplete rows create no claim.",
    }
    output = artifact_root / "experiments" / "formal-readiness.json"
    write_json(output, report)
    e19_result = artifact_root / "experiments" / "E19" / "result.json"
    if e19["pass"]:
        write_json(e19_result, e19)
    else:
        e19_result.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(output),
                "E10": e10["status"],
                "E19": e19["status"],
                "E25": prerequisites["experiments"]["E25"]["status"],
                "E28": prerequisites["experiments"]["E28"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
