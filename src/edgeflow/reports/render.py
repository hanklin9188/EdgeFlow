from __future__ import annotations

import json
from pathlib import Path

from edgeflow.core.serialization import read_json
from edgeflow.metrics.statistics import describe


def render_run_report(run_dir: Path) -> str:
    manifest = read_json(run_dir / "run_manifest.json")
    plan = read_json(run_dir / "execution_plan.json")
    workload = read_json(run_dir / "workload.json")
    validation = read_json(run_dir / "validation_verdict.json")
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    measured = [row for row in rows if row["phase"] == "end_to_end"]
    latency = [row["metrics"]["request_latency_ms"] for row in measured if row["metrics"]["request_latency_ms"] is not None]
    summary = describe(latency) if latency else None
    lines = [
        f"# Run report: {manifest['run_id']}", "", f"> Source: **{manifest['source_type'].upper()}** · Verdict: **{validation['verdict']}**", "",
        "## Scope", "",
        f"- Experiment: `{manifest['experiment_id']}` / `{manifest['protocol_version']}`",
        f"- Model: `{workload['model_id']}` @ `{manifest['model_revision']}`",
        f"- Plan: `{plan['plan_id']}` (`{plan['backend']}`)",
        f"- Workload: `{workload['workload_id']}`",
        f"- Policy eligible: `{str(validation['policy_eligible']).lower()}`", "",
        "## Raw-derived summary", "",
    ]
    if summary:
        lines.extend(
            [
                f"- Requests: {summary['count']}",
                f"- Request latency median: {summary['median']:.3f} ms",
                f"- Request latency p95: {summary['p95']:.3f} ms",
                f"- Robust CV: {summary['robust_cv']:.3%}",
            ]
        )
    else:
        lines.append("- No valid production timing rows.")
    lines.extend(["", "## Validation issues", ""])
    lines.extend(
        [f"- `{issue['code']}` — {issue['message']}" for issue in validation["issues"]]
        or ["- None."]
    )
    lines.extend(["", "All values above are regenerated from `metrics.jsonl`; no value is inferred.", ""])
    return "\n".join(lines)
