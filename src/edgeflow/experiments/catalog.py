from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from edgeflow.core.serialization import read_json

PHASE_LABELS = {
    "measurement_integrity": "Measurement integrity",
    "runtime_baselines": "Runtime baselines",
    "policy_causal": "Policy & causal loop",
    "amortization": "Amortization",
    "custom_kernel": "Custom kernel",
    "optional_learned": "Optional learned layer",
    "external_validation": "External validation",
}


def load_experiment_catalog(root: Path) -> dict[str, Any]:
    path = root / "specs" / "experiment_matrix.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("experiments"), dict):
        raise ValueError("experiment matrix must contain an experiments mapping")
    return value


def _standalone_result(artifact_root: Path, experiment_id: str) -> dict[str, Any] | None:
    candidates = [artifact_root / "experiments" / experiment_id / "result.json"]
    if experiment_id == "E22":
        candidates.append(artifact_root / "kernel" / "rmsnorm-correctness.json")
    elif experiment_id == "E23":
        candidates.append(artifact_root / "kernel" / "rmsnorm-benchmark.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            result = read_json(path)
        except (OSError, ValueError):
            continue
        result["artifact_name"] = path.relative_to(artifact_root).as_posix()
        return result
    return None


def experiment_progress(
    *,
    root: Path,
    artifact_root: Path,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog = load_experiment_catalog(root)
    runs_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        experiment_id = str(row.get("manifest", {}).get("experiment_id", ""))
        if experiment_id:
            runs_by_experiment[experiment_id].append(row)

    rows: list[dict[str, Any]] = []
    phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for experiment_id, definition in catalog["experiments"].items():
        experiment_runs = runs_by_experiment.get(experiment_id, [])
        eligible = sum(
            bool((row.get("validation") or {}).get("policy_eligible")) for row in experiment_runs
        )
        measured = sum(row.get("manifest", {}).get("source_type") == "measured" for row in experiment_runs)
        standalone = _standalone_result(artifact_root, experiment_id)
        standalone_pass = bool(
            standalone
            and (
                standalone.get("status") in {"PASS", "PASSED", "VALIDATED"}
                or standalone.get("pass") is True
                or standalone.get("all_correct") is True
            )
        )
        engineering = str(definition.get("engineering_status", "planned"))
        if eligible or standalone_pass:
            status = "validated"
        elif measured or standalone:
            status = "measured"
        elif engineering == "external":
            status = "external_required"
        elif engineering == "deferred":
            status = "deferred"
        elif engineering in {"implemented", "partial"}:
            status = "ready"
        else:
            status = "planned"
        row = {
            "experiment_id": experiment_id,
            "name": definition.get("name", experiment_id),
            "phase": definition.get("phase", "unassigned"),
            "kind": definition.get("kind", "experiment"),
            "engineering_status": engineering,
            "validation_status": status,
            "run_count": len(experiment_runs),
            "measured_run_count": measured,
            "eligible_run_count": eligible,
            "external_requirement": definition.get("external_requirement"),
            "artifact_name": standalone.get("artifact_name") if standalone else None,
        }
        rows.append(row)
        phases[row["phase"]].append(row)

    counts = Counter(row["validation_status"] for row in rows)
    validated = counts["validated"]
    return {
        "schema_version": catalog["schema_version"],
        "total": len(rows),
        "validated": validated,
        "measured": counts["measured"],
        "ready": counts["ready"],
        "planned": counts["planned"],
        "deferred": counts["deferred"],
        "external_required": counts["external_required"],
        "validated_percent": round(validated / len(rows) * 100, 1) if rows else 0.0,
        "phases": [
            {
                "phase": phase,
                "label": PHASE_LABELS.get(phase, phase.replace("_", " ").title()),
                "experiments": phase_rows,
            }
            for phase, phase_rows in phases.items()
        ],
        "experiments": rows,
        "notice": "Validated means a local artifact passed its registered gate; it does not imply every headline claim is established.",
    }
