from __future__ import annotations

from pathlib import Path
from typing import Any

from edgeflow.core.models import ExecutionPlan
from edgeflow.core.serialization import read_json


def find_compatible_quality_report(
    *,
    artifact_root: Path,
    model_id: str,
    model_revision: str,
    plan: ExecutionPlan,
) -> dict[str, Any] | None:
    """Return an exact-scope quality report; never extrapolate across a quantization or runtime."""

    quality_root = artifact_root / "quality"
    if not quality_root.is_dir():
        return None
    for path in sorted(quality_root.glob("*.json"), reverse=True):
        try:
            report = read_json(path)
        except (OSError, ValueError):
            continue
        scope = report.get("scope", {})
        if not report.get("pass") or report.get("protocol_status") != "FORMAL":
            continue
        if scope.get("model_id") != model_id or scope.get("model_revision") != model_revision:
            continue
        if scope.get("model_format") != plan.model_format:
            continue
        if scope.get("dtype") != plan.dtype or scope.get("quantization") != plan.quantization:
            continue
        if plan.backend not in scope.get("applicable_backends", []):
            continue
        report["registry_artifact"] = path.relative_to(artifact_root).as_posix()
        return report
    return None
