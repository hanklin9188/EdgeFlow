#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.copilot.grounded import (  # noqa: E402
    evaluate_grounding_set,
    grounding_source_hashes,
    load_grounding_set,
)
from edgeflow.core.serialization import write_json  # noqa: E402


def main() -> int:
    question_path = ROOT / "datasets" / "grounded_questions.jsonl"
    questions = load_grounding_set(question_path)
    report = evaluate_grounding_set(ROOT, questions)
    report["question_set"] = str(question_path.relative_to(ROOT))
    report["source_sha256"] = grounding_source_hashes(ROOT, questions)
    output = ROOT / "artifacts" / "experiments" / "E28" / "result.json"
    write_json(output, report)
    print(json.dumps({"output": str(output), "status": report["status"]}, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
