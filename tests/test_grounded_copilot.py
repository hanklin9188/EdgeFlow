from __future__ import annotations

from pathlib import Path

from edgeflow.copilot.grounded import answer_grounded_question, evaluate_grounding_set
from edgeflow.core.serialization import write_json


def _question(value: object) -> dict[str, object]:
    return {
        "question_id": "GQ-001",
        "question": "What is the value?",
        "expected_status": "MEASURED",
        "expected_value": value,
        "source": "artifacts/result.json",
        "json_pointer": "/metric",
    }


def test_grounded_answer_requires_exact_cited_value(tmp_path: Path) -> None:
    write_json(tmp_path / "artifacts" / "result.json", {"metric": 12.5})
    answer = answer_grounded_question(tmp_path, _question(12.5))
    drifted = answer_grounded_question(tmp_path, _question(13.0))

    assert answer["status"] == "MEASURED"
    assert answer["numeric_claims"] == [12.5]
    assert drifted["status"] == "NOT_AVAILABLE"


def test_grounding_gate_requires_fixed_size_and_zero_unsupported_claims(tmp_path: Path) -> None:
    write_json(tmp_path / "artifacts" / "result.json", {"metric": 12.5})
    questions = [{**_question(12.5), "question_id": f"GQ-{index:03d}"} for index in range(1, 21)]
    report = evaluate_grounding_set(tmp_path, questions)

    assert report["pass"] is True
    assert report["question_count"] == 20
    assert report["unsupported_numeric_claims"] == 0
