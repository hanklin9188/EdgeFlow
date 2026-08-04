from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edgeflow.core.models import utc_now
from edgeflow.core.serialization import sha256_file, sha256_value


def _json_pointer(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    current = payload
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise KeyError(pointer)
    return current


def _source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    allowed = ((root / "artifacts").resolve(), (root / "specs").resolve())
    if not any(path == base or base in path.parents for base in allowed):
        raise ValueError("grounding sources must stay within artifacts/ or specs/")
    return path


def answer_grounded_question(root: Path, question: dict[str, Any]) -> dict[str, Any]:
    """Return only a value that can be resolved from the declared local citation."""

    expected_status = str(question.get("expected_status", "MEASURED"))
    if expected_status == "NOT_AVAILABLE":
        return {
            "question_id": question["question_id"],
            "status": "NOT_AVAILABLE",
            "value": None,
            "answer": "NOT AVAILABLE: the fixed evidence set does not support this claim.",
            "citations": [],
            "numeric_claims": [],
        }
    source = str(question["source"])
    pointer = str(question["json_pointer"])
    try:
        path = _source_path(root, source)
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = _json_pointer(payload, pointer)
    except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return {
            "question_id": question["question_id"],
            "status": "NOT_AVAILABLE",
            "value": None,
            "answer": "NOT AVAILABLE: the cited artifact or field cannot be resolved.",
            "citations": [],
            "numeric_claims": [],
        }
    if value != question.get("expected_value"):
        return {
            "question_id": question["question_id"],
            "status": "NOT_AVAILABLE",
            "value": None,
            "answer": "NOT AVAILABLE: the artifact changed from the fixed expected value.",
            "citations": [{"source": source, "json_pointer": pointer}],
            "numeric_claims": [],
        }
    rendered = json.dumps(value, ensure_ascii=False)
    return {
        "question_id": question["question_id"],
        "status": expected_status,
        "value": value,
        "answer": f"{expected_status}: {rendered}",
        "citations": [{"source": source, "json_pointer": pointer}],
        "numeric_claims": [value]
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else [],
    }


def evaluate_grounding_set(root: Path, questions: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    unsupported_numeric_claims = 0
    citation_failures = 0
    answer_failures = 0
    refusal_failures = 0
    for question in questions:
        answer = answer_grounded_question(root, question)
        expected_status = str(question.get("expected_status", "MEASURED"))
        expected_value = question.get("expected_value")
        value_match = answer["value"] == expected_value
        status_match = answer["status"] == expected_status
        citation_valid = expected_status == "NOT_AVAILABLE"
        if expected_status != "NOT_AVAILABLE" and answer["citations"]:
            citation = answer["citations"][0]
            try:
                payload = json.loads(
                    _source_path(root, citation["source"]).read_text(encoding="utf-8")
                )
                citation_valid = _json_pointer(payload, citation["json_pointer"]) == answer["value"]
            except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
                citation_valid = False
        if not value_match or not status_match:
            answer_failures += 1
        if not citation_valid:
            citation_failures += 1
        if expected_status == "NOT_AVAILABLE" and answer["status"] != "NOT_AVAILABLE":
            refusal_failures += 1
        allowed_numbers = (
            [expected_value]
            if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool)
            else []
        )
        extra_numbers = [
            value for value in answer["numeric_claims"] if value not in allowed_numbers
        ]
        unsupported_numeric_claims += len(extra_numbers)
        results.append(
            {
                "question_id": question["question_id"],
                "status_match": status_match,
                "value_match": value_match,
                "citation_valid": citation_valid,
                "answer": answer,
            }
        )
    formal = len(questions) >= 20
    passed = bool(
        formal
        and answer_failures == 0
        and citation_failures == 0
        and refusal_failures == 0
        and unsupported_numeric_claims == 0
    )
    identity = {
        "questions": questions,
        "results": results,
    }
    return {
        "schema_version": "1.0",
        "experiment_id": "E28",
        "created_at": utc_now(),
        "source_type": "measured",
        "protocol_status": "FORMAL" if formal else "DEVELOPMENT",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "question_count": len(questions),
        "required_question_count": 20,
        "answer_failures": answer_failures,
        "citation_failures": citation_failures,
        "refusal_failures": refusal_failures,
        "unsupported_numeric_claims": unsupported_numeric_claims,
        "question_set_sha256": sha256_value(questions),
        "evaluation_sha256": sha256_value(identity),
        "results": results,
        "claim_scope": (
            "Deterministic answers over the fixed local artifact fields only; no free-form factual extrapolation."
        ),
    }


def load_grounding_set(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def grounding_source_hashes(root: Path, questions: list[dict[str, Any]]) -> dict[str, str]:
    sources = {
        str(question["source"])
        for question in questions
        if question.get("expected_status", "MEASURED") != "NOT_AVAILABLE"
    }
    return {source: sha256_file(_source_path(root, source)) for source in sorted(sources)}
