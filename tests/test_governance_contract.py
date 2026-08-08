from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "specs" / "validation_requirements.yaml"


def load_matrix() -> dict:
    return yaml.safe_load(MATRIX.read_text(encoding="utf-8"))


def test_validation_requirement_ids_are_unique() -> None:
    requirements = load_matrix()["requirements"]
    counts = Counter(item["id"] for item in requirements)
    assert [item for item, count in counts.items() if count > 1] == []


def test_implemented_rules_name_real_enforcement_and_test_paths() -> None:
    requirements = load_matrix()["requirements"]
    implemented = [item for item in requirements if item["status"] == "implemented"]
    assert implemented
    for item in implemented:
        implementation = item["implementation"]
        assert implementation["enforcement_paths"], item["id"]
        assert implementation["test_paths"], item["id"]
        assert item["acceptance"]["executable"] is True
        for relative in implementation["enforcement_paths"] + implementation["test_paths"]:
            assert (ROOT / relative).exists(), f"{item['id']}: {relative}"


def test_unfinished_rules_have_a_backlog_owner() -> None:
    requirements = load_matrix()["requirements"]
    unfinished = [item for item in requirements if item["status"] in {"partial", "planned"}]
    assert unfinished
    for item in unfinished:
        assert item["backlog_ticket"], item["id"]
        assert item["owner"], item["id"]


def test_every_hard_rule_blocks_at_least_one_release_profile() -> None:
    data = load_matrix()
    profiles = set(data["release_profiles"])
    hard = [item for item in data["requirements"] if item["kind"] == "hard"]
    for item in hard:
        assert item["release_gates"], item["id"]
        assert set(item["release_gates"]) <= profiles, item["id"]
        assert item["acceptance"]["criterion"].strip(), item["id"]
