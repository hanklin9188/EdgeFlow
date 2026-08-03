from __future__ import annotations

from pathlib import Path

import pytest

from edgeflow.models import ModelRegistry
from edgeflow.quality import evaluate_quality
from edgeflow.reports import render_run_report
from edgeflow.validation.correctness import compare_tensors, compare_token_sequences


def test_registry_resolves_only_pinned_sources() -> None:
    repository, revision = ModelRegistry().resolve_source("smollm2-360m-instruct", "safetensors")
    assert repository == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert len(revision) == 40


def test_quality_gate_is_hard_constraint() -> None:
    passing = evaluate_quality(
        reference={"arc_c_accuracy": 0.50, "perplexity": 10.0},
        candidate={"arc_c_accuracy": 0.495, "perplexity": 10.4},
        profile="balanced",
        protocol_match=True,
    )
    assert passing["pass"] is True
    failing = evaluate_quality(
        reference={"arc_c_accuracy": 0.50, "perplexity": 10.0},
        candidate={"arc_c_accuracy": 0.45, "perplexity": 12.0},
        profile="balanced",
        protocol_match=True,
    )
    assert failing["pass"] is False


def test_tensor_and_token_correctness() -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([1.0, 2.0])
    assert compare_tensors(reference, reference.clone(), dtype="fp32")["pass"] is True
    assert compare_token_sequences([1, 2, 3], [1, 9, 3])["first_divergent_token"] == 1


def test_report_is_rebuilt_from_raw_rows(valid_run_dir: Path) -> None:
    from edgeflow.validation import ValidationEngine

    ValidationEngine().validate(valid_run_dir, write=True)
    report = render_run_report(valid_run_dir)
    assert "Request latency median: 10.000 ms" in report
    assert "Source: **MEASURED**" in report
