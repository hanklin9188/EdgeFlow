from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edgeflow.api.app import app
from edgeflow.kernels.rmsnorm.reference import reference_residual_rmsnorm
from edgeflow.storage import EdgeFlowDB


def test_database_migration_is_idempotent(tmp_path: Path) -> None:
    first = EdgeFlowDB(tmp_path / "runs.sqlite")
    second = EdgeFlowDB(tmp_path / "runs.sqlite")
    assert first.list_runs() == second.list_runs() == []


def test_api_health_and_comparison() -> None:
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.post(
        "/api/v1/compare",
        json={
            "protocol_a": "engine", "protocol_b": "engine",
            "source_type_a": "measured", "source_type_b": "measured",
            "baseline": [10 + index * 0.1 for index in range(30)],
            "intervention": [9 + index * 0.09 for index in range(30)],
        },
    )
    assert response.status_code == 200
    assert response.json()["claim_direction_supported"] is True


def test_api_rejects_incompatible_protocols() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/compare",
        json={"protocol_a": "engine", "protocol_b": "e2e", "baseline": [1, 2], "intervention": [1, 2]},
    )
    assert response.status_code == 422


def test_reference_kernel_cpu_contract() -> None:
    torch = pytest.importorskip("torch")
    x = torch.randn(4, 17)
    residual = torch.randn_like(x)
    weight = torch.randn(17)
    output = reference_residual_rmsnorm(x, residual, weight)
    expected = (x + residual) * torch.rsqrt((x + residual).float().square().mean(-1, keepdim=True) + 1e-6) * weight
    torch.testing.assert_close(output, expected)
