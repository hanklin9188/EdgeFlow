from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edgeflow.api.app import app
from edgeflow.kernels.rmsnorm import LlamaRMSNormIntegration, dispatch_decision
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


def test_llama_integration_matches_model_and_rolls_back_on_reference_path() -> None:
    torch = pytest.importorskip("torch")

    class RMSNorm(torch.nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden))
            self.variance_epsilon = 1e-6

        def forward(self, values):
            normalized = values.float()
            normalized = normalized * torch.rsqrt(
                normalized.square().mean(-1, keepdim=True) + self.variance_epsilon
            )
            return self.weight * normalized.to(values.dtype)

    class Attention(torch.nn.Module):
        def forward(self, *, hidden_states, **_kwargs):
            return hidden_states * 0.25, None

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = RMSNorm(8)
            self.self_attn = Attention()
            self.post_attention_layernorm = RMSNorm(8)
            self.mlp = torch.nn.Linear(8, 8, bias=False)

        def forward(self, hidden_states, **kwargs):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(hidden_states=hidden_states, **kwargs)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            return residual + self.mlp(hidden_states)

    class Base(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer()])

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Base()
            self.config = type("Config", (), {"model_type": "llama"})()

        def forward(self, values):
            for layer in self.model.layers:
                values = layer(values)
            return values

    model = Model().eval()
    values = torch.randn(2, 3, 8)
    baseline = model(values)
    integration = LlamaRMSNormIntegration(model).enable()
    actual = model(values)

    torch.testing.assert_close(actual, baseline, atol=1e-5, rtol=1e-5)
    assert integration.summary()["fallback_calls"] == 2
    assert integration.summary()["cached_dispatch_scope_count"] == 1
    assert dispatch_decision(values.reshape(-1, 8), values.reshape(-1, 8), torch.ones(8))[
        "reason"
    ] == "cpu_reference"
    integration.disable()
    torch.testing.assert_close(model(values), baseline)
