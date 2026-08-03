from __future__ import annotations

from typer.testing import CliRunner

from edgeflow.cli.app import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_workload_create_json() -> None:
    result = runner.invoke(
        app,
        [
            "workload", "create", "--model", "smoke", "--prompt-distribution", "32",
            "--output", "4", "--session-requests", "5",
        ],
    )
    assert result.exit_code == 0
    assert '"workload_id"' in result.stdout
