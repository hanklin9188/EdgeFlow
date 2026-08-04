from __future__ import annotations

from edgeflow.experiments.foundation import evaluate_thermal_block


def _telemetry(*, temperature: float, clock: float, utilization: float) -> dict[str, float]:
    return {
        "temperature_c": temperature,
        "sm_clock_mhz": clock,
        "power_w": 200.0,
        "utilization_pct": utilization,
    }


def test_thermal_block_requires_every_registered_precondition() -> None:
    stable = [100.0 + (index % 3) * 0.1 for index in range(1_000)]
    background = [150.0 + (index % 7) for index in range(1_000)]
    result = evaluate_thermal_block(
        stabilized=stable,
        background=background,
        stabilized_before=_telemetry(temperature=60.0, clock=2_760.0, utilization=100.0),
        stabilized_after=_telemetry(temperature=62.0, clock=2_730.0, utilization=100.0),
        pre_run=_telemetry(temperature=40.0, clock=210.0, utilization=0.0),
        required_iterations=1_000,
    )

    assert result["pass"] is True
    assert all(result["checks"].values())


def test_thermal_block_rejects_busy_pre_run_or_short_protocol() -> None:
    result = evaluate_thermal_block(
        stabilized=[100.0] * 999,
        background=[150.0] * 999,
        stabilized_before=_telemetry(temperature=60.0, clock=2_760.0, utilization=100.0),
        stabilized_after=_telemetry(temperature=60.0, clock=2_760.0, utilization=100.0),
        pre_run=_telemetry(temperature=50.0, clock=2_000.0, utilization=8.0),
        required_iterations=1_000,
    )

    assert result["pass"] is False
    assert result["checks"]["protocol_iterations"] is False
    assert result["checks"]["pre_run_gpu_idle"] is False
