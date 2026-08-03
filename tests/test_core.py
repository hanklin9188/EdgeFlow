from __future__ import annotations

import pytest

from edgeflow.core.models import ExecutionPlan
from edgeflow.core.serialization import canonical_json, sha256_value
from edgeflow.workloads import create_workload, parse_prompt_distribution


def test_canonical_hash_ignores_dictionary_order() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert sha256_value({"b": 1, "a": 2}) == sha256_value({"a": 2, "b": 1})


def test_plan_hash_ignores_kernel_input_order() -> None:
    base = dict(
        plan_id="plan", model_id="model", backend="pytorch_eager",
        model_format="safetensors", dtype="bf16", backend_args={"b": 2, "a": 1},
    )
    first = ExecutionPlan(**base, custom_kernels=("b", "a"))
    second = ExecutionPlan(**base, custom_kernels=("a", "b"))
    assert first.content_sha256 == second.content_sha256


def test_workload_distribution_requires_unit_probability() -> None:
    buckets = parse_prompt_distribution("512:0.25,1024:0.75")
    assert not isinstance(buckets, int)
    assert sum(bucket.probability for bucket in buckets) == 1.0
    with pytest.raises(ValueError):
        parse_prompt_distribution("512:0.25,1024:0.70")


def test_workload_creation_is_deterministic() -> None:
    arguments = dict(
        workload_id="workload-test", model_id="model", prompt_distribution="128",
        output_tokens=32, seed=7,
    )
    assert create_workload(**arguments).content_sha256 == create_workload(**arguments).content_sha256
