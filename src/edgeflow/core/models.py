from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgeflow.core.serialization import sha256_value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceType(StrEnum):
    MEASURED = "measured"
    DEMO = "demo"
    ESTIMATED = "estimated"


class Verdict(StrEnum):
    PASS = "PASS"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"
    SKIPPED = "SKIPPED"


class PromptSource(StrictModel):
    type: Literal["synthetic", "dataset", "trace"]
    revision: str
    name: str | None = None
    split: str | None = None
    sample_ids_sha256: str | None = None


class PromptBucket(StrictModel):
    tokens: int = Field(ge=1)
    probability: float = Field(gt=0, le=1)


class SamplingSpec(StrictModel):
    strategy: Literal["greedy", "sampling"] = "greedy"
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    ignore_eos: bool = True


class WorkloadSpec(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    workload_id: str = Field(min_length=3)
    model_id: str = Field(min_length=1)
    prompt_source: PromptSource
    prompt_tokens: int | list[PromptBucket]
    output_tokens: int = Field(ge=1)
    batch_size: int = Field(default=1, ge=1)
    concurrency: int = Field(default=1, ge=1)
    arrival_pattern: Literal["closed_loop", "poisson", "burst", "trace_replay"] = "closed_loop"
    request_rate: float | None = Field(default=None, gt=0)
    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    seed: int = Field(default=42, ge=0)
    streaming: bool = True
    session_requests: int = Field(default=20, ge=1)
    quality_profile: Literal["strict", "balanced", "memory_first", "custom"] = "balanced"
    notes: str = ""

    @model_validator(mode="after")
    def validate_distribution(self) -> WorkloadSpec:
        if isinstance(self.prompt_tokens, list):
            total = sum(item.probability for item in self.prompt_tokens)
            if abs(total - 1.0) > 1e-6:
                raise ValueError("prompt token bucket probabilities must sum to 1")
        if self.arrival_pattern == "poisson" and self.request_rate is None:
            raise ValueError("Poisson workloads require request_rate")
        return self

    @property
    def content_sha256(self) -> str:
        return sha256_value(self.model_dump(mode="json"))


class ExecutionPlan(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    plan_id: str = Field(min_length=3)
    model_id: str
    backend: Literal["pytorch_eager", "torch_compile", "llama_cpp", "vllm"]
    model_format: Literal["safetensors", "gguf", "fp8", "other"]
    dtype: Literal["fp32", "fp16", "bf16", "fp8"] | None = None
    quantization: str | None = None
    compile_mode: Literal[
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    ] | None = None
    dynamic_shapes: bool | None = None
    fullgraph: bool | None = None
    cuda_graph: bool | None = None
    max_num_batched_tokens: int | None = Field(default=None, ge=1)
    max_num_seqs: int | None = Field(default=None, ge=1)
    kv_cache_dtype: str | None = None
    flash_attention: bool | None = None
    custom_kernels: tuple[str, ...] = ()
    backend_args: dict[str, Any] = Field(default_factory=dict)
    canonical_sha256: str | None = None
    support_status: Literal["verified", "experimental", "pending", "unsupported"] = "pending"

    def canonical_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("canonical_sha256", None)
        payload["custom_kernels"] = sorted(payload["custom_kernels"])
        return payload

    @property
    def content_sha256(self) -> str:
        return sha256_value(self.canonical_payload())

    def with_hash(self) -> ExecutionPlan:
        return self.model_copy(update={"canonical_sha256": self.content_sha256})


class CapabilityReport(StrictModel):
    backend: str
    available: bool
    version: str | None = None
    features: dict[str, bool | str | int | float | None] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()


class MetricValues(StrictModel):
    wall_ms: float | None = Field(default=None, ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    tpot_ms: float | None = Field(default=None, ge=0)
    request_latency_ms: float | None = Field(default=None, ge=0)
    prompt_tokens_per_s: float | None = Field(default=None, ge=0)
    generation_tokens_per_s: float | None = Field(default=None, ge=0)
    requests_per_s: float | None = Field(default=None, ge=0)
    queue_ms: float | None = Field(default=None, ge=0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    gpu_energy_j: float | None = Field(default=None, ge=0)
    temperature_c: float | None = Field(default=None, ge=0)
    sm_clock_mhz: float | None = Field(default=None, ge=0)
    memory_clock_mhz: float | None = Field(default=None, ge=0)


class MetricRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    request_id: str | None
    source_type: SourceType
    phase: Literal[
        "load", "compile", "autotune", "capture", "warmup", "prefill", "decode", "end_to_end", "profile"
    ]
    iteration: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    metrics: MetricValues
    token_timestamps_ms: tuple[float, ...] = ()
    fallbacks: tuple[str, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def timestamps_are_monotonic(self) -> MetricRecord:
        if any(b < a for a, b in pairwise(self.token_timestamps_ms)):
            raise ValueError("token timestamps must be monotonic")
        return self


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
