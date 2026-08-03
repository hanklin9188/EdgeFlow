from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgeflow.core.models import (
    ExecutionPlan,
    PromptBucket,
    PromptSource,
    SamplingSpec,
    WorkloadSpec,
)
from edgeflow.models import ModelRegistry


class BenchmarkSubmission(BaseModel):
    """Bounded input accepted by the local control plane.

    The browser selects registered models and typed runtime options. It never sends a
    command line, executable, filesystem output path, or arbitrary environment variable.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        default="local-agent", min_length=3, max_length=48, pattern=r"^[a-z0-9][a-z0-9-]*$"
    )
    model_id: str = Field(min_length=1, max_length=128)
    model_format: Literal["safetensors", "gguf"] = "safetensors"
    backend: Literal["pytorch_eager", "torch_compile", "llama_cpp", "vllm"] = "pytorch_eager"
    prompt_tokens: int | list[PromptBucket] = Field(default=128)
    output_tokens: int = Field(default=32, ge=1, le=4096)
    batch_size: int = Field(default=1, ge=1, le=64)
    concurrency: int = Field(default=1, ge=1, le=64)
    session_requests: int = Field(default=20, ge=1, le=10000)
    quality_profile: Literal["strict", "balanced", "memory_first", "custom"] = "balanced"
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    dtype: Literal["fp32", "fp16", "bf16"] = "bf16"
    compile_mode: Literal[
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    ] = "default"
    dynamic_shapes: bool = False
    fullgraph: bool = False
    cuda_graph: bool = False
    quantization: str | None = Field(default=None, max_length=32)
    external_base_url: str | None = Field(default=None, max_length=200)
    repetitions: int = Field(default=30, ge=1, le=1000)
    warmup_requests: int = Field(default=5, ge=1, le=100)
    experiment_id: str = Field(default="E04", pattern=r"^E\d{2}$")
    allow_download: bool = False
    allow_busy_gpu: bool = False

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> BenchmarkSubmission:
        if isinstance(self.prompt_tokens, int):
            if not 1 <= self.prompt_tokens <= 131072:
                raise ValueError("prompt_tokens must be between 1 and 131072")
        else:
            total = sum(item.probability for item in self.prompt_tokens)
            if not self.prompt_tokens or abs(total - 1.0) > 1e-6:
                raise ValueError("prompt bucket probabilities must sum to 1")
            if any(item.tokens > 131072 for item in self.prompt_tokens):
                raise ValueError("prompt token buckets cannot exceed 131072")
        if self.backend == "llama_cpp" and self.model_format != "gguf":
            raise ValueError("llama.cpp jobs require a registered GGUF source")
        if (
            self.backend in {"pytorch_eager", "torch_compile", "vllm"}
            and self.model_format != "safetensors"
        ):
            raise ValueError(f"{self.backend} jobs require a safetensors source")
        if self.backend == "torch_compile" and self.compile_mode == "reduce-overhead":
            raise ValueError(
                "reduce-overhead enables an internal CUDA Graph path that is incompatible with "
                "this adapter's mutable token-by-token KV cache"
            )
        if self.backend in {"pytorch_eager", "torch_compile"} and self.concurrency != 1:
            raise ValueError("PyTorch runtimes use batch_size; concurrency must be 1")
        if self.backend in {"llama_cpp", "vllm"} and self.batch_size != 1:
            raise ValueError("HTTP runtimes use concurrency; batch_size must be 1")
        if self.backend in {"llama_cpp", "vllm"}:
            base_url = self.external_base_url or (
                "http://127.0.0.1:8001"
                if self.backend == "llama_cpp"
                else "http://127.0.0.1:8002"
            )
            parsed = urlparse(base_url)
            if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("external runtime servers must use a loopback HTTP URL")
        return self

    def workload(self) -> WorkloadSpec:
        prompt_slug = (
            str(self.prompt_tokens)
            if isinstance(self.prompt_tokens, int)
            else "mix-" + "-".join(str(item.tokens) for item in self.prompt_tokens)
        )
        return WorkloadSpec(
            workload_id=(
                f"{self.label}-p{prompt_slug}-o{self.output_tokens}"
                f"-b{self.batch_size}-c{self.concurrency}"
            ),
            model_id=self.model_id,
            prompt_source=PromptSource(type="synthetic", name="edgeflow-corpus-v1", revision="1.0"),
            prompt_tokens=self.prompt_tokens,
            output_tokens=self.output_tokens,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            sampling=SamplingSpec(),
            seed=self.seed,
            streaming=True,
            session_requests=self.session_requests,
            quality_profile=self.quality_profile,
            notes="Created by the localhost-only EdgeFlow control plane.",
        )

    def resolve(self, registry: ModelRegistry) -> tuple[str, str, ExecutionPlan]:
        model_ref, revision = registry.resolve_source(self.model_id, self.model_format)
        support = registry.support(self.model_id, self.backend)
        if support == "unsupported":
            raise ValueError(f"{self.model_id} does not support backend {self.backend}")
        backend_args: dict[str, str | bool] = {
            "revision": revision,
            "trust_remote_code": False,
        }
        if self.backend in {"llama_cpp", "vllm"}:
            base_url = self.external_base_url or (
                "http://127.0.0.1:8001"
                if self.backend == "llama_cpp"
                else "http://127.0.0.1:8002"
            )
            tokenizer_ref = model_ref
            tokenizer_revision = revision
            if self.backend == "llama_cpp":
                tokenizer_ref, tokenizer_revision = registry.resolve_source(
                    self.model_id, "safetensors"
                )
            backend_args.update(
                {
                    "base_url": base_url,
                    "tokenizer": tokenizer_ref,
                    "tokenizer_revision": tokenizer_revision,
                }
            )
        safe_label = re.sub(r"[^a-z0-9-]", "-", self.label.lower())
        plan = ExecutionPlan(
            plan_id=f"{safe_label}-{self.backend}-{self.dtype}",
            model_id=self.model_id,
            backend=self.backend,
            model_format=self.model_format,
            dtype=self.dtype,
            quantization=self.quantization,
            compile_mode=self.compile_mode if self.backend == "torch_compile" else None,
            dynamic_shapes=self.dynamic_shapes if self.backend == "torch_compile" else None,
            fullgraph=self.fullgraph if self.backend == "torch_compile" else None,
            cuda_graph=self.cuda_graph if self.backend == "torch_compile" else None,
            backend_args=backend_args,
            support_status=("verified" if support in {"required", "verified"} else "experimental"),
        ).with_hash()
        return model_ref, revision, plan
