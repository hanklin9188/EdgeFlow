from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from edgeflow.core.models import CapabilityReport, ExecutionPlan, WorkloadSpec


class RuntimeUnavailable(RuntimeError):
    """Raised when a matrix row targets an unavailable capability."""


@dataclass(frozen=True)
class GenerationResult:
    output_token_ids: tuple[int, ...]
    token_timestamps_ms: tuple[float, ...]
    wall_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    peak_vram_bytes: int | None
    native_metrics: dict[str, Any]


class PreparedRuntime(ABC):
    load_ms: float
    compile_ms: float

    @abstractmethod
    def warmup(self, token_ids: list[int], output_tokens: int) -> GenerationResult: ...

    @abstractmethod
    def generate(self, token_ids: list[int], output_tokens: int) -> GenerationResult: ...

    @abstractmethod
    def shutdown(self) -> None: ...


class RuntimeAdapter(ABC):
    name: str

    @abstractmethod
    def probe(self) -> CapabilityReport: ...

    @abstractmethod
    def prepare(
        self,
        model_ref: str,
        plan: ExecutionPlan,
        workload: WorkloadSpec,
        *,
        local_files_only: bool = True,
    ) -> tuple[PreparedRuntime, Any]:
        """Return a prepared runtime and the tokenizer used for exact token construction."""
