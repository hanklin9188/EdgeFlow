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

    def generate_batch(
        self, token_id_batches: list[list[int]], output_tokens: int
    ) -> list[GenerationResult]:
        """Generate one measured request group.

        Adapters must override this method before advertising batch or concurrent
        execution. The sequential default is intentionally limited to one request.
        """

        if len(token_id_batches) != 1:
            raise RuntimeUnavailable("runtime does not implement measured request grouping")
        return [self.generate(token_id_batches[0], output_tokens)]

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
