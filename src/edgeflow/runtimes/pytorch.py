from __future__ import annotations

import time
from typing import Any

from edgeflow.core.models import CapabilityReport, ExecutionPlan, WorkloadSpec
from edgeflow.runtimes.base import (
    GenerationResult,
    PreparedRuntime,
    RuntimeAdapter,
    RuntimeUnavailable,
)


class PytorchRuntime(PreparedRuntime):
    def __init__(
        self,
        model: Any,
        *,
        load_ms: float,
        compile_ms: float,
        compiled: bool,
        cuda_graph: bool,
    ) -> None:
        self.model = model
        self.load_ms = load_ms
        self.compile_ms = compile_ms
        self.compiled = compiled
        self.cuda_graph = cuda_graph

    def _synchronize(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def generate(self, token_ids: list[int], output_tokens: int) -> GenerationResult:
        return self.generate_batch([token_ids], output_tokens)[0]

    def generate_batch(
        self, token_id_batches: list[list[int]], output_tokens: int
    ) -> list[GenerationResult]:
        import torch

        if output_tokens < 1:
            raise ValueError("output_tokens must be positive")
        if not token_id_batches:
            raise ValueError("token_id_batches cannot be empty")
        prompt_lengths = {len(token_ids) for token_ids in token_id_batches}
        if len(prompt_lengths) != 1:
            raise ValueError("PyTorch measured batches require equal prompt lengths")
        device = next(self.model.parameters()).device
        inputs = torch.tensor(token_id_batches, dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self._synchronize()
        started = time.perf_counter_ns()
        token_events: list[Any] = []
        generated_steps: list[Any] = []
        engine_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        engine_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if engine_start is not None:
            engine_start.record()
        past_key_values = None
        if self.compiled or self.cuda_graph:
            from transformers.cache_utils import StaticCache

            model_config = getattr(self.model, "config", None) or self.model._orig_mod.config
            past_key_values = StaticCache(
                config=model_config,
                max_cache_len=len(token_id_batches[0]) + output_tokens,
            )
        current = inputs
        with torch.inference_mode():
            for _ in range(output_tokens):
                if self.cuda_graph and hasattr(torch, "compiler"):
                    torch.compiler.cudagraph_mark_step_begin()
                outputs = self.model(
                    input_ids=current,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                past_key_values = outputs.past_key_values
                generated_steps.append(next_token)
                if device.type == "cuda":
                    token_event = torch.cuda.Event(enable_timing=True)
                    token_event.record()
                    token_events.append(token_event)
                current = next_token
        if engine_end is not None and engine_start is not None:
            engine_end.record()
            engine_end.synchronize()
            wall_ms = float(engine_start.elapsed_time(engine_end))
            timestamps = [float(engine_start.elapsed_time(event)) for event in token_events]
        else:
            self._synchronize()
            wall_ms = (time.perf_counter_ns() - started) / 1_000_000
            timestamps = []
        host_wall_ms = (time.perf_counter_ns() - started) / 1_000_000
        generated = torch.cat(generated_steps, dim=1).tolist()
        ttft_ms = timestamps[0] if timestamps else None
        tpot_ms = None
        if len(timestamps) > 1:
            tpot_ms = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        return [
            GenerationResult(
                output_token_ids=tuple(output_ids),
                token_timestamps_ms=tuple(timestamps),
                wall_ms=wall_ms,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                peak_vram_bytes=peak,
                native_metrics={
                    "device": str(device),
                    "cache_enabled": True,
                    "measured_batch_size": len(token_id_batches),
                    "timer": "cuda_event" if device.type == "cuda" else "synchronized_perf_counter",
                    "host_wall_ms": host_wall_ms,
                },
            )
            for output_ids in generated
        ]

    def warmup(self, token_ids: list[int], output_tokens: int) -> GenerationResult:
        return self.generate(token_ids, output_tokens)

    def shutdown(self) -> None:
        try:
            import torch

            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, AttributeError):
            pass


class PytorchAdapter(RuntimeAdapter):
    def __init__(self, *, compiled: bool = False) -> None:
        self.compiled = compiled
        self.name = "torch_compile" if compiled else "pytorch_eager"

    def probe(self) -> CapabilityReport:
        try:
            import torch
            import transformers
        except ImportError as exc:
            return CapabilityReport(backend=self.name, available=False, reasons=(str(exc),))
        available = torch.cuda.is_available()
        reasons = () if available else ("torch.cuda.is_available() is false",)
        return CapabilityReport(
            backend=self.name,
            available=available,
            version=str(torch.__version__),
            features={
                "cuda": str(torch.version.cuda),
                "transformers": str(transformers.__version__),
                "compile": hasattr(torch, "compile"),
                "bf16": bool(available and torch.cuda.is_bf16_supported()),
                "manual_decode_cudagraph": False,
            },
            reasons=reasons,
        )

    def prepare(
        self,
        model_ref: str,
        plan: ExecutionPlan,
        workload: WorkloadSpec,
        *,
        local_files_only: bool = True,
    ) -> tuple[PytorchRuntime, Any]:
        report = self.probe()
        if not report.available:
            raise RuntimeUnavailable("; ".join(report.reasons))
        if plan.backend != self.name:
            raise ValueError(f"Plan backend {plan.backend} does not match adapter {self.name}")
        if self.compiled and plan.compile_mode == "reduce-overhead" and plan.cuda_graph:
            raise RuntimeUnavailable(
                "manual token-by-token decode with a mutated KV cache is not CUDA-Graph-safe in this "
                "adapter; use compile_mode=default or max-autotune-no-cudagraphs"
            )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(plan.dtype or "bf16", torch.bfloat16)
        trust_remote_code = bool(plan.backend_args.get("trust_remote_code", False))
        started = time.perf_counter_ns()
        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            revision=plan.backend_args.get("revision"),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        revision = plan.backend_args.get("revision")
        if revision:
            model_kwargs["revision"] = revision
        attention = plan.backend_args.get("attn_implementation")
        if attention:
            model_kwargs["attn_implementation"] = attention
        model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs).eval().to("cuda")
        torch.cuda.synchronize()
        load_ms = (time.perf_counter_ns() - started) / 1_000_000
        compile_ms = 0.0
        if self.compiled:
            if not hasattr(torch, "compile"):
                raise RuntimeUnavailable("torch.compile is unavailable")
            compile_started = time.perf_counter_ns()
            model = torch.compile(
                model,
                mode=plan.compile_mode or "default",
                fullgraph=bool(plan.fullgraph),
                dynamic=plan.dynamic_shapes,
            )
            compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000
        return PytorchRuntime(
            model,
            load_ms=load_ms,
            compile_ms=compile_ms,
            compiled=self.compiled,
            cuda_graph=bool(plan.cuda_graph),
        ), tokenizer
