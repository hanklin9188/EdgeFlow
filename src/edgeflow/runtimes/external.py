from __future__ import annotations

import importlib.metadata
import json
import shutil
import time
import urllib.error
import urllib.request
from typing import Any

from edgeflow.core.models import CapabilityReport, ExecutionPlan, WorkloadSpec
from edgeflow.runtimes.base import (
    GenerationResult,
    PreparedRuntime,
    RuntimeAdapter,
    RuntimeUnavailable,
)


class _HTTPRuntime(PreparedRuntime):
    def __init__(self, base_url: str, model: str, tokenizer: Any) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tokenizer = tokenizer
        self.load_ms = 0.0
        self.compile_ms = 0.0

    def generate(self, token_ids: list[int], output_tokens: int) -> GenerationResult:
        prompt = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": output_tokens,
                "temperature": 0,
                "stream": True,
                "ignore_eos": True,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter_ns()
        timestamps: list[float] = []
        text_parts: list[str] = []
        token_count = 0
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                item = json.loads(line[6:])
                text = item.get("choices", [{}])[0].get("text", "")
                if text:
                    text_parts.append(text)
                    current_ids = self.tokenizer.encode("".join(text_parts), add_special_tokens=False)
                    current_time = (time.perf_counter_ns() - started) / 1_000_000
                    timestamps.extend([current_time] * max(0, len(current_ids) - token_count))
                    token_count = len(current_ids)
        wall_ms = (time.perf_counter_ns() - started) / 1_000_000
        output_ids = self.tokenizer.encode("".join(text_parts), add_special_tokens=False)
        if len(timestamps) != len(output_ids):
            # A backend that rewrites previous text cannot support token-boundary normalization.
            timestamps = []
        ttft = timestamps[0] if timestamps else None
        tpot = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1) if len(timestamps) > 1 else None
        return GenerationResult(
            tuple(output_ids), tuple(timestamps), wall_ms, ttft, tpot, None, {"protocol": "openai_http"}
        )

    def warmup(self, token_ids: list[int], output_tokens: int) -> GenerationResult:
        return self.generate(token_ids, output_tokens)

    def shutdown(self) -> None:
        return None


class _OpenAIAdapter(RuntimeAdapter):
    package_name: str
    executable: str | None

    def __init__(self, name: str, package_name: str, executable: str | None) -> None:
        self.name = name
        self.package_name = package_name
        self.executable = executable

    def _installed_version(self) -> str | None:
        try:
            return importlib.metadata.version(self.package_name)
        except importlib.metadata.PackageNotFoundError:
            executable_path = shutil.which(self.executable) if self.executable else None
            return f"executable:{executable_path}" if executable_path else None

    def _server_ready(self, base_url: str) -> bool:
        for endpoint in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(base_url.rstrip("/") + endpoint, timeout=1) as response:
                    if response.status < 500:
                        return True
            except (urllib.error.URLError, TimeoutError):
                continue
        return False

    def probe(self) -> CapabilityReport:
        version = self._installed_version()
        return CapabilityReport(
            backend=self.name,
            available=version is not None,
            version=version,
            features={"openai_compatible": True, "managed_process": False},
            reasons=() if version else (f"{self.package_name} / {self.executable} not installed",),
        )

    def prepare(
        self,
        model_ref: str,
        plan: ExecutionPlan,
        workload: WorkloadSpec,
        *,
        local_files_only: bool = True,
    ) -> tuple[_HTTPRuntime, Any]:
        base_url = str(plan.backend_args.get("base_url", "http://127.0.0.1:8000"))
        if not self._server_ready(base_url):
            raise RuntimeUnavailable(f"{self.name} server is not ready at {base_url}")
        from transformers import AutoTokenizer

        tokenizer_ref = str(plan.backend_args.get("tokenizer", model_ref))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, local_files_only=local_files_only)
        served_model = str(plan.backend_args.get("served_model_name", model_ref))
        return _HTTPRuntime(base_url, served_model, tokenizer), tokenizer


class LlamaCppAdapter(_OpenAIAdapter):
    def __init__(self) -> None:
        super().__init__("llama_cpp", "llama-cpp-python", "llama-server")


class VllmAdapter(_OpenAIAdapter):
    def __init__(self) -> None:
        super().__init__("vllm", "vllm", "vllm")
