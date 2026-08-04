from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from edgeflow.core.models import CapabilityReport, ExecutionPlan, WorkloadSpec
from edgeflow.core.serialization import project_root
from edgeflow.runtimes.base import (
    GenerationResult,
    PreparedRuntime,
    RuntimeAdapter,
    RuntimeUnavailable,
)


class _HTTPRuntime(PreparedRuntime):
    def __init__(
        self,
        base_url: str,
        model: str,
        tokenizer: Any,
        api_key: str | None = None,
        *,
        exact_token_prompts: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tokenizer = tokenizer
        self.api_key = api_key
        self.exact_token_prompts = exact_token_prompts
        self.load_ms = 0.0
        self.compile_ms = 0.0

    def generate(self, token_ids: list[int], output_tokens: int) -> GenerationResult:
        # OpenAI-compatible servers apply their model's BOS policy. Removing the
        # synthetic leading BOS here avoids a double-BOS prompt while preserving
        # the requested token count once the server adds its own special token.
        prompt: str | list[int] = (
            token_ids
            if self.exact_token_prompts
            else self.tokenizer.decode(token_ids, skip_special_tokens=True)
        )
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": output_tokens,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": True,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        started = time.perf_counter_ns()
        streamed_parts: list[tuple[str, float]] = []
        reported_output_tokens: int | None = None
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                item = json.loads(line[6:])
                usage = item.get("usage") or {}
                if isinstance(usage.get("completion_tokens"), int):
                    reported_output_tokens = int(usage["completion_tokens"])
                text = item.get("choices", [{}])[0].get("text", "")
                if text:
                    current_time = (time.perf_counter_ns() - started) / 1_000_000
                    streamed_parts.append((text, current_time))
        wall_ms = (time.perf_counter_ns() - started) / 1_000_000

        # Reconstruct token boundaries only after the timed response has closed.
        # Re-tokenizing the complete prefix for every streamed chunk is O(n^2)
        # client work and used to contaminate engine latency and its stability
        # estimate. Deferring the identical reconstruction preserves TTFT/TPOT
        # evidence without charging host-side reporting work to the runtime.
        output_text = "".join(text for text, _ in streamed_parts)
        output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
        server_output_tokens = reported_output_tokens or len(output_ids)
        if len(streamed_parts) == server_output_tokens:
            # vLLM and llama.cpp normally emit one non-empty delta per token.
            # This is an exact O(n) mapping and avoids 512 repeated tokenizer
            # passes for a 512-token completion.
            timestamps = [current_time for _, current_time in streamed_parts]
        else:
            # A backend may buffer multiple byte-level tokens into one Unicode
            # chunk. Reconstruct those uncommon boundaries conservatively.
            text_parts: list[str] = []
            timestamps = []
            token_count = 0
            for text, current_time in streamed_parts:
                text_parts.append(text)
                current_ids = self.tokenizer.encode("".join(text_parts), add_special_tokens=False)
                timestamps.extend([current_time] * max(0, len(current_ids) - token_count))
                token_count = len(current_ids)
        if len(timestamps) != server_output_tokens:
            # A backend that rewrites previous text cannot support token-boundary normalization.
            timestamps = []
        ttft = timestamps[0] if timestamps else None
        tpot = (
            (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
            if len(timestamps) > 1
            else None
        )
        return GenerationResult(
            tuple(output_ids),
            tuple(timestamps),
            wall_ms,
            ttft,
            tpot,
            None,
            {
                "protocol": "openai_http",
                "prompt_transport": "exact_token_ids" if self.exact_token_prompts else "text",
                "timestamp_reconstruction": (
                    "direct_stream_delta"
                    if len(streamed_parts) == server_output_tokens
                    else "retokenized_prefix_fallback"
                ),
            },
            server_output_tokens,
            hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        )

    def warmup(self, token_ids: list[int], output_tokens: int) -> GenerationResult:
        return self.generate(token_ids, output_tokens)

    def generate_batch(
        self, token_id_batches: list[list[int]], output_tokens: int
    ) -> list[GenerationResult]:
        if not token_id_batches:
            raise ValueError("token_id_batches cannot be empty")
        with ThreadPoolExecutor(max_workers=len(token_id_batches)) as executor:
            futures = [
                executor.submit(self.generate, token_ids, output_tokens)
                for token_ids in token_id_batches
            ]
            return [future.result() for future in futures]

    def shutdown(self) -> None:
        return None


class _OpenAIAdapter(RuntimeAdapter):
    package_name: str
    executable: str | None

    def __init__(self, name: str, package_name: str, executable: str | None) -> None:
        self.name = name
        self.package_name = package_name
        self.executable = executable

    def _local_executable(self) -> str | None:
        environment_name = (
            "EDGEFLOW_LLAMA_CPP_SERVER" if self.name == "llama_cpp" else "EDGEFLOW_VLLM_EXECUTABLE"
        )
        configured = os.environ.get(environment_name)
        candidates = [Path(configured).expanduser()] if configured else []
        root = project_root()
        if self.name == "llama_cpp":
            candidates.append(root / ".runtime" / "llama.cpp" / "build" / "bin" / "llama-server")
        else:
            candidates.append(root / ".runtime" / "vllm" / ".venv" / "bin" / "vllm")
        located = shutil.which(self.executable) if self.executable else None
        if located:
            candidates.append(Path(located))
        return next(
            (
                str(path.resolve())
                for path in candidates
                if path.is_file() and os.access(path, os.X_OK)
            ),
            None,
        )

    def _installed_version(self) -> tuple[str | None, str | None]:
        try:
            return importlib.metadata.version(self.package_name), self._local_executable()
        except importlib.metadata.PackageNotFoundError:
            executable_path = self._local_executable()
            return (f"isolated:{executable_path}" if executable_path else None), executable_path

    @staticmethod
    def _request(base_url: str, endpoint: str, api_key: str | None = None) -> Any:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        request = urllib.request.Request(base_url.rstrip("/") + endpoint, headers=headers)
        return urllib.request.urlopen(request, timeout=3)

    def _server_ready(self, base_url: str, api_key: str | None = None) -> bool:
        for endpoint in ("/health", "/v1/models"):
            try:
                with self._request(base_url, endpoint, api_key) as response:
                    if response.status < 500:
                        return True
            except (urllib.error.URLError, TimeoutError):
                continue
        return False

    @staticmethod
    def _require_loopback(base_url: str) -> str:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeUnavailable("external runtime servers must use a loopback HTTP URL")
        return base_url.rstrip("/")

    def _served_model(self, base_url: str, fallback: str, api_key: str | None = None) -> str:
        """Resolve the server's actual model id instead of assuming its source repo name."""
        try:
            with self._request(base_url, "/v1/models", api_key) as response:
                payload = json.load(response)
            models = payload.get("data", [])
            if models and isinstance(models[0].get("id"), str):
                return str(models[0]["id"])
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError):
            pass
        return fallback

    def probe(self) -> CapabilityReport:
        version, executable_path = self._installed_version()
        return CapabilityReport(
            backend=self.name,
            available=version is not None,
            version=version,
            features={
                "openai_compatible": True,
                "managed_process": False,
                "isolated_executable": executable_path,
                "loopback_only": True,
            },
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
        default_port = 8001 if self.name == "llama_cpp" else 8002
        base_url = self._require_loopback(
            str(plan.backend_args.get("base_url", f"http://127.0.0.1:{default_port}"))
        )
        api_key = os.environ.get("EDGEFLOW_RUNTIME_API_KEY")
        if not self._server_ready(base_url, api_key):
            raise RuntimeUnavailable(f"{self.name} server is not ready at {base_url}")
        from transformers import AutoConfig, AutoTokenizer

        tokenizer_ref = str(plan.backend_args.get("tokenizer", model_ref))
        tokenizer_revision = plan.backend_args.get("tokenizer_revision")
        config = AutoConfig.from_pretrained(
            tokenizer_ref,
            revision=str(tokenizer_revision) if tokenizer_revision else None,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_ref,
            revision=str(tokenizer_revision) if tokenizer_revision else None,
            local_files_only=local_files_only,
            trust_remote_code=False,
            fix_mistral_regex=config.model_type == "mistral3",
        )
        configured_model = str(plan.backend_args.get("served_model_name", "")).strip()
        served_model = configured_model or self._served_model(base_url, model_ref, api_key)
        return _HTTPRuntime(
            base_url,
            served_model,
            tokenizer,
            api_key,
            # Both registered servers accept token-id arrays on /v1/completions.
            # Text round-trips can change the prompt length and are not eligible for
            # an exact cross-runtime workload comparison.
            exact_token_prompts=True,
        ), tokenizer


class LlamaCppAdapter(_OpenAIAdapter):
    def __init__(self) -> None:
        super().__init__("llama_cpp", "llama-cpp-python", "llama-server")


class VllmAdapter(_OpenAIAdapter):
    def __init__(self) -> None:
        super().__init__("vllm", "vllm", "vllm")
