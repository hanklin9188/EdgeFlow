#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.models import ModelRegistry  # noqa: E402
from edgeflow.quality.llama_cpp import evaluate_llama_cpp_quality  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pinned llama.cpp GGUF quality")
    parser.add_argument("--model-id", default="ministral3-3b-instruct-2512")
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--reference-filename", default="Ministral-3-3B-Instruct-2512-BF16.gguf")
    parser.add_argument("--candidate-filename", default="Ministral-3-3B-Instruct-2512-Q4_K_M.gguf")
    parser.add_argument(
        "--executable",
        type=Path,
        default=ROOT / ".runtime" / "llama.cpp" / "build" / "bin" / "llama-perplexity",
    )
    parser.add_argument("--wikitext-tokens", type=int, default=8192)
    parser.add_argument("--arc-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    arguments = parser.parse_args()

    from huggingface_hub import hf_hub_download

    model_ref, revision = ModelRegistry(ROOT / "specs" / "model_registry.yaml").resolve_source(
        arguments.model_id, "gguf"
    )
    reference_model = Path(
        hf_hub_download(
            model_ref,
            arguments.reference_filename,
            revision=revision,
            local_files_only=not arguments.allow_download,
        )
    )
    candidate_model = Path(
        hf_hub_download(
            model_ref,
            arguments.candidate_filename,
            revision=revision,
            local_files_only=not arguments.allow_download,
        )
    )
    output = evaluate_llama_cpp_quality(
        root=ROOT,
        artifact_root=ROOT / "artifacts",
        executable=arguments.executable,
        model_id=arguments.model_id,
        model_ref=model_ref,
        model_revision=revision,
        reference_model=reference_model,
        candidate_model=candidate_model,
        quantization=arguments.quantization,
        wikitext_token_limit=arguments.wikitext_tokens,
        arc_samples=arguments.arc_samples,
        seed=arguments.seed,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": "PASS" if payload["pass"] else "FAIL"}))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
