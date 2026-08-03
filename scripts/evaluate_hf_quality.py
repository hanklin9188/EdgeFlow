#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.models import ModelRegistry  # noqa: E402
from edgeflow.quality import evaluate_hf_reference_quality  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the pinned local Transformers quality reference"
    )
    parser.add_argument("--model-id", default="llama-3.2-3b-instruct")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--wikitext-tokens", type=int, default=8192)
    parser.add_argument("--arc-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    arguments = parser.parse_args()
    registry = ModelRegistry(ROOT / "specs" / "model_registry.yaml")
    model_ref, revision = registry.resolve_source(arguments.model_id, "safetensors")
    output = evaluate_hf_reference_quality(
        root=ROOT,
        artifact_root=ROOT / "artifacts",
        model_id=arguments.model_id,
        model_ref=model_ref,
        model_revision=revision,
        dtype=arguments.dtype,
        wikitext_token_limit=arguments.wikitext_tokens,
        arc_samples=arguments.arc_samples,
        seed=arguments.seed,
        local_files_only=not arguments.allow_download,
    )
    print(json.dumps({"output": str(output), "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
