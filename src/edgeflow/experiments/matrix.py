from __future__ import annotations

import hashlib
import itertools
from typing import Any


def matrix_case_label(case_id: str, *, maximum_length: int = 48) -> str:
    """Create a bounded, deterministic, collision-resistant job label."""

    slug = case_id.lower()
    if len(slug) <= maximum_length:
        return slug
    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:8]
    prefix_length = maximum_length - len(digest) - 1
    return f"{slug[:prefix_length]}-{digest}"


def matrix_progress_status(
    cases: list[dict[str, Any]], *, total_case_count: int
) -> tuple[str, bool]:
    """Return a truthful resumable-matrix status and aggregate pass flag."""

    failed = any(row.get("status") == "FAILED" for row in cases)
    running = any(row.get("status") == "RUNNING" for row in cases)
    pruned = any(row.get("status") == "PRUNED" for row in cases)
    settled = all(row.get("status") in {"COMPLETED", "FAILED", "PRUNED"} for row in cases)
    complete = len(cases) == total_case_count and settled
    if running:
        return "RUNNING", False
    if not complete:
        return "PARTIAL", False
    if failed:
        return "COMPLETE_WITH_FAILURES", False
    if pruned:
        return "COMPLETE_WITH_PRUNES", bool(cases)
    return "PASS", bool(cases)


def pytorch_matrix_cases(experiment_id: str, *, quick: bool = False) -> list[dict[str, Any]]:
    """Expand the preregistered E04/E05 matrix without silently dropping failed candidates."""

    if experiment_id == "E04":
        prompts = [128] if quick else [128, 1024, 4096]
        outputs = [32] if quick else [32, 128]
        batches = [1] if quick else [1, 4]
        return [
            {
                "case_id": f"E04-p{prompt}-o{output}-b{batch}",
                "backend": "pytorch_eager",
                "prompt_tokens": prompt,
                "output_tokens": output,
                "batch_size": batch,
                "concurrency": 1,
                "compile_mode": "default",
                "dynamic_shapes": False,
            }
            for prompt, output, batch in itertools.product(prompts, outputs, batches)
        ]
    if experiment_id == "E05":
        prompts = [128] if quick else [128, 1024]
        outputs = [32] if quick else [32, 128]
        modes = (
            ["default"]
            if quick
            else [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        )
        dynamic_values = [False] if quick else [False, True]
        return [
            {
                "case_id": (f"E05-p{prompt}-o{output}-{mode}-{'dynamic' if dynamic else 'static'}"),
                "backend": "torch_compile",
                "prompt_tokens": prompt,
                "output_tokens": output,
                "batch_size": 1,
                "concurrency": 1,
                "compile_mode": mode,
                "dynamic_shapes": dynamic,
            }
            for prompt, output, mode, dynamic in itertools.product(
                prompts, outputs, modes, dynamic_values
            )
        ]
    raise ValueError(f"matrix runner does not implement {experiment_id}")
