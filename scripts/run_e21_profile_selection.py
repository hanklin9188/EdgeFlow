#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgeflow.core.models import utc_now  # noqa: E402
from edgeflow.core.serialization import sha256_file, sha256_value, write_json  # noqa: E402
from edgeflow.hardware.inspector import inspect_hardware  # noqa: E402
from edgeflow.models import ModelRegistry  # noqa: E402
from edgeflow.profiler import rank_kernel_candidates  # noqa: E402


def _device_time(event: object) -> float:
    for name in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, name, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def main() -> int:
    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("E21 requires CUDA profiling")
    model_id = "llama-3.2-3b-instruct"
    model_ref, revision = ModelRegistry(ROOT / "specs" / "model_registry.yaml").resolve_source(
        model_id, "safetensors"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_ref,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
        )
        .eval()
        .to("cuda")
    )
    token_ids = torch.tensor(
        [[tokenizer.bos_token_id or 1] + [100 + index % 1000 for index in range(127)]],
        dtype=torch.long,
        device="cuda",
    )
    with torch.inference_mode():
        for _ in range(10):
            model(input_ids=token_ids, use_cache=False, return_dict=True)
    torch.cuda.synchronize()

    output_root = ROOT / "artifacts" / "experiments" / "E21"
    output_root.mkdir(parents=True, exist_ok=True)
    trace_path = output_root / "profile-trace.json"
    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            acc_events=True,
        ) as profiler,
        torch.inference_mode(),
    ):
        for _ in range(10):
            model(input_ids=token_ids, use_cache=False, return_dict=True)
            profiler.step()
    torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    operations = sorted(
        [
            {
                "name": str(event.key),
                "calls": int(event.count),
                "self_device_time_us": _device_time(event),
                "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
            }
            for event in profiler.key_averages()
        ],
        key=lambda row: (-row["self_device_time_us"], row["name"]),
    )
    profile_completed_at = utc_now()
    raw_summary = {
        "schema_version": "1.0",
        "model_id": model_id,
        "model_revision": revision,
        "prompt_tokens": 128,
        "profiled_iterations": 10,
        "source_type": "measured_profile",
        "profile_completed_at": profile_completed_at,
        "operations": operations,
        "trace_sha256": sha256_file(trace_path),
    }
    raw_path = output_root / "profile-summary.json"
    write_json(raw_path, raw_summary)

    # Candidate selection occurs only after the immutable literal summary exists.
    ranked = rank_kernel_candidates(
        operations,
        excluded={"fused-residual-rmsnorm-v1"},
    )
    selected = ranked[0] if ranked else None
    hardware = inspect_hardware(ROOT)
    report = {
        "schema_version": "1.0",
        "experiment_id": "E21",
        "created_at": utc_now(),
        "profile_completed_at": profile_completed_at,
        "profile_before_selection": True,
        "source_type": "measured",
        "status": "PASS" if selected else "INCOMPLETE",
        "pass": selected is not None,
        "model_id": model_id,
        "model_revision": revision,
        "hardware_fingerprint_sha256": hardware["sha256"],
        "profile_summary": str(raw_path.relative_to(ROOT)),
        "profile_summary_sha256": sha256_file(raw_path),
        "ranked_candidates": ranked,
        "selected_candidate": selected,
        "selection_identity_sha256": sha256_value(
            {"profile_summary_sha256": sha256_file(raw_path), "ranked_candidates": ranked}
        ),
        "historical_scope": (
            "This fresh profile selects the next kernel cycle only. It does not retroactively justify the already-completed E24 RMSNorm result."
        ),
        "required_next_steps": [
            "E22 candidate microbenchmark and correctness gate",
            "E23 randomized search-scope intervention",
            "E24 untouched holdout confirmation",
        ],
        "claim_scope": "Profile-grounded candidate hypothesis only; no speedup or causal claim.",
    }
    output = output_root / "result.json"
    write_json(output, report)
    print(json.dumps({"output": str(output), "status": report["status"]}, sort_keys=True))
    del model
    torch.cuda.empty_cache()
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
