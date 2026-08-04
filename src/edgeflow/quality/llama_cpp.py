from __future__ import annotations

import math
import random
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from edgeflow.core.models import utc_now
from edgeflow.core.serialization import sha256_file, sha256_value, write_json
from edgeflow.quality.gates import evaluate_quality
from edgeflow.quality.hf_reference import _dataset_pin

_PPL_PATTERN = re.compile(r"Final estimate: PPL =\s*([0-9]+(?:\.[0-9]+)?)")
_PPL_PROGRESS_PATTERN = re.compile(r"\[\d+\]([0-9]+(?:\.[0-9]+)?),")
_ARC_PATTERN = re.compile(r"Final result:\s*([0-9]+(?:\.[0-9]+)?)\s*\+/-")
_TOKEN_PATTERN = re.compile(r"have\s+(\d+)\s+tokens")
_CHUNK_PATTERN = re.compile(r"computing over\s+(\d+)\s+chunks,\s*n_ctx=(\d+)")


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def build_multiple_choice_payload(rows: list[dict[str, Any]]) -> bytes:
    """Serialize fixed ARC rows in llama-perplexity's documented task format."""

    tasks: list[bytes] = []
    for row in rows:
        labels = [str(value) for value in row["choices"]["label"]]
        choices = [str(value) for value in row["choices"]["text"]]
        answer = str(row["answerKey"])
        if answer not in labels or not choices or len(labels) != len(choices):
            raise ValueError(f"invalid ARC multiple-choice item: {row.get('id', '')}")
        task = bytearray(_pack_string(f"Question: {row['question']}\nAnswer:"))
        task.extend(struct.pack("<I", len(choices)))
        for choice in choices:
            task.extend(_pack_string(choice))
        task.extend(struct.pack(f"<{len(choices)}i", *[int(label == answer) for label in labels]))
        task.extend(struct.pack("<I", 0))  # mc2 answers are intentionally empty.
        tasks.append(bytes(task))

    header_size = 4 + 4 * len(tasks)
    offsets: list[int] = []
    position = header_size
    for task in tasks:
        offsets.append(position)
        position += len(task)
    return (
        struct.pack("<I", len(tasks)) + struct.pack(f"<{len(offsets)}I", *offsets) + b"".join(tasks)
    )


def parse_llama_perplexity(output: str) -> dict[str, int | float]:
    ppl = _PPL_PATTERN.search(output)
    progress = _PPL_PROGRESS_PATTERN.findall(output)
    token_count = _TOKEN_PATTERN.search(output)
    chunk_count = _CHUNK_PATTERN.search(output)
    if (not ppl and not progress) or not token_count or not chunk_count:
        tail = "\n".join(output.splitlines()[-40:])
        raise ValueError(
            "llama-perplexity output omitted the final PPL protocol counters:\n" + tail
        )
    return {
        "perplexity": float(ppl.group(1) if ppl else progress[-1]),
        "input_tokens": int(token_count.group(1)),
        "chunks": int(chunk_count.group(1)),
        "context_size": int(chunk_count.group(2)),
    }


def parse_llama_multiple_choice(output: str) -> float:
    match = _ARC_PATTERN.search(output)
    if not match:
        raise ValueError("llama-perplexity output omitted the final multiple-choice result")
    return float(match.group(1)) / 100.0


def _run(executable: Path, arguments: list[str]) -> str:
    completed = subprocess.run(
        [str(executable), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=3600,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(f"llama-perplexity failed with exit {completed.returncode}:\n{tail}")
    return completed.stdout


def _measure_model(
    *,
    executable: Path,
    model_path: Path,
    wiki_path: Path,
    arc_path: Path,
    chunks: int,
) -> tuple[dict[str, float], dict[str, int | float]]:
    common = [
        "--model",
        str(model_path),
        "--gpu-layers",
        "all",
        "--flash-attn",
        "on",
        "--no-warmup",
    ]
    ppl_output = _run(
        executable,
        [
            *common,
            "--file",
            str(wiki_path),
            "--ctx-size",
            "512",
            "--ppl-stride",
            "511",
            "--chunks",
            str(chunks),
            "--batch-size",
            "512",
            "--ubatch-size",
            "512",
        ],
    )
    ppl = parse_llama_perplexity(ppl_output)
    arc_output = _run(
        executable,
        [
            *common,
            "--file",
            str(arc_path),
            "--multiple-choice",
            "--multiple-choice-tasks",
            "0",
            "--ctx-size",
            "512",
            "--batch-size",
            "256",
            "--ubatch-size",
            "256",
            "--parallel",
            "16",
        ],
    )
    return (
        {
            "perplexity": float(ppl["perplexity"]),
            "arc_c_accuracy": parse_llama_multiple_choice(arc_output),
        },
        ppl,
    )


def evaluate_llama_cpp_quality(
    *,
    root: Path,
    artifact_root: Path,
    executable: Path,
    model_id: str,
    model_ref: str,
    model_revision: str,
    reference_model: Path,
    candidate_model: Path,
    quantization: str,
    wikitext_token_limit: int = 8192,
    arc_samples: int = 50,
    seed: int = 42,
) -> Path:
    """Measure GGUF BF16 and a quantized candidate with one pinned llama.cpp protocol."""

    if not executable.is_file() or not reference_model.is_file() or not candidate_model.is_file():
        raise FileNotFoundError("llama.cpp executable and both GGUF files must exist locally")
    if wikitext_token_limit < 2 or arc_samples < 1:
        raise ValueError("quality sample sizes must be positive")
    from datasets import load_dataset

    wiki = _dataset_pin(root, "wikitext-2-raw")
    arc = _dataset_pin(root, "arc-challenge")
    wiki_rows = load_dataset(
        wiki["source"], wiki.get("config"), revision=wiki["revision"], split=wiki["split"]
    )
    # llama.cpp tokenizes the shared raw text.  A margin guarantees at least the
    # requested number of scored tokens even when tokenizer implementations differ
    # slightly; exact counters from each run are checked below.
    nonempty = [str(row["text"]) for row in wiki_rows if str(row["text"]).strip()]
    text = "\n\n".join(nonempty)
    chunks = max(1, math.ceil(wikitext_token_limit / 511))

    arc_rows = load_dataset(
        arc["source"], arc.get("config"), revision=arc["revision"], split=arc["split"]
    )
    if arc_samples > len(arc_rows):
        raise ValueError(f"ARC sample count {arc_samples} exceeds split size {len(arc_rows)}")
    indices = sorted(random.Random(seed).sample(range(len(arc_rows)), arc_samples))
    selected = [dict(arc_rows[index]) for index in indices]
    arc_payload = build_multiple_choice_payload(selected)

    with tempfile.TemporaryDirectory(prefix="edgeflow-llama-quality-") as temporary:
        temporary_root = Path(temporary)
        wiki_path = temporary_root / "wikitext.txt"
        arc_path = temporary_root / "arc.bin"
        wiki_path.write_text(text, encoding="utf-8")
        arc_path.write_bytes(arc_payload)
        reference, reference_protocol = _measure_model(
            executable=executable,
            model_path=reference_model,
            wiki_path=wiki_path,
            arc_path=arc_path,
            chunks=chunks,
        )
        candidate, candidate_protocol = _measure_model(
            executable=executable,
            model_path=candidate_model,
            wiki_path=wiki_path,
            arc_path=arc_path,
            chunks=chunks,
        )

    protocol_match = bool(
        reference_protocol["input_tokens"] == candidate_protocol["input_tokens"]
        and reference_protocol["chunks"] == candidate_protocol["chunks"] == chunks
        and reference_protocol["context_size"] == candidate_protocol["context_size"]
    )
    formal = bool(wikitext_token_limit >= 8192 and arc_samples >= 50 and chunks >= 17)
    gate = evaluate_quality(
        reference=reference,
        candidate=candidate,
        profile="balanced",
        protocol_match=protocol_match,
    )
    version = _run(executable, ["--version"]).strip().splitlines()[0]
    sample_ids = [row.get("id") for row in selected]
    selection_hash = sha256_value(
        {
            "wiki_text": text,
            "arc_sample_ids": sample_ids,
            "seed": seed,
            "chunks": chunks,
        }
    )
    payload = {
        "schema_version": "1.0",
        "report_id": f"quality-llama-cpp-{sha256_value([model_id, model_revision, quantization, candidate])[:12]}",
        "pass": bool(formal and gate["pass"]),
        "protocol_status": "FORMAL" if formal else "DEVELOPMENT",
        "quality_role": "quantized_cross_runtime_candidate",
        "source_type": "measured",
        "created_at": utc_now(),
        "scope": {
            "model_id": model_id,
            "model_ref": model_ref,
            "model_revision": model_revision,
            "model_format": "gguf",
            "dtype": None,
            "quantization": quantization,
            "applicable_backends": ["llama_cpp"],
            "reference_format": "gguf",
            "reference_quantization": "BF16",
        },
        "metrics": candidate,
        "reference": reference,
        "candidate": candidate,
        "protocol_match": protocol_match,
        "quality_gate": gate,
        "datasets": {
            "wikitext": {
                "dataset_id": wiki["dataset_id"],
                "revision": wiki["revision"],
                "split": wiki["split"],
                "requested_token_limit": wikitext_token_limit,
                "chunks": chunks,
                "stride": 511,
                "reference_input_tokens": reference_protocol["input_tokens"],
                "candidate_input_tokens": candidate_protocol["input_tokens"],
                "effective_context_size": reference_protocol["context_size"],
            },
            "arc_challenge": {
                "dataset_id": arc["dataset_id"],
                "revision": arc["revision"],
                "split": arc["split"],
                "sample_count": len(selected),
                "sample_ids_sha256": sha256_value(sample_ids),
                "scoring": "mean_candidate_log_likelihood",
            },
            "selection_sha256": selection_hash,
        },
        "runtime": {
            "executable": str(executable.resolve()),
            "version": version,
            "request_protocol": "llama-perplexity pinned stride and multiple-choice scorer",
            "flags": {
                "gpu_layers": "all",
                "flash_attention": True,
                "ctx_size_ppl": 512,
                "ppl_stride": 511,
                "ctx_size_arc": 512,
            },
        },
        "artifacts": {
            "reference_model_sha256": sha256_file(reference_model),
            "candidate_model_sha256": sha256_file(candidate_model),
            "executable_sha256": sha256_file(executable),
        },
        "summary": gate["summary"],
    }
    destination = artifact_root / "quality" / f"{model_id}-llama_cpp-{quantization}.json"
    write_json(destination, payload)
    return destination
