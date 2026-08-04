from __future__ import annotations

import json
import math
import random
import urllib.request
from pathlib import Path
from typing import Any

from edgeflow.core.models import utc_now
from edgeflow.core.serialization import read_json, sha256_value, write_json
from edgeflow.quality.gates import evaluate_quality
from edgeflow.quality.hf_reference import _dataset_pin


def prompt_token_logprobs(payload: dict[str, Any], expected_tokens: int) -> list[float | None]:
    """Extract the echoed prompt scores while excluding the generated probe token."""

    try:
        values = payload["choices"][0]["logprobs"]["token_logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenAI-compatible response omitted prompt token logprobs") from exc
    if not isinstance(values, list) or len(values) < expected_tokens:
        raise ValueError(
            f"runtime returned {len(values) if isinstance(values, list) else 0} token scores; "
            f"expected at least {expected_tokens}"
        )
    prompt_values = values[:expected_tokens]
    if prompt_values[0] is not None or any(value is None for value in prompt_values[1:]):
        raise ValueError("runtime prompt logprobs do not follow the registered causal alignment")
    return [None, *(float(value) for value in prompt_values[1:])]


def _request_json(base_url: str, endpoint: str, payload: dict[str, Any] | None = None) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def _score_prompt(base_url: str, served_model: str, token_ids: list[int]) -> list[float | None]:
    response = _request_json(
        base_url,
        "/v1/completions",
        {
            "model": served_model,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "echo": True,
            "logprobs": 1,
            "seed": 42,
        },
    )
    usage = response.get("usage", {})
    if int(usage.get("prompt_tokens", -1)) != len(token_ids):
        raise ValueError("runtime changed the exact prompt-token count")
    return prompt_token_logprobs(response, len(token_ids))


def evaluate_openai_runtime_quality(
    *,
    root: Path,
    artifact_root: Path,
    model_id: str,
    model_ref: str,
    model_revision: str,
    backend: str,
    base_url: str,
    served_model: str,
    dtype: str = "bf16",
    wikitext_token_limit: int = 8192,
    arc_samples: int = 50,
    seed: int = 42,
    local_files_only: bool = True,
) -> Path:
    """Evaluate a pinned loopback runtime with the exact HF reference protocol."""

    if backend not in {"vllm", "llama_cpp"}:
        raise ValueError(f"unsupported OpenAI-compatible quality backend: {backend}")
    if not base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("quality runtime must use a loopback HTTP endpoint")
    from transformers import AutoConfig, AutoTokenizer

    from datasets import load_dataset

    config = AutoConfig.from_pretrained(
        model_ref,
        revision=model_revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        revision=model_revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
        fix_mistral_regex=config.model_type == "mistral3",
    )
    models = _request_json(base_url, "/v1/models")
    served_ids = [str(row.get("id")) for row in models.get("data", [])]
    if served_model not in served_ids:
        raise ValueError(f"runtime does not serve the registered model name {served_model!r}")

    wiki = _dataset_pin(root, "wikitext-2-raw")
    wiki_rows = load_dataset(
        wiki["source"],
        wiki.get("config"),
        revision=wiki["revision"],
        split=wiki["split"],
    )
    token_ids: list[int] = []
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    for row in wiki_rows:
        text = str(row["text"])
        if not text.strip():
            continue
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        token_ids.extend(separator)
        if len(token_ids) >= wikitext_token_limit:
            break
    encoded = token_ids[:wikitext_token_limit]
    total_log_probability = 0.0
    scored_tokens = 0
    for start in range(0, len(encoded) - 1, 512):
        chunk = encoded[start : start + 512]
        if len(chunk) < 2:
            continue
        scores = _score_prompt(base_url, served_model, chunk)
        total_log_probability += sum(float(value) for value in scores[1:] if value is not None)
        scored_tokens += len(chunk) - 1
    if scored_tokens == 0:
        raise ValueError("WikiText selection produced no scoreable runtime tokens")
    perplexity = math.exp(-total_log_probability / scored_tokens)

    arc = _dataset_pin(root, "arc-challenge")
    arc_rows = load_dataset(
        arc["source"],
        arc.get("config"),
        revision=arc["revision"],
        split=arc["split"],
    )
    if arc_samples > len(arc_rows):
        raise ValueError(f"ARC sample count {arc_samples} exceeds split size {len(arc_rows)}")
    indices = sorted(random.Random(seed).sample(range(len(arc_rows)), arc_samples))
    selected = [dict(arc_rows[index]) for index in indices]
    arc_results: list[dict[str, Any]] = []
    correct = 0
    for row in selected:
        prompt = f"Question: {row['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        choices = [str(text) for text in row["choices"]["text"]]
        labels = [str(label) for label in row["choices"]["label"]]
        choice_scores: list[float] = []
        for choice in choices:
            continuation_ids = tokenizer.encode(f" {choice}", add_special_tokens=False)
            scores = _score_prompt(base_url, served_model, prompt_ids + continuation_ids)
            continuation_scores = scores[len(prompt_ids) :]
            choice_scores.append(
                sum(float(value) for value in continuation_scores if value is not None)
                / len(continuation_ids)
            )
        predicted_index = max(range(len(choice_scores)), key=choice_scores.__getitem__)
        predicted = labels[predicted_index]
        answer = str(row["answerKey"])
        correct += predicted == answer
        arc_results.append(
            {
                "id": str(row.get("id", "")),
                "answer": answer,
                "predicted": predicted,
                "correct": predicted == answer,
                "scores": choice_scores,
            }
        )
    candidate = {
        "perplexity": perplexity,
        "arc_c_accuracy": correct / len(selected),
    }
    reference_path = artifact_root / "quality" / f"{model_id}-{dtype}-reference.json"
    reference_report = read_json(reference_path)
    reference = reference_report["metrics"]
    sample_ids_sha256 = sha256_value([row.get("id") for row in selected])
    protocol_match = bool(
        reference_report.get("protocol_status") == "FORMAL"
        and reference_report.get("datasets", {}).get("arc_challenge", {}).get("sample_ids_sha256")
        == sample_ids_sha256
        and reference_report.get("datasets", {}).get("wikitext", {}).get("scored_tokens")
        == scored_tokens
    )
    gate = evaluate_quality(
        reference=reference,
        candidate=candidate,
        profile="balanced",
        protocol_match=protocol_match,
    )
    formal = wikitext_token_limit >= 8192 and arc_samples >= 50
    version = _request_json(base_url, "/version")
    payload = {
        "schema_version": "1.0",
        "report_id": f"quality-{backend}-{sha256_value([model_id, model_revision, candidate])[:12]}",
        "pass": bool(formal and gate["pass"]),
        "protocol_status": "FORMAL" if formal else "DEVELOPMENT",
        "quality_role": "cross_runtime_candidate",
        "source_type": "measured",
        "created_at": utc_now(),
        "scope": {
            "model_id": model_id,
            "model_ref": model_ref,
            "model_revision": model_revision,
            "model_format": "safetensors",
            "dtype": dtype,
            "quantization": None,
            "applicable_backends": [backend],
            "served_model": served_model,
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
                "scored_tokens": scored_tokens,
            },
            "arc_challenge": {
                "dataset_id": arc["dataset_id"],
                "revision": arc["revision"],
                "split": arc["split"],
                "sample_count": len(selected),
                "sample_ids_sha256": sample_ids_sha256,
                "scoring": "mean_candidate_log_likelihood",
            },
        },
        "runtime": {
            "base_url": base_url,
            "version": version,
            "request_protocol": "OpenAI completions exact token IDs with echoed logprobs",
        },
        "arc_results": arc_results,
        "summary": gate["summary"],
    }
    destination = artifact_root / "quality" / f"{model_id}-{backend}-{dtype}.json"
    write_json(destination, payload)
    return destination
