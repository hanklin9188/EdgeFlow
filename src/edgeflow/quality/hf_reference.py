from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import yaml

from edgeflow.core.models import utc_now
from edgeflow.core.serialization import sha256_value, write_json


def _dataset_pin(root: Path, dataset_id: str) -> dict[str, Any]:
    payload = yaml.safe_load((root / "specs" / "dataset_registry.yaml").read_text(encoding="utf-8"))
    for row in payload["datasets"]:
        if row["dataset_id"] == dataset_id:
            revision = str(row.get("revision", ""))
            if not revision or revision.startswith(("PIN_", "CHECK_")):
                raise ValueError(f"dataset {dataset_id} does not have a usable pinned revision")
            return row
    raise KeyError(f"dataset is not registered: {dataset_id}")


def _perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    token_limit: int,
    sequence_length: int,
) -> tuple[float, int]:
    import torch

    token_ids: list[int] = []
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    for text in texts:
        if not text.strip():
            continue
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        token_ids.extend(separator)
        if len(token_ids) >= token_limit:
            break
    encoded = torch.tensor(token_ids[:token_limit], dtype=torch.long)
    if len(encoded) < 2:
        raise ValueError("WikiText selection produced fewer than two tokens")
    total_nll = 0.0
    scored_tokens = 0
    for start in range(0, len(encoded) - 1, sequence_length):
        chunk = encoded[start : start + sequence_length].unsqueeze(0).to(model.device)
        if chunk.shape[1] < 2:
            continue
        with torch.inference_mode():
            output = model(input_ids=chunk, labels=chunk, use_cache=False, return_dict=True)
        count = int(chunk.shape[1] - 1)
        total_nll += float(output.loss.float().item()) * count
        scored_tokens += count
    if scored_tokens == 0:
        raise ValueError("WikiText selection produced no scoreable continuation tokens")
    return math.exp(total_nll / scored_tokens), scored_tokens


def _continuation_score(model: Any, tokenizer: Any, prompt: str, continuation: str) -> float:
    import torch

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not prompt_ids or not continuation_ids:
        raise ValueError("ARC item produced an empty prompt or continuation")
    combined = torch.tensor([prompt_ids + continuation_ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=combined, use_cache=False, return_dict=True).logits.float()
    start = len(prompt_ids) - 1
    token_logits = logits[0, start : start + len(continuation_ids)]
    targets = torch.tensor(continuation_ids, dtype=torch.long, device=model.device)
    log_probabilities = torch.log_softmax(token_logits, dim=-1)
    # Mean continuation likelihood is fixed for every candidate and avoids a raw length bias.
    return float(log_probabilities.gather(1, targets[:, None]).mean().item())


def _arc_accuracy(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    correct = 0
    for row in rows:
        labels = [str(label) for label in row["choices"]["label"]]
        choices = [str(text) for text in row["choices"]["text"]]
        prompt = f"Question: {row['question']}\nAnswer:"
        scores = [
            _continuation_score(model, tokenizer, prompt, f" {choice}") for choice in choices
        ]
        predicted_index = max(range(len(scores)), key=scores.__getitem__)
        predicted = labels[predicted_index]
        answer = str(row["answerKey"])
        correct += predicted == answer
        results.append(
            {
                "id": str(row.get("id", "")),
                "answer": answer,
                "predicted": predicted,
                "correct": predicted == answer,
                "scores": scores,
            }
        )
    return correct / len(rows), results


def evaluate_hf_reference_quality(
    *,
    root: Path,
    artifact_root: Path,
    model_id: str,
    model_ref: str,
    model_revision: str,
    dtype: str = "bf16",
    wikitext_token_limit: int = 8192,
    arc_samples: int = 50,
    seed: int = 42,
    local_files_only: bool = True,
) -> Path:
    """Measure the BF16 Transformers reference used by later hard quality gates."""

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered HF quality reference")
    if dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"unsupported quality dtype: {dtype}")
    dtype_value = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    wiki = _dataset_pin(root, "wikitext-2-raw")
    arc = _dataset_pin(root, "arc-challenge")
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        revision=model_revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        revision=model_revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
        dtype=dtype_value,
    ).eval().to("cuda")
    wiki_rows = load_dataset(
        wiki["source"],
        wiki.get("config"),
        revision=wiki["revision"],
        split=wiki["split"],
    )
    perplexity, scored_tokens = _perplexity(
        model,
        tokenizer,
        [str(row["text"]) for row in wiki_rows],
        token_limit=wikitext_token_limit,
        sequence_length=512,
    )
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
    arc_accuracy, arc_results = _arc_accuracy(model, tokenizer, selected)
    metrics = {"perplexity": perplexity, "arc_c_accuracy": arc_accuracy}
    payload = {
        "schema_version": "1.0",
        "pass": bool(math.isfinite(perplexity) and perplexity > 0 and 0 <= arc_accuracy <= 1),
        "quality_role": "bf16_transformers_reference",
        "source_type": "measured",
        "created_at": utc_now(),
        "scope": {
            "model_id": model_id,
            "model_ref": model_ref,
            "model_revision": model_revision,
            "model_format": "safetensors",
            "dtype": dtype,
            "quantization": None,
            "applicable_backends": ["pytorch_eager", "torch_compile"],
        },
        "metrics": metrics,
        "reference": metrics,
        "candidate": metrics,
        "protocol_match": True,
        "datasets": {
            "wikitext": {
                "dataset_id": wiki["dataset_id"],
                "revision": wiki["revision"],
                "split": wiki["split"],
                "scored_tokens": scored_tokens,
            },
            "arc_challenge": {
                "dataset_id": arc["dataset_id"],
                "revision": arc["revision"],
                "split": arc["split"],
                "sample_count": len(selected),
                "sample_ids_sha256": sha256_value([row.get("id") for row in selected]),
                "scoring": "mean_candidate_log_likelihood",
            },
        },
        "arc_results": arc_results,
        "summary": "Pinned BF16 Transformers reference quality evaluation completed.",
    }
    destination = artifact_root / "quality" / f"{model_id}-{dtype}-reference.json"
    write_json(destination, payload)
    del model
    torch.cuda.empty_cache()
    return destination
