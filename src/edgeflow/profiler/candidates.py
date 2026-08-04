from __future__ import annotations

from typing import Any

_CANDIDATES = {
    "fused-swiglu-v1": {
        "patterns": ("aten::silu", "aten::mul"),
        "description": "Fuse SiLU activation and elementwise gate multiplication.",
    },
    "fused-rope-v1": {
        "patterns": ("aten::cos", "aten::sin", "aten::mul", "aten::add"),
        "description": "Fuse rotary-position elementwise transforms around attention projection.",
    },
    "fused-scaled-softmax-v1": {
        "patterns": ("aten::softmax", "aten::mul"),
        "description": "Fuse registered scaling with attention softmax where shape support is bounded.",
    },
}


def rank_kernel_candidates(
    operations: list[dict[str, Any]], *, excluded: set[str] | None = None
) -> list[dict[str, Any]]:
    """Rank bounded fusion candidates from literal profiler operation totals."""

    excluded = excluded or set()
    normalized = [
        {
            "name": str(row["name"]).lower(),
            "calls": int(row.get("calls", 0)),
            "self_device_time_us": float(row.get("self_device_time_us", 0.0)),
        }
        for row in operations
    ]
    ranked: list[dict[str, Any]] = []
    for candidate_id, definition in _CANDIDATES.items():
        if candidate_id in excluded:
            continue
        matched: dict[str, list[dict[str, Any]]] = {}
        for pattern in definition["patterns"]:
            matched[pattern] = [row for row in normalized if row["name"] == pattern]
        if any(not rows for rows in matched.values()):
            continue
        calls = sum(row["calls"] for rows in matched.values() for row in rows)
        device_time = sum(row["self_device_time_us"] for rows in matched.values() for row in rows)
        # Device time is primary; call count is an explicit launch-overhead proxy.
        score = device_time + 2.0 * calls
        ranked.append(
            {
                "candidate_id": candidate_id,
                "score": score,
                "matched_patterns": {
                    pattern: [row["name"] for row in rows] for pattern, rows in matched.items()
                },
                "observed_calls": calls,
                "self_device_time_us": device_time,
                "description": definition["description"],
                "status": "HYPOTHESIS",
            }
        )
    return sorted(ranked, key=lambda row: (-row["score"], row["candidate_id"]))
