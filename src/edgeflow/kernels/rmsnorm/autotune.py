from __future__ import annotations

from typing import Any


def launch_config(hidden_size: int) -> dict[str, Any]:
    """Deterministic baseline configuration; measured autotune results remain artifact-scoped."""

    if hidden_size <= 1024:
        return {"num_warps": 4, "block": 1 << (hidden_size - 1).bit_length()}
    return {"num_warps": 8, "block": 1 << (hidden_size - 1).bit_length()}
