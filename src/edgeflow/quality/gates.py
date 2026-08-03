from __future__ import annotations

from typing import Any

THRESHOLDS = {
    "strict": {"arc_c_drop_pp_max": 0.5, "ppl_ratio_max": 1.02},
    "balanced": {"arc_c_drop_pp_max": 1.0, "ppl_ratio_max": 1.05},
    "memory_first": {"arc_c_drop_pp_max": 2.0, "ppl_ratio_max": 1.10},
}


def evaluate_quality(
    *,
    reference: dict[str, float],
    candidate: dict[str, float],
    profile: str,
    protocol_match: bool,
) -> dict[str, Any]:
    """Evaluate ARC-C and perplexity as hard constraints, never as a hidden penalty."""

    if not protocol_match:
        return {
            "pass": False,
            "profile": profile,
            "protocol_match": False,
            "checks": [],
            "summary": "Quality protocols are incompatible; comparison is invalid.",
        }
    if profile not in THRESHOLDS:
        raise ValueError(f"unknown built-in quality profile: {profile}")
    required = {"arc_c_accuracy", "perplexity"}
    missing = required - reference.keys() | required - candidate.keys()
    if missing:
        raise ValueError(f"missing quality metric(s): {sorted(missing)}")
    arc_drop = (reference["arc_c_accuracy"] - candidate["arc_c_accuracy"]) * 100.0
    ppl_ratio = candidate["perplexity"] / reference["perplexity"]
    thresholds = THRESHOLDS[profile]
    checks = [
        {
            "name": "arc_c_drop_pp",
            "value": arc_drop,
            "limit": thresholds["arc_c_drop_pp_max"],
            "pass": arc_drop <= thresholds["arc_c_drop_pp_max"],
        },
        {
            "name": "ppl_ratio",
            "value": ppl_ratio,
            "limit": thresholds["ppl_ratio_max"],
            "pass": ppl_ratio <= thresholds["ppl_ratio_max"],
        },
    ]
    passed = all(check["pass"] for check in checks)
    return {
        "pass": passed,
        "profile": profile,
        "protocol_match": True,
        "reference": reference,
        "candidate": candidate,
        "checks": checks,
        "summary": "Quality hard gate passed." if passed else "Quality hard gate failed; policy eligibility is blocked.",
    }
