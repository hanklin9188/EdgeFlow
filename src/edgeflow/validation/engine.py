from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

from jsonschema import Draft202012Validator, FormatChecker

from edgeflow import __version__
from edgeflow.core.models import MetricRecord, Verdict, utc_now
from edgeflow.core.serialization import project_root, read_json, sha256_value, write_json
from edgeflow.metrics.statistics import describe, third_drift


@dataclass
class CheckCollector:
    checks: list[dict[str, Any]]
    issues: list[dict[str, Any]]

    def add(
        self,
        gate: str,
        name: str,
        status: str,
        message: str,
        artifact: str | None = None,
    ) -> None:
        self.checks.append(
            {"gate": gate, "name": name, "status": status, "message": message, "artifact": artifact}
        )
        if status in {"FAIL", "WARN"}:
            self.issues.append(
                {
                    "severity": "error" if status == "FAIL" else "warning",
                    "code": f"{gate.upper()}_{name.upper()}",
                    "message": message,
                }
            )


class ValidationEngine:
    required: ClassVar[dict[str, str]] = {
        "run_manifest.json": "run_manifest.schema.json",
        "workload.json": "workload.schema.json",
        "execution_plan.json": "execution_plan.schema.json",
        "hardware_fingerprint.json": "hardware_fingerprint.schema.json",
    }

    def __init__(self, *, root: Path | None = None) -> None:
        self.root = (root or project_root()).resolve()
        self.schema_root = self.root / "specs"

    def _validate_schema(
        self, collector: CheckCollector, instance: dict[str, Any], schema_name: str, artifact: str
    ) -> None:
        schema = read_json(self.schema_root / schema_name)
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(instance),
            key=lambda item: list(item.path),
        )
        if errors:
            first = errors[0]
            location = "/".join(str(value) for value in first.path) or "<root>"
            collector.add(
                "G0", f"schema_{artifact}", "FAIL", f"{location}: {first.message}", artifact
            )
        else:
            collector.add("G0", f"schema_{artifact}", "PASS", f"Matches {schema_name}.", artifact)

    def validate(self, run_dir: Path, *, write: bool = True) -> dict[str, Any]:
        run_dir = run_dir.resolve()
        collector = CheckCollector([], [])
        parsed: dict[str, dict[str, Any]] = {}
        for filename, schema in self.required.items():
            path = run_dir / filename
            if not path.exists():
                collector.add(
                    "G0", f"required_{filename}", "FAIL", "Required artifact is missing.", filename
                )
                continue
            try:
                parsed[filename] = read_json(path)
                self._validate_schema(collector, parsed[filename], schema, filename)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                collector.add("G0", f"parse_{filename}", "FAIL", str(exc), filename)

        manifest = parsed.get("run_manifest.json", {})
        run_id = str(manifest.get("run_id", run_dir.name))
        execution_status = manifest.get("status")
        if execution_status in {"FAILED", "CANCELLED", "PRECHECK_FAILED"}:
            collector.add(
                "G0",
                "execution_status",
                "FAIL",
                f"Runtime execution ended with {execution_status}; partial metrics cannot validate it.",
                "run_manifest.json",
            )
        metrics_path = run_dir / "metrics.jsonl"
        metric_rows: list[dict[str, Any]] = []
        if not metrics_path.exists():
            collector.add("G0", "raw_metrics", "FAIL", "metrics.jsonl is missing.", "metrics.jsonl")
        else:
            try:
                for number, line in enumerate(
                    metrics_path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    MetricRecord.model_validate(value)
                    schema = read_json(self.schema_root / "metric_record.schema.json")
                    errors = list(Draft202012Validator(schema).iter_errors(value))
                    if errors:
                        raise ValueError(f"line {number}: {errors[0].message}")
                    metric_rows.append(value)
                if not metric_rows:
                    raise ValueError("no metric records")
                collector.add(
                    "G0",
                    "raw_metrics",
                    "PASS",
                    f"Parsed {len(metric_rows)} raw records.",
                    "metrics.jsonl",
                )
            except (ValueError, json.JSONDecodeError) as exc:
                collector.add("G0", "raw_metrics", "FAIL", str(exc), "metrics.jsonl")

        plan = parsed.get("execution_plan.json", {})
        workload = parsed.get("workload.json", {})
        hardware = parsed.get("hardware_fingerprint.json", {})
        if manifest:
            ids_match = all(
                [
                    all(row.get("run_id") == run_id for row in metric_rows),
                    manifest.get("plan_id") == plan.get("plan_id"),
                    manifest.get("workload_id") == workload.get("workload_id"),
                ]
            )
            collector.add(
                "G0",
                "id_consistency",
                "PASS" if ids_match else "FAIL",
                "Artifact IDs agree." if ids_match else "Artifact IDs disagree.",
            )
            plan_for_hash = dict(plan)
            plan_for_hash.pop("canonical_sha256", None)
            if "custom_kernels" in plan_for_hash:
                plan_for_hash["custom_kernels"] = sorted(plan_for_hash["custom_kernels"])
            hash_ok = manifest.get("plan_sha256") == sha256_value(plan_for_hash) and manifest.get(
                "workload_sha256"
            ) == sha256_value(workload)
            collector.add(
                "G0",
                "content_hashes",
                "PASS" if hash_ok else "FAIL",
                "Plan/workload hashes match." if hash_ok else "Plan or workload hash mismatch.",
            )

        background = hardware.get("measurement_controls", {}).get("background_gpu_processes", [])
        backend = str(plan.get("backend", ""))
        runtime_marker = {
            "llama_cpp": "llama-server",
            "vllm": "VLLM::EngineCore",
        }.get(backend)
        selected_runtime_only = bool(
            runtime_marker and len(background) == 1 and runtime_marker in str(background[0])
        )
        background_ok = not background or selected_runtime_only
        if not background:
            background_message = "No background compute process recorded."
        elif selected_runtime_only:
            background_message = (
                f"Only the selected {backend} service was recorded: {background[0]}"
            )
        else:
            background_message = f"Unexpected background GPU processes: {background}"
        collector.add(
            "G1",
            "background_gpu",
            "PASS" if background_ok else "FAIL",
            background_message,
            "hardware_fingerprint.json",
        )
        expected_gpu = "RTX 4080 SUPER" in hardware.get("gpu", {}).get("name", "")
        collector.add(
            "G1",
            "target_gpu",
            "PASS" if expected_gpu else "WARN",
            hardware.get("gpu", {}).get("name", "unknown"),
        )

        correctness_path = run_dir / "correctness.json"
        correctness: dict[str, Any] | None = None
        if correctness_path.exists():
            correctness = read_json(correctness_path)
            correct = bool(correctness.get("pass")) and correctness.get("nan_count", 0) == 0
            collector.add(
                "G2",
                "functional_correctness",
                "PASS" if correct else "FAIL",
                correctness.get("summary", "Correctness report loaded."),
                "correctness.json",
            )
        else:
            collector.add(
                "G2",
                "functional_correctness",
                "WARN",
                "No reference correctness artifact; run cannot enter policy.",
            )

        measured_rows = [row for row in metric_rows if row.get("phase") == "end_to_end"]
        timing_ok = bool(measured_rows)
        for row in measured_rows:
            values = row.get("metrics", {})
            wall = values.get("wall_ms")
            ttft = values.get("ttft_ms")
            timestamps = row.get("token_timestamps_ms", [])
            timing_ok = timing_ok and bool(wall and wall > 0)
            timing_ok = timing_ok and (ttft is None or (0 <= ttft <= wall))
            timing_ok = timing_ok and all(b >= a for a, b in pairwise(timestamps))
            if row.get("output_tokens", 0) > 1:
                timing_ok = timing_ok and values.get("tpot_ms") is not None
        profiler_clean = manifest.get("profiler_level") == "none" if manifest else False
        timing_ok = timing_ok and profiler_clean
        collector.add(
            "G3",
            "timing_integrity",
            "PASS" if timing_ok else "FAIL",
            "Positive monotonic unprofiled timing boundaries."
            if timing_ok
            else "Timing boundary missing, impossible, or profiler-contaminated.",
            "metrics.jsonl",
        )

        latency = [
            float(row["metrics"]["request_latency_ms"])
            for row in measured_rows
            if row.get("metrics", {}).get("request_latency_ms") is not None
        ]
        stable = False
        statistics: dict[str, Any] = {}
        if latency:
            statistics = describe(latency)
            drift = third_drift(latency)
            statistics["first_last_third_drift"] = drift
            clocks = [
                float(row["metrics"]["sm_clock_mhz"])
                for row in measured_rows
                if row.get("metrics", {}).get("sm_clock_mhz")
                and float(row.get("metrics", {}).get("gpu_utilization_pct") or 0) >= 5.0
            ]
            clock_ratio = min(clocks) / max(clocks) if clocks else 1.0
            temperatures = [
                float(row["metrics"]["temperature_c"])
                for row in measured_rows
                if row.get("metrics", {}).get("temperature_c") is not None
            ]
            temperature_range = max(temperatures) - min(temperatures) if temperatures else 0.0
            statistics["minimum_to_maximum_sm_clock_ratio"] = clock_ratio
            statistics["active_clock_samples"] = len(clocks)
            statistics["temperature_range_c"] = temperature_range
            stable = (
                drift <= 0.03 and float(statistics["robust_cv"]) <= 0.10 and clock_ratio >= 0.85
            )
            collector.add(
                "G4",
                "stability",
                "PASS" if stable else "WARN",
                f"drift={drift:.3%}, robust_cv={float(statistics['robust_cv']):.3%}, "
                f"active_clock_ratio={clock_ratio:.3f} ({len(clocks)} samples), "
                f"temp_range={temperature_range:.1f}°C",
                "metrics.jsonl",
            )
        else:
            collector.add("G4", "stability", "FAIL", "No request latency series.")

        required_repetitions = 100 if manifest.get("run_type") == "kernel_microbenchmark" else 30
        enough = len(measured_rows) >= required_repetitions
        collector.add(
            "G5",
            "repetitions",
            "PASS" if enough else "WARN",
            f"{len(measured_rows)} measured repetitions; protocol requires {required_repetitions}.",
        )

        quality_path = run_dir / "quality.json"
        quality_pass: bool | None = None
        if quality_path.exists():
            quality = read_json(quality_path)
            quality_pass = bool(quality.get("pass"))
            collector.add(
                "G6",
                "quality_gate",
                "PASS" if quality_pass else "FAIL",
                quality.get("summary", "Quality report loaded."),
                "quality.json",
            )
        elif plan.get("quantization"):
            collector.add(
                "G6", "quality_gate", "WARN", "Quantized plan lacks required quality report."
            )
        else:
            collector.add(
                "G6", "quality_gate", "WARN", "Quality was not evaluated in this performance block."
            )

        source_type = manifest.get("source_type")
        pinned = bool(manifest.get("model_revision")) and manifest.get("model_revision") not in {
            "unknown",
            "main",
            "latest",
        }
        clean_git = not bool(manifest.get("git_dirty", True))
        exact_command = bool(manifest.get("command"))
        provenance_ok = pinned and exact_command
        collector.add(
            "G7",
            "provenance",
            "PASS" if provenance_ok else "WARN",
            f"model_pinned={pinned}, exact_command={exact_command}, git_clean={clean_git}, source={source_type}",
        )
        collector.add(
            "G7",
            "source_label",
            "PASS" if source_type == "measured" else "WARN",
            "Measured source."
            if source_type == "measured"
            else "Demo/estimated source cannot support public claims.",
        )

        hard_fail = any(
            check["status"] == "FAIL"
            for check in collector.checks
            if check["gate"] in {"G0", "G1", "G3"}
        )
        correctness_fail = any(
            check["status"] == "FAIL" for check in collector.checks if check["gate"] in {"G2", "G6"}
        )
        if manifest.get("status") == "SKIPPED":
            verdict = Verdict.SKIPPED
        elif hard_fail:
            verdict = Verdict.INVALID
        elif correctness_fail:
            verdict = Verdict.FAIL
        elif not all(
            [
                correctness is not None,
                stable,
                enough,
                provenance_ok,
                clean_git,
                quality_pass is True,
            ]
        ):
            verdict = Verdict.CONDITIONAL_PASS
        else:
            verdict = Verdict.PASS

        policy_eligible = verdict == Verdict.PASS and source_type == "measured"
        public_claim_eligible = policy_eligible and clean_git
        collector.add(
            "G8",
            "policy_eligibility",
            "PASS" if policy_eligible else "WARN",
            "Run may enter policy synthesis."
            if policy_eligible
            else "Run remains exploratory/diagnostic and is excluded from policy ranking.",
        )
        result = {
            "schema_version": "1.0",
            "run_id": run_id,
            "verdict": verdict.value,
            "policy_eligible": policy_eligible,
            "public_claim_eligible": public_claim_eligible,
            "quality_pass": quality_pass,
            "evidence_level": "E2" if enough and stable else ("E1" if measured_rows else "E0"),
            "checks": collector.checks,
            "issues": collector.issues,
            "scope": {
                "gpu": hardware.get("gpu", {}).get("name"),
                "model": workload.get("model_id"),
                "backend": plan.get("backend"),
                "workload_id": workload.get("workload_id"),
                "statistics": statistics,
            },
            "validated_at": utc_now(),
            "validator_version": __version__,
        }
        self._validate_schema(
            collector, result, "validation_verdict.schema.json", "validation_verdict.json"
        )
        # Include the verdict-schema check itself after validation.
        result["checks"] = collector.checks
        result["issues"] = collector.issues
        if write:
            write_json(run_dir / "validation_verdict.json", result)
            (run_dir / "VALIDATION.md").write_text(self.render_markdown(result), encoding="utf-8")
        return result

    @staticmethod
    def render_markdown(verdict: dict[str, Any]) -> str:
        lines = [
            f"# Validation: {verdict['run_id']}",
            "",
            "## Verdict",
            "",
            f"**{verdict['verdict']}** — policy eligible: `{str(verdict['policy_eligible']).lower()}`",
            "",
            "## Gate Summary",
            "",
            "| Gate | Check | Status | Evidence |",
            "|---|---|---|---|",
        ]
        for check in verdict["checks"]:
            message = str(check["message"]).replace("|", "\\|")
            lines.append(f"| {check['gate']} | {check['name']} | {check['status']} | {message} |")
        lines.extend(
            [
                "",
                "## Eligibility",
                "",
                f"Public claim eligible: `{str(verdict['public_claim_eligible']).lower()}`",
                "",
                "## Issues and Required Actions",
                "",
            ]
        )
        if verdict["issues"]:
            lines.extend(f"- `{issue['code']}` — {issue['message']}" for issue in verdict["issues"])
        else:
            lines.append("No open validation issues.")
        return "\n".join(lines) + "\n"


def validate_run(run_dir: Path, *, root: Path | None = None, write: bool = True) -> dict[str, Any]:
    return ValidationEngine(root=root).validate(run_dir, write=write)
