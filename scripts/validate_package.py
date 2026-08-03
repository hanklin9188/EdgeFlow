#!/usr/bin/env python3
"""Validate the EdgeFlow planning package.

This script is intentionally dependency-light. JSON and Markdown checks always run.
YAML and JSON Schema instance checks run when PyYAML/jsonschema are available.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".runtime",
    ".cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
}


def project_files(pattern: str) -> list[Path]:
    return sorted(
        path for path in ROOT.rglob(pattern)
        if path.is_file() and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def ok(self, message: str) -> None:
        self.passed.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def print(self) -> None:
        print("\nEdgeFlow package validation")
        print("=" * 31)
        for item in self.passed:
            print(f"[PASS] {item}")
        for item in self.warnings:
            print(f"[WARN] {item}")
        for item in self.errors:
            print(f"[FAIL] {item}")
        print("-" * 31)
        print(f"{len(self.passed)} passed · {len(self.warnings)} warnings · {len(self.errors)} failed")


def load_json_files(report: Report) -> dict[Path, object]:
    parsed: dict[Path, object] = {}
    for path in project_files("*.json"):
        try:
            parsed[path] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.fail(f"JSON parse: {path.relative_to(ROOT)} — {exc}")
    if parsed:
        report.ok(f"Parsed {len(parsed)} JSON files")
    return parsed


def validate_yaml(report: Report) -> None:
    paths = [*project_files("*.yaml"), *project_files("*.yml")]
    try:
        import yaml  # type: ignore
    except ImportError:
        report.warn("PyYAML unavailable; YAML syntax validation skipped")
        return
    for path in paths:
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.fail(f"YAML parse: {path.relative_to(ROOT)} — {exc}")
    if paths:
        report.ok(f"Parsed {len(paths)} YAML files")


def validate_schema_examples(report: Report, parsed: dict[Path, object]) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker  # type: ignore
    except ImportError:
        report.warn("jsonschema unavailable; example-instance validation skipped")
        return

    mapping = {
        "sample_workload.json": "workload.schema.json",
        "sample_execution_plan.json": "execution_plan.schema.json",
        "sample_run_manifest.json": "run_manifest.schema.json",
        "sample_evidence_record.json": "evidence_record.schema.json",
        "sample_validation_verdict.json": "validation_verdict.schema.json",
        "sample_hardware_fingerprint.json": "hardware_fingerprint.schema.json",
        "sample_metric_record.json": "metric_record.schema.json",
        "sample_profiler_summary.json": "profiler_summary.schema.json",
        "sample_deployment_policy.json": "deployment_policy.schema.json",
    }
    validated = 0
    for example_name, schema_name in mapping.items():
        example_path = ROOT / "examples" / example_name
        schema_path = ROOT / "specs" / schema_name
        if example_path not in parsed:
            report.fail(f"Missing or invalid example: examples/{example_name}")
            continue
        if schema_path not in parsed:
            report.fail(f"Missing or invalid schema: specs/{schema_name}")
            continue
        validator = Draft202012Validator(parsed[schema_path], format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(parsed[example_path]), key=lambda err: list(err.path))
        if errors:
            for error in errors:
                loc = "/".join(map(str, error.path)) or "<root>"
                report.fail(f"Schema mismatch {example_name} at {loc}: {error.message}")
        else:
            validated += 1
    if validated:
        report.ok(f"Validated {validated}/{len(mapping)} schema examples")


LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def markdown_files() -> Iterable[Path]:
    return project_files("*.md")


def validate_markdown_links(report: Report) -> None:
    checked = 0
    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>\"'")
            if not target or target.startswith(("http://", "https://", "mailto:", "#", "data:")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            checked += 1
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                report.fail(f"Markdown link escapes package: {path.relative_to(ROOT)} → {raw_target}")
                continue
            if not resolved.exists():
                report.fail(f"Broken Markdown link: {path.relative_to(ROOT)} → {target}")
    report.ok(f"Checked {checked} internal Markdown links")


def parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    result: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def validate_skills(report: Report) -> None:
    skill_paths = sorted(ROOT.glob("skills/*/SKILL.md"))
    for path in skill_paths:
        frontmatter = parse_frontmatter(path)
        text = path.read_text(encoding="utf-8")
        if not frontmatter.get("name") or not frontmatter.get("description"):
            report.fail(f"Skill frontmatter incomplete: {path.relative_to(ROOT)}")
        if not re.search(r"^#\s+", text, re.MULTILINE):
            report.fail(f"Skill missing H1: {path.relative_to(ROOT)}")
        if not re.search(r"^##\s+", text, re.MULTILINE):
            report.fail(f"Skill missing workflow sections: {path.relative_to(ROOT)}")
        if "Never" not in text and "不得" not in text and "Non-negotiable" not in text:
            report.warn(f"Skill has no explicit refusal/never-do rules: {path.relative_to(ROOT)}")
    agent = ROOT / "skills" / "performance-copilot" / "AGENT.md"
    if not agent.exists():
        report.fail("Missing performance-copilot/AGENT.md")
    else:
        agent_text = agent.read_text(encoding="utf-8")
        for phrase in ["System Rules", "Available Tools", "Decision Procedure"]:
            if phrase not in agent_text:
                report.fail(f"AGENT.md missing section: {phrase}")
    report.ok(f"Inspected {len(skill_paths)} SKILL.md files and the Copilot agent contract")


def validate_ui(report: Report) -> None:
    ui = ROOT / "ui-prototype"
    required = [ui / "index.html", ui / "styles.css", ui / "app.js", ui / "README.md", ui / "preview.png"]
    missing = [p.relative_to(ROOT) for p in required if not p.exists()]
    if missing:
        report.fail("UI prototype missing: " + ", ".join(map(str, missing)))
        return
    html = (ui / "index.html").read_text(encoding="utf-8")
    css = (ui / "styles.css").read_text(encoding="utf-8")
    if "DEMO DATA" not in html or 'data-source-type="demo"' not in html:
        report.fail("UI lacks prominent demo/provenance labeling")
    if "prefers-reduced-motion" not in css:
        report.fail("UI lacks reduced-motion handling")
    if "data-theme" not in html or "themeToggle" not in html:
        report.fail("UI theme contract incomplete")
    if "http://" in html or "https://" in html:
        report.warn("UI HTML contains an external URL; confirm offline policy")
    report.ok("UI prototype files, provenance labeling, theme, and reduced-motion contract present")

    production = ROOT / "dashboard"
    production_required = [
        production / "index.html",
        production / "styles.css",
        production / "app.js",
    ]
    production_missing = [path.relative_to(ROOT) for path in production_required if not path.exists()]
    if production_missing:
        report.fail("Production local UI missing: " + ", ".join(map(str, production_missing)))
        return
    production_html = (production / "index.html").read_text(encoding="utf-8")
    production_css = (production / "styles.css").read_text(encoding="utf-8")
    production_js = (production / "app.js").read_text(encoding="utf-8")
    api_source = (ROOT / "src" / "edgeflow" / "api" / "app.py").read_text(encoding="utf-8")
    required_ids = [
        'id="overview"',
        'id="serviceList"',
        'id="tune"',
        'id="jobs"',
        'id="runs"',
        'id="evidence"',
    ]
    for required_id in required_ids:
        if required_id not in production_html:
            report.fail(f"Production local UI missing workflow surface: {required_id}")
    if "Local Control Console" not in production_html or "local-first" not in production_html.lower():
        report.fail("Production UI does not clearly identify the localhost-only control mode")
    if "X-EdgeFlow-Token" not in production_js or "/api/v1/jobs/benchmark" not in production_js:
        report.fail("Production UI lacks typed CSRF-protected local job control")
    if "/api/v1/runtime-services" not in production_js or "LocalRuntimeServiceManager" not in api_source:
        report.fail("Production UI lacks allowlisted managed runtime control")
    if "prefers-reduced-motion" not in production_css:
        report.fail("Production UI lacks reduced-motion handling")
    if 'data-source-type="demo"' in production_html:
        report.fail("Production local UI must not embed demo result rows")
    if "TrustedHostMiddleware" not in api_source or "MAX_REQUEST_BYTES" not in api_source:
        report.fail("Local web API is missing Host or request-size enforcement")
    if "https://" in production_html:
        report.fail("Production local UI contains an unexpected remote URL")
    report.ok("Local-first production UI, typed job control, evidence surfaces, and web safety contract present")


def validate_demo_safety(report: Report, parsed: dict[Path, object]) -> None:
    for path, obj in parsed.items():
        if "examples" not in path.parts:
            continue
        if isinstance(obj, dict) and "source_type" in obj and obj.get("source_type") != "demo":
            report.fail(f"Example artifact not marked demo: {path.relative_to(ROOT)}")
    html = (ROOT / "ui-prototype" / "index.html").read_text(encoding="utf-8")
    if html.count("DEMO") < 4:
        report.fail("UI demo labels are not repeated near result-bearing surfaces")
    forbidden_claims = ["measured on our RTX", "verified speedup", "production result"]
    lower = html.lower()
    for phrase in forbidden_claims:
        if phrase in lower:
            report.fail(f"UI contains unsafe demo claim: {phrase}")
    report.ok("Demo artifacts are isolated from measured-result eligibility")


def validate_required_structure(report: Report) -> None:
    required = [
        "README.md",
        "docs/00_EXECUTIVE_BLUEPRINT.md",
        "docs/03_EXPERIMENT_CATALOG.md",
        "docs/04_MODELS_DATASETS_AND_TRAINING.md",
        "docs/07_VALIDATION_AND_STATISTICS.md",
        "docs/09_UI_UX_GITHUB_PRESENTATION.md",
        "docs/10_IMPLEMENTATION_ROADMAP.md",
        "skills/edgeflow-validation/SKILL.md",
        "specs/experiment_matrix.yaml",
        "templates/README_TEMPLATE.md",
        "github/workflows/ci.yml",
        "diagrams/system_architecture.mmd",
        "SECURITY.md",
        "THIRD_PARTY.md",
        "MODEL_LICENSES.md",
        "DATA_LICENSES.md",
        "NOTICE.md",
        "specs/runtime_registry.yaml",
        "scripts/bootstrap_llama_cpp.sh",
        "scripts/bootstrap_vllm.sh",
        "scripts/start_llama_cpp_server.sh",
        "scripts/start_vllm_server.sh",
        "src/edgeflow/local/jobs.py",
        "src/edgeflow/local/services.py",
    ]
    missing = [item for item in required if not (ROOT / item).exists()]
    if missing:
        for item in missing:
            report.fail(f"Required package file missing: {item}")
    else:
        report.ok(f"Required package structure present ({len(required)} anchors)")


def validate_manifest(report: Report) -> None:
    path = ROOT / "PACKAGE_MANIFEST.json"
    if not path.exists():
        report.warn("PACKAGE_MANIFEST.json not generated yet")
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.fail(f"PACKAGE_MANIFEST.json invalid: {exc}")
        return
    mismatches = []
    for item in manifest.get("files", []):
        rel = item.get("path")
        expected = item.get("sha256")
        if rel == "PACKAGE_MANIFEST.json":
            continue
        file_path = ROOT / rel
        if not file_path.exists():
            mismatches.append(f"missing {rel}")
            continue
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(f"hash {rel}")
    if mismatches:
        report.fail("Package manifest mismatches: " + ", ".join(mismatches[:8]))
    else:
        report.ok(f"Package manifest verified ({len(manifest.get('files', []))} entries)")


def main() -> int:
    report = Report()
    validate_required_structure(report)
    parsed = load_json_files(report)
    validate_yaml(report)
    validate_schema_examples(report, parsed)
    validate_markdown_links(report)
    validate_skills(report)
    validate_ui(report)
    validate_demo_safety(report, parsed)
    validate_manifest(report)
    report.print()
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
