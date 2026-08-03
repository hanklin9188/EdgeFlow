#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PACKAGE_MANIFEST.json"
EXCLUDE = {OUTPUT.resolve()}
EXCLUDED_PARTS = {
    ".git", ".venv", ".cache", ".pytest_cache", ".ruff_cache", "__pycache__", "artifacts"
}

files = []
for path in sorted(
    p for p in ROOT.rglob("*")
    if p.is_file() and not EXCLUDED_PARTS.intersection(p.relative_to(ROOT).parts)
):
    if path.resolve() in EXCLUDE:
        continue
    data = path.read_bytes()
    files.append({
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    })

manifest = {
    "schema_version": "1.0",
    "package": "EdgeFlow",
    "generated_at": datetime.now(UTC).isoformat(),
    "file_count": len(files),
    "total_bytes": sum(item["bytes"] for item in files),
    "files": files,
}
OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(files)} files")
