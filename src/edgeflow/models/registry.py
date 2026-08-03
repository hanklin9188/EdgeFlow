from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from edgeflow.core.serialization import project_root


class ModelRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or project_root() / "specs" / "model_registry.yaml"
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported model registry schema")
        self._models = {item["model_id"]: item for item in payload["models"]}

    def list(self) -> list[dict[str, Any]]:
        return list(self._models.values())

    def get(self, model_id: str) -> dict[str, Any]:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"model is not registered: {model_id}") from exc

    def resolve_source(self, model_id: str, model_format: str) -> tuple[str, str]:
        model = self.get(model_id)
        key = "gguf" if model_format == "gguf" else "safetensors"
        source = model.get("sources", {}).get(key)
        if not isinstance(source, dict) or "repo" not in source:
            raise ValueError(f"{model_id} has no {key} source")
        revision = source.get("revision")
        if not revision or str(revision).startswith(("PIN_", "CHECK_")):
            raise ValueError(f"{model_id}/{key} is not pinned and cannot enter a formal run")
        return str(source["repo"]), str(revision)

    def support(self, model_id: str, backend: str) -> str:
        return str(self.get(model_id).get("backends", {}).get(backend, "unsupported"))

