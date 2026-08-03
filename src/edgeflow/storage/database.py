from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

MIGRATION_001 = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS hardware (
    sha256 TEXT PRIMARY KEY,
    fingerprint_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    captured_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workloads (
    workload_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    backend TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    status TEXT NOT NULL,
    source_type TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    workload_id TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY(plan_id) REFERENCES plans(plan_id),
    FOREIGN KEY(workload_id) REFERENCES workloads(workload_id)
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    phase TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS validations (
    run_id TEXT PRIMARY KEY,
    verdict TEXT NOT NULL,
    policy_eligible INTEGER NOT NULL,
    public_claim_eligible INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    evidence_level TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_edges (
    source_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_id TEXT NOT NULL,
    PRIMARY KEY(source_id, relation, target_id)
);
CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_plan ON runs(plan_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
"""


class EdgeFlowDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        with self.connect() as connection:
            connection.executescript(MIGRATION_001)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (SCHEMA_VERSION,)
            )

    @staticmethod
    def _json(value: dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def save_hardware(self, payload: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO hardware(sha256,fingerprint_id,payload_json,captured_at) VALUES(?,?,?,?)",
                (payload["sha256"], payload["fingerprint_id"], self._json(payload), payload["captured_at"]),
            )

    def save_workload(self, payload: dict[str, Any], sha256: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO workloads(workload_id,sha256,payload_json) VALUES(?,?,?)",
                (payload["workload_id"], sha256, self._json(payload)),
            )

    def save_plan(self, payload: dict[str, Any], sha256: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO plans(plan_id,sha256,backend,payload_json) VALUES(?,?,?,?)",
                (payload["plan_id"], sha256, payload["backend"], self._json(payload)),
            )

    def save_run(self, manifest: dict[str, Any], artifact_path: Path) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO runs(
                    run_id,experiment_id,status,source_type,plan_id,workload_id,artifact_path,
                    manifest_json,created_at,completed_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    manifest["run_id"], manifest["experiment_id"], manifest["status"],
                    manifest["source_type"], manifest["plan_id"], manifest["workload_id"],
                    str(artifact_path), self._json(manifest), manifest["created_at"], manifest.get("completed_at"),
                ),
            )

    def save_metrics(self, run_id: str, records: list[dict[str, Any]]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM metrics WHERE run_id=?", (run_id,))
            connection.executemany(
                "INSERT INTO metrics(run_id,iteration,phase,payload_json) VALUES(?,?,?,?)",
                [(run_id, row["iteration"], row["phase"], self._json(row)) for row in records],
            )

    def save_validation(self, payload: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO validations(
                    run_id,verdict,policy_eligible,public_claim_eligible,payload_json
                ) VALUES(?,?,?,?,?)""",
                (
                    payload["run_id"], payload["verdict"], int(payload["policy_eligible"]),
                    int(payload["public_claim_eligible"]), self._json(payload),
                ),
            )

    def save_evidence(self, payload: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO evidence(evidence_id,status,evidence_level,payload_json) VALUES(?,?,?,?)",
                (payload["evidence_id"], payload["status"], payload["evidence_level"], self._json(payload)),
            )

    def link_evidence(self, source_id: str, relation: str, target_id: str) -> None:
        allowed = {"observes", "supports", "tests", "rejects", "justifies", "validated_by"}
        if relation not in allowed:
            raise ValueError(f"unsupported evidence relation: {relation}")
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO evidence_edges(source_id,relation,target_id) VALUES(?,?,?)",
                (source_id, relation, target_id),
            )

    def save_policy(self, payload: dict[str, Any], *, status: str = "VALID") -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO policies(policy_id,model_id,status,payload_json,created_at) VALUES(?,?,?,?,?)",
                (payload["policy_id"], payload["model_id"], status, self._json(payload), payload["created_at"]),
            )

    def list_runs(self, *, eligible_only: bool = False, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT r.manifest_json, v.payload_json AS validation_json FROM runs r LEFT JOIN validations v USING(run_id)"
        parameters: list[Any] = []
        if eligible_only:
            query += " WHERE v.policy_eligible=1"
        query += " ORDER BY r.created_at DESC LIMIT ?"
        parameters.append(limit)
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "manifest": json.loads(row["manifest_json"]),
                "validation": json.loads(row["validation_json"]) if row["validation_json"] else None,
            }
            for row in rows
        ]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT manifest_json,artifact_path FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return {"manifest": json.loads(row["manifest_json"]), "artifact_path": row["artifact_path"]}

    def list_policies(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT payload_json,status FROM policies ORDER BY created_at DESC").fetchall()
        return [{**json.loads(row["payload_json"]), "status": row["status"]} for row in rows]

    def get_policy(self, policy_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT payload_json,status FROM policies WHERE policy_id=?", (policy_id,)).fetchone()
        return None if row is None else {**json.loads(row["payload_json"]), "status": row["status"]}

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT payload_json FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
        return None if row is None else json.loads(row["payload_json"])

    def get_evidence_chain(self, evidence_id: str) -> dict[str, Any] | None:
        node = self.get_evidence(evidence_id)
        if node is None:
            return None
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT source_id,relation,target_id FROM evidence_edges
                   WHERE source_id=? OR target_id=? ORDER BY source_id,relation,target_id""",
                (evidence_id, evidence_id),
            ).fetchall()
        return {"evidence": node, "edges": [dict(row) for row in rows]}
