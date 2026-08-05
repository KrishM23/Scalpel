"""Durable job store (SQLite) + background execution of edit jobs.

Jobs survive process restarts; execution happens on a bounded thread pool so
the API stays responsive while surgeries run. Each job's full audit report is
persisted alongside its status.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from scalpel.api.schemas import EditJobRequest
from scalpel.biases.catalog import get_bias_spec, spec_from_payload
from scalpel.config import Settings
from scalpel.editing.surgeon import SurgeryConfig
from scalpel.pipelines.debias import run_debias_pipeline

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    model_id TEXT NOT NULL,
    bias_name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error TEXT,
    artifact_dir TEXT,
    report_json TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def create(self, tenant: str, model_id: str, bias_name: str) -> str:
        job_id = f"job_{uuid.uuid4().hex[:20]}"
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, tenant, model_id, bias_name, status, created_at,"
                " updated_at) VALUES (?, ?, ?, ?, 'queued', ?, ?)",
                (job_id, tenant, model_id, bias_name, now, now),
            )
        return job_id

    def update(self, job_id: str, **fields) -> None:
        fields["updated_at"] = _now()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                (*fields.values(), job_id),
            )

    def get(self, job_id: str, tenant: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ? AND tenant = ?", (job_id, tenant)
            ).fetchone()
        return dict(row) if row else None

    def list_for_tenant(self, tenant: str, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE tenant = ? ORDER BY created_at DESC LIMIT ?",
                (tenant, limit),
            ).fetchall()
        return [dict(row) for row in rows]


class JobRunner:
    """Executes edit jobs on a bounded worker pool."""

    def __init__(self, store: JobStore, settings: Settings):
        self.store = store
        self.settings = settings
        self.executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_jobs, thread_name_prefix="scalpel-job"
        )

    def submit(self, job_id: str, tenant: str, request: EditJobRequest) -> None:
        self.executor.submit(self._run, job_id, tenant, request)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, tenant: str, request: EditJobRequest) -> None:
        self.store.update(job_id, status="running")
        try:
            if isinstance(request.bias, str):
                spec = get_bias_spec(request.bias)
            else:
                spec = spec_from_payload(request.bias.model_dump())

            config = SurgeryConfig(
                max_components=request.options.max_components,
                cumulative_share=request.options.cumulative_share,
                harden_projection=request.options.harden_projection,
                edit_bias_terms=request.options.edit_bias_terms,
                direction_layer=request.options.direction_layer,
                device=self.settings.device,
            )
            save_dir = None
            if request.save_artifact:
                save_dir = self.settings.artifact_dir / tenant / job_id

            result = run_debias_pipeline(
                model_id=request.model_id, bias=spec, config=config, save_dir=save_dir
            )
            self.store.update(
                job_id,
                status="succeeded",
                report_json=json.dumps(result.report),
                artifact_dir=str(result.artifact_path) if result.artifact_path else None,
            )
        except Exception as exc:  # noqa: BLE001 - job boundary
            self.store.update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
