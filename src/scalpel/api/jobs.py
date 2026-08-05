"""Durable job store (SQLite) + background execution of edit jobs.

Jobs survive process restarts; execution happens on a bounded thread pool so
the API stays responsive while surgeries run. Each job's full audit report is
persisted alongside its status.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

from scalpel.api.schemas import EditJobRequest
from scalpel.biases.catalog import get_bias_spec, spec_from_payload
from scalpel.config import Settings
from scalpel.editing.surgeon import SurgeryConfig
from scalpel.pipelines.debias import run_debias_pipeline
from scalpel.reporting import render_report_html

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    model_id TEXT NOT NULL,
    bias_name TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'edit',
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
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Additive migrations for databases created by older versions."""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "mode" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN mode TEXT NOT NULL DEFAULT 'edit'")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def create(self, tenant: str, model_id: str, bias_name: str, mode: str = "edit") -> str:
        job_id = f"job_{uuid.uuid4().hex[:20]}"
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, tenant, model_id, bias_name, mode, status,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
                (job_id, tenant, model_id, bias_name, mode, now, now),
            )
        return job_id

    def count_jobs_this_month(self, tenant: str) -> int:
        month_start = datetime.now(timezone.utc).strftime("%Y-%m-01T00:00:00")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE tenant = ? AND created_at >= ?",
                (tenant, month_start),
            ).fetchone()
        return int(row[0])

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
                num_directions=request.options.num_directions,
                calibrate=request.options.calibrate,
                harden_projection=request.options.harden_projection,
                edit_bias_terms=request.options.edit_bias_terms,
                direction_layer=request.options.direction_layer,
                device=self.settings.device,
            )
            save_dir = None
            if request.save_artifact:
                save_dir = self.settings.artifact_dir / tenant / job_id

            result = run_debias_pipeline(
                model_id=request.model_id,
                bias=spec,
                config=config,
                save_dir=save_dir,
                audit_only=(request.mode == "audit"),
            )

            if result.artifact_path is not None:
                self._package_artifacts(result.artifact_path, result.report)

            self.store.update(
                job_id,
                status="succeeded",
                report_json=json.dumps(result.report),
                artifact_dir=str(result.artifact_path) if result.artifact_path else None,
            )
            self._notify(request, job_id, "succeeded", result.report)
        except Exception as exc:  # noqa: BLE001 - job boundary
            error = f"{type(exc).__name__}: {exc}"
            self.store.update(job_id, status="failed", error=error)
            self._notify(request, job_id, "failed", {"error": error})

    @staticmethod
    def _package_artifacts(artifact_path: Path, report: dict) -> None:
        """Write the HTML compliance report and zip edited weights for download."""
        (artifact_path / "report.html").write_text(render_report_html(report))
        model_dir = artifact_path / "model"
        if model_dir.is_dir():
            shutil.make_archive(str(artifact_path / "model"), "zip", model_dir)

    @staticmethod
    def _notify(request: EditJobRequest, job_id: str, status: str, report: dict) -> None:
        """Fire-and-forget webhook on job completion."""
        if not request.webhook_url:
            return
        payload = {
            "job_id": job_id,
            "status": status,
            "model_id": request.model_id,
            "mode": request.mode,
            "metrics": report.get("metrics"),
            "error": report.get("error"),
        }
        try:
            httpx.post(request.webhook_url, json=payload, timeout=10.0)
        except httpx.HTTPError:
            pass  # delivery is best-effort; the job result is still queryable
