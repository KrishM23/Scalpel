"""Durable job store (SQLite) + background execution of edit jobs.

Jobs survive process restarts; execution happens on a bounded thread pool so
the API stays responsive while surgeries run. Each job's full audit report is
persisted alongside its status. Incomplete jobs are recovered on startup when
the original request payload was stored.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import shutil
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from scalpel.api.schemas import EditJobRequest
from scalpel.biases.catalog import get_bias_spec, spec_from_payload
from scalpel.config import Settings
from scalpel.editing.surgeon import SurgeryConfig
from scalpel.pipelines.debias import run_debias_pipeline
from scalpel.reporting import render_report_html

log = logging.getLogger(__name__)

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
    report_json TEXT,
    request_json TEXT
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
        if "request_json" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN request_json TEXT")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def create(
        self,
        tenant: str,
        model_id: str,
        bias_name: str,
        mode: str = "edit",
        request_json: str | None = None,
    ) -> str:
        job_id = f"job_{uuid.uuid4().hex[:20]}"
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, tenant, model_id, bias_name, mode, status,"
                " created_at, updated_at, request_json) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
                (job_id, tenant, model_id, bias_name, mode, now, now, request_json),
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

    def list_incomplete(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') "
                "ORDER BY created_at ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def reap_stale(
        self, *, queued_timeout_s: int, running_timeout_s: int
    ) -> list[str]:
        """Fail jobs that have been queued/running longer than the allowed window."""
        now = datetime.now(timezone.utc)
        failed_ids: list[str] = []
        for row in self.list_incomplete():
            try:
                updated = datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
            except (TypeError, ValueError):
                updated = now - timedelta(days=1)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            age = (now - updated).total_seconds()
            status = row["status"]
            if status == "queued" and age > queued_timeout_s:
                msg = (
                    f"Timed out after {int(age)}s in queue "
                    f"(limit {queued_timeout_s}s). Resubmit or raise "
                    "SCALPEL_MAX_CONCURRENT_JOBS / use a smaller model."
                )
            elif status == "running" and age > running_timeout_s:
                msg = (
                    f"Timed out after {int(age)}s running "
                    f"(limit {running_timeout_s}s). Resubmit with "
                    "save_artifact=false or a smaller model."
                )
            else:
                continue
            self.update(row["id"], status="failed", error=msg)
            failed_ids.append(row["id"])
            log.warning("Reaped stale job %s (%s)", row["id"], status)
        return failed_ids


class JobRunner:
    """Executes edit jobs on a bounded worker pool."""

    def __init__(self, store: JobStore, settings: Settings):
        self.store = store
        self.settings = settings
        self._inflight: set[str] = set()
        self.executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_jobs, thread_name_prefix="scalpel-job"
        )

    def submit(self, job_id: str, tenant: str, request: EditJobRequest) -> None:
        if job_id in self._inflight:
            return
        self._inflight.add(job_id)
        self.executor.submit(self._run_guarded, job_id, tenant, request)

    def reap_stale(self) -> list[str]:
        return self.store.reap_stale(
            queued_timeout_s=self.settings.job_queued_timeout_seconds,
            running_timeout_s=self.settings.job_running_timeout_seconds,
        )

    def recover(self) -> int:
        """Requeue incomplete jobs after a process restart.

        Stale jobs are failed first. Jobs without a persisted request payload
        cannot be resumed and are marked failed so they do not starve the pool.
        """
        self.reap_stale()
        recovered = 0
        for row in self.store.list_incomplete():
            job_id = row["id"]
            raw = row.get("request_json")
            if not raw:
                self.store.update(
                    job_id,
                    status="failed",
                    error="Interrupted by server restart; resubmit the job",
                )
                log.warning("Abandoned incomplete job %s (no request payload)", job_id)
                continue
            try:
                request = EditJobRequest.model_validate_json(raw)
            except Exception as exc:  # noqa: BLE001 - corrupt payload boundary
                self.store.update(
                    job_id,
                    status="failed",
                    error=f"Invalid stored request after restart: {exc}",
                )
                continue
            self.store.update(job_id, status="queued", error=None)
            self.submit(job_id, row["tenant"], request)
            recovered += 1
            log.info("Requeued job %s after restart", job_id)
        return recovered

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _run_guarded(self, job_id: str, tenant: str, request: EditJobRequest) -> None:
        try:
            self._run(job_id, tenant, request)
        finally:
            self._inflight.discard(job_id)

    def _run(self, job_id: str, tenant: str, request: EditJobRequest) -> None:
        self.store.update(job_id, status="running", error=None)
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
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.exception("Job %s failed", job_id)
            self.store.update(job_id, status="failed", error=error)
            self._notify(request, job_id, "failed", {"error": error})

    @staticmethod
    def _package_artifacts(artifact_path: Path, report: dict) -> None:
        """Write the HTML compliance report and zip edited weights for download."""
        (artifact_path / "report.html").write_text(render_report_html(report))
        model_dir = artifact_path / "model"
        if model_dir.is_dir():
            shutil.make_archive(str(artifact_path / "model"), "zip", model_dir)

    def _notify(self, request: EditJobRequest, job_id: str, status: str, report: dict) -> None:
        """Fire-and-forget webhook on job completion (optionally HMAC-signed)."""
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
        body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        secret = self.settings.webhook_secret
        if secret:
            digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
            headers["X-Scalpel-Signature"] = f"sha256={digest}"
        try:
            httpx.post(
                request.webhook_url, content=body, headers=headers, timeout=10.0
            )
        except httpx.HTTPError:
            pass  # delivery is best-effort; the job result is still queryable
