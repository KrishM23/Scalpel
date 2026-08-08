"""Platform configuration, sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_mapping(name: str) -> dict[str, str]:
    """Parse "key:value,key:value" env entries."""
    mapping: dict[str, str] = {}
    for entry in _env_list(name):
        if ":" in entry:
            key, value = entry.split(":", 1)
            mapping[key.strip()] = value.strip()
    return mapping


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime settings for the Scalpel API service."""

    api_keys: list[str] = field(default_factory=lambda: _env_list("SCALPEL_API_KEYS"))
    artifact_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("SCALPEL_ARTIFACT_DIR", "artifacts"))
    )
    database_path: Path = field(
        default_factory=lambda: Path(os.environ.get("SCALPEL_DB_PATH", "artifacts/scalpel.db"))
    )
    # Postgres URL for durable accounts (Neon/Supabase/RDS). When set, signup
    # users are stored in Postgres; job store still uses database_path (SQLite)
    # unless you point both at the same operational volume.
    database_url: str | None = field(
        default_factory=lambda: (
            os.environ.get("DATABASE_URL")
            or os.environ.get("SCALPEL_DATABASE_URL")
            or None
        )
    )
    # Public origin of this API when the marketing site is hosted separately
    # (e.g. Netlify). Injected into static auth pages as window.SCALPEL_API_BASE.
    public_api_url: str = field(
        default_factory=lambda: (
            os.environ.get("SCALPEL_PUBLIC_API_URL")
            or os.environ.get("PUBLIC_API_URL")
            or ""
        ).rstrip("/")
    )
    device: str = field(default_factory=lambda: os.environ.get("SCALPEL_DEVICE", "cpu"))
    max_concurrent_jobs: int = field(
        default_factory=lambda: int(os.environ.get("SCALPEL_MAX_CONCURRENT_JOBS", "1"))
    )
    tenant_plans: dict[str, str] = field(
        default_factory=lambda: _env_mapping("SCALPEL_TENANT_PLANS")
    )
    default_plan: str = field(
        default_factory=lambda: os.environ.get("SCALPEL_DEFAULT_PLAN", "enterprise")
    )
    # Ops / alerts
    alert_weat_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SCALPEL_ALERT_WEAT_THRESHOLD", "0.5"))
    )
    alert_overcorrection_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("SCALPEL_ALERT_OVERCORRECTION_THRESHOLD", "0.3")
        )
    )
    # Jobs stuck queued/running longer than this are failed so the pool unblocks.
    job_queued_timeout_seconds: int = field(
        default_factory=lambda: int(os.environ.get("SCALPEL_JOB_QUEUED_TIMEOUT_S", "900"))
    )
    job_running_timeout_seconds: int = field(
        default_factory=lambda: int(os.environ.get("SCALPEL_JOB_RUNNING_TIMEOUT_S", "3600"))
    )
    # Deployment
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("SCALPEL_CORS_ORIGINS")
    )
    webhook_secret: str = field(
        default_factory=lambda: os.environ.get("SCALPEL_WEBHOOK_SECRET", "").strip()
    )
    require_api_keys: bool = field(
        default_factory=lambda: _env_bool("SCALPEL_REQUIRE_API_KEYS", True)
    )
    # Consumer signup (workspace accounts). Disable to run key-provisioned only.
    public_signup: bool = field(
        default_factory=lambda: _env_bool("SCALPEL_PUBLIC_SIGNUP", True)
    )
    # Max signups per client IP per hour (in-memory; resets on process restart).
    signup_rate_limit_per_hour: int = field(
        default_factory=lambda: int(os.environ.get("SCALPEL_SIGNUP_RATE_LIMIT", "10"))
    )


def get_settings() -> Settings:
    return Settings()
