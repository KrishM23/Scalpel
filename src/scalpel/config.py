"""Platform configuration, sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


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
    device: str = field(default_factory=lambda: os.environ.get("SCALPEL_DEVICE", "cpu"))
    max_concurrent_jobs: int = field(
        default_factory=lambda: int(os.environ.get("SCALPEL_MAX_CONCURRENT_JOBS", "1"))
    )


def get_settings() -> Settings:
    return Settings()
