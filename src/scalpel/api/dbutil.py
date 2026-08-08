"""Tiny SQLite / Postgres helper for account storage.

Local default: SQLite file (``SCALPEL_DB_PATH``).
Production / Netlify API: set ``DATABASE_URL`` or ``SCALPEL_DATABASE_URL``
to a Postgres URL (Neon, Supabase, RDS, Railway, …).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal


Backend = Literal["sqlite", "postgres"]


@dataclass
class DbConfig:
    backend: Backend
    sqlite_path: Path | None = None
    database_url: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        database_url: str | None,
        sqlite_path: Path,
    ) -> DbConfig:
        url = (database_url or "").strip()
        if url:
            if not (url.startswith("postgres://") or url.startswith("postgresql://")):
                raise ValueError(
                    "DATABASE_URL must be a postgres:// or postgresql:// URL "
                    f"(got scheme from {url.split(':', 1)[0]!r})"
                )
            # psycopg accepts postgresql://; normalize postgres://
            if url.startswith("postgres://"):
                url = "postgresql://" + url[len("postgres://") :]
            return cls(backend="postgres", database_url=url)
        return cls(backend="sqlite", sqlite_path=sqlite_path)


class Db:
    """Connection factory with `?` placeholders on both backends."""

    def __init__(self, config: DbConfig):
        self.config = config
        if config.backend == "sqlite":
            assert config.sqlite_path is not None
            config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        elif config.backend == "postgres":
            try:
                import psycopg  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Postgres DATABASE_URL is set but psycopg is not installed. "
                    'Install with: pip install "scalpel-ai[postgres]"'
                ) from exc

    def _sql(self, sql: str) -> str:
        if self.config.backend == "postgres":
            return sql.replace("?", "%s")
        return sql

    @contextmanager
    def connect(self) -> Iterator[Any]:
        if self.config.backend == "sqlite":
            conn = sqlite3.connect(self.config.sqlite_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                yield _SqliteConn(conn, self)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            return

        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(self.config.database_url, row_factory=dict_row) as conn:
            try:
                yield _PgConn(conn, self)
                conn.commit()
            except Exception:
                conn.rollback()
                raise


class _SqliteConn:
    def __init__(self, conn: sqlite3.Connection, db: Db):
        self._conn = conn
        self._db = db

    def execute(self, sql: str, params: tuple | list = ()):
        return self._conn.execute(self._db._sql(sql), params)

    def executescript(self, sql: str) -> None:
        self._conn.executescript(sql)


class _PgConn:
    def __init__(self, conn: Any, db: Db):
        self._conn = conn
        self._db = db

    def execute(self, sql: str, params: tuple | list = ()):
        cur = self._conn.cursor()
        cur.execute(self._db._sql(sql), params)
        return cur

    def executescript(self, sql: str) -> None:
        # Split on semicolons for simple schema scripts.
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                self.execute(stmt)
