"""Persistent workspace accounts (signup / login).

Backed by SQLite locally, or Postgres when ``DATABASE_URL`` /
``SCALPEL_DATABASE_URL`` is set (required for durable signup behind Netlify).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scalpel.api.dbutil import Db, DbConfig

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    company TEXT NOT NULL,
    tenant TEXT NOT NULL UNIQUE,
    api_key TEXT NOT NULL UNIQUE,
    plan TEXT NOT NULL DEFAULT 'free',
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"pbkdf2_sha256$200000${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algo, rounds, salt_hex, digest_hex = encoded.split("$", 3)
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, int(rounds)
    )
    return hmac.compare_digest(actual, expected)


def _slug_tenant(company: str, email: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", (company or email.split("@")[0]).lower()).strip("_")
    base = (base or "workspace")[:28]
    return f"{base}_{secrets.token_hex(3)}"


def _issue_api_key() -> str:
    return f"sk_live_{secrets.token_urlsafe(24)}"


@dataclass(frozen=True)
class UserAccount:
    id: str
    email: str
    name: str
    company: str
    tenant: str
    api_key: str
    plan: str
    created_at: str

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "company": self.company,
            "tenant": self.tenant,
            "api_key": self.api_key,
            "plan": self.plan,
            "created_at": self.created_at,
        }


class UserStore:
    def __init__(self, db: Db):
        self.db = db
        with self.db.connect() as conn:
            conn.executescript(_SCHEMA)

    @classmethod
    def open(
        cls,
        *,
        database_url: str | None = None,
        sqlite_path: Path | None = None,
    ) -> UserStore:
        path = sqlite_path or Path("artifacts/scalpel.db")
        config = DbConfig.from_env(database_url=database_url, sqlite_path=path)
        return cls(Db(config))

    @property
    def backend(self) -> str:
        return self.db.config.backend

    @staticmethod
    def _row(row: Any) -> UserAccount:
        return UserAccount(
            id=row["id"],
            email=row["email"],
            name=row["name"],
            company=row["company"],
            tenant=row["tenant"],
            api_key=row["api_key"],
            plan=row["plan"],
            created_at=row["created_at"],
        )

    def list_api_key_entries(self) -> list[str]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT tenant, api_key FROM users").fetchall()
        return [f"{row['tenant']}:{row['api_key']}" for row in rows]

    def list_tenant_plans(self) -> dict[str, str]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT tenant, plan FROM users").fetchall()
        return {row["tenant"]: row["plan"] for row in rows}

    def get_by_email(self, email: str) -> UserAccount | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
            ).fetchone()
        return self._row(row) if row else None

    def create(
        self,
        *,
        email: str,
        password: str,
        name: str,
        company: str,
        plan: str = "free",
    ) -> UserAccount:
        email = email.lower().strip()
        if not email or "@" not in email:
            raise ValueError("A valid email is required")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        name = (name or "").strip() or email.split("@")[0]
        company = (company or "").strip() or name
        if self.get_by_email(email):
            raise ValueError("An account with this email already exists")

        account = UserAccount(
            id=f"usr_{uuid.uuid4().hex[:16]}",
            email=email,
            name=name,
            company=company,
            tenant=_slug_tenant(company, email),
            api_key=_issue_api_key(),
            plan=plan if plan in {"free", "pro", "enterprise"} else "free",
            created_at=_now(),
        )
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, name, company, tenant,"
                " api_key, plan, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    account.id,
                    account.email,
                    _hash_password(password),
                    account.name,
                    account.company,
                    account.tenant,
                    account.api_key,
                    account.plan,
                    account.created_at,
                ),
            )
        return account

    def authenticate(self, email: str, password: str) -> UserAccount | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
            ).fetchone()
        if row is None:
            return None
        if not _verify_password(password, row["password_hash"]):
            return None
        return self._row(row)
