"""API-key authentication for enterprise tenants.

Keys are provisioned out of band and supplied via the SCALPEL_API_KEYS
environment variable (comma-separated ``tenant:key`` entries, or bare keys
which map to the ``default`` tenant). Comparison is constant-time.

Accepts either ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.
When ``SCALPEL_OPEN_KEYS`` is enabled (local/dev), any non-empty key is
accepted and mapped to a stable per-key tenant so every key gets a full
isolated workspace.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re

from fastapi import HTTPException, Request, status


def parse_key_entries(entries: list[str]) -> dict[str, str]:
    """Map raw key -> tenant name. Strips whitespace; ignores empty entries."""
    keys: dict[str, str] = {}
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            tenant, key = entry.split(":", 1)
            tenant, key = tenant.strip(), key.strip()
        else:
            tenant, key = "default", entry
        if key:
            keys[key] = tenant or "default"
    return keys


def extract_api_key(request: Request) -> str:
    """Pull the presented key from X-API-Key or Authorization: Bearer."""
    presented = (request.headers.get("X-API-Key") or "").strip()
    if presented:
        return presented
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def open_keys_enabled() -> bool:
    """True when any non-empty key is accepted (local/dev / demos)."""
    return os.environ.get("SCALPEL_OPEN_KEYS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tenant_for_open_key(key: str) -> str:
    """Stable tenant id derived from the key so each open key is isolated."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    # Prefer a readable slug if the key looks like sk_tenant_xxx
    match = re.match(r"^sk[_-]?([a-zA-Z][a-zA-Z0-9_-]{1,32})", key)
    if match:
        return match.group(1).lower().replace("-", "_")
    return f"tenant_{digest}"


def require_tenant(request: Request) -> str:
    """FastAPI dependency: validate API key and return the tenant name."""
    keys: dict[str, str] = request.app.state.api_keys
    presented = extract_api_key(request)

    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Pass X-API-Key or Authorization: Bearer <key>",
        )

    for key, tenant in keys.items():
        if hmac.compare_digest(presented, key):
            return tenant

    # Local/dev escape hatch: any key works, each gets its own tenant + full plan.
    if open_keys_enabled():
        return _tenant_for_open_key(presented)

    if not keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No API keys configured; set SCALPEL_API_KEYS (or SCALPEL_OPEN_KEYS=1 for local use)",
        )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
    )
