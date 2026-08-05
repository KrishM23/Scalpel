"""API-key authentication for enterprise tenants.

Keys are provisioned out of band and supplied via the SCALPEL_API_KEYS
environment variable (comma-separated ``tenant:key`` entries, or bare keys
which map to the ``default`` tenant). Comparison is constant-time.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status


def parse_key_entries(entries: list[str]) -> dict[str, str]:
    """Map raw key -> tenant name."""
    keys: dict[str, str] = {}
    for entry in entries:
        if ":" in entry:
            tenant, key = entry.split(":", 1)
        else:
            tenant, key = "default", entry
        if key:
            keys[key] = tenant
    return keys


def require_tenant(request: Request) -> str:
    """FastAPI dependency: validate X-API-Key and return the tenant name."""
    keys: dict[str, str] = request.app.state.api_keys
    if not keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No API keys configured; set SCALPEL_API_KEYS",
        )
    presented = request.headers.get("X-API-Key", "")
    for key, tenant in keys.items():
        if hmac.compare_digest(presented, key):
            return tenant
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
    )
