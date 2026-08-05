"""Commercial plans and quota enforcement.

The pricing structure implements the standard land-and-expand motion:

- ``free``       — audit-only. Prospects measure their bias and see exactly
                   which circuit causes it, but must upgrade to remove it.
- ``pro``        — full editing, capped monthly volume.
- ``enterprise`` — unlimited volume (custom contract).

Tenants are mapped to plans via SCALPEL_TENANT_PLANS ("acme:pro,beta:free");
unmapped tenants get SCALPEL_DEFAULT_PLAN (default: enterprise, so
single-tenant self-hosted deployments are unrestricted out of the box).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    name: str
    monthly_job_limit: int | None  # None = unlimited
    allows_edit: bool


PLANS: dict[str, Plan] = {
    "free": Plan(name="free", monthly_job_limit=10, allows_edit=False),
    "pro": Plan(name="pro", monthly_job_limit=100, allows_edit=True),
    "enterprise": Plan(name="enterprise", monthly_job_limit=None, allows_edit=True),
}


def plan_for_tenant(tenant: str, tenant_plans: dict[str, str], default_plan: str) -> Plan:
    """Resolve the commercial plan for a tenant.

    When ``SCALPEL_OPEN_KEYS`` is enabled, unmapped tenants (including
    per-key open-mode tenants) always receive enterprise so every key can
    run full edit surgeries.
    """
    from scalpel.api.auth import open_keys_enabled

    if tenant in tenant_plans:
        name = tenant_plans[tenant]
    elif open_keys_enabled():
        name = "enterprise"
    else:
        name = default_plan
    if name not in PLANS:
        raise ValueError(f"Unknown plan '{name}' (available: {sorted(PLANS)})")
    return PLANS[name]
