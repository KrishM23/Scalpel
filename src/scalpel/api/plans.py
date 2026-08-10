"""Commercial plans and quota enforcement.

Land-and-expand is the product: free proves you *need* a cut; paid ships it.

Primary buyer: ML / Trust & Safety leads at adtech, brand-safety, and creative-
intelligence companies that already run open CLIP (or similar) for ranking,
tagging, or moderation. They pay when a brand escalation needs a deployable
edit plus a PDF — not a prompt wrapper.

- ``free``       — audit-only. Measure ad creative gaps (gender×product,
                   age×luxury, ethnicity×brand) and see the circuit. You cannot
                   export edited weights. That pain is intentional.
- ``pro``        — full Measure→Locate→Cut→Prove, capped monthly volume,
                   shareable PDF/recipe/weight artifacts for models you deploy.
- ``enterprise`` — unlimited volume, custom contract, multi-market bias packs.

Why they pay: closed APIs never give you a file you can version in the ad
stack. Scalpel edits *your* weights so mitigation is owned, not rented.

Tenants are mapped via SCALPEL_TENANT_PLANS ("acme:pro,beta:free");
unmapped tenants get SCALPEL_DEFAULT_PLAN (default: enterprise for
single-tenant self-host).
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
