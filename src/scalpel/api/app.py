"""Scalpel enterprise API (v1).

Public pages:
- GET  /                          marketing landing
- GET  /product · /developers · /security
- GET  /privacy · /terms
- GET  /login · /signup           workspace auth
- GET  /app                       ops console
- GET  /static/*                  brand assets + shared CSS

Endpoints (all under /v1, authenticated via X-API-Key or Bearer):

- POST /v1/auth/signup            create workspace + API key
- POST /v1/auth/login             exchange credentials for API key
- GET  /v1/models                 featured models + supported families
- GET  /v1/models/probe           classify any Hugging Face model id
- GET  /v1/biases                 built-in bias benchmark catalog
- POST /v1/edit-jobs              queue a debiasing surgery (202)
- GET  /v1/edit-jobs              list this tenant's jobs
- GET  /v1/edit-jobs/{id}         job status + summary metrics
- GET  /v1/edit-jobs/{id}/report  full audit report (compliance artifact)
- GET  /v1/alerts                 active workspace alerts
- GET  /v1/usage                  plan + quota

GET /health and /ready are unauthenticated for load balancers.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import scalpel
from scalpel.api.alerts import collect_alerts
from scalpel.api.auth import open_keys_enabled, parse_key_entries, require_tenant
from scalpel.api.jobs import JobRunner, JobStore
from scalpel.api.marketing import render_marketing_page
from scalpel.api.plans import plan_for_tenant
from scalpel.api.schemas import (
    AlertEntry,
    AuthResponse,
    BiasCatalogEntry,
    EditJobDetail,
    EditJobRequest,
    EditJobSummary,
    LoginRequest,
    ModelCatalogResponse,
    ModelProbeResponse,
    SignupRequest,
    UsageResponse,
)
from scalpel.api.users import UserAccount, UserStore
from scalpel.biases.catalog import bias_catalog, spec_from_payload
from scalpel.config import Settings, get_settings
from scalpel.models.adapters import supported_families
from scalpel.models.registry import (
    UnsupportedArchitectureError,
    featured_models,
    probe_model,
)
from scalpel.reporting import render_report_html

log = logging.getLogger("scalpel.api")

_STATIC = Path(__file__).parent / "static"
_SIGNUP_HITS: dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    forwarded = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


def _enforce_signup_rate(ip: str, *, limit: int, window_s: int = 3600) -> None:
    if limit <= 0:
        return
    now = time.time()
    recent = [t for t in _SIGNUP_HITS[ip] if now - t < window_s]
    if len(recent) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many signup attempts from this network. Try again later.",
        )
    recent.append(now)
    _SIGNUP_HITS[ip] = recent


def _html_page(name: str, *, replacements: dict[str, str] | None = None) -> HTMLResponse:
    html = (_STATIC / name).read_text()
    html = html.replace("__SCALPEL_VERSION__", scalpel.__version__)
    reps = dict(replacements or {})
    if "__API_BASE__" not in reps:
        reps["__API_BASE__"] = ""
    for key, value in reps.items():
        html = html.replace(key, value)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        app.state.user_store = UserStore.open(
            database_url=settings.database_url,
            sqlite_path=settings.database_path,
        )
        log.info("User store backend: %s", app.state.user_store.backend)
        if app.state.user_store.backend == "sqlite" and settings.public_signup:
            log.warning(
                "Public signup is using SQLite (%s). For Netlify/production, set "
                "DATABASE_URL to Postgres so accounts persist across deploys.",
                settings.database_path,
            )
        app.state.api_keys = parse_key_entries(settings.api_keys)
        app.state.api_keys.update(
            parse_key_entries(app.state.user_store.list_api_key_entries())
        )
        app.state.tenant_plans = {
            **settings.tenant_plans,
            **app.state.user_store.list_tenant_plans(),
        }
        if settings.require_api_keys and not app.state.api_keys and not open_keys_enabled():
            log.error(
                "No SCALPEL_API_KEYS configured and SCALPEL_OPEN_KEYS is off. "
                "Set API keys or allow signup before accepting production traffic."
            )
        app.state.store = JobStore(settings.database_path)
        app.state.runner = JobRunner(app.state.store, settings)
        # Resume (or cleanly fail) jobs left incomplete by a prior process.
        recovered = app.state.runner.recover()
        if recovered:
            log.info("Requeued %d incomplete job(s) after restart", recovered)
        yield
        app.state.runner.shutdown()

    def _register_account(account: UserAccount) -> None:
        app.state.api_keys[account.api_key] = account.tenant
        app.state.tenant_plans[account.tenant] = account.plan

    app = FastAPI(
        title="Scalpel — Model Editing & Bias Mitigation API",
        version=scalpel.__version__,
        description=__doc__,
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def security_and_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        # Console is same-origin; APIs are key-authenticated.
        if request.url.path.startswith("/v1"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        store: UserStore = app.state.user_store
        return {
            "status": "ok",
            "version": scalpel.__version__,
            "users_db": store.backend,
        }

    @app.get("/ready", tags=["ops"])
    def ready() -> JSONResponse:
        """Readiness: DB reachable and auth path available."""
        checks: dict[str, str] = {}
        ok = True
        try:
            store: JobStore = app.state.store
            with store._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            checks["database"] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks["database"] = f"error: {exc}"
            ok = False
        # Ready when keys exist, open-keys (dev), or public signup can mint keys.
        keys_ok = (
            bool(getattr(app.state, "api_keys", None))
            or open_keys_enabled()
            or settings.public_signup
        )
        if settings.require_api_keys and not keys_ok:
            checks["auth"] = "no_api_keys"
            ok = False
        else:
            checks["auth"] = "ok"
        body = {"status": "ready" if ok else "not_ready", "checks": checks,
                "version": scalpel.__version__}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/", include_in_schema=False)
    def landing() -> HTMLResponse:
        return render_marketing_page(
            "landing.html",
            title="Scalpel — Surgical model editing",
            description=(
                "Locate latent bias circuits. Erase them with calibrated rank-one "
                "edits. Ship models that hold up under audit."
            ),
        )

    @app.get("/product", include_in_schema=False)
    def product_page() -> HTMLResponse:
        return render_marketing_page(
            "product.html",
            title="Scalpel — Product",
            description="Measure, locate, cut, and prove — surgical bias editing for production models.",
            active="product",
        )

    @app.get("/developers", include_in_schema=False)
    def developers_page() -> HTMLResponse:
        return render_marketing_page(
            "developers.html",
            title="Scalpel — Developers",
            description="API quickstart for queuing edit jobs and fetching audit reports.",
            active="developers",
        )

    @app.get("/security", include_in_schema=False)
    def security_page() -> HTMLResponse:
        return render_marketing_page(
            "security.html",
            title="Scalpel — Security",
            description="Tenant isolation, API authentication, and compliance artifacts.",
            active="security",
        )

    @app.get("/privacy", include_in_schema=False)
    def privacy_page() -> HTMLResponse:
        return render_marketing_page(
            "privacy.html",
            title="Scalpel — Privacy",
            description="How Scalpel handles account and workspace data.",
        )

    @app.get("/terms", include_in_schema=False)
    def terms_page() -> HTMLResponse:
        return render_marketing_page(
            "terms.html",
            title="Scalpel — Terms",
            description="Terms of service for Scalpel’s website, console, and API.",
        )

    @app.get("/app", include_in_schema=False)
    def console() -> HTMLResponse:
        """Ops console. Public HTML; every API call requires a tenant API key."""
        return _html_page("index.html")

    @app.get("/login", include_in_schema=False)
    def login_page() -> HTMLResponse:
        return _html_page(
            "auth.html",
            replacements={
                "__AUTH_MODE__": "login",
                "__AUTH_TITLE__": "Log in",
                "__API_BASE__": settings.public_api_url,
            },
        )

    @app.get("/signup", include_in_schema=False)
    def signup_page() -> HTMLResponse:
        return _html_page(
            "auth.html",
            replacements={
                "__AUTH_MODE__": "signup",
                "__AUTH_TITLE__": "Sign up",
                "__API_BASE__": settings.public_api_url,
            },
        )

    @app.post("/v1/auth/signup", response_model=AuthResponse, tags=["auth"])
    def signup(request: Request, body: SignupRequest) -> AuthResponse:
        if not settings.public_signup:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Public signup is disabled. Request an API key from your admin.",
            )
        _enforce_signup_rate(
            _client_ip(request),
            limit=settings.signup_rate_limit_per_hour,
        )
        store: UserStore = app.state.user_store
        try:
            account = store.create(
                email=body.email,
                password=body.password,
                name=body.name,
                company=body.company,
                plan="free",
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        _register_account(account)
        return AuthResponse(**account.public_dict())

    @app.post("/v1/auth/login", response_model=AuthResponse, tags=["auth"])
    def login(body: LoginRequest) -> AuthResponse:
        store: UserStore = app.state.user_store
        account = store.authenticate(body.email, body.password)
        if account is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        # Ensure runtime key map is current (e.g. after restart already loaded).
        _register_account(account)
        return AuthResponse(**account.public_dict())

    @app.get("/v1/models", response_model=ModelCatalogResponse, tags=["catalog"])
    def list_models(_tenant: str = Depends(require_tenant)) -> ModelCatalogResponse:
        """Featured models plus the architecture families any HF id can use."""
        return ModelCatalogResponse(
            accepts_any_huggingface_id=True,
            families=supported_families(),
            featured=[
                {
                    "model_id": model_id,
                    "family": meta["family"],
                    "description": meta["description"],
                    "featured": True,
                }
                for model_id, meta in featured_models().items()
            ],
        )

    @app.get("/v1/models/probe", response_model=ModelProbeResponse, tags=["catalog"])
    def probe_model_endpoint(
        model_id: str, _tenant: str = Depends(require_tenant)
    ) -> ModelProbeResponse:
        """Classify a Hugging Face model id without downloading weights."""
        try:
            info = probe_model(model_id)
        except UnsupportedArchitectureError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        return ModelProbeResponse(
            model_id=info.model_id,
            family=info.family,
            model_type=info.model_type,
            architecture_key=info.architecture_key,
            description=info.description,
            supported=True,
        )

    @app.get("/v1/biases", response_model=list[BiasCatalogEntry], tags=["catalog"])
    def list_biases(_tenant: str = Depends(require_tenant)) -> list[BiasCatalogEntry]:
        return [
            BiasCatalogEntry(
                name=spec.name,
                description=spec.description,
                groups=[spec.group_a_label, spec.group_b_label],
                num_contrastive_pairs=len(spec.paired_prompts),
                num_probes=len(spec.probe_set_1) + len(spec.probe_set_2),
            )
            for spec in bias_catalog().values()
        ]

    @app.get("/v1/alerts", response_model=list[AlertEntry], tags=["ops"])
    def list_alerts(tenant: str = Depends(require_tenant)) -> list[AlertEntry]:
        """Active alerts for this tenant (failures, unremediated / residual bias)."""
        app.state.runner.reap_stale()
        jobs = app.state.store.list_for_tenant(tenant)
        alerts = collect_alerts(
            jobs,
            weat_threshold=settings.alert_weat_threshold,
            overcorrection_threshold=settings.alert_overcorrection_threshold,
        )
        return [AlertEntry(**a.to_dict()) for a in alerts]

    @app.post(
        "/v1/edit-jobs",
        response_model=EditJobSummary,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["editing"],
    )
    def create_edit_job(
        request: EditJobRequest, tenant: str = Depends(require_tenant)
    ) -> EditJobSummary:
        if isinstance(request.bias, str):
            if request.bias not in bias_catalog():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Unknown bias spec '{request.bias}'",
                )
            bias_name = request.bias
        else:
            try:
                spec_from_payload(request.bias.model_dump())
            except (ValueError, KeyError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
                ) from exc
            bias_name = request.bias.name

        try:
            probe_model(request.model_id)
        except UnsupportedArchitectureError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

        # Commercial plan enforcement.
        plan = plan_for_tenant(tenant, app.state.tenant_plans, settings.default_plan)
        if request.mode == "edit" and not plan.allows_edit:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"Plan '{plan.name}' is audit-only. Run mode='audit' to locate the "
                    "bias circuit, or upgrade to a paid plan to apply edits."
                ),
            )
        used = app.state.store.count_jobs_this_month(tenant)
        if plan.monthly_job_limit is not None and used >= plan.monthly_job_limit:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"Monthly job quota reached ({used}/{plan.monthly_job_limit} on "
                    f"plan '{plan.name}'). Upgrade to raise the limit."
                ),
            )

        job_id = app.state.store.create(
            tenant,
            request.model_id,
            bias_name,
            request.mode,
            request_json=request.model_dump_json(),
        )
        app.state.runner.submit(job_id, tenant, request)
        return _summary(app.state.store.get(job_id, tenant))

    @app.get("/v1/usage", response_model=UsageResponse, tags=["billing"])
    def usage(tenant: str = Depends(require_tenant)) -> UsageResponse:
        plan = plan_for_tenant(tenant, app.state.tenant_plans, settings.default_plan)
        return UsageResponse(
            tenant=tenant,
            plan=plan.name,
            jobs_this_month=app.state.store.count_jobs_this_month(tenant),
            monthly_job_limit=plan.monthly_job_limit,
            allows_edit=plan.allows_edit,
        )

    @app.get("/v1/edit-jobs", response_model=list[EditJobSummary], tags=["editing"])
    def list_edit_jobs(tenant: str = Depends(require_tenant)) -> list[EditJobSummary]:
        # Opportunistic cleanup so a wedged queue cannot linger forever.
        app.state.runner.reap_stale()
        return [_summary(row) for row in app.state.store.list_for_tenant(tenant)]

    @app.get("/v1/edit-jobs/{job_id}", response_model=EditJobDetail, tags=["editing"])
    def get_edit_job(job_id: str, tenant: str = Depends(require_tenant)) -> EditJobDetail:
        row = _get_or_404(job_id, tenant)
        report = json.loads(row["report_json"]) if row["report_json"] else None
        summary = report.get("metrics") if report else None
        return EditJobDetail(**_summary(row).model_dump(), report=summary)

    @app.get("/v1/edit-jobs/{job_id}/report", tags=["editing"])
    def get_edit_job_report(job_id: str, tenant: str = Depends(require_tenant)) -> dict:
        row = _get_or_404(job_id, tenant)
        if not row["report_json"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job is '{row['status']}'; report not available yet",
            )
        return json.loads(row["report_json"])

    @app.get("/v1/edit-jobs/{job_id}/report.html", tags=["editing"])
    def get_edit_job_report_html(
        job_id: str, tenant: str = Depends(require_tenant)
    ) -> HTMLResponse:
        """Standalone HTML compliance report (printable, shareable)."""
        row = _get_or_404(job_id, tenant)
        if not row["report_json"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job is '{row['status']}'; report not available yet",
            )
        return HTMLResponse(render_report_html(json.loads(row["report_json"])))

    @app.get("/v1/edit-jobs/{job_id}/artifact", tags=["editing"])
    def download_artifact(job_id: str, tenant: str = Depends(require_tenant)):
        """Download the edited model weights as a zip archive."""
        from fastapi.responses import FileResponse

        row = _get_or_404(job_id, tenant)
        archive = Path(row["artifact_dir"] or "") / "model.zip"
        if not row["artifact_dir"] or not archive.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No weight artifact for this job (audit-only, unfinished, "
                "or save_artifact=false)",
            )
        return FileResponse(archive, filename=f"{job_id}-model.zip")

    def _get_or_404(job_id: str, tenant: str) -> dict:
        row = app.state.store.get(job_id, tenant)
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
            )
        return row

    def _summary(row: dict) -> EditJobSummary:
        return EditJobSummary(
            id=row["id"],
            tenant=row["tenant"],
            model_id=row["model_id"],
            bias_name=row["bias_name"],
            mode=row["mode"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error=row["error"],
            artifact_dir=row["artifact_dir"],
        )

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app
