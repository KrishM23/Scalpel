"""Scalpel enterprise API (v1).

Endpoints (all under /v1, authenticated via X-API-Key):

- GET  /v1/models                 supported foundation models
- GET  /v1/biases                 built-in bias benchmark catalog
- POST /v1/edit-jobs              queue a debiasing surgery (202)
- GET  /v1/edit-jobs              list this tenant's jobs
- GET  /v1/edit-jobs/{id}         job status + summary metrics
- GET  /v1/edit-jobs/{id}/report  full audit report (compliance artifact)

GET /health is unauthenticated for load balancers.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse

import scalpel
from scalpel.api.auth import parse_key_entries, require_tenant
from scalpel.api.jobs import JobRunner, JobStore
from scalpel.api.plans import plan_for_tenant
from scalpel.api.schemas import (
    BiasCatalogEntry,
    EditJobDetail,
    EditJobRequest,
    EditJobSummary,
    ModelCatalogResponse,
    ModelProbeResponse,
    UsageResponse,
)
from scalpel.biases.catalog import bias_catalog, spec_from_payload
from scalpel.config import Settings, get_settings
from scalpel.models.adapters import supported_families
from scalpel.models.registry import (
    UnsupportedArchitectureError,
    featured_models,
    probe_model,
)
from scalpel.reporting import render_report_html


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.api_keys = parse_key_entries(settings.api_keys)
        app.state.store = JobStore(settings.database_path)
        app.state.runner = JobRunner(app.state.store, settings)
        # Resume (or cleanly fail) jobs left incomplete by a prior process.
        app.state.runner.recover()
        yield
        app.state.runner.shutdown()

    app = FastAPI(
        title="Scalpel — Model Editing & Bias Mitigation API",
        version=scalpel.__version__,
        description=__doc__,
        lifespan=lifespan,
    )

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok", "version": scalpel.__version__}

    @app.get("/", include_in_schema=False)
    def console() -> HTMLResponse:
        """Web console. The page itself is public; every API call it makes
        requires the tenant's API key."""
        page = Path(__file__).parent / "static" / "index.html"
        return HTMLResponse(page.read_text())

    @app.get("/v1/models", response_model=ModelCatalogResponse, tags=["catalog"])
    def list_models(_tenant: str = Depends(require_tenant)) -> ModelCatalogResponse:
        """Featured models plus the architecture families any HF id can use.

        Scalpel is not limited to the featured list — POST /v1/edit-jobs with
        any Hugging Face model id whose ``model_type`` is in a supported family.
        """
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
        plan = plan_for_tenant(tenant, settings.tenant_plans, settings.default_plan)
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
        plan = plan_for_tenant(tenant, settings.tenant_plans, settings.default_plan)
        return UsageResponse(
            tenant=tenant,
            plan=plan.name,
            jobs_this_month=app.state.store.count_jobs_this_month(tenant),
            monthly_job_limit=plan.monthly_job_limit,
            allows_edit=plan.allows_edit,
        )

    @app.get("/v1/edit-jobs", response_model=list[EditJobSummary], tags=["editing"])
    def list_edit_jobs(tenant: str = Depends(require_tenant)) -> list[EditJobSummary]:
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

    return app
