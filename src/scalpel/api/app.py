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

from fastapi import Depends, FastAPI, HTTPException, status

import scalpel
from scalpel.api.auth import parse_key_entries, require_tenant
from scalpel.api.jobs import JobRunner, JobStore
from scalpel.api.schemas import (
    BiasCatalogEntry,
    EditJobDetail,
    EditJobRequest,
    EditJobSummary,
    ModelCatalogEntry,
)
from scalpel.biases.catalog import bias_catalog, spec_from_payload
from scalpel.config import Settings, get_settings
from scalpel.models.registry import supported_models


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.api_keys = parse_key_entries(settings.api_keys)
        app.state.store = JobStore(settings.database_path)
        app.state.runner = JobRunner(app.state.store, settings)
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

    @app.get("/v1/models", response_model=list[ModelCatalogEntry], tags=["catalog"])
    def list_models(_tenant: str = Depends(require_tenant)) -> list[ModelCatalogEntry]:
        return [
            ModelCatalogEntry(model_id=model_id, **meta)
            for model_id, meta in supported_models().items()
        ]

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

        if request.model_id not in supported_models():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported model '{request.model_id}'",
            )

        job_id = app.state.store.create(tenant, request.model_id, bias_name)
        app.state.runner.submit(job_id, tenant, request)
        return _summary(app.state.store.get(job_id, tenant))

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
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error=row["error"],
            artifact_dir=row["artifact_dir"],
        )

    return app
