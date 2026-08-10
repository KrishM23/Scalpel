"""Pydantic schemas for the public v1 API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomBiasSpecPayload(BaseModel):
    """Tenant-supplied bias benchmark."""

    name: str = Field(min_length=1, max_length=64)
    description: str = ""
    group_a_label: str = "group_a"
    group_b_label: str = "group_b"
    paired_prompts: list[tuple[str, str]] = Field(min_length=4)
    probe_set_1: list[str] = Field(min_length=1)
    probe_set_2: list[str] = Field(min_length=1)
    retention_prompts: list[str] = Field(min_length=4)


class SurgeryOptions(BaseModel):
    max_components: int = Field(default=12, ge=1, le=128)
    cumulative_share: float = Field(default=0.8, gt=0.0, le=1.0)
    num_directions: int = Field(default=1, ge=1, le=16)
    calibrate: bool = Field(
        default=False,
        description="Sweep erasure strength and keep the one minimizing residual |WEAT|",
    )
    harden_projection: bool = True
    edit_bias_terms: bool = True
    direction_layer: int | None = Field(default=None, ge=1)


class EditJobRequest(BaseModel):
    model_id: str = Field(examples=["openai/clip-vit-base-patch32"])
    bias: str | CustomBiasSpecPayload = "gender_profession"
    mode: Literal["edit", "audit"] = "edit"
    options: SurgeryOptions = SurgeryOptions()
    save_artifact: bool = Field(
        default=False,
        description="Persist edited weights to disk (slower). Default off for snappy API jobs.",
    )
    webhook_url: str | None = Field(
        default=None, description="POSTed a completion payload when the job finishes"
    )


class EditJobSummary(BaseModel):
    id: str
    tenant: str
    model_id: str
    bias_name: str
    mode: Literal["edit", "audit"] = "edit"
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    updated_at: str
    error: str | None = None
    artifact_dir: str | None = None


class UsageResponse(BaseModel):
    tenant: str
    plan: str
    jobs_this_month: int
    monthly_job_limit: int | None
    allows_edit: bool


class EditJobDetail(EditJobSummary):
    report: dict | None = None


class BiasCatalogEntry(BaseModel):
    name: str
    description: str
    groups: list[str]
    num_contrastive_pairs: int
    num_probes: int


class ModelCatalogEntry(BaseModel):
    model_id: str
    family: str = "clip"
    description: str = ""
    featured: bool = True


class ModelCatalogResponse(BaseModel):
    """Featured suggestions + the architecture families any HF id may use."""

    accepts_any_huggingface_id: bool = True
    families: dict
    featured: list[ModelCatalogEntry]


class ModelProbeResponse(BaseModel):
    model_id: str
    family: str
    model_type: str
    architecture_key: str
    description: str
    supported: bool = True


class AlertEntry(BaseModel):
    id: str
    job_id: str
    kind: str
    severity: str
    title: str
    detail: str
    bias_name: str
    mode: str
    created_at: str


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(default="", max_length=120)
    company: str = Field(default="", max_length=120)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    id: str
    email: str
    name: str
    company: str
    tenant: str
    api_key: str
    plan: str
    created_at: str


class PublicDemoRequest(BaseModel):
    """Kick off (or reuse) a public landing-page surgery demo."""

    bias: str = "global_language_prestige"
    model_id: str | None = Field(
        default=None,
        description="Any supported Hugging Face model id (CLIP, text encoder, or LM)",
    )
    force: bool = Field(
        default=False,
        description="Skip cache and queue a fresh surgery even if a recent report exists",
    )
    export_weights: bool = Field(
        default=False,
        description="Persist edited weights for download (slower; shareable zip)",
    )


class PublicDemoJobResponse(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    model_id: str
    bias_name: str
    mode: Literal["edit", "audit"] = "edit"
    cached: bool = False
    error: str | None = None
    share_token: str | None = None
    share_url: str | None = None
    pdf_url: str | None = None
    recipe_url: str | None = None
    artifact_url: str | None = None
    report: dict | None = None
    created_at: str | None = None
    reproduce_curl: str | None = None


class ShareLinkResponse(BaseModel):
    token: str
    share_url: str
    pdf_url: str
    recipe_url: str
    artifact_url: str | None = None
    expires_at: str | None = None
