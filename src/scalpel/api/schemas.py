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
    harden_projection: bool = True
    edit_bias_terms: bool = True
    direction_layer: int | None = Field(default=None, ge=1)


class EditJobRequest(BaseModel):
    model_id: str = Field(examples=["openai/clip-vit-base-patch32"])
    bias: str | CustomBiasSpecPayload = "gender_profession"
    options: SurgeryOptions = SurgeryOptions()
    save_artifact: bool = True


class EditJobSummary(BaseModel):
    id: str
    tenant: str
    model_id: str
    bias_name: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    updated_at: str
    error: str | None = None
    artifact_dir: str | None = None


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
    family: str
    description: str
