from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    doc_id: str
    storage_id: str | None = None
    title: str | None = None
    uri: str | None = None
    source: str | None = None
    lang: str | None = None
    tags: list[str] = Field(default_factory=list)
    tenant_id: str | None = None
    project_id: str | None = None
    acl: list[str] = Field(default_factory=list)
    content_type: str | None = None
    size: int | None = None
    stored_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class StoreDocumentRequest(BaseModel):
    doc_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoreDocumentResponse(BaseModel):
    ok: bool
    storage_id: str | None = None
    doc_id: str
    size: int | None = None
    content_type: str | None = None
    stored_at: datetime | None = None
    # Exact deduplication (bytes-level)
    duplicate: bool | None = None
    duplicate_of: str | None = None
    error: str | None = None


class DocumentSearchRequest(BaseModel):
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    lang: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    # "Collections" support: allow selecting multiple project_ids at query time.
    project_ids: list[str] = Field(default_factory=list)
    date_range: dict[str, datetime] | None = None  # {"from": ..., "to": ...}
    limit: int = 100
    offset: int = 0


class DocumentSearchResponse(BaseModel):
    ok: bool
    documents: list[DocumentMetadata] = Field(default_factory=list)
    total: int = 0
    error: str | None = None


class PatchExtraRequest(BaseModel):
    """
    Patch (merge) the `extra` JSON field for a document.
    Top-level keys in `patch` overwrite existing keys.
    """

    patch: dict[str, Any] = Field(default_factory=dict)


class PatchExtraByIdRequest(BaseModel):
    """Request body for /by-id/extra endpoint (doc_id in body to avoid query param issues)."""
    doc_id: str
    patch: dict[str, Any] = Field(default_factory=dict)

