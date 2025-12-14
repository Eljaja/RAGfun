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
    error: str | None = None


class DocumentSearchRequest(BaseModel):
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    lang: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    date_range: dict[str, datetime] | None = None  # {"from": ..., "to": ...}
    limit: int = 100
    offset: int = 0


class DocumentSearchResponse(BaseModel):
    ok: bool
    documents: list[DocumentMetadata] = Field(default_factory=list)
    total: int = 0
    error: str | None = None

