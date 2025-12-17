from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    doc_id: str
    source: str | None = None
    title: str | None = None
    uri: str | None = None
    lang: str | None = None
    tags: list[str] = Field(default_factory=list)
    acl: list[str] = Field(default_factory=list)
    tenant_id: str | None = None
    project_id: str | None = None


class Locator(BaseModel):
    page: int | None = None


class ChunkMeta(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    locator: Locator | None = None
    title: str | None = None
    uri: str | None = None
    lang: str | None = None
    tags: list[str] = Field(default_factory=list)
    acl: list[str] = Field(default_factory=list)
    source: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None


class ProcessRequest(BaseModel):
    """
    Process a stored document (by doc_id) into text chunks and upsert them into retrieval.
    The processor will fetch bytes from document-storage.
    """

    document: DocumentMeta
    refresh: bool = False


class ProcessResponse(BaseModel):
    ok: bool
    doc_id: str
    content_type: str | None = None
    pages: int | None = None
    extracted_chars: int | None = None
    chunks: int | None = None
    retrieval: dict[str, Any] | None = None
    partial: bool = False
    degraded: list[str] = Field(default_factory=list)
    error: str | None = None
    detail: Any | None = None






