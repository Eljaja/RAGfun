from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Locator(BaseModel):
    page: int | None = None
    heading_path: list[str] | None = None
    anchor: str | None = None
    line_range: list[int] | None = None
    timecode: str | None = None
    extra: dict[str, Any] | None = None


class DocumentMeta(BaseModel):
    doc_id: str
    source: str | None = None
    title: str | None = None
    uri: str | None = None
    lang: str | None = None
    tags: list[str] = Field(default_factory=list)
    acl: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    version: str | None = None
    content_hash: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None


class ChunkMeta(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str

    # Optional linkage back to source
    uri: str | None = None
    title: str | None = None
    lang: str | None = None
    tags: list[str] = Field(default_factory=list)
    acl: list[str] = Field(default_factory=list)
    locator: Locator | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None
    content_hash: str | None = None
    token_count: int | None = None

    source: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None


class IndexUpsertRequest(BaseModel):
    mode: Literal["chunks", "document"] = "chunks"
    document: DocumentMeta | None = None
    chunks: list[ChunkMeta] | None = None

    # for mode=document
    text: str | None = None

    # tuning
    refresh: bool = False


class IndexUpsertResponse(BaseModel):
    ok: bool
    upserted: int = 0
    skipped_unchanged: int = 0
    partial: bool = False
    errors: list[dict[str, Any]] = Field(default_factory=list)


class IndexDeleteRequest(BaseModel):
    doc_id: str | None = None
    chunk_id: str | None = None
    refresh: bool = False


class IndexDeleteResponse(BaseModel):
    ok: bool
    deleted: int = 0
    partial: bool = False
    errors: list[dict[str, Any]] = Field(default_factory=list)


class SearchFilters(BaseModel):
    source: str | None = None
    tags: list[str] | None = None
    lang: str | None = None
    doc_ids: list[str] | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    # "Collections" support: allow selecting multiple project_ids at query time.
    # UI/backend treat project_id as "collection id".
    project_ids: list[str] | None = None


class SearchRequest(BaseModel):
    query: str
    mode: Literal["bm25", "vector", "hybrid"] = "hybrid"
    top_k: int | None = None
    include_sources: bool = False
    sources_level: Literal["none", "basic", "full"] = "basic"

    filters: SearchFilters | None = None
    acl: list[str] = Field(default_factory=list)

    group_by_doc: bool = True
    max_chunks_per_doc: int | None = None

    rerank: bool | None = None  # override config


class SourceObj(BaseModel):
    doc_id: str
    title: str | None = None
    uri: str | None = None
    locator: Locator | None = None


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    source_scores: dict[str, float] = Field(default_factory=dict)
    rerank_score: float | None = None
    text: str | None = None
    highlight: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: SourceObj | None = None


class SearchResponse(BaseModel):
    ok: bool = True
    mode: str
    partial: bool = False
    partial_rerank: bool = False
    degraded: list[str] = Field(default_factory=list)
    hits: list[SearchHit] = Field(default_factory=list)
    sources: list[SourceObj] | None = None


class IndexExistsRequest(BaseModel):
    doc_ids: list[str] = Field(default_factory=list)


class IndexExistsResponse(BaseModel):
    ok: bool = True
    indexed_doc_ids: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)

