from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class GateFilters(BaseModel):
    source: str | None = None
    tags: list[str] | None = None
    lang: str | None = None
    doc_ids: list[str] | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    # "Collections" support: allow selecting multiple project_ids at query time.
    project_ids: list[str] | None = None


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)

    # pass-through to retrieval
    retrieval_mode: Literal["bm25", "vector", "hybrid"] | None = None
    top_k: int | None = None
    rerank: bool | None = None
    filters: GateFilters | None = None
    acl: list[str] = Field(default_factory=list)

    include_sources: bool = True


class Source(BaseModel):
    ref: int | None = None
    doc_id: str
    title: str | None = None
    uri: str | None = None
    locator: dict[str, Any] | None = None


class ContextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str | None = None
    score: float
    source: Source | None = None


class ChatResponse(BaseModel):
    ok: bool = True
    answer: str
    used_mode: str
    degraded: list[str] = Field(default_factory=list)
    partial: bool = False
    context: list[ContextChunk] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    # Debug/traceability: what retrieval returned (e.g. hits/sources/partial/degraded).
    # Used by UI to show retrieval result.
    retrieval: dict[str, Any] | None = None


