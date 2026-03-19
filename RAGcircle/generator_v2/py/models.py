from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChunkResult(BaseModel):
    text: str
    source_id: str
    chunk_index: int
    score: float


class ChatRequest(BaseModel):
    project_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = True
    strategy: str = Field(default="hybrid", pattern="^(hybrid|vector|bm25)$")
    max_retries: int = Field(default=1, ge=0, le=5)
    reflection_enabled: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    retries_used: int
    query: str


class AgentRequest(BaseModel):
    project_id: str
    query: str
    history: list[dict[str, str]] = Field(default_factory=list)
    include_sources: bool = True
    top_k: int = Field(default=8, ge=1, le=50)
    rerank: bool = True
    strategy: str = Field(default="hybrid", pattern="^(hybrid|vector|bm25)$")
    mode: str | None = Field(default=None, pattern="^(minimal|conservative|aggressive)?$")


class AgentResponse(BaseModel):
    trace_id: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    context: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = "hybrid"
    partial: bool = False
    degraded: list[str] = Field(default_factory=list)
