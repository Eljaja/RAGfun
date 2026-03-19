from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    source: str | None = None
    tags: list[str] | None = None
    lang: str | None = None
    doc_ids: list[str] | None = None


class ChatRequest(BaseModel):
    project_id: str
    query: str
    preset: str = Field(default="hybrid", pattern="^(fast|hybrid|thorough|budget)$")
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = True
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
    filters: SearchFilters | None = None
    include_sources: bool = True
    top_k: int | None = None
    max_llm_calls: int | None = None
    max_fact_queries: int | None = None
    use_hyde: bool | None = None
    hyde_num: int | None = None
    use_fact_queries: bool | None = None
    use_retry: bool | None = None
    use_tools: bool | None = None
    mode: str | None = Field(None, pattern="^(minimal|conservative|aggressive)?$")


class AgentResponse(BaseModel):
    trace_id: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    context: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = "hybrid"
    partial: bool = False
    degraded: list[str] = Field(default_factory=list)


class ExecuteRequest(BaseModel):
    project_id: str
    query: str
    history: list[dict[str, str]] = Field(default_factory=list)
    brain_plan: Any  # BrainPlan — typed loosely to avoid circular import
    include_sources: bool = True
