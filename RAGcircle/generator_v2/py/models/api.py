from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SearchFilters(BaseModel):
    """Optional filters narrowing the retrieval scope."""
    model_config = ConfigDict(json_schema_extra={"example": {"lang": "en"}})

    source: str | None = Field(None, description="Restrict to a specific source document ID")
    tags: list[str] | None = Field(None, description="Filter by document tags")
    lang: str | None = Field(None, description="Filter by language code")
    doc_ids: list[str] | None = Field(None, description="Restrict to specific document IDs")


class ChatRequest(BaseModel):
    """Simple synchronous chat — retrieve, generate, optionally reflect."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "project_id": "my-project",
                "query": "What is retrieval-augmented generation?",
            }
        }
    )

    project_id: str = Field(description="Project / collection to search")
    query: str = Field(description="User question")
    preset: Literal["fast", "hybrid", "thorough", "budget"] = Field(
        default="hybrid", description="Retrieval strategy preset",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")
    rerank: bool = Field(default=True, description="Whether to rerank results")
    max_retries: int = Field(default=1, ge=0, le=5, description="Max reflect-retry rounds")
    reflection_enabled: bool = Field(default=True, description="Enable reflection step")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    retries_used: int
    query: str


class AgentRequest(BaseModel):
    """Full agent pipeline — plan, expand, retrieve, generate, evaluate.

    Only `project_id` and `query` are required. Everything else has
    sensible server-side defaults. Pass `mode` for a named preset,
    or set individual knobs to override.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "project_id": "my-project",
                "query": "Compare vector search and BM25 for RAG",
                "history": [
                    {"role": "user", "content": "What is RAG?"},
                    {"role": "assistant", "content": "RAG stands for retrieval-augmented generation..."},
                ],
            }
        }
    )

    project_id: str = Field(description="Project / collection to search")
    query: str = Field(description="User question")
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior conversation turns as {role, content} dicts",
    )
    filters: SearchFilters | None = Field(None, description="Optional retrieval filters")
    include_sources: bool = Field(default=True, description="Include source metadata in response")
    mode: Literal["minimal", "conservative", "aggressive"] | None = Field(
        None, description="Named preset — overrides individual knobs below",
    )
    top_k: int | None = Field(None, ge=1, le=50, description="Override: chunks to retrieve")
    max_llm_calls: int | None = Field(None, ge=1, le=100, description="Override: LLM call budget")
    max_fact_queries: int | None = Field(None, ge=0, le=10, description="Override: fact sub-queries")
    use_hyde: bool | None = Field(None, description="Override: enable HyDE expansion")
    use_fact_queries: bool | None = Field(None, description="Override: enable fact query split")
    use_retry: bool | None = Field(None, description="Override: enable assess+retry loop")
    use_tools: bool | None = Field(None, description="Override: enable calculator tool")


class AgentResponse(BaseModel):
    trace_id: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    context: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = "hybrid"
    partial: bool = False
    degraded: list[str] = Field(default_factory=list)


class ExecuteRequest(BaseModel):
    """Execute a raw BrainPlan (advanced). Streams SSE events."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "project_id": "my-project",
                "query": "What is the capital of France?",
            }
        }
    )

    project_id: str = Field(description="Project / collection to search")
    query: str = Field(description="User question")
    history: list[dict[str, str]] = Field(default_factory=list)
    brain_plan: Any = Field(description="Full BrainPlan JSON")
    include_sources: bool = True
