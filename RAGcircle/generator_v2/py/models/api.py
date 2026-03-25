from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from generator_contract import (
    AgentRequest,
    AgentResponse,
    ChatRequest,
    ChatResponse,
    SearchFilters,
)
from models.plan import BrainRound

__all__ = [
    "SearchFilters",
    "ChatRequest",
    "ChatResponse",
    "AgentRequest",
    "AgentResponse",
    "ExecuteRequest",
    "PlanRunRequest",
    "PlanRunResponse",
]


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


# ── Dynamic plan endpoint models ─────────────────────────


class PlanRunRequest(BaseModel):
    """Run an explicit BrainPlan for testing / experimentation.

    Unlike ExecuteRequest, the brain_plan field is strongly typed:
    Pydantic validates the full plan tree at parse time, giving clear
    errors for invalid step combinations.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "project_id": "my-project",
                "query": "Compare vector search and BM25 for RAG",
                "history": [
                    {"role": "user", "content": "What is RAG?"},
                    {"role": "assistant", "content": "RAG stands for retrieval-augmented generation..."},
                ],
                "brain_plan": {
                    "configure": [{"kind": "detect_lang"}],
                    "retrieval": {
                        "default_config": {"kind": "retrieve", "preset": "hybrid", "top_k": 10, "rerank": True},
                        "initial_expand": [
                            {"kind": "hyde", "num_passages": 1},
                            {"kind": "fact_queries", "max_queries": 2},
                        ],
                        "loop_check": [{"kind": "quality_check", "min_hits": 3, "min_score": 0.5}],
                        "loop_expand": [{"kind": "two_pass", "min_unique_sources": 3}],
                        "finalize": [{"kind": "stitch", "max_per_segment": 4}],
                        "max_rounds": 2,
                    },
                    "generate": {"kind": "generate", "temperature": 0.2},
                    "evaluate": [{"kind": "assess"}],
                    "max_llm_calls": 12,
                },
                "include_sources": True,
                "include_traces": True,
            }
        },
    )

    project_id: str = Field(description="Project / collection to search")
    query: str = Field(min_length=1, max_length=2000, description="User question")
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior conversation turns as {role, content} dicts",
    )
    brain_plan: BrainRound
    include_sources: bool = Field(default=True, description="Include source metadata")
    include_traces: bool = Field(default=True, description="Include execution traces")


class PlanRunResponse(BaseModel):
    """Structured result echoing the validated plan alongside outputs."""

    brain_plan: BrainRound
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    context: list[dict[str, Any]] = Field(default_factory=list)
    traces: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = "hybrid"
    lang: str = "English"
    is_factoid: bool = False
    needs_retry: bool = False
    missing_terms: list[str] = Field(default_factory=list)
    requery: str | None = None
