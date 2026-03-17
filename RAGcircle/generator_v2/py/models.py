from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Score source (mirrors retrieval service) ─────────────


class ScoreSource(StrEnum):
    RETRIEVAL = "retrieval"
    RRF = "rrf"
    RERANK = "rerank"


# ── Chunk result (mirrors retrieval service) ─────────────


class ChunkResult(BaseModel):
    text: str
    source_id: str
    chunk_index: int
    score: float
    score_source: ScoreSource = ScoreSource.RETRIEVAL


# ── Lightweight plan models (for building plans to send) ─


class BM25SearchStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["bm25_search"] = "bm25_search"
    top_k: int = Field(default=20, ge=1, le=2000)
    query: str = Field(default="")


class VectorSearchStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["vector_search"] = "vector_search"
    top_k: int = Field(default=20, ge=1, le=2000)
    query: str = Field(default="")


class FuseStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["fuse"] = "fuse"
    method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, ge=1, le=5000)


class RerankStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["rerank"] = "rerank"
    top_n: int = Field(default=8, ge=1, le=1000)
    query: str = Field(default="")


class AdaptiveKStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["adaptive_k"] = "adaptive_k"
    min_k: int = Field(default=3, ge=1, le=1000)
    max_k: int = Field(default=24, ge=1, le=1000)


class TrimStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["trim"] = "trim"
    top_k: int = Field(default=8, ge=1, le=2000)


RetrievalStep = Annotated[BM25SearchStep | VectorSearchStep, Field(discriminator="kind")]
RankStep = Annotated[RerankStep | AdaptiveKStep, Field(discriminator="kind")]
FinalizeStep = Annotated[TrimStep, Field(discriminator="kind")]


class PlanRound(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retrieve: list[RetrievalStep] = Field(default_factory=list)
    combine: FuseStep | None = None
    rank: list[RankStep] = Field(default_factory=list)
    finalize: list[FinalizeStep] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round: PlanRound


# ── Reflection result ────────────────────────────────────


class ReflectionResult(BaseModel):
    complete: bool
    missing_context: str | None = None
    requery: str | None = None


# ── Assessment result ────────────────────────────────────


class AssessmentResult(BaseModel):
    incomplete: bool = False
    missing_terms: list[str] = Field(default_factory=list)
    reason: str = ""


# ── Filters ──────────────────────────────────────────────


class SearchFilters(BaseModel):
    source: str | None = None
    tags: list[str] | None = None
    lang: str | None = None
    doc_ids: list[str] | None = None


# ── /chat request / response ─────────────────────────────


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


# ── /agent request / response ────────────────────────────


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
