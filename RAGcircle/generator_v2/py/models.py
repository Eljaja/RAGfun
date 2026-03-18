from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


# ── Brain plan step types ────────────────────────────────


class PlanLLMStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["plan_retrieval"] = "plan_retrieval"


class DetectLangStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["detect_lang"] = "detect_lang"


class HyDEStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["hyde"] = "hyde"
    num_passages: int = Field(default=1, ge=1, le=7)


class FactQueryStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["fact_queries"] = "fact_queries"
    max_queries: int = Field(default=2, ge=1, le=10)


class KeywordStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["keywords"] = "keywords"


class BrainRetrieveStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["retrieve"] = "retrieve"
    preset: str = Field(default="hybrid", pattern="^(fast|hybrid|thorough|budget)$")
    top_k: int = Field(default=10, ge=1, le=50)
    rerank: bool = True


class QualityCheckStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["quality_check"] = "quality_check"
    min_hits: int = Field(default=3, ge=1, le=50)
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)


class GenerateStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["generate"] = "generate"
    use_tools: bool = False
    stream: bool = True
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class ReflectStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["reflect"] = "reflect"


class AssessStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["assess"] = "assess"


class QueryVariantsStep(BaseModel):
    """Heuristic multi-query: keyword, quoted-phrase, year-based variants."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["query_variants"] = "query_variants"


class TwoPassStep(BaseModel):
    """Conditional re-retrieval when unique source count is low."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["two_pass"] = "two_pass"
    min_unique_sources: int = Field(default=3, ge=1, le=20)


class BM25AnchorStep(BaseModel):
    """BM25-only keyword search merged with main results via RRF."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["bm25_anchor"] = "bm25_anchor"
    top_k: int = Field(default=15, ge=1, le=100)


class FactoidExpandStep(BaseModel):
    """Factoid pre-expansion: detect factoid and expand within-doc context."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["factoid_expand"] = "factoid_expand"


class FactoidRetryStep(BaseModel):
    """Factoid post-generation grounding check + conditional re-generation."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["factoid_retry"] = "factoid_retry"


class StitchStep(BaseModel):
    """Segment stitching: combine adjacent chunks from the same document."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["stitch"] = "stitch"
    max_per_segment: int = Field(default=4, ge=1, le=10)


ExpandStep = Annotated[
    PlanLLMStep | DetectLangStep | HyDEStep | FactQueryStep | KeywordStep | QueryVariantsStep,
    Field(discriminator="kind"),
]
PostRetrieveStep = Annotated[
    QualityCheckStep | TwoPassStep | BM25AnchorStep | FactoidExpandStep | StitchStep,
    Field(discriminator="kind"),
]
EvalStep = Annotated[
    ReflectStep | AssessStep | FactoidRetryStep,
    Field(discriminator="kind"),
]


# ── Brain round and plan ─────────────────────────────────


class BrainRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expand: list[ExpandStep] = Field(default_factory=list)
    retrieve: BrainRetrieveStep = Field(default_factory=BrainRetrieveStep)
    post_retrieve: list[PostRetrieveStep] = Field(default_factory=list)
    generate: GenerateStep = Field(default_factory=GenerateStep)
    evaluate: list[EvalStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_structure(self):
        expand_kinds = [s.kind for s in self.expand]
        if expand_kinds.count("plan_retrieval") > 1:
            raise ValueError("at most one plan_retrieval step per round")
        if expand_kinds.count("detect_lang") > 1:
            raise ValueError("at most one detect_lang step per round")
        if expand_kinds.count("query_variants") > 1:
            raise ValueError("at most one query_variants step per round")
        eval_kinds = [s.kind for s in self.evaluate]
        if "reflect" in eval_kinds and "assess" in eval_kinds:
            raise ValueError("reflect and assess are mutually exclusive in a single round")
        post_kinds = [s.kind for s in self.post_retrieve]
        if post_kinds.count("stitch") > 1:
            raise ValueError("at most one stitch step per round")
        return self


class BrainPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rounds: list[BrainRound] = Field(min_length=1)
    max_llm_calls: int = Field(default=12, ge=1, le=100)


# ── /execute request ─────────────────────────────────────


class ExecuteRequest(BaseModel):
    project_id: str
    query: str
    history: list[dict[str, str]] = Field(default_factory=list)
    brain_plan: BrainPlan
    include_sources: bool = True
